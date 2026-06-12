# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import torch

from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.conv._utils import BLOCK_K, _out_dhw
from aiter.ops.triton.conv._prepack import get_or_make_weight_pack_3d
from aiter.ops.triton.conv._launch import (
    _launch_conv3d_general,
    _select_conv3d_method,
    specialized_enabled,
)

_LOGGER = AiterTritonLogger()

# Tracks the last Triton kernel selected by conv3d routing (read by the bench
# to label per-layer rows). Mirrors conv2d._last_triton_kernel.
_last_triton_kernel: Optional[str] = None


def conv3d(
    x,
    w_oidhw,
    bias=None,
    stride=(1, 1, 1),
    padding=(0, 0, 0),
    dilation=(1, 1, 1),
    activation="none",
    layout="ncdhw",
):
    """Forward 3-D conv on AMD ROCm via Triton. Drop-in for the forward of
    ``torch.nn.functional.conv3d`` (no backward).

    A shape-driven router picks among the conv3d kernel families. Phase 1 ships
    the general (implicit-GEMM) kernel, which covers every kernel size / stride /
    dilation / padding; 1x1x1 and 3x3x3 specializations route through the same
    entry point as they land.

    Inputs must be fp16 or bf16. The output matches the input dtype (like
    ``torch.nn.Conv3d`` — there is no dtype override). ``layout="ndhwc"`` runs an
    NDHWC-native kernel (no NCDHW round-trip).

    Notes
    -----
    - Only ``groups=1`` (depthwise/grouped raises ``AssertionError`` downstream).
    - Only ``padding_mode="zeros"``.
    - ``stride``/``padding``/``dilation`` are length-3 tuples (d, h, w).
    """
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"conv3d only supports fp16 and bf16 inputs, got {x.dtype}")
    layout = layout.lower()
    if layout not in ("ncdhw", "ndhwc"):
        raise ValueError(f"layout must be 'ncdhw' or 'ndhwc', got '{layout}'")

    _LOGGER.info(
        f"CONV3D: x={tuple(x.shape)} w={tuple(w_oidhw.shape)} stride={stride} "
        f"padding={padding} dilation={dilation} layout={layout} "
        f"dtype={x.dtype} bias={'yes' if bias is not None else 'no'} "
        f"act={activation}"
    )

    if layout == "ndhwc":
        return conv3d_ndhwc(x, w_oidhw, bias, stride, padding, dilation, activation)
    else:
        return conv3d_ncdhw(x, w_oidhw, bias, stride, padding, dilation, activation)


def conv3d_general(
    x,
    w_oidhw,
    bias=None,
    stride=(1, 1, 1),
    padding=(0, 0, 0),
    dilation=(1, 1, 1),
    activation="none",
    block_k=BLOCK_K,
    layout="ncdhw",
):
    """NCDHW/NDHWC conv3d using the general kernel with K-major prepacked weights.
    Output dtype always matches the input."""
    assert x.is_cuda and w_oidhw.is_cuda
    N, C, D, H, W_in = x.shape
    K_out, Cw, KD, KH, KW = w_oidhw.shape
    assert Cw == C, f"weight C ({Cw}) != input C ({C})"
    D_out, P, Q = _out_dhw(D, H, W_in, KD, KH, KW, stride, padding, dilation)

    if layout == "ndhwc":
        y = torch.empty((N, K_out, D_out, P, Q), device=x.device, dtype=x.dtype).to(
            memory_format=torch.channels_last_3d
        )
    else:
        y = torch.empty((N, K_out, D_out, P, Q), device=x.device, dtype=x.dtype)
    bias_fp32 = bias.float().contiguous() if bias is not None else None
    w_k, (_, K_pad) = get_or_make_weight_pack_3d(w_oidhw.contiguous(), block_k)
    _launch_conv3d_general(
        x,
        w_k,
        bias_fp32,
        y,
        N,
        C,
        D,
        H,
        W_in,
        K_out,
        KD,
        KH,
        KW,
        D_out,
        P,
        Q,
        K_pad,
        stride,
        padding,
        dilation,
        block_k,
        activation,
        layout=layout,
    )
    return y


def conv3d_1x1x1(*args, **kwargs):
    """Specialized 1x1x1 GEMM kernel — Phase 2. Not yet implemented."""
    raise NotImplementedError(
        "conv3d_1x1x1 lands in Phase 2; the router currently falls back to "
        "conv3d_general for 1x1x1 shapes."
    )


def conv3d_ncdhw_cblocked(*args, **kwargs):
    """Specialized NCDHWc 3x3x3 kernel — Phase 3. Not yet implemented."""
    raise NotImplementedError(
        "conv3d 3x3x3 cblocked lands in Phase 3; the router currently falls "
        "back to conv3d_general for 3x3x3 shapes."
    )


def conv3d_ndhwc_3x3x3(*args, **kwargs):
    """Specialized NDHWC 3x3x3 kernel — Phase 3. Not yet implemented."""
    raise NotImplementedError(
        "conv3d 3x3x3 NDHWC lands in Phase 3; the router currently falls back "
        "to conv3d_general for 3x3x3 shapes."
    )


def conv3d_ncdhw(
    x,
    w_oidhw,
    bias=None,
    stride=(1, 1, 1),
    padding=(0, 0, 0),
    dilation=(1, 1, 1),
    activation="none",
    block_k=BLOCK_K,
):
    """Hybrid NCDHW conv3d: routes to the best available kernel for the shape.
    Output dtype always matches the input."""
    assert x.is_cuda and w_oidhw.is_cuda
    N, C, D, H, W_in = x.shape
    K_out, Cw, KD, KH, KW = w_oidhw.shape
    assert Cw == C, f"weight C ({Cw}) != input C ({C})"

    global _last_triton_kernel
    method = _select_conv3d_method(
        N, C, D, H, W_in, K_out, KD, KH, KW, stride, dilation
    )
    # Phase 1: specialized kernels not built yet; collapse to general unless the
    # feature flag is on (and even then they raise NotImplementedError). This
    # keeps routing live without breaking real calls.
    if method != "general" and not specialized_enabled():
        method = "general"

    if method == "1x1x1":
        _last_triton_kernel = "_conv3d_1x1x1_kernel"
        return conv3d_1x1x1(
            x, w_oidhw, bias, stride, padding, dilation, activation, block_k
        )
    elif method == "3x3x3":
        _last_triton_kernel = "_conv3d_3x3x3_cblocked_kernel"
        return conv3d_ncdhw_cblocked(
            x, w_oidhw, bias, stride, padding, dilation, activation, block_k
        )
    else:
        _last_triton_kernel = "_conv3d_general_kernel"
        return conv3d_general(
            x,
            w_oidhw,
            bias,
            stride,
            padding,
            dilation,
            activation,
            block_k,
            layout="ncdhw",
        )


def conv3d_ndhwc(
    x,
    w_oidhw,
    bias=None,
    stride=(1, 1, 1),
    padding=(0, 0, 0),
    dilation=(1, 1, 1),
    activation="none",
    block_k=BLOCK_K,
):
    """Conv3d with NDHWC (channels-last-3d) input and output.

    Input x can be NCDHW or NDHWC — it is converted to channels_last_3d. Output
    is allocated channels_last_3d and returned in logical NCDHW shape with
    channels_last_3d strides. Output dtype always matches the input.
    """
    assert x.is_cuda and w_oidhw.is_cuda
    x = x.to(memory_format=torch.channels_last_3d)
    N, C, D, H, W_in = x.shape
    K_out, Cw, KD, KH, KW = w_oidhw.shape
    assert Cw == C, f"weight C ({Cw}) != input C ({C})"

    global _last_triton_kernel
    method = _select_conv3d_method(
        N, C, D, H, W_in, K_out, KD, KH, KW, stride, dilation
    )
    if method != "general" and not specialized_enabled():
        method = "general"

    if method == "1x1x1":
        _last_triton_kernel = "_conv3d_1x1x1_kernel"
        return conv3d_1x1x1(
            x, w_oidhw, bias, stride, padding, dilation, activation, block_k
        )
    elif method == "3x3x3":
        _last_triton_kernel = "_conv3d_3x3x3_ndhwc_kernel"
        return conv3d_ndhwc_3x3x3(
            x, w_oidhw, bias, stride, padding, dilation, activation, block_k
        )
    else:
        _last_triton_kernel = "_conv3d_general_kernel"
        return conv3d_general(
            x,
            w_oidhw,
            bias,
            stride,
            padding,
            dilation,
            activation,
            block_k,
            layout="ndhwc",
        )
