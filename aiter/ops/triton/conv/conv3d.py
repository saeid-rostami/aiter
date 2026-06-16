# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import torch

from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.conv._utils import (
    BLOCK_K,
    _out_dhw,
    _is_1x1x1_conv,
    _is_3x3x3_conv,
    _is_winograd3d_eligible,
)
from aiter.ops.triton.conv._prepack import (
    get_or_make_weight_pack_3d,
    get_or_make_weight_pack_3x3x3,
    get_or_make_input_pack_cblocked_3d,
    get_or_make_winograd3d_filter_f2x3,
)
from aiter.ops.triton.conv._launch import (
    _launch_conv3d_general,
    _launch_conv3d_1x1x1,
    _launch_conv3d_3x3x3_ndhwc,
    _launch_conv3d_3x3x3_cblocked,
    _launch_winograd3d_f2x3,
    _launch_winograd3d_f2x3_cblocked,
    _select_conv3d_method,
    conv3d_method_implemented,
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
        y = torch.empty(
            (N, K_out, D_out, P, Q),
            device=x.device,
            dtype=x.dtype,
            memory_format=torch.channels_last_3d,
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


def conv3d_1x1x1(
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
    """NCDHW/NDHWC conv3d for 1x1x1 kernels (pure channel-reduction GEMM).
    Output dtype always matches the input. Raises ValueError for non-1x1x1."""
    assert x.is_cuda and w_oidhw.is_cuda
    N, C, D, H, W_in = x.shape
    K_out, Cw, KD, KH, KW = w_oidhw.shape
    assert Cw == C, f"weight C ({Cw}) != input C ({C})"
    if not _is_1x1x1_conv(KD, KH, KW, dilation):
        raise ValueError(
            f"conv3d_1x1x1 requires a 1x1x1 kernel with dilation=(1,1,1), "
            f"got {KD}x{KH}x{KW} dilation={dilation}"
        )
    D_out, P, Q = _out_dhw(D, H, W_in, KD, KH, KW, stride, padding, dilation)

    if layout == "ndhwc":
        # Allocate channels-last directly — avoids a per-call reorder of the
        # (large) output that .to(channels_last_3d) on a fresh tensor would do.
        y = torch.empty(
            (N, K_out, D_out, P, Q),
            device=x.device,
            dtype=x.dtype,
            memory_format=torch.channels_last_3d,
        )
    else:
        y = torch.empty((N, K_out, D_out, P, Q), device=x.device, dtype=x.dtype)
    bias_fp32 = bias.float().contiguous() if bias is not None else None
    _launch_conv3d_1x1x1(
        x,
        w_oidhw.contiguous(),
        bias_fp32,
        y,
        N,
        C,
        D,
        H,
        W_in,
        K_out,
        D_out,
        P,
        Q,
        stride,
        padding,
        activation,
        layout=layout,
    )
    return y


def conv3d_winograd_f2x3(
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
    """NCDHW/NDHWC conv3d via 3D Winograd F(2x2x2, 3x3x3) (experimental).
    Reads the input layout directly (no channel-block repack). Output dtype matches
    the input. Raises ValueError for non-eligible (non-3x3x3 / strided / dilated)."""
    assert x.is_cuda and w_oidhw.is_cuda
    if layout == "ndhwc":
        x = x.to(memory_format=torch.channels_last_3d)
    N, C, D, H, W_in = x.shape
    K_out, Cw, KD, KH, KW = w_oidhw.shape
    assert Cw == C, f"weight C ({Cw}) != input C ({C})"
    if not _is_winograd3d_eligible(KD, KH, KW, stride, dilation, C):
        raise ValueError(
            "conv3d_winograd_f2x3 requires a 3x3x3 kernel with stride=1, dilation=1, "
            f"C>=4, got {KD}x{KH}x{KW} stride={stride} dilation={dilation} C={C}"
        )
    D_out, P, Q = _out_dhw(D, H, W_in, KD, KH, KW, stride, padding, dilation)

    if layout == "ndhwc":
        y = torch.empty(
            (N, K_out, D_out, P, Q),
            device=x.device,
            dtype=x.dtype,
            memory_format=torch.channels_last_3d,
        )
    else:
        y = torch.empty((N, K_out, D_out, P, Q), device=x.device, dtype=x.dtype)
    bias_fp32 = bias.float().contiguous() if bias is not None else None
    U, (_, C_pad) = get_or_make_winograd3d_filter_f2x3(w_oidhw.contiguous(), block_k)
    _launch_winograd3d_f2x3(
        x, U, bias_fp32, y,
        N, C, D, H, W_in, K_out, D_out, P, Q, C_pad,
        padding, activation, layout=layout,
    )
    return y


def conv3d_winograd_f2x3_cblocked(
    x,
    w_oidhw,
    bias=None,
    stride=(1, 1, 1),
    padding=(0, 0, 0),
    dilation=(1, 1, 1),
    activation="none",
    block_k=BLOCK_K,
):
    """NCDHW conv3d via 3D Winograd F(2x2x2,3x3x3) with NCDHWc input packing for
    coalesced channel loads; writes NCDHW-contiguous output (experimental)."""
    assert x.is_cuda and w_oidhw.is_cuda
    N, C, D, H, W_in = x.shape
    K_out, Cw, KD, KH, KW = w_oidhw.shape
    assert Cw == C, f"weight C ({Cw}) != input C ({C})"
    if not _is_winograd3d_eligible(KD, KH, KW, stride, dilation, C):
        raise ValueError(
            "conv3d_winograd_f2x3_cblocked requires a 3x3x3 kernel with stride=1, "
            f"dilation=1, C>=4, got {KD}x{KH}x{KW} stride={stride} dilation={dilation}"
        )
    D_out, P, Q = _out_dhw(D, H, W_in, KD, KH, KW, stride, padding, dilation)

    y = torch.empty((N, K_out, D_out, P, Q), device=x.device, dtype=x.dtype)
    bias_fp32 = bias.float().contiguous() if bias is not None else None
    U, (_, C_pad) = get_or_make_winograd3d_filter_f2x3(w_oidhw.contiguous(), block_k)
    Cb = block_k
    x_blocked, C_pad_x = get_or_make_input_pack_cblocked_3d(x, Cb)
    assert C_pad_x == C_pad, f"channel padding mismatch: input {C_pad_x} vs weight {C_pad}"
    _launch_winograd3d_f2x3_cblocked(
        x_blocked, Cb, U, bias_fp32, y,
        N, C, D, H, W_in, K_out, D_out, P, Q, C_pad,
        padding, activation,
    )
    return y


def conv3d_ncdhw_cblocked(
    x,
    w_oidhw,
    bias=None,
    stride=(1, 1, 1),
    padding=(0, 0, 0),
    dilation=(1, 1, 1),
    activation="none",
    block_k=BLOCK_K,
):
    """NCDHW conv3d for 3x3x3 kernels via channel-blocked (NCDHWc) input packing.
    Repacks the NCDHW input to NCDHWc for coalesced channel loads and writes
    NCDHW-contiguous output. Output dtype matches the input. Raises ValueError for
    non-3x3x3."""
    assert x.is_cuda and w_oidhw.is_cuda
    N, C, D, H, W_in = x.shape
    K_out, Cw, KD, KH, KW = w_oidhw.shape
    assert Cw == C, f"weight C ({Cw}) != input C ({C})"
    if not _is_3x3x3_conv(KD, KH, KW):
        raise ValueError(
            f"conv3d_ncdhw_cblocked requires a 3x3x3 kernel, got {KD}x{KH}x{KW}"
        )
    D_out, P, Q = _out_dhw(D, H, W_in, KD, KH, KW, stride, padding, dilation)

    y = torch.empty((N, K_out, D_out, P, Q), device=x.device, dtype=x.dtype)
    bias_fp32 = bias.float().contiguous() if bias is not None else None
    w_3x3x3, (_, C_pad) = get_or_make_weight_pack_3x3x3(w_oidhw.contiguous(), block_k)
    Cb = block_k  # channel-block size matches the weight channel padding block
    x_blocked, C_pad_x = get_or_make_input_pack_cblocked_3d(x, Cb)
    assert C_pad_x == C_pad, f"channel padding mismatch: input {C_pad_x} vs weight {C_pad}"
    _launch_conv3d_3x3x3_cblocked(
        x_blocked,
        w_3x3x3,
        bias_fp32,
        y,
        N,
        C,
        D,
        H,
        W_in,
        K_out,
        D_out,
        P,
        Q,
        C_pad,
        Cb,
        stride,
        padding,
        dilation,
        activation,
    )
    return y


def conv3d_ndhwc_3x3x3(
    x,
    w_oidhw,
    bias=None,
    stride=(1, 1, 1),
    padding=(0, 0, 0),
    dilation=(1, 1, 1),
    activation="none",
    block_k=BLOCK_K,
):
    """NDHWC conv3d for 3x3x3 kernels (direct, channels-contiguous, no input repack).
    Input must already be channels_last_3d. Output dtype matches the input.
    Raises ValueError for non-3x3x3."""
    assert x.is_cuda and w_oidhw.is_cuda
    N, C, D, H, W_in = x.shape
    K_out, Cw, KD, KH, KW = w_oidhw.shape
    assert Cw == C, f"weight C ({Cw}) != input C ({C})"
    if not _is_3x3x3_conv(KD, KH, KW):
        raise ValueError(f"conv3d_ndhwc_3x3x3 requires a 3x3x3 kernel, got {KD}x{KH}x{KW}")
    D_out, P, Q = _out_dhw(D, H, W_in, KD, KH, KW, stride, padding, dilation)

    y = torch.empty(
        (N, K_out, D_out, P, Q),
        device=x.device,
        dtype=x.dtype,
        memory_format=torch.channels_last_3d,
    )
    bias_fp32 = bias.float().contiguous() if bias is not None else None
    w_3x3x3, (_, C_pad) = get_or_make_weight_pack_3x3x3(w_oidhw.contiguous(), block_k)
    _launch_conv3d_3x3x3_ndhwc(
        x,
        w_3x3x3,
        bias_fp32,
        y,
        N,
        C,
        D,
        H,
        W_in,
        K_out,
        D_out,
        P,
        Q,
        C_pad,
        stride,
        padding,
        dilation,
        activation,
    )
    return y


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
    # A specialized method whose kernel isn't built yet (for this layout) collapses
    # to general. NCDHW 3x3x3 has no direct kernel yet → falls back to general.
    if not conv3d_method_implemented(method, "ncdhw"):
        method = "general"

    if method == "1x1x1":
        _last_triton_kernel = "_conv3d_1x1x1_kernel"
        return conv3d_1x1x1(
            x, w_oidhw, bias, stride, padding, dilation, activation, block_k,
            layout="ncdhw",
        )
    elif method == "winograd_f2x3":
        _last_triton_kernel = "_winograd3d_f2x3_cblocked_* (fused: input+gemm/output)"
        return conv3d_winograd_f2x3_cblocked(
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
    if not conv3d_method_implemented(method, "ndhwc"):
        method = "general"

    if method == "1x1x1":
        _last_triton_kernel = "_conv3d_1x1x1_kernel"
        return conv3d_1x1x1(
            x, w_oidhw, bias, stride, padding, dilation, activation, block_k,
            layout="ndhwc",
        )
    elif method == "winograd_f2x3":
        _last_triton_kernel = "_winograd3d_f2x3_* ndhwc (fused: input+gemm/output)"
        return conv3d_winograd_f2x3(
            x, w_oidhw, bias, stride, padding, dilation, activation, block_k,
            layout="ndhwc",
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
