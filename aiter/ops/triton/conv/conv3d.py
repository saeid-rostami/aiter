# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Forward 3-D convolution on AMD ROCm via Triton.

A shape-driven router picks between a specialized 1x1x1 kernel (pure channel
GEMM) and an im2col-free "general" GEMM kernel that handles any other 3D conv
(any kernel size / stride / padding / dilation). Both run in NCDHW and NDHWC
layouts. This mirrors the conv2d ``general`` / ``1x1`` split and is the seed for
further specialization (3x3x3 direct / cblocked / Winograd-3D). See
``conv/DESIGN.md`` for the 2D design this mirrors.
"""

from enum import Enum

import torch

from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.conv._utils import (
    BLOCK_K,
    _conv3d_dims,
    _alloc_output_3d,
    _prep_bias,
    _is_1x1x1_conv,
    _is_3x3x3_conv,
    _is_winograd3d_hw_eligible,
)
from aiter.ops.triton.conv._prepack import (
    get_or_make_weight_pack_3d,
    get_or_make_weight_pack_3x3x3,
    prepack_ncdhw_to_cblocked,
    get_or_make_winograd_hw_filter_f4x3,
)
from aiter.ops.triton.conv._launch import (
    _launch_general_3d,
    _launch_1x1x1_3d,
    _launch_3x3x3_ndhwc,
    _launch_3x3x3_cblocked,
    _launch_winograd_hw_f4x3,
)

_LOGGER = AiterTritonLogger()


# NCDHW 3x3x3 shapes with at least this many channels take the (faster but
# lower-precision) 2.5D Winograd path; below it, the direct cblocked kernel wins
# and is numerically tighter. Heuristic from the RDNA4 sweep — retune if needed.
WINOGRAD_HW_MIN_C = 32


class Route3D(Enum):
    # Values are kernel display names (parallel to conv2d's Route enum).
    ONE_X_ONE_X_ONE = "_conv3d_1x1x1_kernel"
    WINOGRAD_HW = "_winograd_hw_f4x3_* (3 kernels)"
    CBLOCKED_NCDHW = "_conv3d_3x3x3_cblocked_kernel"
    NDHWC_3X3X3 = "_conv3d_3x3x3_ndhwc_kernel"
    GENERAL = "_conv3d_general_kernel"


def _resolve_route(T, R, S, stride, dilation, C, layout):
    """Single source of dispatch for conv3d (parallel to conv2d._resolve_route).

    1x1x1 -> specialized channel-GEMM kernel. 3x3x3 -> specialized kernel picked
    by layout: NCDHW uses 2.5D Winograd when eligible (unit stride/dilation,
    C >= threshold), else the cblocked input repack; NDHWC uses the
    channels-last kernel. Everything else -> the im2col-free general kernel.
    """
    if _is_1x1x1_conv(T, R, S, dilation):
        return Route3D.ONE_X_ONE_X_ONE
    if _is_3x3x3_conv(T, R, S) and dilation == (1, 1, 1):
        if layout == "ndhwc":
            return Route3D.NDHWC_3X3X3
        if (
            _is_winograd3d_hw_eligible(T, R, S, stride, dilation, C)
            and C >= WINOGRAD_HW_MIN_C
        ):
            return Route3D.WINOGRAD_HW
        return Route3D.CBLOCKED_NCDHW
    return Route3D.GENERAL


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

    Parameters
    ----------
    x : Tensor
        Input with logical shape ``[N, C, D, H, W]``, fp16 or bf16. For
        ``layout="ndhwc"`` it may carry channels-last-3d strides (or be
        converted to them internally).
    w_oidhw : Tensor
        Weight in PyTorch-canonical ``[K_out, C, T, R, S]`` layout
        (T = depth tap).
    bias : Tensor, optional
        1-D bias of length ``K_out``, cast to fp32 once at entry.
    stride, padding, dilation : 3-tuple of int
        Standard ``Conv3d`` semantics (depth, height, width).
    activation : str
        ``"none" / "relu" / "relu6" / "gelu"`` — fused into the epilogue.
    layout : str
        ``"ncdhw"`` (channels-first) or ``"ndhwc"`` (channels-last-3d).
        Case-insensitive. NDHWC runs a channels-last-3d kernel with no internal
        layout conversion of the compute; the output matches the input layout.

    Notes
    -----
    - Output dtype always matches the input dtype (fp32 accumulator downcast
      at store), mirroring ``torch.nn.Conv3d``.
    - Only ``groups=1``; only ``padding_mode="zeros"``.
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
    """conv3d using the general im2col-free kernel with K-major prepacked
    weights. Handles any kernel size / stride / padding / dilation in either
    layout. ``x`` is trusted to already carry the physical strides implied by
    ``layout`` (the ``conv3d_ncdhw`` / ``conv3d_ndhwc`` wrappers normalize it)."""
    N, C, D, H, W_in, K_out, T, R, S, OD, P, Q = _conv3d_dims(
        x, w_oidhw, stride, padding, dilation
    )

    y = _alloc_output_3d(N, K_out, OD, P, Q, x, layout)
    bias_fp32 = _prep_bias(bias)
    w_k, K_pad = get_or_make_weight_pack_3d(w_oidhw.contiguous(), block_k)
    _launch_general_3d(
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
        T,
        R,
        S,
        OD,
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
    """conv3d for 1x1x1 kernels — a pure channel-reduction GEMM. Raises
    ValueError for non-1x1x1. ``x`` is trusted to carry the physical strides
    implied by ``layout`` (the ncdhw/ndhwc wrappers normalize it)."""
    N, C, D, H, W_in, K_out, T, R, S, OD, P, Q = _conv3d_dims(
        x, w_oidhw, stride, padding, dilation
    )
    if not _is_1x1x1_conv(T, R, S, dilation):
        raise ValueError(
            f"conv3d_1x1x1 requires 1x1x1 kernel with dilation=1, "
            f"got {T}x{R}x{S} dilation={dilation}"
        )

    y = _alloc_output_3d(N, K_out, OD, P, Q, x, layout)
    bias_fp32 = _prep_bias(bias)
    _launch_1x1x1_3d(
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
        OD,
        P,
        Q,
        stride,
        padding,
        activation,
        layout=layout,
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
    """NDHWC-native 3x3x3 conv3d (channels-last-3d). Raises ValueError for
    non-3x3x3. ``x`` is trusted to carry channels_last_3d strides."""
    N, C, D, H, W_in, K_out, T, R, S, OD, P, Q = _conv3d_dims(
        x, w_oidhw, stride, padding, dilation
    )
    if not _is_3x3x3_conv(T, R, S):
        raise ValueError(f"conv3d_ndhwc_3x3x3 requires 3x3x3 kernel, got {T}x{R}x{S}")

    y = _alloc_output_3d(N, K_out, OD, P, Q, x, "ndhwc")
    bias_fp32 = _prep_bias(bias)
    w_3x3x3, C_pad = get_or_make_weight_pack_3x3x3(w_oidhw.contiguous(), block_k)
    _launch_3x3x3_ndhwc(
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
        OD,
        P,
        Q,
        C_pad,
        stride,
        padding,
        dilation,
        activation,
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
    x_blocked=None,
):
    """NCDHW 3x3x3 conv3d with channel-blocked (NCDHWc) input packing for
    coalesced channel loads. Raises ValueError for non-3x3x3.

    x_blocked: optional pre-packed NCDHWc input (used by the benchmark to time
    the kernel without host-side packing); when None the input is packed here."""
    N, C, D, H, W_in, K_out, T, R, S, OD, P, Q = _conv3d_dims(
        x, w_oidhw, stride, padding, dilation
    )
    if not _is_3x3x3_conv(T, R, S):
        raise ValueError(
            f"conv3d_ncdhw_cblocked requires 3x3x3 kernel, got {T}x{R}x{S}"
        )

    y = _alloc_output_3d(N, K_out, OD, P, Q, x, "ncdhw")
    bias_fp32 = _prep_bias(bias)
    w_3x3x3, C_pad = get_or_make_weight_pack_3x3x3(w_oidhw.contiguous(), block_k)
    if x_blocked is None:
        x_blocked, C_pad_x = prepack_ncdhw_to_cblocked(x, block_k)
    else:
        C_pad_x = x_blocked.shape[-1] * x_blocked.shape[1]
    assert (
        C_pad_x == C_pad
    ), f"Channel padding mismatch: input {C_pad_x} vs weight {C_pad}"
    _launch_3x3x3_cblocked(
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
        OD,
        P,
        Q,
        C_pad,
        block_k,
        stride,
        padding,
        dilation,
        activation,
    )
    return y


def conv3d_winograd_hw_f4x3(
    x,
    w_oidhw,
    bias=None,
    stride=(1, 1, 1),
    padding=(0, 0, 0),
    dilation=(1, 1, 1),
    activation="none",
    block_k=BLOCK_K,
):
    """NCDHW 3x3x3 conv3d via 2.5D Winograd (F(4x4,3x3) on H,W + direct depth).
    Raises ValueError for non-eligible convs (needs 3x3x3, stride=1, dilation=1,
    C>=4)."""
    N, C, D, H, W_in, K_out, T, R, S, OD, P, Q = _conv3d_dims(
        x, w_oidhw, stride, padding, dilation
    )
    if not _is_winograd3d_hw_eligible(T, R, S, stride, dilation, C):
        raise ValueError(
            "conv3d_winograd_hw_f4x3 requires 3x3x3 kernel with stride=1, "
            f"dilation=1, C>=4, got {T}x{R}x{S} stride={stride} "
            f"dilation={dilation} C={C}"
        )

    y = _alloc_output_3d(N, K_out, OD, P, Q, x, "ncdhw")
    bias_fp32 = _prep_bias(bias)
    U, C_pad = get_or_make_winograd_hw_filter_f4x3(w_oidhw.contiguous(), block_k)
    _launch_winograd_hw_f4x3(
        x.contiguous(),
        U,
        bias_fp32,
        y,
        N,
        C,
        D,
        H,
        W_in,
        K_out,
        OD,
        P,
        Q,
        C_pad,
        padding,
        activation,
        block_k=block_k,
    )
    return y


def conv3d_winograd_hw_f4x3_cblocked(
    x,
    w_oidhw,
    bias=None,
    stride=(1, 1, 1),
    padding=(0, 0, 0),
    dilation=(1, 1, 1),
    activation="none",
    block_k=BLOCK_K,
    x_blocked=None,
):
    """2.5D Winograd with a channel-blocked (NCDHWc) input transform for
    coalesced channel loads. Same GEMM/output transform as
    :func:`conv3d_winograd_hw_f4x3`; only the input read changes. Raises
    ValueError for non-eligible convs.

    x_blocked: optional pre-packed NCDHWc input (used by the benchmark to time
    the kernel without host-side packing); when None the input is packed here."""
    N, C, D, H, W_in, K_out, T, R, S, OD, P, Q = _conv3d_dims(
        x, w_oidhw, stride, padding, dilation
    )
    if not _is_winograd3d_hw_eligible(T, R, S, stride, dilation, C):
        raise ValueError(
            "conv3d_winograd_hw_f4x3_cblocked requires 3x3x3 kernel with "
            f"stride=1, dilation=1, C>=4, got {T}x{R}x{S} stride={stride} "
            f"dilation={dilation} C={C}"
        )

    y = _alloc_output_3d(N, K_out, OD, P, Q, x, "ncdhw")
    bias_fp32 = _prep_bias(bias)
    U, C_pad = get_or_make_winograd_hw_filter_f4x3(w_oidhw.contiguous(), block_k)
    if x_blocked is None:
        x_blocked, C_pad_x = prepack_ncdhw_to_cblocked(x, block_k)
    else:
        C_pad_x = x_blocked.shape[-1] * x_blocked.shape[1]
    assert (
        C_pad_x == C_pad
    ), f"Channel padding mismatch: input {C_pad_x} vs weight {C_pad}"
    _launch_winograd_hw_f4x3(
        x,
        U,
        bias_fp32,
        y,
        N,
        C,
        D,
        H,
        W_in,
        K_out,
        OD,
        P,
        Q,
        C_pad,
        padding,
        activation,
        block_k=block_k,
        x_blocked=x_blocked,
    )
    return y


def _route_and_run(
    x, w_oidhw, bias, stride, padding, dilation, activation, block_k, layout
):
    """Shared dispatch body for conv3d_ncdhw / conv3d_ndhwc: resolve the route
    once and dispatch to the matching wrapper (parallel to conv2d._route_and_run).
    """
    K_out, C, T, R, S = w_oidhw.shape
    route = _resolve_route(T, R, S, stride, dilation, C, layout)

    if route == Route3D.ONE_X_ONE_X_ONE:
        return conv3d_1x1x1(
            x,
            w_oidhw,
            bias,
            stride,
            padding,
            dilation,
            activation,
            block_k,
            layout=layout,
        )
    if route == Route3D.WINOGRAD_HW:
        return conv3d_winograd_hw_f4x3(
            x, w_oidhw, bias, stride, padding, dilation, activation, block_k
        )
    if route == Route3D.CBLOCKED_NCDHW:
        return conv3d_ncdhw_cblocked(
            x, w_oidhw, bias, stride, padding, dilation, activation, block_k
        )
    if route == Route3D.NDHWC_3X3X3:
        return conv3d_ndhwc_3x3x3(
            x, w_oidhw, bias, stride, padding, dilation, activation, block_k
        )
    return conv3d_general(
        x,
        w_oidhw,
        bias,
        stride,
        padding,
        dilation,
        activation,
        block_k,
        layout=layout,
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
    """NCDHW (channels-first) conv3d: routes to the general kernel (only kernel
    today)."""
    assert x.is_cuda and w_oidhw.is_cuda
    x = x.contiguous()
    return _route_and_run(
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

    Input ``x`` has logical NCDHW shape but is converted to channels_last_3d so
    channels are the inner contiguous axis (coalesced loads). Output is
    allocated channels_last_3d and returned in logical NCDHW shape with
    channels_last_3d strides — mirrors ``conv2d_nhwc``.
    """
    assert x.is_cuda and w_oidhw.is_cuda
    x = x.to(memory_format=torch.channels_last_3d)
    return _route_and_run(
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
