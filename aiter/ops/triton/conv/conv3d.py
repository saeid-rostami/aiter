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
)
from aiter.ops.triton.conv._prepack import get_or_make_weight_pack_3d
from aiter.ops.triton.conv._launch import _launch_general_3d, _launch_1x1x1_3d

_LOGGER = AiterTritonLogger()


class Route3D(Enum):
    # Values are kernel display names (parallel to conv2d's Route enum).
    ONE_X_ONE_X_ONE = "_conv3d_1x1x1_kernel"
    GENERAL = "_conv3d_general_kernel"


def _resolve_route(T, R, S, dilation):
    """Single source of dispatch for conv3d (parallel to conv2d._resolve_route)."""
    if _is_1x1x1_conv(T, R, S, dilation):
        return Route3D.ONE_X_ONE_X_ONE
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


def _route_and_run(
    x, w_oidhw, bias, stride, padding, dilation, activation, block_k, layout
):
    """Shared dispatch body for conv3d_ncdhw / conv3d_ndhwc: resolve the route
    once and dispatch to the matching wrapper (parallel to conv2d._route_and_run).
    """
    K_out, _, T, R, S = w_oidhw.shape
    route = _resolve_route(T, R, S, dilation)

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
