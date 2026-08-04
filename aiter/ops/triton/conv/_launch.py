# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.conv.conv_1x1 import (
    _conv2d_1x1_kernel,
)
from aiter.ops.triton._triton_kernels.conv.conv_1x1 import (
    _get_config as _get_config_1x1,
)
from aiter.ops.triton._triton_kernels.conv.conv_3x3 import (
    _conv2d_3x3_cblocked_kernel,
    _conv2d_3x3_nchw_kernel,
    _conv2d_3x3_nhwc_kernel,
    _get_config_cblocked,
    _get_config_nchw,
    _get_config_nhwc,
)
from aiter.ops.triton._triton_kernels.conv.conv_3x3_winograd_f4x3 import (
    _get_config_gemm as _get_config_wino_gemm,
)
from aiter.ops.triton._triton_kernels.conv.conv_3x3_winograd_f4x3 import (
    _get_config_input as _get_config_wino_input,
)
from aiter.ops.triton._triton_kernels.conv.conv_3x3_winograd_f4x3 import (
    _get_config_output as _get_config_wino_output,
)
from aiter.ops.triton._triton_kernels.conv.conv_3x3_winograd_f4x3 import (
    _winograd_f4x3_batched_gemm_kernel,
    _winograd_f4x3_cblocked_input_transform_kernel,
    _winograd_f4x3_input_transform_kernel,
    _winograd_f4x3_output_transform_kernel,
)
from aiter.ops.triton._triton_kernels.conv.conv_general import (
    _conv2d_general_kernel,
)
from aiter.ops.triton._triton_kernels.conv.conv_general import (
    _get_config as _get_config_general,
)
from aiter.ops.triton.conv._prepack import is_nchw_direct_view
from aiter.ops.triton.conv._utils import BLOCK_K, _is_winograd_eligible, _out_hw
from aiter.ops.triton.utils.conv_config_utils import format_shape_key


def _make_mn_grid(M_total, K_out):
    """Grid for the GEMM-style conv kernels (1x1, 3x3 nhwc/cblocked, general):
    one program per (BLOCK_M tile of M_total) x (BLOCK_N tile of K_out)."""

    def grid(meta):
        return (
            triton.cdiv(M_total, meta["BLOCK_M"]) * triton.cdiv(K_out, meta["BLOCK_N"]),
        )

    return grid


def _make_wino_input_grid(T, C_pad):
    """Grid for the Winograd F(4,3) input-transform kernels: one program per
    tile T x (BLOCK_C tile of C_pad)."""

    def grid(meta):
        return (T, triton.cdiv(C_pad, meta["BLOCK_C"]))

    return grid


def _make_wino_gemm_grid(T, K_out):
    """Grid for the Winograd F(4,3) batched GEMM: (BLOCK_M tile of T) x
    (BLOCK_N tile of K_out) program blocks, batched over the 36 tile elements."""

    def grid(meta):
        return (
            triton.cdiv(T, meta["BLOCK_M"]) * triton.cdiv(K_out, meta["BLOCK_N"]),
            36,
        )

    return grid


def _make_wino_output_grid(T, K_out):
    """Grid for the Winograd F(4,3) output-transform kernels: one program per
    tile T x (BLOCK_K tile of K_out)."""

    def grid(meta):
        return (T, triton.cdiv(K_out, meta["BLOCK_K"]))

    return grid


def _select_3x3_method(N, C, H, W, K_out, stride, dilation, block_c=BLOCK_K):
    """Pick the best 3x3 kernel method based on shape heuristics.

    Decision tree (from benchmark sweep on RDNA4):
    0. Narrower than one channel block (C < block_c) -> general. Every 3x3
       kernel packs the weight to C_pad = ceil(C/block_c)*block_c and reduces
       over that padded extent, so a shape with C < block_c spends most of its
       `tl.dot` lanes on masked-out padding. `conv_general` decodes (c, r, s)
       on the fly against the true C and does not amplify. See below.
    1. Non-Winograd-eligible (stride>1, dilation>1, or C<4) -> cblocked
    2. Winograd only wins when BOTH C and K >= 512 with enough tiles (T >= 98).
       At 256x256 channels, cblocked is tied or slightly better.
    3. Among Winograd variants: WF4cb (NCHWc input) beats WF4 (NCHW input)
       when T >= 392 (large batch * spatial gives more coalescing benefit).
       Below that, WF4 is slightly faster (less repacking overhead).

    Rule 0 is a *structural* consequence of the pack block, not a tuned
    constant: it fires exactly when the channel pad is non-empty. Measured on
    gfx1201 (fp16, 3x3, pad 1), 3x3-NCHW vs general:
    C=3 (1,3,512,512)->128  374.4us -> 158.9us (2.36x);
    C=32 (1,32,64,64)->512   48.7us ->  33.6us (1.45x);
    C=64 and above the pad is empty and the 3x3 kernels win, so the rule stops
    at `< block_c`. Only the NCHW router acts on it — the NHWC path was not
    measured and keeps its existing kernel (see `_resolve_route`).
    """
    if C < block_c:
        return "general"
    if not _is_winograd_eligible(3, 3, stride, dilation, C):
        return "cblocked"
    P, Q = _out_hw(H, W, 3, 3, stride, (1, 1), dilation)
    tile_H = (P + 3) // 4
    tile_W = (Q + 3) // 4
    T = N * tile_H * tile_W
    if C >= 512 and K_out >= 512 and T >= 98:
        if T >= 392:
            return "winograd_f4x3_cblocked"
        return "winograd_f4x3"
    return "cblocked"


def _launch_1x1(
    x,
    w_oihw,
    bias,
    y,
    N,
    C,
    H,
    W_in,
    K_out,
    P,
    Q,
    stride,
    padding,
    activation,
    layout="nchw",
):
    """Launch specialized 1x1 kernel.
    layout: "nchw" or "nhwc" (case-insensitive).
    """
    sh, sw = stride
    ph, pw = padding

    w = w_oihw.squeeze(-1).squeeze(-1).contiguous()  # [K_out, C]

    M_total = N * P * Q

    shape_key = format_shape_key(
        N=N,
        C=C,
        H=H,
        W=W_in,
        K=K_out,
        R=1,
        S=1,
        sh=sh,
        sw=sw,
        ph=ph,
        pw=pw,
        dh=1,
        dw=1,
    )
    config = _get_config_1x1(shape_key=shape_key, M=M_total)

    _conv2d_1x1_kernel[_make_mn_grid(M_total, K_out)](
        x,
        w,
        bias,
        y,
        N,
        C,
        H,
        W_in,
        K_out,
        P,
        Q,
        sh,
        sw,
        ph,
        pw,
        M_total,
        HAS_BIAS=bias is not None,
        ACTIVATION=activation,
        LAYOUT=layout,
        **config,
    )


def _launch_3x3_nhwc(
    x,
    w_3x3,
    bias,
    y,
    N,
    C,
    H,
    W_in,
    K_out,
    P,
    Q,
    C_pad,
    stride,
    padding,
    dilation,
    activation,
):
    """Launch specialized 3x3 NHWC kernel (hardcoded stride_c=1, stride_k=1)."""
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

    M_total = N * P * Q

    shape_key = format_shape_key(
        N=N,
        C=C,
        H=H,
        W=W_in,
        K=K_out,
        R=3,
        S=3,
        sh=sh,
        sw=sw,
        ph=ph,
        pw=pw,
        dh=dh,
        dw=dw,
    )
    config = _get_config_nhwc(shape_key=shape_key, M=M_total)

    _conv2d_3x3_nhwc_kernel[_make_mn_grid(M_total, K_out)](
        x,
        w_3x3,
        bias,
        y,
        N,
        C,
        H,
        W_in,
        K_out,
        P,
        Q,
        C_pad,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        M_total,
        HAS_BIAS=bias is not None,
        ACTIVATION=activation,
        **config,
    )


def _launch_3x3_cblocked(
    x_blocked,
    w_3x3,
    bias,
    y,
    N,
    C,
    H,
    W_in,
    K_out,
    P,
    Q,
    C_pad,
    Cb,
    stride,
    padding,
    dilation,
    activation,
):
    """Launch the 3x3 NCHW kernel for the packed-or-viewed input.

    ``prepack_nchw_to_cblocked`` returns either a materialized NCHWc pack or a
    zero-copy NCHW view of the caller's activation (both shaped
    ``[N, C_blocks, H, W, Cb]``). The view case dispatches to
    ``_conv2d_3x3_nchw_kernel``, which reads NCHW in place and so skips the
    repack entirely; the packed case keeps the original cblocked kernel.
    """
    if is_nchw_direct_view(x_blocked):
        _launch_3x3_nchw(
            x_blocked,
            w_3x3,
            bias,
            y,
            N,
            C,
            H,
            W_in,
            K_out,
            P,
            Q,
            C_pad,
            stride,
            padding,
            dilation,
            activation,
        )
        return

    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

    M_total = N * P * Q

    shape_key = format_shape_key(
        N=N,
        C=C,
        H=H,
        W=W_in,
        K=K_out,
        R=3,
        S=3,
        sh=sh,
        sw=sw,
        ph=ph,
        pw=pw,
        dh=dh,
        dw=dw,
    )
    config = _get_config_cblocked(shape_key=shape_key, M=M_total)

    _conv2d_3x3_cblocked_kernel[_make_mn_grid(M_total, K_out)](
        x_blocked,
        w_3x3,
        bias,
        y,
        N,
        C,
        H,
        W_in,
        K_out,
        P,
        Q,
        C_pad,
        Cb,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        M_total,
        HAS_BIAS=bias is not None,
        ACTIVATION=activation,
        **config,
    )


def _launch_3x3_nchw(
    x,
    w_3x3,
    bias,
    y,
    N,
    C,
    H,
    W_in,
    K_out,
    P,
    Q,
    C_pad,
    stride,
    padding,
    dilation,
    activation,
):
    """Launch the repack-free 3x3 kernel that reads plain NCHW activations.

    ``x`` may be the 5-D NCHW view produced by ``prepack_nchw_to_cblocked``;
    only its base pointer is used, the kernel derives NCHW strides from
    ``C_pad``/``H``/``W_in``.
    """
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

    M_total = N * P * Q

    shape_key = format_shape_key(
        N=N,
        C=C,
        H=H,
        W=W_in,
        K=K_out,
        R=3,
        S=3,
        sh=sh,
        sw=sw,
        ph=ph,
        pw=pw,
        dh=dh,
        dw=dw,
    )
    config = _get_config_nchw(shape_key=shape_key, M=M_total)

    # A BLOCK_M tile stays inside one output row exactly when BLOCK_M divides Q;
    # the kernel then skips the //,% row decode and keeps the pixel axis a
    # provable unit-stride arange (vectorized loads/stores). Purely a legality
    # test on the tile size, not a tuning knob.
    # (Under AITER_TRITON_CONV_AUTOTUNE the config is empty and BLOCK_M is only
    # known inside the autotuner, so the specialization is simply off.)
    row_aligned = "BLOCK_M" in config and Q % config["BLOCK_M"] == 0

    _conv2d_3x3_nchw_kernel[_make_mn_grid(M_total, K_out)](
        x,
        w_3x3,
        bias,
        y,
        N,
        C,
        H,
        W_in,
        K_out,
        P,
        Q,
        C_pad,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        M_total,
        HAS_BIAS=bias is not None,
        ACTIVATION=activation,
        ROW_ALIGNED=row_aligned,
        **config,
    )


def _launch_general(
    x,
    w_k,
    bias,
    y,
    N,
    C,
    H,
    W_in,
    K_out,
    R,
    S,
    P,
    Q,
    K_pad,
    stride,
    padding,
    dilation,
    block_k,
    activation,
    layout="nchw",
):
    """Launch general conv kernel.
    layout: "nchw" or "nhwc" (case-insensitive).
    """
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

    M_total = N * P * Q

    shape_key = format_shape_key(
        N=N,
        C=C,
        H=H,
        W=W_in,
        K=K_out,
        R=R,
        S=S,
        sh=sh,
        sw=sw,
        ph=ph,
        pw=pw,
        dh=dh,
        dw=dw,
    )
    config = _get_config_general(shape_key=shape_key, M=M_total)

    _conv2d_general_kernel[_make_mn_grid(M_total, K_out)](
        x,
        w_k,
        bias,
        y,
        N,
        C,
        H,
        W_in,
        K_out,
        R,
        S,
        P,
        Q,
        K_pad,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        M_total,
        HAS_BIAS=bias is not None,
        ACTIVATION=activation,
        LAYOUT=layout,
        **config,
    )


def _launch_winograd_f4x3(
    x,
    U,
    bias,
    y,
    N,
    C,
    H,
    W_in,
    K_out,
    P,
    Q,
    C_pad,
    padding,
    activation,
    layout="nchw",
):
    """Launch Winograd F(4x4,3x3) pipeline: input transform -> batched GEMM -> output transform."""
    ph, pw = padding
    tile_H = (P + 3) // 4
    tile_W = (Q + 3) // 4
    T = N * tile_H * tile_W

    input_dtype = x.dtype
    V = torch.empty((36, T, C_pad), device=x.device, dtype=input_dtype)
    M = torch.empty((36, T, K_out), device=x.device, dtype=torch.float32)

    shape_key = format_shape_key(
        N=N,
        C=C,
        H=H,
        W=W_in,
        K=K_out,
        R=3,
        S=3,
        sh=1,
        sw=1,
        ph=ph,
        pw=pw,
        dh=1,
        dw=1,
    )
    input_config = _get_config_wino_input(shape_key=shape_key, M=T)
    gemm_config = _get_config_wino_gemm(shape_key=shape_key, M=T)
    output_config = _get_config_wino_output(shape_key=shape_key, M=T)

    # 1. Input transform
    _winograd_f4x3_input_transform_kernel[_make_wino_input_grid(T, C_pad)](
        x,
        V,
        N,
        C,
        C_pad,
        H,
        W_in,
        tile_H,
        tile_W,
        T,
        ph,
        pw,
        LAYOUT=layout,
        **input_config,
    )

    # 2. Batched GEMM
    _winograd_f4x3_batched_gemm_kernel[_make_wino_gemm_grid(T, K_out)](
        V,
        U,
        M,
        T,
        K_out,
        C_pad,
        **gemm_config,
    )

    # 3. Output transform
    _winograd_f4x3_output_transform_kernel[_make_wino_output_grid(T, K_out)](
        M,
        bias,
        y,
        N,
        K_out,
        P,
        Q,
        tile_H,
        tile_W,
        T,
        HAS_BIAS=bias is not None,
        ACTIVATION=activation,
        LAYOUT=layout,
        **output_config,
    )


def _launch_winograd_f4x3_cblocked(
    x_blocked,
    C_pad_blocked,
    U,
    bias,
    y,
    N,
    C,
    H,
    W_in,
    K_out,
    P,
    Q,
    C_pad,
    padding,
    activation,
    block_k,
):
    """Launch Winograd F(4x4,3x3) with NCHWc input layout: cblocked input transform -> batched GEMM -> output transform."""
    # The cblocked input transform reads real NCHWc storage. When
    # prepack_nchw_to_cblocked handed back its zero-copy NCHW view (the
    # NCHW-direct path used by _conv2d_3x3_nchw_kernel), materialize it here —
    # `.contiguous()` on the view is exactly the pack the old code produced.
    if is_nchw_direct_view(x_blocked):
        x_blocked = x_blocked.contiguous()
    ph, pw = padding
    tile_H = (P + 3) // 4
    tile_W = (Q + 3) // 4
    T = N * tile_H * tile_W

    Cb = block_k
    input_dtype = x_blocked.dtype
    V = torch.empty((36, T, C_pad), device=x_blocked.device, dtype=input_dtype)
    M = torch.empty((36, T, K_out), device=x_blocked.device, dtype=torch.float32)

    shape_key = format_shape_key(
        N=N,
        C=C,
        H=H,
        W=W_in,
        K=K_out,
        R=3,
        S=3,
        sh=1,
        sw=1,
        ph=ph,
        pw=pw,
        dh=1,
        dw=1,
    )
    input_config = _get_config_wino_input(shape_key=shape_key, M=T)
    gemm_config = _get_config_wino_gemm(shape_key=shape_key, M=T)
    output_config = _get_config_wino_output(shape_key=shape_key, M=T)

    # 1. Cblocked input transform
    _winograd_f4x3_cblocked_input_transform_kernel[_make_wino_input_grid(T, C_pad)](
        x_blocked,
        V,
        N,
        C,
        C_pad,
        H,
        W_in,
        tile_H,
        tile_W,
        T,
        ph,
        pw,
        Cb,
        **input_config,
    )

    _winograd_f4x3_batched_gemm_kernel[_make_wino_gemm_grid(T, K_out)](
        V,
        U,
        M,
        T,
        K_out,
        C_pad,
        **gemm_config,
    )

    _winograd_f4x3_output_transform_kernel[_make_wino_output_grid(T, K_out)](
        M,
        bias,
        y,
        N,
        K_out,
        P,
        Q,
        tile_H,
        tile_W,
        T,
        HAS_BIAS=bias is not None,
        ACTIVATION=activation,
        **output_config,
    )
