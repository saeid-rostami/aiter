# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import os

import torch

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None

from aiter.ops.triton.conv._utils import (
    _out_hw,
    _out_dhw,
    _is_winograd_eligible,
    _is_winograd3d_eligible,
    _is_1x1x1_conv,
    _is_3x3x3_conv,
    get_winograd3d_kernel_matrices,
)
from aiter.ops.triton.utils.conv_config_utils import (
    format_shape_key,
    format_shape_key_3d,
)
from aiter.ops.triton._triton_kernels.conv.conv_1x1 import (
    _conv2d_1x1_kernel,
    _get_config as _get_config_1x1,
)
from aiter.ops.triton._triton_kernels.conv.conv_general import (
    _conv2d_general_kernel,
    _get_config as _get_config_general,
)
from aiter.ops.triton._triton_kernels.conv.conv_3x3 import (
    _conv2d_3x3_nhwc_kernel,
    _conv2d_3x3_cblocked_kernel,
    _get_config_nhwc,
    _get_config_cblocked,
)
from aiter.ops.triton._triton_kernels.conv.conv_3x3_winograd_f4x3 import (
    _winograd_f4x3_input_transform_kernel,
    _winograd_f4x3_cblocked_input_transform_kernel,
    _winograd_f4x3_batched_gemm_kernel,
    _winograd_f4x3_output_transform_kernel,
    _winograd_f4x3_fused_gemm_output_kernel,
    _get_config_input as _get_config_wino_input,
    _get_config_gemm as _get_config_wino_gemm,
    _get_config_output as _get_config_wino_output,
    _get_config_fused as _get_config_wino_fused,
)
from aiter.ops.triton._triton_kernels.conv.conv3d_general import (
    _conv3d_general_kernel,
    _get_config as _get_config_general_3d,
)
from aiter.ops.triton._triton_kernels.conv.conv3d_1x1x1 import (
    _conv3d_1x1x1_kernel,
    _get_config as _get_config_1x1x1_3d,
)
from aiter.ops.triton._triton_kernels.conv.conv3d_3x3x3 import (
    _conv3d_3x3x3_ndhwc_kernel,
    _conv3d_3x3x3_cblocked_kernel,
    _get_config_ndhwc as _get_config_3x3x3_ndhwc_3d,
    _get_config_cblocked as _get_config_3x3x3_cblocked_3d,
)
from aiter.ops.triton._triton_kernels.conv.conv3d_3x3x3_winograd_f2x3 import (
    _winograd3d_f2x3_input_transform_kernel,
    _winograd3d_f2x3_cblocked_input_transform_kernel,
    _winograd3d_f2x3_batched_gemm_kernel,
    _winograd3d_f2x3_output_transform_kernel,
    _winograd3d_f2x3_fused_gemm_output_kernel,
    _get_config_input as _get_config_wino3d_input,
    _get_config_gemm as _get_config_wino3d_gemm,
    _get_config_output as _get_config_wino3d_output,
    _get_config_fused as _get_config_wino3d_fused,
)


def _select_3x3_method(N, C, H, W, K_out, stride, dilation):
    """Pick the best 3x3 kernel method based on shape heuristics.

    Decision tree (from benchmark sweep on RDNA4):
    1. Non-Winograd-eligible (stride>1, dilation>1, or C<4) -> cblocked
    2. Winograd only wins when BOTH C and K >= 512 with enough tiles (T >= 98).
       At 256x256 channels, cblocked is tied or slightly better.
    3. Among Winograd variants: WF4cb (NCHWc input) beats WF4 (NCHW input)
       when T >= 392 (large batch * spatial gives more coalescing benefit).
       Below that, WF4 is slightly faster (less repacking overhead).
    """
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
    bias_fp32,
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
    if triton is None:
        raise RuntimeError("Triton not available")

    sh, sw = stride
    ph, pw = padding

    w = w_oihw.squeeze(-1).squeeze(-1).contiguous()  # [K_out, C]

    def grid(meta):
        BM = meta["BLOCK_M"]
        BN = meta["BLOCK_N"]
        return (triton.cdiv(N * P * Q, BM) * triton.cdiv(K_out, BN),)

    bias_arg = bias_fp32 if bias_fp32 is not None else w.new_empty(1)

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

    _conv2d_1x1_kernel[grid](
        x,
        w,
        bias_arg,
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
        HAS_BIAS=1 if bias_fp32 is not None else 0,
        ACTIVATION=activation,
        LAYOUT=layout,
        **config,
    )


def _launch_3x3_nhwc(
    x,
    w_3x3,
    bias_fp32,
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
    if triton is None:
        raise RuntimeError("Triton not available")

    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

    def grid(meta):
        BM = meta["BLOCK_M"]
        BN = meta["BLOCK_N"]
        return (triton.cdiv(N * P * Q, BM) * triton.cdiv(K_out, BN),)

    bias_arg = bias_fp32 if bias_fp32 is not None else w_3x3.new_empty(1)

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

    _conv2d_3x3_nhwc_kernel[grid](
        x,
        w_3x3,
        bias_arg,
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
        HAS_BIAS=1 if bias_fp32 is not None else 0,
        ACTIVATION=activation,
        **config,
    )


def _launch_3x3_cblocked(
    x_blocked,
    w_3x3,
    bias_fp32,
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
    """Launch specialized 3x3 kernel for channel-blocked input."""
    if triton is None:
        raise RuntimeError("Triton not available")

    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

    def grid(meta):
        BM = meta["BLOCK_M"]
        BN = meta["BLOCK_N"]
        return (triton.cdiv(N * P * Q, BM) * triton.cdiv(K_out, BN),)

    bias_arg = bias_fp32 if bias_fp32 is not None else w_3x3.new_empty(1)

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

    _conv2d_3x3_cblocked_kernel[grid](
        x_blocked,
        w_3x3,
        bias_arg,
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
        HAS_BIAS=1 if bias_fp32 is not None else 0,
        ACTIVATION=activation,
        **config,
    )


def _launch_general(
    x,
    w_k,
    bias_fp32,
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
    if triton is None:
        raise RuntimeError("Triton not available")

    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

    def grid(meta):
        BM = meta["BLOCK_M"]
        BN = meta["BLOCK_N"]
        return (triton.cdiv(N * P * Q, BM) * triton.cdiv(K_out, BN),)

    bias_arg = bias_fp32 if bias_fp32 is not None else w_k.new_empty(1)

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

    _conv2d_general_kernel[grid](
        x,
        w_k,
        bias_arg,
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
        HAS_BIAS=1 if bias_fp32 is not None else 0,
        ACTIVATION=activation,
        LAYOUT=layout,
        **config,
    )


def _launch_winograd_f4x3_fused(
    x,
    U,
    bias_fp32,
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
    """Launch Winograd F(4x4,3x3) with fused GEMM+output transform (2 kernels instead of 3)."""
    if triton is None:
        raise RuntimeError("Triton not available")
    ph, pw = padding
    tile_H = (P + 3) // 4
    tile_W = (Q + 3) // 4
    T = N * tile_H * tile_W

    input_dtype = x.dtype
    V = torch.empty((36, T, C_pad), device=x.device, dtype=input_dtype)

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
    fused_config = _get_config_wino_fused(shape_key=shape_key, M=T)

    # 1. Input transform
    def input_grid_f4(meta):
        return (T, triton.cdiv(C_pad, meta["BLOCK_C"]))

    _winograd_f4x3_input_transform_kernel[input_grid_f4](
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

    # 2. Fused GEMM + output transform
    bias_arg = bias_fp32 if bias_fp32 is not None else x.new_empty(1)

    def fused_grid_f4(meta):
        return (triton.cdiv(T, meta["BLOCK_T"]), triton.cdiv(K_out, meta["BLOCK_K"]))

    _winograd_f4x3_fused_gemm_output_kernel[fused_grid_f4](
        V,
        U,
        bias_arg,
        y,
        N,
        K_out,
        P,
        Q,
        C_pad,
        tile_H,
        tile_W,
        T,
        HAS_BIAS=1 if bias_fp32 is not None else 0,
        ACTIVATION=activation,
        LAYOUT=layout,
        **fused_config,
    )


def _launch_winograd_f4x3(
    x,
    U,
    bias_fp32,
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
    if triton is None:
        raise RuntimeError("Triton not available")
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
    def input_grid_f4(meta):
        return (T, triton.cdiv(C_pad, meta["BLOCK_C"]))

    _winograd_f4x3_input_transform_kernel[input_grid_f4](
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
    def gemm_grid(meta):
        BM = meta["BLOCK_M"]
        BN = meta["BLOCK_N"]
        return (triton.cdiv(T, BM) * triton.cdiv(K_out, BN), 36)

    _winograd_f4x3_batched_gemm_kernel[gemm_grid](
        V,
        U,
        M,
        T,
        K_out,
        C_pad,
        **gemm_config,
    )

    # 3. Output transform
    bias_arg = bias_fp32 if bias_fp32 is not None else x.new_empty(1)

    def output_grid_f4(meta):
        return (T, triton.cdiv(K_out, meta["BLOCK_K"]))

    _winograd_f4x3_output_transform_kernel[output_grid_f4](
        M,
        bias_arg,
        y,
        N,
        K_out,
        P,
        Q,
        tile_H,
        tile_W,
        T,
        HAS_BIAS=1 if bias_fp32 is not None else 0,
        ACTIVATION=activation,
        LAYOUT=layout,
        **output_config,
    )


def _launch_winograd_f4x3_cblocked(
    x_blocked,
    C_pad_blocked,
    U,
    bias_fp32,
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
    if triton is None:
        raise RuntimeError("Triton not available")
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
    def input_grid_f4(meta):
        return (T, triton.cdiv(C_pad, meta["BLOCK_C"]))

    _winograd_f4x3_cblocked_input_transform_kernel[input_grid_f4](
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

    def gemm_grid(meta):
        BM = meta["BLOCK_M"]
        BN = meta["BLOCK_N"]
        return (triton.cdiv(T, BM) * triton.cdiv(K_out, BN), 36)

    _winograd_f4x3_batched_gemm_kernel[gemm_grid](
        V,
        U,
        M,
        T,
        K_out,
        C_pad,
        **gemm_config,
    )

    bias_arg = bias_fp32 if bias_fp32 is not None else x_blocked.new_empty(1)

    def output_grid_f4(meta):
        return (T, triton.cdiv(K_out, meta["BLOCK_K"]))

    _winograd_f4x3_output_transform_kernel[output_grid_f4](
        M,
        bias_arg,
        y,
        N,
        K_out,
        P,
        Q,
        tile_H,
        tile_W,
        T,
        HAS_BIAS=1 if bias_fp32 is not None else 0,
        ACTIVATION=activation,
        **output_config,
    )


# -- 3D convolution -----------------------------------------------------------

# Methods with a built kernel today, per layout. The router (_select_conv3d_method)
# may return a specialized name before its kernel exists for that layout;
# conv3d_method_implemented gates that so unbuilt (method, layout) pairs fall back
# to "general". 3x3x3 is built for NDHWC but not yet NCDHW (cblocked TBD — see
# the Phase-3 decision in the handoff). Delete this gate once all are built.
_IMPLEMENTED_CONV3D_METHODS = {
    "ncdhw": frozenset({"general", "1x1x1", "3x3x3", "winograd_f2x3"}),
    "ndhwc": frozenset({"general", "1x1x1", "3x3x3", "winograd_f2x3"}),
}

# 3D Winograd F(2,3) auto-route win-region (measured on gfx1201, Phase 5; fused kernel).
# F(2,3) only yields a 3.375x MAC reduction, so the 64-alpha transform overhead pays off
# only in a *bounded* region. With the fused GEMM+output kernel (default — no M[64,T,K_out]
# round-trip), best-of-5 sweeps show reliable 1.2-2.4x wins for:
#   C,K >= 256  AND  512 <= T <= 2048  AND  min(D,H,W) >= 4
# Outside that it regresses: small T (<~512) is too little work to amortize (0.63-0.70x at
# 256-ch/8^3), large T (>~2300) is memory-bound on V (0.85-0.90x), a tiny spatial axis
# wastes the 4-wide tiles (D=3 -> ~0.96x), and C/K<256 loses outright (0.45-0.59x). Gate
# tightly so every auto-routed shape actually beats the direct 3x3x3 kernel.
_WINO3D_MIN_C = 256
_WINO3D_MIN_K = 256
_WINO3D_MIN_T = 512
_WINO3D_MAX_T = 2048
_WINO3D_MIN_DIM = 4


def conv3d_method_implemented(method, layout) -> bool:
    return method in _IMPLEMENTED_CONV3D_METHODS[layout]


def _select_conv3d_method(N, C, D, H, W, K_out, KD, KH, KW, stride, dilation):
    """Pick the best 3D conv kernel for a shape.

    Returns "1x1x1" / "winograd_f2x3" / "3x3x3" / "general". 3x3x3 stride-1 convs
    route to the Winograd F(2,3) path only inside the measured win-region (large
    C/K, moderate tile count, no tiny spatial axis); otherwise the direct kernel.
    """
    if _is_1x1x1_conv(KD, KH, KW, dilation):
        return "1x1x1"
    if _is_3x3x3_conv(KD, KH, KW):
        if _is_winograd3d_eligible(KD, KH, KW, stride, dilation, C):
            # T proxy from input dims (pad=1 'same' conv keeps output ~= input).
            T = N * ((D + 1) // 2) * ((H + 1) // 2) * ((W + 1) // 2)
            if (
                C >= _WINO3D_MIN_C
                and K_out >= _WINO3D_MIN_K
                and _WINO3D_MIN_T <= T <= _WINO3D_MAX_T
                and min(D, H, W) >= _WINO3D_MIN_DIM
            ):
                return "winograd_f2x3"
        return "3x3x3"
    return "general"


def _launch_conv3d_general(
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
    layout="ncdhw",
):
    """Launch the general 3D conv kernel. layout: 'ncdhw' or 'ndhwc'."""
    if triton is None:
        raise RuntimeError("Triton not available")

    sd, sh, sw = stride
    pd, ph, pw = padding
    dd, dh, dw = dilation
    # LAYOUT travels to the kernel as a string constexpr ("ncdhw"/"ndhwc"),
    # matching conv2d and the ACTIVATION style. Validated at the conv3d() entry.

    def grid(meta):
        BM = meta["BLOCK_M"]
        BN = meta["BLOCK_N"]
        return (triton.cdiv(N * D_out * P * Q, BM) * triton.cdiv(K_out, BN),)

    bias_arg = bias_fp32 if bias_fp32 is not None else w_k.new_empty(1)

    M_total = N * D_out * P * Q

    shape_key = format_shape_key_3d(
        N=N,
        C=C,
        D=D,
        H=H,
        W=W_in,
        K=K_out,
        KD=KD,
        KH=KH,
        KW=KW,
        sd=sd,
        sh=sh,
        sw=sw,
        pd=pd,
        ph=ph,
        pw=pw,
        dd=dd,
        dh=dh,
        dw=dw,
    )
    config = _get_config_general_3d(shape_key=shape_key, M=M_total, layout=layout)

    _conv3d_general_kernel[grid](
        x,
        w_k,
        bias_arg,
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
        sd,
        sh,
        sw,
        pd,
        ph,
        pw,
        dd,
        dh,
        dw,
        M_total,
        HAS_BIAS=1 if bias_fp32 is not None else 0,
        ACTIVATION=activation,
        LAYOUT=layout,
        **config,
    )


def _launch_conv3d_1x1x1(
    x,
    w_oidhw,
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
    layout="ncdhw",
):
    """Launch the specialized 1x1x1 conv kernel (pure GEMM). layout: 'ncdhw'/'ndhwc'."""
    if triton is None:
        raise RuntimeError("Triton not available")

    sd, sh, sw = stride
    pd, ph, pw = padding

    # [K_out, C, 1, 1, 1] -> [K_out, C]
    w = w_oidhw.squeeze(-1).squeeze(-1).squeeze(-1).contiguous()

    def grid(meta):
        BM = meta["BLOCK_M"]
        BN = meta["BLOCK_N"]
        return (triton.cdiv(N * D_out * P * Q, BM) * triton.cdiv(K_out, BN),)

    bias_arg = bias_fp32 if bias_fp32 is not None else w.new_empty(1)

    M_total = N * D_out * P * Q

    shape_key = format_shape_key_3d(
        N=N,
        C=C,
        D=D,
        H=H,
        W=W_in,
        K=K_out,
        KD=1,
        KH=1,
        KW=1,
        sd=sd,
        sh=sh,
        sw=sw,
        pd=pd,
        ph=ph,
        pw=pw,
        dd=1,
        dh=1,
        dw=1,
    )
    config = _get_config_1x1x1_3d(shape_key=shape_key, M=M_total, layout=layout)

    _conv3d_1x1x1_kernel[grid](
        x,
        w,
        bias_arg,
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
        sd,
        sh,
        sw,
        pd,
        ph,
        pw,
        M_total,
        HAS_BIAS=1 if bias_fp32 is not None else 0,
        ACTIVATION=activation,
        LAYOUT=layout,
        **config,
    )


def _launch_conv3d_3x3x3_ndhwc(
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
):
    """Launch the specialized NDHWC 3x3x3 kernel (hardcoded stride_c=1, stride_k=1)."""
    if triton is None:
        raise RuntimeError("Triton not available")

    sd, sh, sw = stride
    pd, ph, pw = padding
    dd, dh, dw = dilation

    def grid(meta):
        BM = meta["BLOCK_M"]
        BN = meta["BLOCK_N"]
        return (triton.cdiv(N * D_out * P * Q, BM) * triton.cdiv(K_out, BN),)

    bias_arg = bias_fp32 if bias_fp32 is not None else w_3x3x3.new_empty(1)

    M_total = N * D_out * P * Q

    shape_key = format_shape_key_3d(
        N=N,
        C=C,
        D=D,
        H=H,
        W=W_in,
        K=K_out,
        KD=3,
        KH=3,
        KW=3,
        sd=sd,
        sh=sh,
        sw=sw,
        pd=pd,
        ph=ph,
        pw=pw,
        dd=dd,
        dh=dh,
        dw=dw,
    )
    config = _get_config_3x3x3_ndhwc_3d(shape_key=shape_key, M=M_total)

    _conv3d_3x3x3_ndhwc_kernel[grid](
        x,
        w_3x3x3,
        bias_arg,
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
        sd,
        sh,
        sw,
        pd,
        ph,
        pw,
        dd,
        dh,
        dw,
        M_total,
        HAS_BIAS=1 if bias_fp32 is not None else 0,
        ACTIVATION=activation,
        **config,
    )


def _launch_conv3d_3x3x3_cblocked(
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
):
    """Launch the NCDHWc cblocked 3x3x3 kernel (NCDHW-contiguous output)."""
    if triton is None:
        raise RuntimeError("Triton not available")

    sd, sh, sw = stride
    pd, ph, pw = padding
    dd, dh, dw = dilation

    def grid(meta):
        BM = meta["BLOCK_M"]
        BN = meta["BLOCK_N"]
        return (triton.cdiv(N * D_out * P * Q, BM) * triton.cdiv(K_out, BN),)

    bias_arg = bias_fp32 if bias_fp32 is not None else w_3x3x3.new_empty(1)

    M_total = N * D_out * P * Q

    shape_key = format_shape_key_3d(
        N=N,
        C=C,
        D=D,
        H=H,
        W=W_in,
        K=K_out,
        KD=3,
        KH=3,
        KW=3,
        sd=sd,
        sh=sh,
        sw=sw,
        pd=pd,
        ph=ph,
        pw=pw,
        dd=dd,
        dh=dh,
        dw=dw,
    )
    config = _get_config_3x3x3_cblocked_3d(shape_key=shape_key, M=M_total)

    _conv3d_3x3x3_cblocked_kernel[grid](
        x_blocked,
        w_3x3x3,
        bias_arg,
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
        sd,
        sh,
        sw,
        pd,
        ph,
        pw,
        dd,
        dh,
        dw,
        M_total,
        HAS_BIAS=1 if bias_fp32 is not None else 0,
        ACTIVATION=activation,
        **config,
    )


# -- 3D Winograd F(2x2x2, 3x3x3) (experimental, Phase 5) ----------------------


# Default the GEMM+output stage to the fused kernel (skips the M[64,T,K_out]
# round-trip). Override with AITER_TRITON_WINO3D_FUSED=0 for the 3-kernel path.
_WINO3D_FUSED_DEFAULT = os.environ.get("AITER_TRITON_WINO3D_FUSED", "1") == "1"


def _winograd3d_finish(
    V, U, A3d, bias_fp32, y, x_for_alloc,
    N, C, D, H, W_in, K_out, D_out, P, Q, C_pad,
    T, tile_D, tile_H, tile_W, shape_key, activation, layout, fused,
):
    """Stage after the input transform: either the 3-kernel (batched GEMM ->
    output transform via M) path, or the fused GEMM+output kernel (no M)."""
    bias_arg = bias_fp32 if bias_fp32 is not None else x_for_alloc.new_empty(1)
    has_bias = 1 if bias_fp32 is not None else 0

    if fused:
        fused_config = _get_config_wino3d_fused(shape_key=shape_key, M=T)

        def fused_grid(meta):
            return (
                triton.cdiv(T, meta["BLOCK_T"]),
                triton.cdiv(K_out, meta["BLOCK_K"]),
            )

        _winograd3d_f2x3_fused_gemm_output_kernel[fused_grid](
            V, U, A3d, bias_arg, y,
            N, K_out, D_out, P, Q, C_pad,
            tile_D, tile_H, tile_W, T,
            HAS_BIAS=has_bias,
            ACTIVATION=activation,
            LAYOUT=layout,
            **fused_config,
        )
        return

    M = torch.empty((64, T, K_out), device=x_for_alloc.device, dtype=V.dtype)
    gemm_config = _get_config_wino3d_gemm(shape_key=shape_key, M=T)
    output_config = _get_config_wino3d_output(shape_key=shape_key, M=T)

    def gemm_grid(meta):
        BM = meta["BLOCK_M"]
        BN = meta["BLOCK_N"]
        return (triton.cdiv(T, BM) * triton.cdiv(K_out, BN), 64)

    _winograd3d_f2x3_batched_gemm_kernel[gemm_grid](
        V, U, M, T, K_out, C_pad, **gemm_config,
    )

    def output_grid(meta):
        return (T, triton.cdiv(K_out, meta["BLOCK_K"]))

    _winograd3d_f2x3_output_transform_kernel[output_grid](
        M, A3d, bias_arg, y,
        N, K_out, D_out, P, Q,
        tile_D, tile_H, tile_W, T,
        HAS_BIAS=has_bias,
        ACTIVATION=activation,
        LAYOUT=layout,
        **output_config,
    )


def _winograd3d_run(
    input_kernel,
    input_extra_args,
    x_for_alloc,
    U,
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
    padding,
    activation,
    layout,
    fused,
):
    """Driver: input transform -> (fused GEMM+output, or 3-kernel GEMM->output).
    input_kernel/input_extra_args differ between the NCDHW/NDHWC and cblocked paths."""
    pd, ph, pw = padding
    tile_D = (D_out + 1) // 2
    tile_H = (P + 1) // 2
    tile_W = (Q + 1) // 2
    T = N * tile_D * tile_H * tile_W

    input_dtype = x_for_alloc.dtype
    BBB, A3d = get_winograd3d_kernel_matrices(x_for_alloc.device, input_dtype)
    V = torch.empty((64, T, C_pad), device=x_for_alloc.device, dtype=input_dtype)

    shape_key = format_shape_key_3d(
        N=N, C=C, D=D, H=H, W=W_in, K=K_out,
        KD=3, KH=3, KW=3,
        sd=1, sh=1, sw=1, pd=pd, ph=ph, pw=pw, dd=1, dh=1, dw=1,
    )
    input_config = _get_config_wino3d_input(shape_key=shape_key, M=T)

    def input_grid(meta):
        return (T, triton.cdiv(C_pad, meta["BLOCK_C"]))

    input_kernel[input_grid](
        *input_extra_args,
        BBB,
        V,
        N, C, C_pad, D, H, W_in,
        tile_D, tile_H, tile_W, T,
        pd, ph, pw,
        LAYOUT=layout,
        **input_config,
    )

    _winograd3d_finish(
        V, U, A3d, bias_fp32, y, x_for_alloc,
        N, C, D, H, W_in, K_out, D_out, P, Q, C_pad,
        T, tile_D, tile_H, tile_W, shape_key, activation, layout, fused,
    )


def _launch_winograd3d_f2x3(
    x, U, bias_fp32, y,
    N, C, D, H, W_in, K_out, D_out, P, Q, C_pad,
    padding, activation, layout="ncdhw", fused=None,
):
    """Winograd F(2,3) reading NCDHW or NDHWC input directly (LAYOUT)."""
    if triton is None:
        raise RuntimeError("Triton not available")
    if fused is None:
        fused = _WINO3D_FUSED_DEFAULT
    _winograd3d_run(
        _winograd3d_f2x3_input_transform_kernel,
        (x,),
        x, U, bias_fp32, y,
        N, C, D, H, W_in, K_out, D_out, P, Q, C_pad,
        padding, activation, layout, fused,
    )


def _launch_winograd3d_f2x3_cblocked(
    x_blocked, Cb, U, bias_fp32, y,
    N, C, D, H, W_in, K_out, D_out, P, Q, C_pad,
    padding, activation, fused=None,
):
    """Winograd F(2,3) reading channel-blocked NCDHWc input; writes NCDHW output."""
    if triton is None:
        raise RuntimeError("Triton not available")
    if fused is None:
        fused = _WINO3D_FUSED_DEFAULT

    pd, ph, pw = padding
    tile_D = (D_out + 1) // 2
    tile_H = (P + 1) // 2
    tile_W = (Q + 1) // 2
    T = N * tile_D * tile_H * tile_W

    input_dtype = x_blocked.dtype
    BBB, A3d = get_winograd3d_kernel_matrices(x_blocked.device, input_dtype)
    V = torch.empty((64, T, C_pad), device=x_blocked.device, dtype=input_dtype)

    shape_key = format_shape_key_3d(
        N=N, C=C, D=D, H=H, W=W_in, K=K_out,
        KD=3, KH=3, KW=3,
        sd=1, sh=1, sw=1, pd=pd, ph=ph, pw=pw, dd=1, dh=1, dw=1,
    )
    input_config = _get_config_wino3d_input(shape_key=shape_key, M=T)

    def input_grid(meta):
        return (T, triton.cdiv(C_pad, meta["BLOCK_C"]))

    _winograd3d_f2x3_cblocked_input_transform_kernel[input_grid](
        x_blocked, BBB, V,
        N, C, C_pad, D, H, W_in,
        tile_D, tile_H, tile_W, T,
        pd, ph, pw, Cb,
        **input_config,
    )

    _winograd3d_finish(
        V, U, A3d, bias_fp32, y, x_blocked,
        N, C, D, H, W_in, K_out, D_out, P, Q, C_pad,
        T, tile_D, tile_H, tile_W, shape_key, activation, "ncdhw", fused,
    )
