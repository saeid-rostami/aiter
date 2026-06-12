# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl
from aiter.ops.triton.utils.conv_config_utils import get_conv_config
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from .helpers import CONV_AUTOTUNE_ENABLED
from ..activation import _relu, _relu6, _gelu_tanh


def _get_config(shape_key=None, M=None):
    if CONV_AUTOTUNE_ENABLED:
        return {}
    return get_conv_config("CONV3D-GENERAL", shape_key=shape_key, M=M)


_conv3d_general_kernel_repr = make_kernel_repr(
    "_conv3d_general_kernel",
    [
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_K",
        "GROUP_SIZE_M",
        "HAS_BIAS",
        "ACTIVATION",
        "LAYOUT",
    ],
)


@triton.jit(repr=_conv3d_general_kernel_repr)
def _conv3d_general_kernel(
    X,
    WK,
    BIAS,
    Y,
    N: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W_in: tl.constexpr,
    K_out: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    D_out: tl.constexpr,
    P: tl.constexpr,
    Q: tl.constexpr,
    K_pad: tl.constexpr,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_d: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    dil_d: tl.constexpr,
    dil_h: tl.constexpr,
    dil_w: tl.constexpr,
    M_total: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACTIVATION: tl.constexpr,
    LAYOUT: tl.constexpr,
):
    """General 3D conv kernel: implicit GEMM with on-the-fly (c, kd, kh, kw)
    decode — no im2col buffer. LAYOUT: "ncdhw" or "ndhwc".

    3D analog of _conv2d_general_kernel: adds the depth reduction tap (kd) and
    the depth output axis (D_out). Same L2 GROUP_SIZE_M swizzle, fp32 acc, and
    fused bias+activation epilogue.
    """
    # WK is always [K_out, K_pad] contiguous
    stride_wk_kout: tl.constexpr = K_pad
    stride_wk_kred: tl.constexpr = 1
    if LAYOUT == "ncdhw":
        # NCDHW: X[N, C, D, H, W_in], Y[N, K_out, D_out, P, Q]
        stride_x_n: tl.constexpr = C * D * H * W_in
        stride_x_c: tl.constexpr = D * H * W_in
        stride_x_d: tl.constexpr = H * W_in
        stride_x_h: tl.constexpr = W_in
        stride_x_w: tl.constexpr = 1
        stride_y_n: tl.constexpr = K_out * D_out * P * Q
        stride_y_k: tl.constexpr = D_out * P * Q
        stride_y_d: tl.constexpr = P * Q
        stride_y_p: tl.constexpr = Q
        stride_y_q: tl.constexpr = 1
    else:
        # NDHWC: X[N, D, H, W_in, C], Y[N, D_out, P, Q, K_out]
        stride_x_n: tl.constexpr = D * H * W_in * C
        stride_x_c: tl.constexpr = 1
        stride_x_d: tl.constexpr = H * W_in * C
        stride_x_h: tl.constexpr = W_in * C
        stride_x_w: tl.constexpr = C
        stride_y_n: tl.constexpr = D_out * P * Q * K_out
        stride_y_k: tl.constexpr = 1
        stride_y_d: tl.constexpr = P * Q * K_out
        stride_y_p: tl.constexpr = Q * K_out
        stride_y_q: tl.constexpr = K_out

    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M_total, BLOCK_M)
    num_pid_n = tl.cdiv(K_out, BLOCK_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    if pid_m >= num_pid_m:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    kout_mask = offs_n < K_out

    # Decode offs_m -> (n_idx, d_idx, p_idx, q_idx)
    dpq = D_out * P * Q
    pq = P * Q
    n_idx = offs_m[:, None] // dpq
    rem = offs_m[:, None] % dpq
    d_idx = rem // pq
    rem2 = rem % pq
    p_idx = rem2 // Q
    q_idx = rem2 % Q

    n_valid = n_idx < N

    # Precompute output-tile base coordinates
    base_od = d_idx * stride_d - pad_d
    base_oh = p_idx * stride_h - pad_h
    base_ow = q_idx * stride_w - pad_w
    stride_dd = dil_d * stride_x_d
    stride_dh = dil_h * stride_x_h
    stride_dw = dil_w * stride_x_w
    x_base = (
        X
        + n_idx * stride_x_n
        + base_od * stride_x_d
        + base_oh * stride_x_h
        + base_ow * stride_x_w
    )
    wk_base = WK + offs_n[None, :] * stride_wk_kout

    Y_ptrs = (
        Y
        + n_idx * stride_y_n
        + offs_n[None, :] * stride_y_k
        + d_idx * stride_y_d
        + p_idx * stride_y_p
        + q_idx * stride_y_q
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    offs_k = tl.arange(0, BLOCK_K)
    khw = KH * KW
    kdhw = KD * KH * KW

    for k0 in range(0, K_pad, BLOCK_K):
        kred = k0 + offs_k

        WK_ptrs = wk_base + kred[:, None] * stride_wk_kred
        w_tile = tl.load(WK_ptrs, mask=kout_mask[None, :], other=0.0)

        # Decode kred -> (c, kd, kh, kw)
        c = kred // kdhw
        krem = kred % kdhw
        kd = krem // khw
        krem2 = krem % khw
        kh = krem2 // KW
        kw = krem2 % KW

        od = base_od + kd * dil_d
        oh = base_oh + kh * dil_h
        ow = base_ow + kw * dil_w

        X_ptrs = (
            x_base
            + c * stride_x_c
            + kd * stride_dd
            + kh * stride_dh
            + kw * stride_dw
        )
        x_mask = (
            n_valid
            & (od >= 0)
            & (od < D)
            & (oh >= 0)
            & (oh < H)
            & (ow >= 0)
            & (ow < W_in)
            & (c[None, :] < C)
        )
        x_tile = tl.load(X_ptrs, mask=x_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    if HAS_BIAS:
        b = tl.load(BIAS + offs_n, mask=offs_n < K_out, other=0.0)
        acc += b[None, :]

    if ACTIVATION == "relu":
        acc = _relu(acc)
    elif ACTIVATION == "relu6":
        acc = _relu6(acc)
    elif ACTIVATION == "gelu":
        acc = _gelu_tanh(acc)

    tl.store(
        Y_ptrs,
        acc,
        mask=(
            n_valid
            & (d_idx < D_out)
            & (p_idx < P)
            & (q_idx < Q)
            & kout_mask[None, :]
        ),
    )


# Autotune search space (used when AITER_TRITON_CONV_AUTOTUNE=1).
AUTOTUNE_CONV3D_GENERAL_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
        num_warps=8,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
        num_warps=8,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 4},
        num_warps=8,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 4},
        num_warps=4,
        num_stages=1,
    ),
]


if CONV_AUTOTUNE_ENABLED:
    _conv3d_general_kernel = triton.autotune(
        configs=AUTOTUNE_CONV3D_GENERAL_CONFIGS,
        key=["M_total", "K_out", "K_pad"],
        cache_results=True,
    )(_conv3d_general_kernel)
