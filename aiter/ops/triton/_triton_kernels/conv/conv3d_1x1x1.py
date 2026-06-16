# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton.utils.conv_config_utils import get_conv_config
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from .helpers import CONV_AUTOTUNE_ENABLED
from ..activation import _relu, _relu6, _gelu_tanh


def _get_config(shape_key=None, M=None, layout="ncdhw"):
    # NCDHW and NDHWC want different tilings (opposite contiguous axes for the
    # input gather), so they use separate config files.
    if CONV_AUTOTUNE_ENABLED:
        return {}
    name = "CONV3D-1X1X1-NDHWC" if layout == "ndhwc" else "CONV3D-1X1X1"
    return get_conv_config(name, shape_key=shape_key, M=M)


_conv3d_1x1x1_kernel_repr = make_kernel_repr(
    "_conv3d_1x1x1_kernel",
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


@triton.jit(repr=_conv3d_1x1x1_kernel_repr)
def _conv3d_1x1x1_kernel(
    X,
    W,
    BIAS,
    Y,
    N: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W_in: tl.constexpr,
    K_out: tl.constexpr,
    D_out: tl.constexpr,
    P: tl.constexpr,
    Q: tl.constexpr,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_d: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    M_total: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACTIVATION: tl.constexpr,
    LAYOUT: tl.constexpr,
):
    """Specialized 1x1x1 convolution kernel — a pure channel-reduction GEMM.
    - No kernel-tap loop (KD=KH=KW=1): reduction is only over input channels C.
    - 3D analog of _conv2d_1x1_kernel (adds the depth output axis).
    LAYOUT: "ncdhw" or "ndhwc".
    """
    # W is always [K_out, C] contiguous (the 1x1x1 weight squeezed)
    stride_w_k: tl.constexpr = C
    stride_w_c: tl.constexpr = 1
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

    # M = N * D_out * P * Q (output spatial), N_dim = K_out (output channels)
    num_pid_m = tl.cdiv(M_total, BLOCK_M)
    num_pid_n = tl.cdiv(K_out, BLOCK_N)

    # L2 cache swizzle pattern
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
    offs_k = tl.arange(0, BLOCK_K)

    # Decode (n, d, p, q) from linear index
    dpq = D_out * P * Q
    pq = P * Q
    n_idx = offs_m // dpq
    rem = offs_m % dpq
    d_idx = rem // pq
    rem2 = rem % pq
    p_idx = rem2 // Q
    q_idx = rem2 % Q

    m_mask = offs_m < M_total
    n_mask = offs_n < K_out

    # Map output position back to input coordinate (stride/padding still apply)
    id_ = d_idx * stride_d - pad_d
    ih = p_idx * stride_h - pad_h
    iw = q_idx * stride_w - pad_w

    spatial_valid = (
        (id_ >= 0)
        & (id_ < D)
        & (ih >= 0)
        & (ih < H)
        & (iw >= 0)
        & (iw < W_in)
        & (n_idx < N)
    )

    x_base = (
        X + n_idx * stride_x_n + id_ * stride_x_d + ih * stride_x_h + iw * stride_x_w
    )  # [BLOCK_M]
    w_base = W + offs_n * stride_w_k  # [BLOCK_N]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Channel reduction loop (the only reduction for 1x1x1)
    for k0 in range(0, C, BLOCK_K):
        k_offs = k0 + offs_k
        k_mask = k_offs < C

        x_ptrs = x_base[:, None] + k_offs[None, :] * stride_x_c
        x_mask = spatial_valid[:, None] & k_mask[None, :]
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        w_ptrs = w_base[None, :] + k_offs[:, None] * stride_w_c
        w_mask = k_mask[:, None] & n_mask[None, :]
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    if HAS_BIAS:
        b = tl.load(BIAS + offs_n, mask=n_mask, other=0.0)
        acc += b[None, :]

    if ACTIVATION == "relu":
        acc = _relu(acc)
    elif ACTIVATION == "relu6":
        acc = _relu6(acc)
    elif ACTIVATION == "gelu":
        acc = _gelu_tanh(acc)

    y_ptrs = (
        Y
        + n_idx[:, None] * stride_y_n
        + offs_n[None, :] * stride_y_k
        + d_idx[:, None] * stride_y_d
        + p_idx[:, None] * stride_y_p
        + q_idx[:, None] * stride_y_q
    )
    y_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)


# Autotune search space (used when AITER_TRITON_CONV_AUTOTUNE=1).
AUTOTUNE_CONV3D_1X1X1_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
        num_warps=8,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
        num_warps=8,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 4},
        num_warps=8,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
        num_warps=4,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 4},
        num_warps=4,
        num_stages=1,
    ),
]


if CONV_AUTOTUNE_ENABLED:
    _conv3d_1x1x1_kernel = triton.autotune(
        configs=AUTOTUNE_CONV3D_1X1X1_CONFIGS,
        key=["M_total", "K_out", "C"],
        cache_results=True,
    )(_conv3d_1x1x1_kernel)
