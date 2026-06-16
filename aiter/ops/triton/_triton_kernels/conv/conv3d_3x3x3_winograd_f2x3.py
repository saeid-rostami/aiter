# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""3D Winograd F(2x2x2, 3x3x3) kernels (experimental, Phase 5).

Mirrors the 2D Winograd F(4x4,3x3) pipeline in conv_3x3_winograd_f4x3.py, but
expresses the separable data/output transforms as small tl.dot matmuls against
the Kronecker matrices BBB = (B^T)^(x)3 [64x64] and A3d = (A^T)^(x)3 [8->16 x 64]
instead of hand-unrolling 64+ scalar combinations per axis. For F(2,3) those
matrices are pure {-1,0,1}, so fp16/bf16 WMMA is exact in fp32 accumulation.

Pipeline (3 kernels):
  1. input transform : V[a, t, c] = BBB @ d(tile t, patch 4x4x4)   -> [64, T, C_pad]
  2. batched GEMM     : M[a]       = V[a] @ U[a]^T                  -> [64, T, K_out]
  3. output transform : Y(2x2x2)   = A3d @ M(tile t)               -> NCDHW / NDHWC

Tiles cover the output in 2x2x2 blocks; the input patch per tile is 4x4x4 = 64.
Alpha index a = ai*16 + aj*4 + ak (depth, height, width); output index
r = od*4 + oh*2 + ow. U (filter) is transformed on the host (G has 0.5 entries).
"""

import triton
import triton.language as tl
from aiter.ops.triton.utils.conv_config_utils import get_conv_config
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from .helpers import CONV_AUTOTUNE_ENABLED
from ..activation import _relu, _relu6, _gelu_tanh


def _get_config_input(shape_key=None, M=None):
    if CONV_AUTOTUNE_ENABLED:
        return {}
    return get_conv_config("CONV3D-WINO3D-F2X3-INPUT", shape_key=shape_key, M=M)


def _get_config_gemm(shape_key=None, M=None):
    if CONV_AUTOTUNE_ENABLED:
        return {}
    return get_conv_config("CONV3D-WINO3D-F2X3-GEMM", shape_key=shape_key, M=M)


def _get_config_output(shape_key=None, M=None):
    if CONV_AUTOTUNE_ENABLED:
        return {}
    return get_conv_config("CONV3D-WINO3D-F2X3-OUTPUT", shape_key=shape_key, M=M)


def _get_config_fused(shape_key=None, M=None):
    if CONV_AUTOTUNE_ENABLED:
        return {}
    return get_conv_config("CONV3D-WINO3D-F2X3-FUSED", shape_key=shape_key, M=M)


_winograd3d_f2x3_input_transform_kernel_repr = make_kernel_repr(
    "_winograd3d_f2x3_input_transform_kernel",
    ["BLOCK_C", "LAYOUT"],
)
_winograd3d_f2x3_cblocked_input_transform_kernel_repr = make_kernel_repr(
    "_winograd3d_f2x3_cblocked_input_transform_kernel",
    ["BLOCK_C"],
)
_winograd3d_f2x3_batched_gemm_kernel_repr = make_kernel_repr(
    "_winograd3d_f2x3_batched_gemm_kernel",
    ["BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_SIZE_M"],
)
_winograd3d_f2x3_output_transform_kernel_repr = make_kernel_repr(
    "_winograd3d_f2x3_output_transform_kernel",
    ["BLOCK_K", "HAS_BIAS", "ACTIVATION", "LAYOUT"],
)
_winograd3d_f2x3_fused_gemm_output_kernel_repr = make_kernel_repr(
    "_winograd3d_f2x3_fused_gemm_output_kernel",
    ["BLOCK_T", "BLOCK_K", "BLOCK_C", "HAS_BIAS", "ACTIVATION", "LAYOUT"],
)


@triton.jit(repr=_winograd3d_f2x3_input_transform_kernel_repr)
def _winograd3d_f2x3_input_transform_kernel(
    X,
    BBB,
    V,
    N: tl.constexpr,
    C: tl.constexpr,
    C_pad: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W_in: tl.constexpr,
    tile_D: tl.constexpr,
    tile_H: tl.constexpr,
    tile_W: tl.constexpr,
    T: tl.constexpr,
    pad_d: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    BLOCK_C: tl.constexpr,
    LAYOUT: tl.constexpr = "ncdhw",
):
    """V[a, tile, c] = (B^T (x) B^T (x) B^T) @ d, where d is the 4x4x4 input patch.
    One program == one (tile, channel-block). Reads NCDHW or NDHWC via LAYOUT."""
    INPUT_DTYPE: tl.constexpr = X.type.element_ty
    if LAYOUT == "ncdhw":
        stride_x_w: tl.constexpr = 1
        stride_x_h: tl.constexpr = W_in
        stride_x_d: tl.constexpr = H * W_in
        stride_x_c: tl.constexpr = D * H * W_in
        stride_x_n: tl.constexpr = C * D * H * W_in
    else:  # ndhwc
        stride_x_w: tl.constexpr = C
        stride_x_h: tl.constexpr = W_in * C
        stride_x_d: tl.constexpr = H * W_in * C
        stride_x_c: tl.constexpr = 1
        stride_x_n: tl.constexpr = D * H * W_in * C
    # V layout: [64, T, C_pad] contiguous
    stride_v_c: tl.constexpr = 1
    stride_v_tile: tl.constexpr = C_pad
    stride_v_alpha: tl.constexpr = T * C_pad

    tile_idx = tl.program_id(0)
    c_block = tl.program_id(1)

    thw = tile_H * tile_W
    n = tile_idx // (tile_D * thw)
    rem = tile_idx % (tile_D * thw)
    td = rem // thw
    rem2 = rem % thw
    th = rem2 // tile_W
    tw = rem2 % tile_W

    d_start = td * 2 - pad_d
    h_start = th * 2 - pad_h
    w_start = tw * 2 - pad_w

    offs_c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = offs_c < C

    # 64 patch positions: p = pi*16 + pj*4 + pk  (pi=depth, pj=h, pk=w in 0..3)
    p = tl.arange(0, 64)
    pi = p // 16
    pr = p % 16
    pj = pr // 4
    pk = pr % 4

    dd_ = d_start + pi
    hh = h_start + pj
    ww = w_start + pk
    n_valid = n < N
    valid = (
        n_valid
        & (dd_ >= 0)
        & (dd_ < D)
        & (hh >= 0)
        & (hh < H)
        & (ww >= 0)
        & (ww < W_in)
    )  # [64]
    spatial_off = dd_ * stride_x_d + hh * stride_x_h + ww * stride_x_w  # [64]

    # d_tile: [64, BLOCK_C] = patch positions x channels
    x_ptrs = (
        X
        + n * stride_x_n
        + spatial_off[:, None]
        + offs_c[None, :] * stride_x_c
    )
    d_tile = tl.load(x_ptrs, mask=valid[:, None] & c_mask[None, :], other=0.0)

    # BBB: [64, 64]
    j = tl.arange(0, 64)
    bbb = tl.load(BBB + p[:, None] * 64 + j[None, :])

    v = tl.dot(bbb, d_tile)  # [64, BLOCK_C], fp32 acc

    v_base = (
        V
        + tile_idx * stride_v_tile
        + offs_c[None, :] * stride_v_c
        + p[:, None] * stride_v_alpha
    )
    c_store_mask = offs_c < C_pad
    tl.store(v_base, v.to(INPUT_DTYPE), mask=c_store_mask[None, :])


@triton.jit(repr=_winograd3d_f2x3_cblocked_input_transform_kernel_repr)
def _winograd3d_f2x3_cblocked_input_transform_kernel(
    X,
    BBB,
    V,
    N: tl.constexpr,
    C: tl.constexpr,
    C_pad: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W_in: tl.constexpr,
    tile_D: tl.constexpr,
    tile_H: tl.constexpr,
    tile_W: tl.constexpr,
    T: tl.constexpr,
    pad_d: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    Cb: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Same as the input transform but reads channel-blocked NCDHWc input
    [N, C_blocks, D, H, W, Cb] for coalesced channel loads."""
    INPUT_DTYPE: tl.constexpr = X.type.element_ty
    # X layout: [N, C_blocks, D, H, W_in, Cb], C_blocks = C_pad // Cb
    stride_x_w: tl.constexpr = Cb
    stride_x_h: tl.constexpr = W_in * Cb
    stride_x_d: tl.constexpr = H * W_in * Cb
    stride_x_cblock: tl.constexpr = D * H * W_in * Cb
    stride_x_n: tl.constexpr = (C_pad // Cb) * D * H * W_in * Cb
    # V layout: [64, T, C_pad] contiguous
    stride_v_c: tl.constexpr = 1
    stride_v_tile: tl.constexpr = C_pad
    stride_v_alpha: tl.constexpr = T * C_pad

    tile_idx = tl.program_id(0)
    c_block = tl.program_id(1)

    thw = tile_H * tile_W
    n = tile_idx // (tile_D * thw)
    rem = tile_idx % (tile_D * thw)
    td = rem // thw
    rem2 = rem % thw
    th = rem2 // tile_W
    tw = rem2 % tile_W

    d_start = td * 2 - pad_d
    h_start = th * 2 - pad_h
    w_start = tw * 2 - pad_w

    offs_c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = offs_c < C
    cblock_idx = offs_c // Cb
    c_local = offs_c % Cb
    chan_off = cblock_idx * stride_x_cblock + c_local  # [BLOCK_C]

    p = tl.arange(0, 64)
    pi = p // 16
    pr = p % 16
    pj = pr // 4
    pk = pr % 4

    dd_ = d_start + pi
    hh = h_start + pj
    ww = w_start + pk
    n_valid = n < N
    valid = (
        n_valid
        & (dd_ >= 0)
        & (dd_ < D)
        & (hh >= 0)
        & (hh < H)
        & (ww >= 0)
        & (ww < W_in)
    )  # [64]
    spatial_off = dd_ * stride_x_d + hh * stride_x_h + ww * stride_x_w  # [64]

    x_ptrs = (
        X
        + n * stride_x_n
        + spatial_off[:, None]
        + chan_off[None, :]
    )
    d_tile = tl.load(x_ptrs, mask=valid[:, None] & c_mask[None, :], other=0.0)

    j = tl.arange(0, 64)
    bbb = tl.load(BBB + p[:, None] * 64 + j[None, :])
    v = tl.dot(bbb, d_tile)

    v_base = (
        V
        + tile_idx * stride_v_tile
        + offs_c[None, :] * stride_v_c
        + p[:, None] * stride_v_alpha
    )
    c_store_mask = offs_c < C_pad
    tl.store(v_base, v.to(INPUT_DTYPE), mask=c_store_mask[None, :])


@triton.jit(repr=_winograd3d_f2x3_batched_gemm_kernel_repr)
def _winograd3d_f2x3_batched_gemm_kernel(
    V,
    U,
    M_out,
    T: tl.constexpr,
    K_out: tl.constexpr,
    C_pad: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Batched GEMM over 64 alphas: M[a] = V[a] @ U[a]^T, a in [0..64)."""
    # V: [64, T, C_pad], U: [64, K_out, C_pad], M: [64, T, K_out]
    stride_v_c: tl.constexpr = 1
    stride_v_tile: tl.constexpr = C_pad
    stride_v_alpha: tl.constexpr = T * C_pad
    stride_u_c: tl.constexpr = 1
    stride_u_k: tl.constexpr = C_pad
    stride_u_alpha: tl.constexpr = K_out * C_pad
    stride_m_k: tl.constexpr = 1
    stride_m_tile: tl.constexpr = K_out
    stride_m_alpha: tl.constexpr = T * K_out

    pid = tl.program_id(0)
    alpha = tl.program_id(1)

    num_pid_m = tl.cdiv(T, BLOCK_M)
    num_pid_n = tl.cdiv(K_out, BLOCK_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    if pid_m >= num_pid_m or pid_n >= num_pid_n:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    v_base = V + alpha * stride_v_alpha
    u_base = U + alpha * stride_u_alpha

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, C_pad, BLOCK_K):
        k_offs = k0 + offs_k
        v_ptrs = v_base + offs_m[:, None] * stride_v_tile + k_offs[None, :] * stride_v_c
        v_mask = (offs_m[:, None] < T) & (k_offs[None, :] < C_pad)
        v_tile = tl.load(v_ptrs, mask=v_mask, other=0.0)

        u_ptrs = u_base + offs_n[:, None] * stride_u_k + k_offs[None, :] * stride_u_c
        u_mask = (offs_n[:, None] < K_out) & (k_offs[None, :] < C_pad)
        u_tile = tl.load(u_ptrs, mask=u_mask, other=0.0)

        acc = tl.dot(v_tile, tl.trans(u_tile), acc=acc)

    m_ptrs = (
        M_out
        + alpha * stride_m_alpha
        + offs_m[:, None] * stride_m_tile
        + offs_n[None, :] * stride_m_k
    )
    m_mask = (offs_m[:, None] < T) & (offs_n[None, :] < K_out)
    # Store as fp16/bf16 to feed the fp16 WMMA output transform (acc stays fp32 there).
    tl.store(m_ptrs, acc.to(M_out.type.element_ty), mask=m_mask)


@triton.jit(repr=_winograd3d_f2x3_output_transform_kernel_repr)
def _winograd3d_f2x3_output_transform_kernel(
    M_in,
    A3D,
    BIAS,
    Y,
    N: tl.constexpr,
    K_out: tl.constexpr,
    D_out: tl.constexpr,
    P: tl.constexpr,
    Q: tl.constexpr,
    tile_D: tl.constexpr,
    tile_H: tl.constexpr,
    tile_W: tl.constexpr,
    T: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACTIVATION: tl.constexpr,
    LAYOUT: tl.constexpr = "ncdhw",
):
    """Y(2x2x2) = (A^T (x) A^T (x) A^T) @ M(tile). A3D is [16,64] (8 real rows,
    padded to 16 for the WMMA M-floor). One program == one (tile, k-block)."""
    # M: [64, T, K_out]
    stride_m_k: tl.constexpr = 1
    stride_m_tile: tl.constexpr = K_out
    stride_m_alpha: tl.constexpr = T * K_out
    if LAYOUT == "ncdhw":
        stride_y_q: tl.constexpr = 1
        stride_y_p: tl.constexpr = Q
        stride_y_d: tl.constexpr = P * Q
        stride_y_k: tl.constexpr = D_out * P * Q
        stride_y_n: tl.constexpr = K_out * D_out * P * Q
    else:  # ndhwc
        stride_y_q: tl.constexpr = K_out
        stride_y_p: tl.constexpr = Q * K_out
        stride_y_d: tl.constexpr = P * Q * K_out
        stride_y_k: tl.constexpr = 1
        stride_y_n: tl.constexpr = D_out * P * Q * K_out

    tile_idx = tl.program_id(0)
    k_block = tl.program_id(1)

    thw = tile_H * tile_W
    n = tile_idx // (tile_D * thw)
    rem = tile_idx % (tile_D * thw)
    td = rem // thw
    rem2 = rem % thw
    th = rem2 // tile_W
    tw = rem2 % tile_W

    d_start = td * 2
    p_start = th * 2
    q_start = tw * 2

    offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = offs_k < K_out

    # M_tile: [64, BLOCK_K] = alphas x output channels
    a = tl.arange(0, 64)
    m_ptrs = (
        M_in
        + tile_idx * stride_m_tile
        + a[:, None] * stride_m_alpha
        + offs_k[None, :] * stride_m_k
    )
    m_tile = tl.load(m_ptrs, mask=k_mask[None, :], other=0.0)  # [64, BLOCK_K]

    # A3D: [16, 64]
    r16 = tl.arange(0, 16)
    a3d = tl.load(A3D + r16[:, None] * 64 + a[None, :])
    yt = tl.dot(a3d, m_tile)  # [16, BLOCK_K], fp32; rows 0..7 are the 2x2x2 outputs

    if HAS_BIAS:
        bias = tl.load(BIAS + offs_k, mask=k_mask, other=0.0)  # [BLOCK_K]
        yt = yt + bias[None, :]

    if ACTIVATION == "relu":
        yt = _relu(yt)
    elif ACTIVATION == "relu6":
        yt = _relu6(yt)
    elif ACTIVATION == "gelu":
        yt = _gelu_tanh(yt)

    # Scatter the 8 outputs (r = od*4 + oh*2 + ow) to their NCDHW/NDHWC positions.
    n_valid = n < N
    y_base = Y + n * stride_y_n + offs_k * stride_y_k
    if n_valid:
        for r in tl.static_range(8):
            od = d_start + (r // 4)
            oh = p_start + ((r % 4) // 2)
            ow = q_start + (r % 2)
            if (od < D_out) and (oh < P) and (ow < Q):
                row = tl.sum(tl.where(tl.arange(0, 16)[:, None] == r, yt, 0.0), axis=0)
                tl.store(
                    y_base + od * stride_y_d + oh * stride_y_p + ow * stride_y_q,
                    row,
                    mask=k_mask,
                )


@triton.jit(repr=_winograd3d_f2x3_fused_gemm_output_kernel_repr)
def _winograd3d_f2x3_fused_gemm_output_kernel(
    V,
    U,
    A3D,
    BIAS,
    Y,
    N: tl.constexpr,
    K_out: tl.constexpr,
    D_out: tl.constexpr,
    P: tl.constexpr,
    Q: tl.constexpr,
    C_pad: tl.constexpr,
    tile_D: tl.constexpr,
    tile_H: tl.constexpr,
    tile_W: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACTIVATION: tl.constexpr,
    LAYOUT: tl.constexpr = "ncdhw",
):
    """Fused batched-GEMM + output transform: avoids materializing M[64,T,K_out].

    One program owns a (tile-block, k-block). It streams the 64 alpha GEMMs
    (M[a] = V[a] @ U[a]^T, reduced over C) and distributes each alpha's result
    into 8 output accumulators y0..y7 weighted by A3D[o, a] (= (A^T)^(x)3). After
    all alphas, it applies bias/activation and scatter-stores the 2x2x2 outputs.
    A3D is the [16,64] matrix (rows 0..7 used). Inputs V/U are fp16/bf16; the
    GEMM and transform accumulate in fp32."""
    stride_v_c: tl.constexpr = 1
    stride_v_tile: tl.constexpr = C_pad
    stride_v_alpha: tl.constexpr = T * C_pad
    stride_u_c: tl.constexpr = 1
    stride_u_k: tl.constexpr = C_pad
    stride_u_alpha: tl.constexpr = K_out * C_pad
    if LAYOUT == "ncdhw":
        stride_y_q: tl.constexpr = 1
        stride_y_p: tl.constexpr = Q
        stride_y_d: tl.constexpr = P * Q
        stride_y_k: tl.constexpr = D_out * P * Q
        stride_y_n: tl.constexpr = K_out * D_out * P * Q
    else:
        stride_y_q: tl.constexpr = K_out
        stride_y_p: tl.constexpr = Q * K_out
        stride_y_d: tl.constexpr = P * Q * K_out
        stride_y_k: tl.constexpr = 1
        stride_y_n: tl.constexpr = D_out * P * Q * K_out

    pid_t = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    t_mask = offs_t < T
    k_mask = offs_k < K_out
    offs_c = tl.arange(0, BLOCK_C)

    y0 = tl.zeros((BLOCK_T, BLOCK_K), dtype=tl.float32)
    y1 = tl.zeros((BLOCK_T, BLOCK_K), dtype=tl.float32)
    y2 = tl.zeros((BLOCK_T, BLOCK_K), dtype=tl.float32)
    y3 = tl.zeros((BLOCK_T, BLOCK_K), dtype=tl.float32)
    y4 = tl.zeros((BLOCK_T, BLOCK_K), dtype=tl.float32)
    y5 = tl.zeros((BLOCK_T, BLOCK_K), dtype=tl.float32)
    y6 = tl.zeros((BLOCK_T, BLOCK_K), dtype=tl.float32)
    y7 = tl.zeros((BLOCK_T, BLOCK_K), dtype=tl.float32)

    for alpha in tl.range(0, 64):
        v_base = V + alpha * stride_v_alpha
        u_base = U + alpha * stride_u_alpha
        m = tl.zeros((BLOCK_T, BLOCK_K), dtype=tl.float32)
        for c0 in range(0, C_pad, BLOCK_C):
            coff = c0 + offs_c
            cm = coff < C_pad
            v = tl.load(
                v_base + offs_t[:, None] * stride_v_tile + coff[None, :] * stride_v_c,
                mask=t_mask[:, None] & cm[None, :],
                other=0.0,
            )
            u = tl.load(
                u_base + offs_k[:, None] * stride_u_k + coff[None, :] * stride_u_c,
                mask=k_mask[:, None] & cm[None, :],
                other=0.0,
            )
            m = tl.dot(v, tl.trans(u), acc=m)
        # Distribute m into the 8 output accumulators (A3D[o, alpha] in {-1,0,1}).
        for o in tl.static_range(8):
            coef = tl.load(A3D + o * 64 + alpha)
            cmm = coef * m
            if o == 0:
                y0 += cmm
            elif o == 1:
                y1 += cmm
            elif o == 2:
                y2 += cmm
            elif o == 3:
                y3 += cmm
            elif o == 4:
                y4 += cmm
            elif o == 5:
                y5 += cmm
            elif o == 6:
                y6 += cmm
            else:
                y7 += cmm

    # Epilogue: decode each tile in the block, then bias/activation/scatter-store.
    thw = tile_H * tile_W
    n = offs_t // (tile_D * thw)
    rem = offs_t % (tile_D * thw)
    td = rem // thw
    rem2 = rem % thw
    th = rem2 // tile_W
    tw = rem2 % tile_W
    n_valid = n < N

    if HAS_BIAS:
        bias = tl.load(BIAS + offs_k, mask=k_mask, other=0.0)  # [BLOCK_K]

    for o in tl.static_range(8):
        od = o // 4
        oh = (o % 4) // 2
        ow = o % 2
        if o == 0:
            y = y0
        elif o == 1:
            y = y1
        elif o == 2:
            y = y2
        elif o == 3:
            y = y3
        elif o == 4:
            y = y4
        elif o == 5:
            y = y5
        elif o == 6:
            y = y6
        else:
            y = y7

        if HAS_BIAS:
            y = y + bias[None, :]
        if ACTIVATION == "relu":
            y = _relu(y)
        elif ACTIVATION == "relu6":
            y = _relu6(y)
        elif ACTIVATION == "gelu":
            y = _gelu_tanh(y)

        d_idx = td * 2 + od
        p_idx = th * 2 + oh
        q_idx = tw * 2 + ow
        y_ptrs = (
            Y
            + n[:, None] * stride_y_n
            + offs_k[None, :] * stride_y_k
            + d_idx[:, None] * stride_y_d
            + p_idx[:, None] * stride_y_p
            + q_idx[:, None] * stride_y_q
        )
        store_mask = (
            t_mask[:, None]
            & k_mask[None, :]
            & n_valid[:, None]
            & (d_idx < D_out)[:, None]
            & (p_idx < P)[:, None]
            & (q_idx < Q)[:, None]
        )
        tl.store(y_ptrs, y.to(Y.type.element_ty), mask=store_mask)
