# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F

# Channel padding granularity for prepacked weights/inputs. Must align with the
# BLOCK_K autotune candidates in _triton_kernels/conv/helpers.py — change with care.
BLOCK_K = 64


def dynamic_conv_tolerances(dtype: torch.dtype, K_red: int, ref: torch.Tensor):
    eps = {
        torch.float16: 2**-10,
        torch.bfloat16: 2**-7,
        torch.float32: 2**-23,
    }.get(dtype, 2**-10)
    rtol = 6e-3 if K_red < 1024 else (8e-3 if K_red < 4096 else 1.2e-2)
    # Error model: fp16 inputs multiplied pairwise have eps relative error per product.
    # Accumulated in fp32 over K_red terms, max absolute error grows as ~eps * sqrt(K_red).
    # The 10x multiplier covers worst-case accumulation ordering differences
    # between our Triton kernels and PyTorch reference.
    atol = max(eps * 8, 10.0 * eps * (K_red**0.5))
    return rtol, atol


def flops_conv(N, C, K_out, R, S, P, Q):
    return 2.0 * N * P * Q * K_out * C * R * S


def _out_hw(H, W, R, S, stride, padding, dilation):
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    P = (H + 2 * ph - dh * (R - 1) - 1) // sh + 1
    Q = (W + 2 * pw - dw * (S - 1) - 1) // sw + 1
    return P, Q


def _out_dhw(D, H, W, KD, KH, KW, stride, padding, dilation):
    """3D analog of _out_hw. stride/padding/dilation are length-3 (d, h, w)."""
    sd, sh, sw = stride
    pd, ph, pw = padding
    dd, dh, dw = dilation
    D_out = (D + 2 * pd - dd * (KD - 1) - 1) // sd + 1
    P = (H + 2 * ph - dh * (KH - 1) - 1) // sh + 1
    Q = (W + 2 * pw - dw * (KW - 1) - 1) // sw + 1
    return D_out, P, Q


def _storage_ptr(t: torch.Tensor) -> int:
    return (
        t.untyped_storage().data_ptr()
        if hasattr(t, "untyped_storage")
        else t.storage().data_ptr()
    )


def _is_1x1_conv(R, S, dilation):
    """Check if this is a 1x1 convolution (no spatial reduction in kernel)."""
    return R == 1 and S == 1 and dilation == (1, 1)


def _is_3x3_conv(R, S):
    """Check if this is a 3x3 convolution."""
    return R == 3 and S == 3


def _is_1x1x1_conv(KD, KH, KW, dilation):
    """Check if this is a 1x1x1 convolution (no spatial reduction in kernel)."""
    return KD == 1 and KH == 1 and KW == 1 and dilation == (1, 1, 1)


def _is_3x3x3_conv(KD, KH, KW):
    """Check if this is a 3x3x3 convolution."""
    return KD == 3 and KH == 3 and KW == 3


def _is_winograd_eligible(R, S, stride, dilation, C=None):
    if not (R == 3 and S == 3 and stride == (1, 1) and dilation == (1, 1)):
        return False
    # F(4,3) output transform amplifies bf16 rounding by up to 361x (AT row3 L1=19).
    # With very few input channels the tolerance budget is too small to absorb this.
    if C is not None and C < 4:
        return False
    return True


def _winograd_tolerances(dtype, K_red, ref, variant="f4x3"):
    """Return (rtol, atol) for Winograd F(4x4,3x3) correctness checks.
    Winograd transforms amplify fp16 rounding errors:
    - F(4x4,3x3): coefficients up to ±8, significant amplification
    """
    rtol, atol = dynamic_conv_tolerances(dtype, K_red, ref)
    if variant == "f4x3":
        rtol *= 6.0
        atol = max(atol * 6.0, 0.6)
    return rtol, atol


def _is_winograd3d_eligible(KD, KH, KW, stride, dilation, C=None):
    """Eligibility for 3D Winograd F(2x2x2, 3x3x3): 3x3x3, unit stride/dilation."""
    if not (
        KD == 3
        and KH == 3
        and KW == 3
        and stride == (1, 1, 1)
        and dilation == (1, 1, 1)
    ):
        return False
    if C is not None and C < 4:
        return False
    return True


def _winograd3d_tolerances(dtype, K_red, ref, variant="f2x3"):
    """Return (rtol, atol) for 3D Winograd F(2x2x2,3x3x3) correctness checks.

    F(2,3) transform matrices (B^T, A^T) are pure {-1,0,1}, so the data/output
    transforms add no rounding beyond the fp32 accumulation. The extra error vs a
    direct conv comes from (a) the fp32->fp16 round of the per-tile GEMM result M
    and (b) the filter transform G (entries up to 0.5) done in fp32 on the host.
    A modest tolerance bump (vs the direct 3x3x3 path) absorbs that.
    """
    rtol, atol = dynamic_conv_tolerances(dtype, K_red, ref)
    if variant == "f2x3":
        rtol *= 3.0
        atol = max(atol * 3.0, 0.3)
    return rtol, atol


# --- 3D Winograd F(2x2x2, 3x3x3) transform matrices --------------------------
# F(2,3) 1-D matrices (Winograd):
#   B^T (4x4) data transform, A^T (2x4) output transform, G (4x3) filter transform.
# The 3-D transforms are the Kronecker (separable) products applied on (d, h, w).
_WINO3D_BT = [
    [1.0, 0.0, -1.0, 0.0],
    [0.0, 1.0, 1.0, 0.0],
    [0.0, -1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, -1.0],
]
_WINO3D_AT = [
    [1.0, 1.0, 1.0, 0.0],
    [0.0, 1.0, -1.0, -1.0],
]
_WINO3D_G = [
    [1.0, 0.0, 0.0],
    [0.5, 0.5, 0.5],
    [0.5, -0.5, 0.5],
    [0.0, 0.0, 1.0],
]

# Cache the device/dtype-resident kernel matrices (BBB: [64,64], A3d: [16,64]).
_WINO3D_MAT_CACHE = {}


def _kron3(m: torch.Tensor) -> torch.Tensor:
    """Separable 3-axis Kronecker product m (x) m (x) m."""
    return torch.kron(torch.kron(m, m), m)


def get_winograd3d_kernel_matrices(device, dtype):
    """Return (BBB, A3d) for the in-kernel transforms, in `dtype` on `device`.

    BBB = B^T (x) B^T (x) B^T  -> [64, 64]   (input/data transform).
    A3d = A^T (x) A^T (x) A^T  -> [8, 64], zero-padded to [16, 64] so the output
    transform's tl.dot has M>=16 (WMMA tile floor). Both are {-1,0,1}, exact in fp16.
    """
    key = (str(device), dtype)
    cached = _WINO3D_MAT_CACHE.get(key)
    if cached is not None:
        return cached
    bt = torch.tensor(_WINO3D_BT, dtype=torch.float32, device=device)
    at = torch.tensor(_WINO3D_AT, dtype=torch.float32, device=device)
    bbb = _kron3(bt).contiguous()  # [64, 64]
    a3d_small = _kron3(at).contiguous()  # [8, 64]
    a3d = torch.zeros((16, 64), dtype=torch.float32, device=device)
    a3d[:8, :] = a3d_small
    item = (bbb.to(dtype).contiguous(), a3d.to(dtype).contiguous())
    _WINO3D_MAT_CACHE[key] = item
    return item


def winograd3d_filter_matrix(device) -> torch.Tensor:
    """G (x) G (x) G  -> [64, 27], fp32. Maps a flat 3x3x3 filter to 64 alphas."""
    g = torch.tensor(_WINO3D_G, dtype=torch.float32, device=device)
    return _kron3(g).contiguous()


def apply_activation(y: torch.Tensor, activation: str):
    if activation == "relu":
        return F.relu(y)
    if activation == "relu6":
        return torch.clamp(y, 0, 6)
    if activation == "gelu":
        return F.gelu(y, approximate="tanh")
    return y
