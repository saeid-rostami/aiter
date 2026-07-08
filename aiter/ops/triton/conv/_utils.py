# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

# Channel padding granularity for prepacked weights/inputs. Must align with the
# BLOCK_K autotune candidates in _triton_kernels/conv/helpers.py — change with care.
BLOCK_K = 64


def _out_hw(H, W, R, S, stride, padding, dilation):
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    P = (H + 2 * ph - dh * (R - 1) - 1) // sh + 1
    Q = (W + 2 * pw - dw * (S - 1) - 1) // sw + 1
    return P, Q


def _conv_dims(x, w_oihw, stride, padding, dilation):
    """Shared wrapper preamble: validate inputs and return the conv dimensions."""
    assert x.is_cuda and w_oihw.is_cuda
    N, C, H, W_in = x.shape
    K_out, Cw, R, S = w_oihw.shape
    assert Cw == C
    P, Q = _out_hw(H, W_in, R, S, stride, padding, dilation)
    return N, C, H, W_in, K_out, R, S, P, Q


def _out_dhw(D, H, W, T, R, S, stride, padding, dilation):
    """3D output extent. `stride`/`padding`/`dilation` are 3-tuples (depth,
    height, width)."""
    sd, sh, sw = stride
    pd, ph, pw = padding
    dd, dh, dw = dilation
    OD = (D + 2 * pd - dd * (T - 1) - 1) // sd + 1
    P = (H + 2 * ph - dh * (R - 1) - 1) // sh + 1
    Q = (W + 2 * pw - dw * (S - 1) - 1) // sw + 1
    return OD, P, Q


def _conv3d_dims(x, w_oidhw, stride, padding, dilation):
    """3D counterpart of :func:`_conv_dims`: validate a NCDHW input against an
    OIDHW weight and return the conv dimensions.

    Weight is PyTorch-canonical ``[K_out, C, T, R, S]`` (T=depth tap).
    """
    assert x.is_cuda and w_oidhw.is_cuda
    N, C, D, H, W_in = x.shape
    K_out, Cw, T, R, S = w_oidhw.shape
    assert Cw == C, f"weight in-channels {Cw} != input channels {C}"
    OD, P, Q = _out_dhw(D, H, W_in, T, R, S, stride, padding, dilation)
    return N, C, D, H, W_in, K_out, T, R, S, OD, P, Q


def _alloc_output(N, K_out, P, Q, x, layout):
    """Allocate the output tensor, channels_last for nhwc else contiguous."""
    y = torch.empty((N, K_out, P, Q), device=x.device, dtype=x.dtype)
    if layout == "nhwc":
        return y.to(memory_format=torch.channels_last)
    return y


def _alloc_output_3d(N, K_out, OD, P, Q, x, layout):
    """Allocate the 3D output tensor: channels_last_3d for ndhwc, else
    contiguous NCDHW. Returned in logical NCDHW shape either way."""
    y = torch.empty((N, K_out, OD, P, Q), device=x.device, dtype=x.dtype)
    if layout == "ndhwc":
        return y.to(memory_format=torch.channels_last_3d)
    return y


def _prep_bias(bias):
    """Cast bias to contiguous fp32 for the kernels, or None when absent."""
    return bias.float().contiguous() if bias is not None else None


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


def _is_3x3x3_conv(T, R, S):
    """Check if this is a 3x3x3 (depth x height x width) convolution."""
    return T == 3 and R == 3 and S == 3


def _is_1x1x1_conv(T, R, S, dilation):
    """Check if this is a 1x1x1 convolution (no spatial reduction in kernel)."""
    return T == 1 and R == 1 and S == 1 and dilation == (1, 1, 1)


def _is_winograd3d_hw_eligible(T, R, S, stride, dilation, C=None):
    """Eligibility for 2.5D Winograd F(4,3)-on-HW: 3x3x3, unit stride/dilation.
    (Depth is a direct 3-tap reduction, so only H,W need the Winograd geometry.)"""
    if not (T == 3 and R == 3 and S == 3):
        return False
    if stride != (1, 1, 1) or dilation != (1, 1, 1):
        return False
    # Winograd output transform amplifies bf16 rounding; too few channels can't
    # absorb it (same rationale as 2D _is_winograd_eligible).
    if C is not None and C < 4:
        return False
    return True


def _is_winograd_eligible(R, S, stride, dilation, C=None):
    if not (R == 3 and S == 3 and stride == (1, 1) and dilation == (1, 1)):
        return False
    # F(4,3) output transform amplifies bf16 rounding by up to 361x (AT row3 L1=19).
    # With very few input channels the tolerance budget is too small to absorb this.
    if C is not None and C < 4:
        return False
    return True


def _require_winograd_eligible(name, R, S, stride, dilation, C):
    """Raise a uniform ValueError if this shape isn't Winograd F(4,3)-eligible."""
    if not _is_winograd_eligible(R, S, stride, dilation, C):
        raise ValueError(
            f"{name} requires 3x3 kernel with stride=1, dilation=1, "
            f"and C >= 4 (F(4,3) output transform amplifies rounding by up to "
            f"361x; C<4 has too few reduction terms to absorb it), "
            f"got {R}x{S} stride={stride} dilation={dilation} C={C}"
        )
