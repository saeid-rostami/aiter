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


def _alloc_output(N, K_out, P, Q, x, layout):
    """Allocate the output tensor, channels_last for nhwc else contiguous.

    The nhwc buffer is allocated directly in channels_last. Going through
    ``torch.empty(...).to(memory_format=...)`` instead would allocate a second
    buffer and launch a full strided transposing copy of uninitialized memory
    before the kernel ever writes to it.
    """
    if layout == "nhwc":
        return torch.empty(
            (N, K_out, P, Q),
            device=x.device,
            dtype=x.dtype,
            memory_format=torch.channels_last,
        )
    return torch.empty((N, K_out, P, Q), device=x.device, dtype=x.dtype)


def _prep_bias(bias):
    """Hand the bias to the kernels in its native dtype, or None when absent.

    The kernels index ``BIAS + offs_n``, so the only requirement is a unit
    innermost stride; the fp16/bf16 -> fp32 promotion happens in the epilogue,
    where it is exact and free. Casting on the host instead would add a whole
    extra GPU dispatch per conv call for a K-element tensor that is a caller-
    owned, loop-invariant ``nn.Conv2d`` parameter.
    """
    if bias is None:
        return None
    return bias if bias.stride(0) == 1 else bias.contiguous()


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


def _is_winograd_eligible(R, S, stride, dilation, C=None):
    if not (R == 3 and S == 3 and stride == (1, 1) and dilation == (1, 1)):
        return False
    # F(4,3) output transform amplifies bf16 rounding by up to 361x (AT row3 L1=19).
    # With very few input channels the tolerance budget is too small to absorb this.
    return not (C is not None and C < 4)


def _require_winograd_eligible(name, R, S, stride, dilation, C):
    """Raise a uniform ValueError if this shape isn't Winograd F(4,3)-eligible."""
    if not _is_winograd_eligible(R, S, stride, dilation, C):
        raise ValueError(
            f"{name} requires 3x3 kernel with stride=1, dilation=1, "
            f"and C >= 4 (F(4,3) output transform amplifies rounding by up to "
            f"361x; C<4 has too few reduction terms to absorb it), "
            f"got {R}x{S} stride={stride} dilation={dilation} C={C}"
        )
