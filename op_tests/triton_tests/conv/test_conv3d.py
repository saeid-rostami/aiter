# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Pytest unit tests for aiter.ops.triton.conv.conv3d.

Correctness only. Compares the Triton general conv3d kernel against
torch.nn.functional.conv3d on synthetic tensors, in both NCDHW and NDHWC
(channels-last-3d) layouts. No model loading, no network.

The headline shapes are the Wan-style VAE encoder/decoder 3x3x3 convs
(stride 1, pad 1) the kernel was first built for; encoder and decoder
collapse to the same 4 distinct shapes. A few small extra shapes exercise
stride/pad/dilation and non-3x3x3 kernels through the same general path.

Tolerances reuse the shared dynamic model with K_red = C*T*R*S.
"""

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.conv.conv3d import conv3d, _resolve_route, Route3D

from ._helpers import ALL_SUPPORTED_ARCHS, dynamic_conv_tolerances

_current_arch = get_arch()
if _current_arch not in ALL_SUPPORTED_ARCHS:
    pytest.skip(
        f"aiter.ops.triton.conv tests run on {sorted(ALL_SUPPORTED_ARCHS)}; "
        f"current arch {_current_arch!r} not supported",
        allow_module_level=True,
    )


# (name, N, C, D, H, W, K) — all 3x3x3, stride 1, pad 1, dil 1.
VAE_SHAPES = [
    ("vae_384_3x46x51", 1, 384, 3, 46, 51, 384),
    ("vae_96_6x354x394", 1, 96, 6, 354, 394, 96),
    ("vae_192_6x178x198", 1, 192, 6, 178, 198, 192),
    ("vae_384_4x90x100", 1, 384, 4, 90, 100, 384),
]

# (name, N, C, D, H, W, K, T, R, S, stride, pad, dil) — covers both the general
# path and the specialized 1x1x1 path (routing verified in test_routing).
EXTRA_SHAPES = [
    ("stride2", 1, 32, 8, 32, 32, 32, 3, 3, 3, (2, 2, 2), (1, 1, 1), (1, 1, 1)),
    ("pad0", 1, 16, 6, 24, 24, 16, 3, 3, 3, (1, 1, 1), (0, 0, 0), (1, 1, 1)),
    ("k1x1x1", 1, 64, 4, 16, 16, 128, 1, 1, 1, (1, 1, 1), (0, 0, 0), (1, 1, 1)),
    ("k1x1x1_proj", 1, 384, 3, 46, 51, 192, 1, 1, 1, (1, 1, 1), (0, 0, 0), (1, 1, 1)),
    ("k1x1x1_s2", 1, 96, 6, 32, 32, 96, 1, 1, 1, (2, 2, 2), (0, 0, 0), (1, 1, 1)),
    ("k5x5x5", 1, 16, 8, 20, 20, 16, 5, 5, 5, (1, 1, 1), (2, 2, 2), (1, 1, 1)),
    ("dilated", 1, 16, 8, 24, 24, 16, 3, 3, 3, (1, 1, 1), (2, 2, 2), (2, 2, 2)),
]

DTYPES = [(torch.float16, "fp16"), (torch.bfloat16, "bf16")]


def _run_case(
    N, C, D, H, W, K, T, R, S, stride, pad, dil, dtype, bias, act, layout="ncdhw"
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.manual_seed(0)
    x = torch.randn(N, C, D, H, W, device="cuda", dtype=dtype)
    w = torch.randn(K, C, T, R, S, device="cuda", dtype=dtype)
    b = torch.randn(K, device="cuda", dtype=dtype) if bias else None

    y = conv3d(
        x, w, b, stride=stride, padding=pad, dilation=dil, activation=act, layout=layout
    )

    ref = F.conv3d(
        x.float(),
        w.float(),
        b.float() if b is not None else None,
        stride=stride,
        padding=pad,
        dilation=dil,
    )
    if act == "relu":
        ref = F.relu(ref)

    K_red = C * T * R * S
    rtol, atol = dynamic_conv_tolerances(dtype, K_red)
    y32 = y.float()
    torch.testing.assert_close(y32, ref.to(y32.dtype), rtol=rtol, atol=atol)


LAYOUTS = ["ncdhw", "ndhwc"]


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype,dtype_id", DTYPES, ids=[d[1] for d in DTYPES])
@pytest.mark.parametrize("shape", VAE_SHAPES, ids=[s[0] for s in VAE_SHAPES])
def test_vae_shapes(shape, dtype, dtype_id, layout):
    _, N, C, D, H, W, K = shape
    _run_case(
        N,
        C,
        D,
        H,
        W,
        K,
        3,
        3,
        3,
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        dtype,
        bias=True,
        act="none",
        layout=layout,
    )


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype,dtype_id", DTYPES, ids=[d[1] for d in DTYPES])
def test_vae_bias_and_relu(dtype, dtype_id, layout):
    # Smallest VAE shape, exercise the fused bias + relu epilogue.
    _run_case(
        1,
        384,
        3,
        46,
        51,
        384,
        3,
        3,
        3,
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        dtype,
        bias=True,
        act="relu",
        layout=layout,
    )


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype,dtype_id", DTYPES, ids=[d[1] for d in DTYPES])
@pytest.mark.parametrize("shape", EXTRA_SHAPES, ids=[s[0] for s in EXTRA_SHAPES])
def test_extra_shapes(shape, dtype, dtype_id, layout):
    name, N, C, D, H, W, K, T, R, S, stride, pad, dil = shape
    _run_case(
        N,
        C,
        D,
        H,
        W,
        K,
        T,
        R,
        S,
        stride,
        pad,
        dil,
        dtype,
        bias=False,
        act="none",
        layout=layout,
    )


def test_routing():
    # 1x1x1 -> specialized channel-GEMM (layout-independent).
    assert _resolve_route(1, 1, 1, (1, 1, 1), "ncdhw") is Route3D.ONE_X_ONE_X_ONE
    assert _resolve_route(1, 1, 1, (1, 1, 1), "ndhwc") is Route3D.ONE_X_ONE_X_ONE
    # 3x3x3 -> specialized kernel picked by layout.
    assert _resolve_route(3, 3, 3, (1, 1, 1), "ncdhw") is Route3D.CBLOCKED_NCDHW
    assert _resolve_route(3, 3, 3, (1, 1, 1), "ndhwc") is Route3D.NDHWC_3X3X3
    # No specialized kernel -> general.
    assert _resolve_route(5, 5, 5, (1, 1, 1), "ncdhw") is Route3D.GENERAL
    # Dilated variants have no specialized kernel yet -> general.
    assert _resolve_route(3, 3, 3, (2, 2, 2), "ncdhw") is Route3D.GENERAL
    assert _resolve_route(1, 1, 1, (2, 2, 2), "ncdhw") is Route3D.GENERAL


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
