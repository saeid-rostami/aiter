# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Pytest unit tests for aiter.ops.triton.conv.conv3d.

Correctness only. All tests compare Triton kernels against
torch.nn.functional.conv3d on synthetic tensors. Mirrors the 2D suite in
op_tests/triton_tests/conv/test_conv2d.py.

Phase 1: only the general (implicit-GEMM) kernel exists, so the (dtype, layout,
method) matrix has a single method ("default"). 1x1x1 / 3x3x3 specializations
extend ORDERED_METHODS in _helpers.py and are picked up automatically here.
"""

import pytest
import torch

from aiter.ops.triton.utils._triton.arch_info import get_arch

from ._helpers import (
    ALL_SUPPORTED_ARCHS,
    TestSuite,
    ORDERED_METHODS,
    run_edge_cases,
    run_activations,
    run_no_bias,
    run_random_fuzzing,
)

_current_arch = get_arch()
if _current_arch not in ALL_SUPPORTED_ARCHS:
    pytest.skip(
        f"aiter.ops.triton.conv.conv3d tests run on {sorted(ALL_SUPPORTED_ARCHS)}; "
        f"current arch {_current_arch!r} not supported",
        allow_module_level=True,
    )


def _build_matrix():
    matrix = []
    for dtype, dtype_id in [(torch.float16, "fp16"), (torch.bfloat16, "bf16")]:
        for method in ORDERED_METHODS:
            matrix.append(((dtype, "ncdhw", method), f"{dtype_id}_ncdhw_{method}"))
        matrix.append(((dtype, "ndhwc", "default"), f"{dtype_id}_ndhwc"))
        # Explicit NDHWC general-kernel coverage (bypasses the router; see
        # run_all_methods' ndhwc branch handling of method="general").
        matrix.append(((dtype, "ndhwc", "general"), f"{dtype_id}_ndhwc_general"))
    return matrix


_MATRIX = _build_matrix()
PARAMS = [params for params, _ in _MATRIX]
IDS = [tid for _, tid in _MATRIX]


def _make_suite(dtype, layout):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return TestSuite(device="cuda", dtype=dtype, layout_mode=layout)


def _assert_suite(suite: TestSuite):
    failed = suite.failed_results()
    assert not failed, f"{len(failed)} tests failed: {[r.name for r in failed]}"


@pytest.mark.parametrize("dtype,layout,method", PARAMS, ids=IDS)
def test_edge(dtype, layout, method):
    suite = _make_suite(dtype, layout)
    run_edge_cases(suite, method=method)
    _assert_suite(suite)


@pytest.mark.parametrize("dtype,layout,method", PARAMS, ids=IDS)
def test_fuzz(dtype, layout, method):
    suite = _make_suite(dtype, layout)
    run_random_fuzzing(suite, num_tests=10, method=method)
    _assert_suite(suite)


@pytest.mark.parametrize("dtype,layout,method", PARAMS, ids=IDS)
def test_no_bias(dtype, layout, method):
    suite = _make_suite(dtype, layout)
    run_no_bias(suite, method=method)
    _assert_suite(suite)


@pytest.mark.parametrize("activation", ["relu", "relu6", "gelu"])
@pytest.mark.parametrize("dtype,layout,method", PARAMS, ids=IDS)
def test_activations(dtype, layout, method, activation):
    suite = _make_suite(dtype, layout)
    run_activations(suite, method=method, activation=activation)
    _assert_suite(suite)
