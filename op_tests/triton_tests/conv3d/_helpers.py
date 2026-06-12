# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Test-side library for conv3d: TestSuite, method registry, and runners.

Library code (no ``test_`` prefix) — pytest does not collect this file.
``test_conv3d.py`` imports the runners and registry from here. Mirrors the 2D
``op_tests/triton_tests/conv/_helpers.py``.

Phase 1 ships only the general (implicit-GEMM) kernel, so METHOD_REGISTRY has a
single entry (``default`` -> ``conv3d_ncdhw``). 1x1x1 / 3x3x3 specializations
add entries here as they land.
"""

import random
import traceback
from collections import namedtuple
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F

from aiter.ops.triton.conv._utils import (
    dynamic_conv_tolerances,
    _out_dhw,
    apply_activation,
)
from aiter.ops.triton.conv.conv3d import conv3d_ncdhw, conv3d_ndhwc

# -- Architecture gating ------------------------------------------------------
SUPPORTED_ARCHS = {
    "RDNA": {"gfx1200", "gfx1201"},
    "CDNA": set(),
}
ALL_SUPPORTED_ARCHS = set().union(*SUPPORTED_ARCHS.values())


# -- Method registry ----------------------------------------------------------

MethodEntry = namedtuple(
    "MethodEntry", ["kernel_fn", "guard_fn", "is_winograd", "bench_tag", "short_name"]
)

METHOD_REGISTRY = {
    "default": MethodEntry(conv3d_ncdhw, None, False, "", "default"),
}

ORDERED_METHODS = list(METHOD_REGISTRY.keys())
ALL_METHODS = ORDERED_METHODS + ["all"]


# -- Result + suite -----------------------------------------------------------


@dataclass
class TestResult:
    name: str
    passed: bool
    max_abs_error: float
    rel_error: float
    message: str = ""


class TestSuite:
    """Correctness-only test runner for conv3d."""

    __test__ = False  # not a pytest TestCase

    def __init__(
        self,
        device: str,
        dtype: torch.dtype,
        verbose: bool = False,
        layout_mode: str = "both",
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.verbose = verbose
        self.layout_mode = layout_mode
        self.results: List[TestResult] = []

    def check_close(
        self,
        name: str,
        got: torch.Tensor,
        ref: torch.Tensor,
        K_red: Optional[int] = None,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
    ) -> TestResult:
        got32 = got.float()
        ref32 = ref.float()
        diff = (got32 - ref32).abs()
        max_abs = float(diff.max().item()) if diff.numel() else 0.0
        rel = max_abs / (float(ref32.abs().max().item()) + 1e-6)
        if rtol is None or atol is None:
            K_est = int(K_red) if K_red is not None else 1024
            rtol_calc, atol_calc = dynamic_conv_tolerances(self.dtype, K_est, ref32)
            rtol = rtol if rtol is not None else rtol_calc
            atol = atol if atol is not None else atol_calc
        try:
            torch.testing.assert_close(got32, ref32, rtol=rtol, atol=atol)
            passed, msg = True, "OK"
        except AssertionError as e:
            passed, msg = False, str(e).split("\n")[0]
        res = TestResult(name, passed, max_abs, rel, msg)
        self.results.append(res)
        if self.verbose:
            mark = "OK" if passed else "XX"
            print(f"  [{mark}] {name:<44} | max_abs={max_abs:.3e} rel={rel:.3e}")
        return res

    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def failed_results(self) -> List[TestResult]:
        return [r for r in self.results if not r.passed]


# -- Dispatch -----------------------------------------------------------------


def run_all_methods(
    suite: TestSuite,
    x: torch.Tensor,
    w: torch.Tensor,
    b: Optional[torch.Tensor],
    stride,
    padding,
    dilation,
    name: str,
    method: str = "default",
    activation: str = "none",
):
    """Correctness-only dispatch: run selected method(s) and check vs F.conv3d."""
    N, C, D, H, W_in = x.shape
    K_out, _, KD, KH, KW = w.shape
    K_red = C * KD * KH * KW

    y_ref = F.conv3d(
        x,
        w,
        b.to(dtype=suite.dtype) if b is not None else None,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    y_ref = apply_activation(y_ref, activation)

    if suite.layout_mode in ("ncdhw", "both"):
        methods_to_run = ORDERED_METHODS if method == "all" else [method]
        for m in methods_to_run:
            entry = METHOD_REGISTRY[m]
            if entry.guard_fn and not entry.guard_fn(KD, KH, KW, stride, dilation, C):
                continue
            y_tri = entry.kernel_fn(
                x,
                w,
                b,
                stride,
                padding,
                dilation,
                activation=activation,
            )
            suite.check_close(
                f"{name} {entry.bench_tag or '[NCDHW]'}",
                y_tri,
                y_ref,
                K_red=K_red,
            )

    if suite.layout_mode in ("ndhwc", "both"):
        y_ndhwc = conv3d_ndhwc(
            x,
            w,
            b,
            stride,
            padding,
            dilation,
            activation=activation,
        )
        suite.check_close(f"{name} [NDHWC]", y_ndhwc, y_ref, K_red=K_red)


# -- Shape sets ---------------------------------------------------------------


def get_edge_case_shapes():
    # (N, C, D, H, W, K_out, KD, KH, KW, stride, padding, dilation, desc)
    return [
        (1, 8, 8, 8, 8, 16, 1, 1, 1, (1, 1, 1), (0, 0, 0), (1, 1, 1), "1x1x1 stride1"),
        (1, 16, 8, 16, 16, 32, 3, 3, 3, (1, 1, 1), (1, 1, 1), (1, 1, 1), "3x3x3 same pad"),
        (2, 16, 8, 16, 16, 32, 3, 3, 3, (2, 2, 2), (1, 1, 1), (1, 1, 1), "3x3x3 stride2"),
        (1, 8, 10, 12, 14, 16, 3, 3, 3, (1, 1, 1), (0, 0, 0), (1, 1, 1), "3x3x3 no pad asym"),
        (1, 8, 8, 16, 16, 16, 5, 5, 5, (1, 1, 1), (2, 2, 2), (1, 1, 1), "5x5x5 same pad"),
        (2, 16, 8, 8, 8, 32, 3, 3, 3, (1, 1, 1), (0, 0, 0), (2, 2, 2), "3x3x3 dilation2"),
        (1, 4, 4, 8, 8, 8, 1, 3, 3, (1, 1, 1), (0, 1, 1), (1, 1, 1), "1x3x3 anisotropic"),
        (1, 3, 6, 16, 16, 16, 3, 3, 3, (1, 1, 1), (1, 1, 1), (1, 1, 1), "low-C 3x3x3"),
    ]


# -- Test runners -------------------------------------------------------------


def run_edge_cases(suite: TestSuite, activation: str = "none", method: str = "default"):
    for (
        N, C, D, H, W, K_out, KD, KH, KW, stride, padding, dilation, desc
    ) in get_edge_case_shapes():
        D_out, P, Q = _out_dhw(D, H, W, KD, KH, KW, stride, padding, dilation)
        if D_out < 1 or P < 1 or Q < 1:
            continue
        x = torch.randn((N, C, D, H, W), device=suite.device, dtype=suite.dtype)
        w = torch.randn((K_out, C, KD, KH, KW), device=suite.device, dtype=suite.dtype)
        b = torch.randn((K_out,), device=suite.device, dtype=suite.dtype)
        run_all_methods(
            suite, x, w, b, stride, padding, dilation,
            name=desc, method=method, activation=activation,
        )


def run_activations(suite: TestSuite, method: str = "default", activation: str = "relu"):
    N, C, D, H, W, K_out = 2, 16, 8, 16, 16, 32
    KD, KH, KW = 3, 3, 3
    stride, padding, dilation = (1, 1, 1), (1, 1, 1), (1, 1, 1)
    x = torch.randn((N, C, D, H, W), device=suite.device, dtype=suite.dtype)
    w = torch.randn((K_out, C, KD, KH, KW), device=suite.device, dtype=suite.dtype)
    b = torch.randn((K_out,), device=suite.device, dtype=suite.dtype)
    run_all_methods(
        suite, x, w, b, stride, padding, dilation,
        name=f"activation_{activation}_{method}", method=method, activation=activation,
    )


def run_no_bias(suite: TestSuite, method: str = "default"):
    shapes = [
        (1, 16, 8, 8, 8, 32, 1, 1, 1, (1, 1, 1), (0, 0, 0), (1, 1, 1), "1x1x1 no bias"),
        (2, 16, 8, 16, 16, 32, 3, 3, 3, (1, 1, 1), (1, 1, 1), (1, 1, 1), "3x3x3 no bias"),
        (1, 8, 8, 8, 8, 16, 5, 5, 5, (1, 1, 1), (2, 2, 2), (1, 1, 1), "5x5x5 no bias"),
    ]
    for N, C, D, H, W, K_out, KD, KH, KW, stride, padding, dilation, desc in shapes:
        x = torch.randn((N, C, D, H, W), device=suite.device, dtype=suite.dtype)
        w = torch.randn((K_out, C, KD, KH, KW), device=suite.device, dtype=suite.dtype)
        run_all_methods(
            suite, x, w, None, stride, padding, dilation, name=desc, method=method
        )


def run_random_fuzzing(
    suite: TestSuite,
    num_tests: int = 10,
    activation: str = "none",
    method: str = "default",
    seed: int = 42,
):
    random.seed(seed)
    for i in range(num_tests):
        N = random.randint(1, 4)
        C = random.choice([1, 3, 8, 16, 32, 64])
        D = random.randint(4, 24)
        H = random.randint(4, 32)
        W = random.randint(4, 32)
        K_out = random.choice([8, 16, 32, 64])
        KD = random.randint(1, min(5, D))
        KH = random.randint(1, min(5, H))
        KW = random.randint(1, min(5, W))
        sd = random.randint(1, 2)
        sh = random.randint(1, 2)
        sw = random.randint(1, 2)
        pd = random.randint(0, KD // 2)
        ph = random.randint(0, KH // 2)
        pw = random.randint(0, KW // 2)
        dd = random.randint(1, 2)
        dh = random.randint(1, 2)
        dw = random.randint(1, 2)
        stride, padding, dilation = (sd, sh, sw), (pd, ph, pw), (dd, dh, dw)
        D_out, P, Q = _out_dhw(D, H, W, KD, KH, KW, stride, padding, dilation)
        if D_out < 1 or P < 1 or Q < 1:
            continue
        try:
            x = torch.randn((N, C, D, H, W), device=suite.device, dtype=suite.dtype)
            w = torch.randn(
                (K_out, C, KD, KH, KW), device=suite.device, dtype=suite.dtype
            )
            b = torch.randn((K_out,), device=suite.device, dtype=suite.dtype)
            tag = (
                f"Random[{i}] ({N},{C},{D},{H},{W})->"
                f"({N},{K_out},{D_out},{P},{Q})"
            )
            run_all_methods(
                suite, x, w, b, stride, padding, dilation,
                name=tag, method=method, activation=activation,
            )
        except Exception as e:
            tb = traceback.format_exc()
            suite.results.append(
                TestResult(
                    f"Random[{i}]", False, float("inf"), float("inf"),
                    f"{type(e).__name__}: {e}\n{tb}",
                )
            )
