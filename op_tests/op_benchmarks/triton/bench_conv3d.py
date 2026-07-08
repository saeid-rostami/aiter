# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Quick benchmark for aiter.ops.triton.conv.conv3d (general NCDHW kernel).

Times the Triton conv3d against torch.nn.functional.conv3d (MIOpen on AMD)
on the Wan-style VAE encoder/decoder 3x3x3 shapes, reports per-shape latency
and TFLOPS, and prints a correctness delta vs the fp32 reference.

TFLOPS uses the universal direct-conv FLOP count
    2 * N * K * C * T * R * S * O * P * Q
(same convention as bench_conv2d.py), so numbers are comparable across
algorithms and against MIOpen.

Usage:
    python -m op_tests.op_benchmarks.triton.bench_conv3d [--dtype fp16|bf16]
"""

import argparse

import torch
import torch.nn.functional as F
import triton

from aiter.ops.triton.conv._utils import _out_dhw
from aiter.ops.triton.conv.conv3d import conv3d


# (label, N, C, D, H, W, K), all 3x3x3 stride1 pad1. Encoder+decoder dedup to
# these 4 distinct shapes; the call-count weights from the workload are noted.
SHAPES = [
    ("384 @ 3x46x51   (enc256/dec320)", 1, 384, 3, 46, 51, 384),
    ("96  @ 6x354x394 (enc120/dec180)", 1, 96, 6, 354, 394, 96),
    ("192 @ 6x178x198 (enc90/dec180)", 1, 192, 6, 178, 198, 192),
    ("384 @ 4x90x100  (enc90/dec150)", 1, 384, 4, 90, 100, 384),
]


def _bench_one(N, C, D, H, W, K, dtype, layout):
    T = R = S = 3
    stride = pad = dil = (1, 1, 1)
    O, P, Q = _out_dhw(D, H, W, T, R, S, stride, pad, dil)

    x = torch.randn(N, C, D, H, W, device="cuda", dtype=dtype)
    w = torch.randn(K, C, T, R, S, device="cuda", dtype=dtype)
    b = torch.randn(K, device="cuda", dtype=dtype)

    # For the torch reference, feed channels_last_3d when benching NDHWC so both
    # backends see the same memory layout.
    x_ref = x.to(memory_format=torch.channels_last_3d) if layout == "ndhwc" else x

    tri = lambda: conv3d(x, w, b, stride=stride, padding=pad, dilation=dil, layout=layout)
    ref = lambda: F.conv3d(x_ref, w, b, stride=stride, padding=pad, dilation=dil)

    # Correctness vs fp32 reference.
    y = tri().float()
    y_ref = F.conv3d(x.float(), w.float(), b.float(), stride=stride, padding=pad).float()
    max_abs = (y - y_ref).abs().max().item()
    rel = max_abs / (y_ref.abs().max().item() + 1e-9)

    t_tri = triton.testing.do_bench(tri, warmup=25, rep=100)
    t_ref = triton.testing.do_bench(ref, warmup=25, rep=100)

    flop = 2.0 * N * K * C * T * R * S * O * P * Q
    tflops_tri = flop / (t_tri * 1e-3) / 1e12
    tflops_ref = flop / (t_ref * 1e-3) / 1e12
    return (O, P, Q), t_tri, t_ref, tflops_tri, tflops_ref, rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--layout", choices=["ncdhw", "ndhwc"], default="ncdhw")
    args = ap.parse_args()
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    hdr = (
        f"{'shape':34s} {'out(O,P,Q)':14s} {'Tri ms':>8s} {'Tor ms':>8s} "
        f"{'Tri TF':>7s} {'Tor TF':>7s} {'speedup':>8s} {'rel':>9s}"
    )
    print(f"\nconv3d general kernel — dtype={args.dtype} layout={args.layout}\n")
    print(hdr)
    print("-" * len(hdr))
    for label, N, C, D, H, W, K in SHAPES:
        (O, P, Q), t_tri, t_ref, tf_tri, tf_ref, rel = _bench_one(
            N, C, D, H, W, K, dtype, args.layout
        )
        print(
            f"{label:34s} {f'{O},{P},{Q}':14s} {t_tri:8.3f} {t_ref:8.3f} "
            f"{tf_tri:7.1f} {tf_ref:7.1f} {t_ref / t_tri:7.2f}x {rel:9.2e}"
        )
    print()


if __name__ == "__main__":
    main()
