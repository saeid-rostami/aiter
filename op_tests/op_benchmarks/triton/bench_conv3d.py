# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Benchmark + offline tuner for aiter.ops.triton.conv.conv3d.

Modes:
  --bench   Time conv3d vs torch.nn.functional.conv3d (MIOpen) on a shape list,
            both layouts, print a table. Default mode.
  --tune    For each JSON write-target (config_name, shape_key) sweep a candidate
            grid, pick the fastest correct config, and write the winner into the
            per-arch config JSON under "shapes". Production (autotune off) then
            picks these tuned configs over the conservative "any" fallback.

Tuning groups bench-points by their JSON write-target and scores each candidate
config by the *combined* median time across all dtypes/layouts that map to that
target (config correct for all). Because the shape_key has no dtype, this yields a
single config per (config_name, shape_key) that is robust across dtypes — rather
than letting a later dtype overwrite an earlier one.

It drives the real public conv3d() path and only overrides where the config comes
from (monkeypatching the kernel _get_config helper that _launch imports), so the
measured kernel/launch is exactly what production runs.

Live model shape extraction needs diffusers (not assumed here); add shapes to
DEFAULT_SHAPES.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F

import aiter.ops.triton.conv._launch as L
from aiter.ops.triton.conv.conv3d import conv3d
from aiter.ops.triton.conv._utils import dynamic_conv_tolerances
from aiter.ops.triton.utils.conv_config_utils import format_shape_key_3d
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH


# (name, N, C, D, H, W, K, KD, KH, KW, stride, padding, dilation)
DEFAULT_SHAPES = [
    # video-VAE 3x3x3 "same" convs (the dominant case)
    ("vae_3x3x3_a", 1, 384, 3, 46, 51, 384, 3, 3, 3, (1, 1, 1), (1, 1, 1), (1, 1, 1)),
    ("vae_3x3x3_b", 1, 96, 6, 354, 394, 96, 3, 3, 3, (1, 1, 1), (1, 1, 1), (1, 1, 1)),
    ("vae_3x3x3_c", 1, 192, 6, 178, 198, 192, 3, 3, 3, (1, 1, 1), (1, 1, 1), (1, 1, 1)),
    ("vae_3x3x3_d", 1, 384, 4, 90, 100, 384, 3, 3, 3, (1, 1, 1), (1, 1, 1), (1, 1, 1)),
    ("vae_3x3x3_e", 1, 128, 8, 64, 64, 128, 3, 3, 3, (1, 1, 1), (1, 1, 1), (1, 1, 1)),
    ("vae_3x3x3_f", 1, 256, 8, 32, 32, 256, 3, 3, 3, (1, 1, 1), (1, 1, 1), (1, 1, 1)),
    # pointwise 1x1x1 (channel mixing)
    ("vae_1x1x1_a", 1, 256, 16, 32, 32, 512, 1, 1, 1, (1, 1, 1), (0, 0, 0), (1, 1, 1)),
    ("vae_1x1x1_b", 1, 512, 8, 64, 64, 512, 1, 1, 1, (1, 1, 1), (0, 0, 0), (1, 1, 1)),
    ("vae_1x1x1_c", 1, 384, 3, 46, 51, 384, 1, 1, 1, (1, 1, 1), (0, 0, 0), (1, 1, 1)),
    ("vae_1x1x1_d", 1, 128, 16, 64, 64, 256, 1, 1, 1, (1, 1, 1), (0, 0, 0), (1, 1, 1)),
    # general fallback (other kernel sizes)
    ("gen_5x5x5", 1, 64, 8, 32, 32, 64, 5, 5, 5, (1, 1, 1), (2, 2, 2), (1, 1, 1)),
]

# Candidate grids as (BLOCK_M, BLOCK_N, tile, GROUP_SIZE_M, num_warps).
# 3x3x3 kernels key the channel tile as BLOCK_C (must be <= Cb=64).
_GRID_3X3X3 = [
    (64, 64, 64, 4, 4),
    (64, 128, 64, 8, 4),
    (128, 64, 64, 8, 4),
    (128, 128, 64, 8, 8),
    (64, 256, 64, 8, 8),
    (256, 64, 64, 8, 8),
    (128, 128, 32, 8, 8),
    (64, 128, 32, 8, 4),
    (64, 64, 32, 4, 4),
    (256, 128, 64, 8, 8),
    (128, 256, 64, 8, 8),
]
# 1x1x1 / general key the reduction tile as BLOCK_K (can be larger).
_GRID_GEMM = [
    (64, 64, 64, 4, 4),
    (64, 128, 64, 8, 4),
    (128, 64, 64, 8, 4),
    (128, 128, 64, 8, 8),
    (64, 256, 64, 8, 8),
    (256, 64, 64, 8, 8),
    (128, 128, 32, 8, 8),
    (64, 64, 128, 4, 4),
    (64, 128, 128, 8, 8),
    (128, 128, 128, 8, 8),
    (256, 128, 64, 8, 8),
    (128, 256, 64, 8, 8),
]


def _bench(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True)
    e = torch.cuda.Event(True)
    ts = []
    for _ in range(iters):
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def _target_for(KD, KH, KW, layout):
    """(getter_attr, config_name, tile_key) for the kernel this (shape, layout) routes
    to. tile_key is 'BLOCK_C' for 3x3x3 (uses _GRID_3X3X3) else 'BLOCK_K' (_GRID_GEMM)."""
    if KD == 1 and KH == 1 and KW == 1:
        name = "CONV3D-1X1X1-NDHWC" if layout == "ndhwc" else "CONV3D-1X1X1"
        return "_get_config_1x1x1_3d", name, "BLOCK_K"
    if KD == 3 and KH == 3 and KW == 3:
        if layout == "ndhwc":
            return "_get_config_3x3x3_ndhwc_3d", "CONV3D-3X3X3-NDHWC", "BLOCK_C"
        return "_get_config_3x3x3_cblocked_3d", "CONV3D-3X3X3-CBLOCKED", "BLOCK_C"
    # general — one config file shared across layouts (LAYOUT constexpr)
    return "_get_config_general_3d", "CONV3D-GENERAL", "BLOCK_K"


def _cfg_dict(bm, bn, tile, gs, nw, tile_key):
    return {
        "BLOCK_M": bm,
        "BLOCK_N": bn,
        tile_key: tile,
        "GROUP_SIZE_M": gs,
        "num_warps": nw,
        "num_stages": 1,
    }


def _make_inputs(shp, dtype, layout):
    _, N, C, D, H, W, K, KD, KH, KW, st, pd, di = shp
    torch.manual_seed(0)
    x = torch.randn(N, C, D, H, W, device="cuda", dtype=dtype)
    w = torch.randn(K, C, KD, KH, KW, device="cuda", dtype=dtype)
    b = torch.randn(K, device="cuda", dtype=dtype)
    xin = x.to(memory_format=torch.channels_last_3d) if layout == "ndhwc" else x
    ref = F.conv3d(x, w, b, stride=st, padding=pd, dilation=di)
    rtol, atol = dynamic_conv_tolerances(dtype, C * KD * KH * KW, ref.float())
    return dict(x=x, w=w, b=b, xin=xin, ref=ref, rtol=rtol, atol=atol)


def _call(io, shp, layout):
    _, N, C, D, H, W, K, KD, KH, KW, st, pd, di = shp
    return conv3d(io["xin"], io["w"], io["b"], stride=st, padding=pd, dilation=di, layout=layout)


def _shape_key(shp):
    _, N, C, D, H, W, K, KD, KH, KW, st, pd, di = shp
    return format_shape_key_3d(
        N=N, C=C, D=D, H=H, W=W, K=K, KD=KD, KH=KH, KW=KW,
        sd=st[0], sh=st[1], sw=st[2], pd=pd[0], ph=pd[1], pw=pd[2],
        dd=di[0], dh=di[1], dw=di[2],
    )


# ---------------------------------------------------------------- bench mode ---
def run_bench(shapes, dtypes, layouts):
    print(f"# conv3d bench on {arch_info.get_arch()}  (Triton vs torch/MIOpen, median ms)")
    hdr = f"{'shape':28s} {'dtype':5s} {'layout':6s} {'corr':5s} {'torch':>8s} {'triton':>8s} {'speedup':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for dtype in dtypes:
        dn = {torch.float16: "fp16", torch.bfloat16: "bf16"}[dtype]
        for shp in shapes:
            name = shp[0]
            st, pd, di = shp[10], shp[11], shp[12]
            for layout in layouts:
                io = _make_inputs(shp, dtype, layout)
                y = _call(io, shp, layout)
                ok = "PASS" if torch.allclose(y.float(), io["ref"].float(), rtol=io["rtol"], atol=io["atol"]) else "FAIL"
                tt = _bench(lambda: F.conv3d(io["x"], io["w"], io["b"], stride=st, padding=pd, dilation=di), 8, 30)
                ty = _bench(lambda: _call(io, shp, layout), 8, 30)
                print(f"{name:28s} {dn:5s} {layout:6s} {ok:5s} {tt:8.3f} {ty:8.3f} {tt/ty:7.2f}x")


# ----------------------------------------------------------------- tune mode ---
def run_tune(shapes, dtypes, layouts, write=True):
    print(f"# Tuning on {arch_info.get_arch()}  (3x3x3 grid={len(_GRID_3X3X3)}, gemm grid={len(_GRID_GEMM)})")

    # group bench-points by JSON write-target (config_name, shape_key)
    groups = defaultdict(lambda: {"getter": None, "tile_key": None, "points": []})
    for shp in shapes:
        KD, KH, KW = shp[7], shp[8], shp[9]
        sk = _shape_key(shp)
        for layout in layouts:
            getter, config_name, tile_key = _target_for(KD, KH, KW, layout)
            g = groups[(config_name, sk)]
            g["getter"] = getter
            g["tile_key"] = tile_key
            for dtype in dtypes:
                g["points"].append((shp, dtype, layout))

    touched = set()
    for (config_name, sk), g in groups.items():
        grid = _GRID_3X3X3 if g["tile_key"] == "BLOCK_C" else _GRID_GEMM
        # materialize inputs once per point
        pts = [(_make_inputs(shp, dtype, layout), shp, layout) for (shp, dtype, layout) in g["points"]]
        orig = getattr(L, g["getter"])
        best = None  # (total_time, cfg)
        try:
            for (bm, bn, tile, gs, nw) in grid:
                cfg = _cfg_dict(bm, bn, tile, gs, nw, g["tile_key"])
                setattr(L, g["getter"], lambda *a, _c=cfg, **k: dict(_c))
                total = 0.0
                ok_all = True
                for (io, shp, layout) in pts:
                    try:
                        y = _call(io, shp, layout)
                        torch.cuda.synchronize()
                    except Exception:
                        ok_all = False
                        break
                    if not torch.allclose(y.float(), io["ref"].float(), rtol=io["rtol"], atol=io["atol"]):
                        ok_all = False
                        break
                    total += _bench(lambda: _call(io, shp, layout), 5, 15)
                if ok_all and (best is None or total < best[0]):
                    best = (total, cfg)
        finally:
            setattr(L, g["getter"], orig)

        if best is None:
            print(f"  {config_name:22s} {sk[:34]:34s} -> NO valid config")
            continue
        cfg = best[1]
        tk = "BLOCK_C" if "BLOCK_C" in cfg else "BLOCK_K"
        cfgstr = f"BM{cfg['BLOCK_M']} BN{cfg['BLOCK_N']} {tk[-1]}{cfg[tk]} GS{cfg['GROUP_SIZE_M']} w{cfg['num_warps']}"
        if write:
            _merge_into_json(config_name, sk, cfg)
            touched.add(config_name)
        print(f"  {config_name:22s} {cfgstr:28s} [{len(g['points'])} pts]  ({best[0]:.3f} ms total)")

    if write and touched:
        print(f"\nWrote tuned shapes into: {', '.join(sorted(touched))} ({arch_info.get_arch()})")
    elif not write:
        print("\n(dry run — no JSON written)")


def _merge_into_json(config_name, shape_key, cfg):
    dev = arch_info.get_arch()
    fpath = os.path.join(AITER_TRITON_CONFIGS_PATH, "conv", f"{dev}-{config_name}.json")
    with open(fpath) as f:
        data = json.load(f)
    data.setdefault("shapes", {})[shape_key] = cfg
    with open(fpath, "w") as f:
        json.dump(data, f, indent=4)
        f.write("\n")
    return fpath


def _dtypes(s):
    return {"fp16": [torch.float16], "bf16": [torch.bfloat16],
            "both": [torch.float16, torch.bfloat16]}[s]


def main(argv=None):
    p = argparse.ArgumentParser(description="conv3d benchmark + offline tuner")
    p.add_argument("--tune", action="store_true", help="sweep configs and pin winners into JSON")
    p.add_argument("--dry-run", action="store_true", help="with --tune: don't write JSON")
    p.add_argument("--dtype", choices=["fp16", "bf16", "both"], default="both")
    p.add_argument("--layout", choices=["ncdhw", "ndhwc", "both"], default="both")
    args = p.parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        sys.exit(1)

    dtypes = _dtypes(args.dtype)
    layouts = {"ncdhw": ["ncdhw"], "ndhwc": ["ndhwc"], "both": ["ncdhw", "ndhwc"]}[args.layout]

    if args.tune:
        run_tune(DEFAULT_SHAPES, dtypes, layouts, write=not args.dry_run)
    else:
        run_bench(DEFAULT_SHAPES, dtypes, layouts)


if __name__ == "__main__":
    main()
