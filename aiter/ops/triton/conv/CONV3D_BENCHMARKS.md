# conv3d — VAE shape benchmarks (Phase 1, general kernel)

Reference results for the Triton `conv3d` general (implicit-GEMM) kernel vs
PyTorch/MIOpen on real video-VAE conv shapes. Captured during Phase 1 (general
kernel only — no 1×1×1 / 3×3×3 / Winograd specialization yet).

## Environment
- **Date:** 2026-06-09
- **GPU / arch:** AMD RDNA4, `gfx1201`
- **Stack:** ROCm 7.2 / PyTorch 2.9.1+gitff65f5b / Triton 3.7
- **Kernel under test:** `aiter.ops.triton.conv.conv3d` → `_conv3d_general_kernel`
- **Reference:** `torch.nn.functional.conv3d` (MIOpen backend)
- **Conv params:** "same" 3×3×3 — `stride=(1,1,1)`, `padding=(1,1,1)`, `dilation=(1,1,1)`, with bias
- **Layout:** NCDHW. **Timing:** median of 20 iters after 5 warmup; weight prepack LRU-cached (steady-state)
- **Correctness:** `torch.testing.assert_close` in fp32 vs `F.conv3d`, tolerance = `dynamic_conv_tolerances(dtype, C·KD·KH·KW)`
- **TFLOPS:** direct-conv-equivalent (`2·N·D_out·P·Q·K·C·KD·KH·KW`), same denominator both backends

## Shapes (video VAE encoder/decoder)
All are `[C,C,3,3,3]` weights (in-channels == out-channels). Counts = how often
each shape appears in the encoder / decoder.

| # | Input (N,C,D,H,W) | Weight | Enc count | Dec count |
|---|---|---|---|---|
| 1 | [1,384,3,46,51]   | [384,384,3³] | 256 | 320 |
| 2 | [1,96,6,354,394]  | [96,96,3³]   | 120 | 180 |
| 3 | [1,192,6,178,198] | [192,192,3³] | 90  | 180 |
| 4 | [1,384,4,90,100]  | [384,384,3³] | 90  | 150 |

## Results

Prepack cache cleared before the run. **warm** = weight prepack LRU-cached
(steady-state; what real inference sees since weights are reused every forward).
**cold** = prepack cache cleared on *every* call, so the K-major weight repack is
included in each measurement.

| # | Shape | dtype | Correct | torch ms | Triton warm | Triton cold | spd(warm) | spd(cold) | torch TFLOPS | TFLOPS warm | TFLOPS cold |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [1,384,3,46,51]   | fp16 | PASS | 2.85  | 1.65  | 1.52  | 1.73× | 1.87× | 19.7 | 34.0 | 36.8 |
| 2 | [1,96,6,354,394]  | fp16 | PASS | 47.27 | 11.02 | 11.03 | 4.29× | 4.28× | 8.8  | 37.8 | 37.7 |
| 3 | [1,192,6,178,198] | fp16 | PASS | 25.47 | 8.21  | 8.13  | 3.10× | 3.13× | 16.5 | 51.3 | 51.8 |
| 4 | [1,384,4,90,100]  | fp16 | PASS | 9.75  | 5.49  | 5.42  | 1.78× | 1.80× | 29.4 | 52.2 | 52.9 |
| 1 | [1,384,3,46,51]   | bf16 | PASS | 1.86  | 1.12  | 1.08  | 1.67× | 1.72× | 30.1 | 50.2 | 51.8 |
| 2 | [1,96,6,354,394]  | bf16 | PASS | 47.47 | 10.99 | 11.00 | 4.32× | 4.32× | 8.8  | 37.9 | 37.9 |
| 3 | [1,192,6,178,198] | bf16 | PASS | 25.45 | 8.18  | 8.08  | 3.11× | 3.15× | 16.5 | 51.5 | 52.1 |
| 4 | [1,384,4,90,100]  | bf16 | PASS | 9.68  | 5.46  | 5.39  | 1.77× | 1.80× | 29.6 | 52.5 | 53.2 |

Triton is faster on every shape/dtype: **1.7×–4.3×**. bf16 ≈ fp16 for Triton
(single code path). MIOpen swings 8.8→30.1 TFLOPS; Triton is steady 34–53.

**warm vs cold are within timing noise** (cold occasionally measures *faster*),
so the reported TFLOPS are unaffected by repacking — see below.

## Repacking cost (why warm ≈ cold)

The only prepack in the general kernel is the **weight** K-major pack
(`reshape(K_out, C·KD·KH·KW)` + zero-pad to a multiple of 64 + `.contiguous()`).
It is input-independent and LRU-cached, so in steady state it runs once per
weight. Measured cost is negligible:

| Shape | K_red | K_pad | pad? | repack ms | as % of conv |
|---|---|---|---|---|---|
| [1,384,3,46,51]   | 10368 | 10368 | no  | 0.0056 | 0.32% |
| [1,96,6,354,394]  | 2592  | 2624  | yes | 0.0156 | 0.14% |
| [1,192,6,178,198] | 5184  | 5184  | no  | 0.0045 | 0.06% |
| [1,384,4,90,100]  | 10368 | 10368 | no  | 0.0046 | 0.09% |

For 3 of 4 shapes `K_red` is already 64-aligned → no pad, no copy (the reshape is
a view and `.contiguous()` is a no-op). Only shape 2 (C=96) does a tiny pad.

**Fairness note:** the asymmetry is correct, not a thumb on the scale. Our repack
is per-*weight* (input-independent) → legitimately amortizable. MIOpen's
`Im3d2Col` is per-*call* (depends on the input, changes every forward) → cannot be
cached → it **is** counted in MIOpen's time. Each backend's one-time setup is
excluded by warmup (our repack; MIOpen's ~46 ms `FindConvolution` solver search).

> NOTE: this clean story holds only for the general kernel. The Phase 3 cblocked
> path adds a per-*call* input repack (NCDHW→NCDHWc) that is **not** amortizable —
> report a separate "kernel + repack" column for it, like the conv2d bench does.

## What PyTorch/MIOpen actually runs

All 8 cases resolve to MIOpen solver **`GemmFwdRest`** (algorithm
`miopenConvolutionFwdAlgoGEMM`) — the **im2col + rocBLAS GEMM fallback**. No
specialized 3D conv solver applies on RDNA4. Two GPU kernels per conv:

1. **`Im3d2Col`** (from `MIOpenIm3d2Col.cpp.o`) — builds the im2col buffer.
2. **rocBLAS `gemm_ex`** (Tensile) — the matmul, `compute_type=f32_r`, bf16/fp16 in/out.
   ```
   rocblas-bench -f gemm_ex --transposeA N --transposeB N \
     -m <N·D_out·P·Q> -n <K_out> -k <C·KD·KH·KW> \
     --a_type {f16_r|bf16_r} --b_type ... --c_type ... --compute_type f32_r --algo 0
   ```

### MIOpen GEMM dims + im2col workspace (constant across dtype)

| # | Shape | GEMM M×N×K | im2col workspace |
|---|---|---|---|
| 1 | [1,384,3,46,51]   | 7,038×384×10,368   | ~0.55 GB |
| 2 | [1,96,6,354,394]  | 836,856×96×2,592   | **4.04 GB** |
| 3 | [1,192,6,178,198] | 211,464×192×5,184  | ~2.04 GB |
| 4 | [1,384,4,90,100]  | 36,000×384×10,368  | ~0.69 GB |

The biggest speedup (shape 2, 4.3×) tracks the biggest im2col blowup: a **4 GB**
buffer (27× the 160 MB input) written by `Im3d2Col` then read by the GEMM. The
Triton kernel is a single fused implicit-GEMM — **no im2col buffer** — which is
why it wins before any 3×3×3 specialization exists.

## Caveats
- These are **general fallback** kernel numbers. The Phase 3 direct 3×3×3 kernel
  (NCDHWc cblocked / NDHWC) is the next lever, esp. on shapes 1 & 4 (~34–52 TFLOPS).
- MIOpen small-shape timings jitter ±~20% run-to-run (shape 1 fp16 seen 1.85→2.81 ms);
  large shapes are stable.
- The literal Tensile kernel symbol (`Cijk_…_MT…`) was not captured (needs
  `ROCBLAS_LAYER=4` / `rocprof --hip-trace`); `gemm_ex` is what rocBLAS dispatches.

## How to reproduce
- Correctness + perf: a standalone script using `aiter.ops.triton.conv.conv3d.conv3d`
  vs `F.conv3d`, `torch.cuda.Event` timing, the 4 shapes above.
- MIOpen solver: run under `MIOPEN_LOG_LEVEL=6`, grep stderr for `Chosen Algorithm:`.
- rocBLAS gemm_ex + im2col kernel + workspace: add `ROCBLAS_LAYER=2`, grep stderr for
  `rocblas-bench -f gemm_ex`, `kernel_name = Im3d2Col`, and `workspace =`.
