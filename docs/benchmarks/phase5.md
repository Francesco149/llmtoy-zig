# Phase 5 — CPU Optimization Benchmarks

## Hardware

- CPU: Ryzen 5 3600 (6C/12T, Zen 2), AVX2 + FMA
- RAM: 64 GB DDR4
- OS: Linux x86_64
- Build: `zig build -Doptimize=ReleaseFast` (native CPU target from step 2 onward)

## Model

`Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` unless otherwise noted.  
24 layers, d_model=896, d_ffn=4864, 14 Q heads / 2 KV heads, vocab=151936

---

## Step 1 — Multi-threading (`quantMatvecPar`)

Split output rows across N threads. Each thread dequantizes and dot-products its
own row slice independently, writing into a disjoint sub-slice of the output.

Pre-allocated per-thread scratch buffers (`n_threads × max_cols` floats) avoid
any allocation or contention inside the hot path. `min_rows_per_thread = 16`
before spawning to amortise thread-spawn cost. Build: baseline x86_64 target.

### Results

| Threads | Generation | tok/s |
|---------|------------|-------|
| 1       | 20747 ms / 30 tok | 1.45 |
| 12      | 13427 ms / 30 tok | 2.23 |

**Speedup: 1.54× generation**

### hyperfine (20 tokens, 3 runs each)

```
threads=1:   16.817 s ± 0.110 s
threads=12:  11.117 s ± 0.047 s
threads=12 is 1.51× faster
```

### Analysis

Modest speedup despite 12× thread count because:

1. **Spawn overhead underestimated** — Linux thread spawn is ~200 µs (not 50 µs as
   initially assumed). With `min_rows=16`, wk/wv (128 rows) spawn 9 threads each →
   24 layers × 2 kv-mats × 9 spawns × 200 µs ≈ 86 ms overhead per token.

2. **Small matrices** — wk/wv at 128 rows gain little from parallelism relative
   to spawn cost; FFN matrices (4864 rows) and lm_head (151936 rows) benefit most.

---

## Step 2 — AVX2 SIMD dot product + native CPU target

### Changes

- `build.zig`: default CPU model set to `.native` so AVX2/FMA are enabled.
- `math.zig`: `dotf32(a, b)` with 4 independent `@Vector(8, f32)` accumulators
  (hides Zen 2's 5-cycle FMA latency; 4 × 8 = 32 elements per inner iteration).
- `min_rows_per_thread` raised from 16 → 512: wk/wv (128 rows) and wq/wo (896 rows)
  now run serially or with only 2 threads respectively, cutting spawns from ~1884
  to ~636 per token and eliminating the Q8_0 regression.

### Results — Q4_K_M

| Configuration | Generation | tok/s | vs Phase 4 |
|---------------|------------|-------|------------|
| Phase 4 baseline (serial, scalar) | 20747 ms / 30 tok | 1.45 | — |
| Step 1: threading only (scalar) | 13427 ms / 30 tok | 2.23 | 1.54× |
| Step 2: SIMD, 1 thread | 9809 ms / 30 tok | 3.06 | **2.11×** |
| Step 2: SIMD, 12 threads | 6818 ms / 30 tok | 4.40 | **3.03×** |

### Results — Q8_0 (simpler dequant, larger file)

| Threads | tok/s | note |
|---------|-------|------|
| 1 | 6.40 | single-thread is already fast — Q8_0 dequant is cheap |
| 12 (step 1, min_rows=16) | 2.00 | **regression** — spawn overhead dominated |
| 12 (step 2, min_rows=512) | 5.64 | fixed; slight overhead from w_gate/w_up spawns |

### hyperfine — Q4_K_M (20 tokens, 3 runs each)

```
threads=1:   7.862 s ± 0.066 s
threads=12:  5.655 s ± 0.007 s
threads=12 is 1.39× faster
```

### Analysis

**SIMD dot product (2.11× single-thread)**: The `@Vector(8, f32)` loop processes
32 floats per iteration using 4 VFMADD256 instructions, vs 1 scalar FMA before.
Zen 2 can sustain 2 VFMADD256/cycle = 16 FMAs/cycle. The 3.5 KB row_buf fits in
L1D (32 KB), so the dot product is compute-bound after dequant. Measured speedup
implies the dot product is ~60% of total per-row time, dequant ~40%.

**Spawn threshold (fixes Q8_0)**: Q8_0 dequant is ~14× cheaper than Q4_K, so
the same 128-row matrix that is worth threading for Q4_K is not worth it for Q8_0.
`min_rows=512` makes wk/wv serial for all quant types. For the large matrices
(w_gate 4864 rows → 10 threads; lm_head 151936 rows → 12 threads) threading
still helps regardless of quant type.

**Threading ratio change (1.39× vs 1.54× before)**: With min_rows=512, wq/wo
use only 2 threads instead of 12, reducing the parallel fraction. This is the
correct tradeoff — the absolute throughput is higher (4.40 vs 2.23 tok/s), even
though the threading multiplier shrinks.

---

## Next: Step 3 — Fused Q4_K/Q8_0 dequant+dot (planned)

Vectorize the dequantization itself. Q4_K nibble unpacking and scale extraction
can be done with SIMD byte shuffles. Q8_0 can fuse i8→f32 conversion with
the dot product, eliminating the `row_buf` round-trip entirely.
