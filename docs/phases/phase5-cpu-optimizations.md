# Phase 5 — CPU Optimizations

## What changed from Phase 4

Phase 4 had a correct but single-threaded forward pass. Every matrix-vector multiply
dequantized one row at a time on the calling thread. Phase 5 makes the forward pass
faster in two steps:

1. **Multi-threading** — split output rows across all CPU threads
2. **AVX2 SIMD** *(planned)* — vectorized dot product using 8-wide f32 FMA

---

## Step 1: Parallel row dispatch (`quantMatvecPar`)

### The bottleneck

`quantMatvec` processes one row at a time: dequantize → dot-product → next row.
All CPU time is on the calling thread. On a 12-thread Ryzen 3600 this wastes 11
out of 12 available cores for the most expensive operation in the model.

### Design

```
quantMatvecPar(out, mat_data, mat_type, vec, rows, cols, scratch, n_threads)

  nt = min(n_threads, rows / min_rows_per_thread)   ← skip tiny matrices

  for t in 0..nt:
    jobs[t] = RowJob { row_start, n_rows, row_buf = scratch[t*cols..] }
    threads[t] = Thread.spawn(RowJob.run, &jobs[t])

  for t in 0..n_spawned: threads[t].join()
```

Key properties:

- **No allocation inside threads.** Scratch is pre-allocated as `n_threads × max_cols`
  floats by the caller. Each thread gets its own disjoint slice `scratch[t*cols..]`
  with no synchronisation needed.

- **No thread pool.** Zig 0.16 has no `std.Thread.Pool`; raw `Thread.spawn` / `Thread.join`
  is used. Threads are spawned per-call, not reused across calls.

- **Spawn-overhead guard.** `min_rows_per_thread = 16` — if the total row count is
  too small, the serial path runs on the calling thread. This prevents spawning 12
  threads to compute 24 output values.

- **Fallback on spawn failure.** If `Thread.spawn` returns an error, that job slice
  is run synchronously on the calling thread. Only successfully spawned threads are
  joined.

### Integration

`forwardOneModel` now receives `n_threads: usize` and allocates:

```zig
const scratch = try allocator.alloc(f32, n_threads * max_cols);
```

Every `quantMatvec` call is replaced with `quantMatvecPar`. `embedLookup` stays
single-threaded (it accesses one row).

`cmdGenerate` detects core count via `std.Thread.getCpuCount() catch 1`, overridable
with `--threads N` for benchmarking.

### Results (Qwen2.5-0.5B Q4_K_M, Ryzen 5 3600, ReleaseFast)

| Configuration | tok/s | vs baseline |
|---------------|-------|-------------|
| Phase 4 serial (1 thread) | 1.45 | — |
| Phase 5 parallel (12 threads) | 2.23 | **1.54×** |

See `docs/benchmarks/phase5.md` for full hyperfine data and analysis.

### Why the speedup is modest

The dominant cost is memory bandwidth: loading quantized weight bytes from RAM.
Every thread shares the same memory bus; 12 threads reading the same cache lines
in parallel doesn't multiply throughput — it saturates bandwidth sooner. The 1.54×
speedup reflects the fraction of work that is compute-bound (the dot product itself)
vs. memory-bound (the dequantize fetch). For larger models with higher compute
intensity the threading benefit will be more pronounced.

Small attention projection matrices (wq: 896 rows, wk/wv: 128 rows for GQA) barely
clear the min-rows threshold and gain little. Large FFN matrices (w_gate/w_up/w_down:
4864 rows) and the lm_head (151936 rows) benefit most.

---

## Step 2: AVX2 SIMD dot product

### Changes

**`build.zig`**: Set default CPU model to `.native` so AVX2/FMA are available
without a special build flag. This is correct for a single-machine project.

**`math.zig`**: Added `dotf32(a, b)`:

```zig
pub inline fn dotf32(a: []const f32, b: []const f32) f32 {
    var acc0: @Vector(8, f32) = @splat(0.0);
    var acc1: @Vector(8, f32) = @splat(0.0);
    var acc2: @Vector(8, f32) = @splat(0.0);
    var acc3: @Vector(8, f32) = @splat(0.0);
    var i: usize = 0;
    while (i + 32 <= n) : (i += 32) {
        acc0 += a[i+ 0..][0..8].* * b[i+ 0..][0..8].*;
        acc1 += a[i+ 8..][0..8].* * b[i+ 8..][0..8].*;
        acc2 += a[i+16..][0..8].* * b[i+16..][0..8].*;
        acc3 += a[i+24..][0..8].* * b[i+24..][0..8].*;
    }
    // ... 8-element tail, scalar tail ...
}
```

4 accumulators because Zen 2 has 5-cycle FMA latency + 2 FMA units per cycle →
need ~10 independent in-flight FMAs to keep both units busy at peak throughput.
4 × 8 = 32 elements per iteration = 4 FMA ops → not perfect but close.

**`min_rows_per_thread` raised to 512**: Benchmarking showed Linux thread spawn
is ~200 µs (not 50 µs). With `min_rows=16`, Q8_0 multithreaded was 3× *slower*
than single-threaded because 1884 spawns × 200 µs = 376 ms overhead per token.
With `min_rows=512`, wk/wv (128 rows) and wq/wo (896 rows with only 2 threads)
reduce total spawns to ~636, fixing the regression.

### Results (Q4_K_M, Qwen2.5-0.5B)

| Step | Threads | tok/s | vs Phase 4 |
|------|---------|-------|------------|
| Phase 4 serial scalar | 1 | 1.45 | — |
| Phase 5 step 1 (threading, scalar) | 12 | 2.23 | 1.54× |
| Phase 5 step 2 (SIMD) | 1 | 3.06 | **2.11×** |
| Phase 5 step 2 (SIMD + threads) | 12 | 4.40 | **3.03×** |

The SIMD dot product alone is a 2.11× single-thread gain. Measured result implies
the dot product accounts for ~60% of total per-row time, dequant ~40%.

See `docs/benchmarks/phase5.md` for full hyperfine data and analysis.

---

## What's missing (Phase 6)

- **MoE routing**: Qwen3.6 and Gemma4 use Mixture-of-Experts FFN layers. The loader
  already maps `ffn_gate/up/down` but MoE models have dozens of expert FFNs and a
  learned router. Phase 6 implements MoE dispatch.
- **BF16 support**: some models store weights as BF16.
- **Batched prefill**: processing all prompt tokens in one batched matrix multiply
  rather than one-at-a-time would give much faster prefill.
