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

## Step 2: AVX2 SIMD dot product *(planned)*

### What it does

Replace the scalar dot-product loop:

```zig
var sum: f32 = 0;
for (row, vec) |w, x| sum += w * x;
```

with an 8-wide Zig `@Vector` loop that compiles to AVX2 FMA instructions:

```zig
var acc: @Vector(8, f32) = @splat(0);
var i: usize = 0;
while (i + 8 <= n) : (i += 8) {
    const w8: @Vector(8, f32) = row[i..][0..8].*;
    const x8: @Vector(8, f32) = vec[i..][0..8].*;
    acc += w8 * x8;
}
var sum = @reduce(.Add, acc);
// handle tail
```

Each VFMADD instruction processes 8 f32 values per cycle. The Ryzen 3600 has two
256-bit FMA units → up to 16 FMAs per cycle. The scalar loop does 1 per cycle.

### Fused Q8_0 dequant+dot

For Q8_0 blocks, the decode is `val = d × qs[i]`. Instead of writing to an
intermediate `row_buf` and then doing a separate dot, both can be fused:

```
for each block b:
    d = scale[b]
    for i in 0..32 step 8:
        q8 = load 8 × i8 from qs
        qf = @intToFloat(@Vector(8, f32), q8)
        acc += (d × qf) * vec[b*32+i..]
```

This eliminates one full write+read of `max_cols` floats per row and fits the hot
data in L1/L2 instead of spilling to L3.

### Expected gain

2–4× on the dot-product component of each row. Combined with threading:
~3–6× total vs Phase 4 baseline, targeting 4–8 tok/s on the 0.5B model.

---

## What's missing (Phase 6)

- **MoE routing**: Qwen3.6 and Gemma4 use Mixture-of-Experts FFN layers. The loader
  already maps `ffn_gate/up/down` but MoE models have dozens of expert FFNs and a
  learned router. Phase 6 implements MoE dispatch.
- **BF16 support**: some models store weights as BF16.
- **Batched prefill**: processing all prompt tokens in one batched matrix multiply
  rather than one-at-a-time would give much faster prefill.
