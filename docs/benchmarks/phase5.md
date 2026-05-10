# Phase 5 — CPU Optimization Benchmarks

## Hardware

- CPU: Ryzen 5 3600 (6C/12T, Zen 2), AVX2 + FMA
- RAM: 64 GB DDR4
- OS: Linux x86_64
- Build: `zig build -Doptimize=ReleaseFast`

## Model

`Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`  
24 layers, d_model=896, d_ffn=4864, 14 Q heads / 2 KV heads, vocab=151936

---

## Step 1 — Multi-threading (`quantMatvecPar`)

Split output rows across N threads. Each thread dequantizes and dot-products its
own row slice independently, writing into a disjoint sub-slice of the output.

Pre-allocated per-thread scratch buffers (`n_threads × max_cols` floats) avoid
any allocation or contention inside the hot path. Minimum 16 rows/thread before
spawning to amortise the ~50 µs thread-spawn cost.

### Results

| Threads | Prefill (4 tokens) | Generation | tok/s |
|---------|-------------------|------------|-------|
| 1       | 2967 ms           | 20747 ms / 30 tok | 1.45 |
| 12      | 1863 ms           | 13427 ms / 30 tok | 2.23 |

**Speedup: 1.54× generation, 1.59× prefill**

### hyperfine (20 tokens, 3 runs each)

```
Benchmark 1 (threads=1):   16.817 s ± 0.110 s
Benchmark 2 (threads=12):  11.117 s ± 0.047 s
Summary: threads=12 is 1.51× faster
```

### Analysis

Modest speedup despite 12× thread count because:

1. **Memory bandwidth bottleneck** — Q4_K dequant is memory-bound; raw bandwidth
   is shared across all cores. Ryzen 3600 has ~47 GB/s theoretical; the
   dequant+matvec loop is limited by how fast we can stream the quantized rows.

2. **Small matrices** — With d_model=896, the wq/wk/wv matrices have only 896 and
   128 output rows respectively (well below 12 × 16 = 192 min threshold for k/v).
   The FFN w_gate/w_up/w_down rows (4864) and lm_head rows (151936) do benefit.

3. **Thread spawn overhead** — ~50 µs per spawn × 24 layers × ~6 matrix ops = ~7 ms
   overhead per token, partially amortised by the larger matrices.

---

## Next: Step 2 — AVX2 SIMD dot product

Replace scalar `for (row, vec) |w, x| sum += w * x` with `@Vector(8, f32)` 8-wide
FMA. Expected 2–4× additional throughput on the dot product itself.

Fused Q8_0 dequant+dot: decode 8 i8 values → f32, multiply, accumulate in a
single inner loop iteration — eliminates the intermediate `row_buf` write for Q8_0.
