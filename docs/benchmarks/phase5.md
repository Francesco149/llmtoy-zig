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

## Step 3 — Fused Q8_0 dequant + dot (`dotQ8_0`)

Added `dq.dotQ8_0(data, vec)`: processes a Q8_0 row directly against the input
vector without writing to `row_buf`. Uses `vpmovsxbd` (i8→i32 sign-extend) +
`vcvtdq2ps` (i32→f32) + `vmulps` in a 32-element unrolled inner loop.

Serial paths in both `quantMatvec` and `RowJob.run` branch on `mat_type == .q8_0`
before dequant, taking the fused path and skipping `row_buf` entirely.

### Results (30 tokens, ReleaseFast)

| Model | Threads | Step 2 tok/s | Step 3 tok/s | gain |
|-------|---------|-------------|-------------|------|
| Q8_0  | 1  | 6.40 | **14.06** | **2.20×** |
| Q8_0  | 12 | 5.64 | 6.48 | 1.15× |
| Q4_K_M | 1 | 3.06 | 3.06 | — |
| Q4_K_M | 12 | 4.40 | 4.40 | — |

### Analysis

**Q8_0 1-thread (2.20×)**: Per-row memory traffic falls from 8.1 KB to 4.5 KB.
`vpmovsxbd ymm, [mem]` reads 8 bytes and produces 8 i32s in one µop — the i8→f32
conversion has near-zero marginal cost over the memory load itself.

**Q8_0 12-thread (1.15×)**: Faster per-row work means spawn overhead is a larger
fraction of total time. Thread pool needed to fully exploit multithreading on
fast paths.

**Q4_K_M unchanged**: The fused path is Q8_0-only; Q4_K still uses dequant+dotf32.

### Cumulative from Phase 4 baseline (Q4_K_M, best-thread config)

| Phase | tok/s (best config) | overall speedup |
|-------|---------------------|----------------|
| 4 serial scalar | 1.45 | — |
| 5.1 threading | 2.23 (12t) | 1.54× |
| 5.2 SIMD dot | 4.40 (12t) | 3.03× |
| 5.3 fused Q8_0 | 4.40 (12t) | 3.03× (Q4_K unchanged) |
| 5.4 fused Q5_0 | **4.82 (1t)** | **3.32×** |

### Q8_0 cumulative

| Phase | tok/s |
|-------|-------|
| 4 baseline | ~1.5 |
| 5.2 SIMD dot | 6.40 (1-thread) |
| 5.3 fused dot | **14.06** (1-thread) |

---

## Step 4 — Fused Q5_0 dequant + dot (`dotQ5_0`)

### Background: actual quant distribution

Inspecting `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` with `llmtoy info` reveals the dominant
quantization type is **Q5_0** (132 tensors), not Q4_K (12 tensors). Token embeddings and
most attention projections are Q5_0; the "K_M" suffix only affects the FFN matrices.

### Q5_0 block layout (22 bytes / 32 elements)

```
[0..1]  f16  d         — scale
[2..5]  u8[4] qh       — 1 high bit per element packed as little-endian u32
[6..21] u8[16] qs      — 4 low bits per element, 2 per byte
```

Reconstruction: `val = d × ((nib | hi_bit<<4) − 16)`, range −16..15.

### `dotQ5_0` SIMD strategy

Process 32 elements in 4 comptime-unrolled groups of 8:

```
g=0: lo nibbles of qs[0..7],  qh bits  0.. 7 → elements  0.. 7
g=1: lo nibbles of qs[8..15], qh bits  8..15 → elements  8..15
g=2: hi nibbles of qs[0..7],  qh bits 16..23 → elements 16..23
g=3: hi nibbles of qs[8..15], qh bits 24..31 → elements 24..31
```

High-bit extraction: shift qh right by a comptime offset, isolate 8 bits with
`{1,2,4,8,16,32,64,128}` masks, then `@min(bits, splat(1))` normalises each to 0/1.
No intermediate row_buf write — values convert directly i8→f32 via `vpmovsxbd + vcvtdq2ps`.

### Compiler output (inner per-block loop)

```asm
; ── per-block loop ───────────────────────────────────────────────────────
.block_loop:
    ; load f16 scale and convert
    vpinsrw     xmm7, xmm0, [rdi], 0x0    ; insert 2-byte f16 into xmm7
    mov         eax, [rdi+0x2]             ; qh (u32, 4 bytes)
    vmovq       xmm8, [rdi+0x6]            ; qs[0..7] — 8 nibble-packed bytes
    vcvtph2ps   xmm7, xmm7                 ; f16 → f32 scale

    ; extract lo nibbles of qs[0..7]
    vpand       xmm9, xmm8, xmm1           ; xmm1 = 0x0F×8: lo nibbles → xmm9
    vpsrlw      xmm8, xmm8, 0x4
    vpand       xmm8, xmm8, xmm6           ; xmm6 = 0x0F×16: hi nibbles → xmm8

    ; high-bit extraction for group 0 (qh bits 0..7)
    ; xmm4 = 0xF0×8  — encodes -16 as i8 (clever: 0xF0|nib = nib-16 after sign-ext)
    ; xmm2 = BIT_MASKS = {1,2,4,8,16,32,64,128} as u32
    vpbroadcastd ymm10, xmm9               ; broadcast qh[0..7] to 8 u32 lanes
    vpand       ymm10, ymm10, ymm2         ; isolate bits 0..7 of qh
    vpcmpeqd    ymm10, ymm10, ymm3         ; ymm3=0: eq-to-zero mask (0xFF where bit was 0)
    vpackssdw   xmm10, xmm10, xmm11       ; pack i32→i16
    vpacksswb   xmm10, xmm10, xmm10       ; pack i16→i8 byte mask
    vpblendvb   xmm10, xmm5, xmm4, xmm10  ; bit=0 → 0xF0 (=-16 as i8), bit=1 → 0

    vpor        xmm9, xmm10, xmm9         ; q5_i8: bit=0 → nib|0xF0 = nib-16, bit=1 → nib
    vpmovsxbd   ymm9, xmm9                ; sign-extend i8→i32 (encodes -16 via 0xF0)

    ; convert and multiply (groups 1-3 follow the same pattern)
    vcvtdq2ps   ymm9, ymm9                ; i32 → f32: bit=0: nib-16, bit=1: nib  (no vsubps!)
    vmulps      ymm9, ymm9, [rsi-0x60]    ; × vec[0..7]
    ; ... groups 1-3 similarly produce ymm10, ymm8, ymm11 ...

    vaddps      ymm8, ymm9, ymm8          ; accumulate all 4 groups
    vaddps      ymm8, ymm8, ymm10

    ; horizontal reduce ymm8 → scalar, multiply by scale, accumulate total
    vmulss      xmm7, xmm_scalar, xmm7   ; block_sum × d
    vaddss      xmm0, xmm0, xmm7         ; total += block_sum

    dec         rcx
    jne         .block_loop
```

**Key insight — `vpblendvb` encodes the −16 offset into the selection constant:**
`xmm4 = {0xF0×8}`. `0xF0` is −16 as a signed i8. When the high bit is **zero**, the blend
selects `0xF0`; OR with the nibble gives `nib | 0xF0 = nib − 16` after sign-extension
(`vpmovsxbd`). When the high bit is **one**, the blend selects `0x00`; nibble passes
through unchanged. The `vsubps −16.0` instruction is eliminated entirely — the subtraction
is absorbed into the integer domain before `vcvtdq2ps`.

`vpinsrw + vcvtph2ps` converts the f16 block scale in one XMM round-trip, identical to
the Q8_0 path.

### Results (30 tokens, ReleaseFast)

| Model | Threads | Step 3 tok/s | Step 4 tok/s | gain |
|-------|---------|-------------|-------------|------|
| Q4_K_M | 1  | 3.06 | **4.82** | **1.57×** |
| Q4_K_M | 12 | 4.40 | 4.65 | 1.06× |
| Q8_0   | 1  | 14.06 | 14.06 | — (unchanged) |

### hyperfine (20 tokens, 3 runs)

```
threads=1:   4.441 s ± 0.073 s
threads=12:  4.726 s ± 0.009 s
threads=1 is 1.06× faster than threads=12
```

Single-thread is now faster than 12-thread because per-row work is so fast that
raw-spawn overhead (~200 µs × ~650 spawns = ~130 ms/token) exceeds parallelism gains.
This repeats the pattern from step 3 for Q8_0: a thread pool is required to benefit
from threading at this throughput level.

### Analysis

**1.57× single-thread gain**: Q5_0 is 132/290 tensors in Q4_K_M (~45% of calls to
`quantMatvec`). For those rows, per-row memory traffic falls from ~3.5 KB store +
3.5 KB load (row_buf round-trip) to read-only (nibbles + vec). The fused path also
eliminates the SIMD-scalar boundary crossing that `dequantQ5_0 + dotf32` required.

**Threading breakeven**: Step 4 makes Q5_0 rows ~1.57× faster. Each row is now so
fast that `min_rows_per_thread=512` no longer adequately amortises the 200 µs
spawn cost for medium matrices (wq at 896 rows spawns 2 threads = 400 µs overhead
vs < 100 µs of useful work gained). A persistent thread pool would fix this.

---

## Step 5 — Persistent thread pool

### Problem

Each `quantMatvecPar` call spawned new threads (200 µs each). After fused Q5_0,
per-row work is fast enough that spawn overhead exceeded parallelism gains at any
matrix size. 12 threads was slower than 1 thread at step 4.

### Changes

- `src/ops/thread_pool.zig`: new `ThreadPool` struct with persistent worker threads
  sleeping on a condition variable, draining a bounded ring-buffer work queue.
  `submit(func, ctx)` enqueues without spawning. `wait()` blocks until all pending
  jobs finish. No allocation inside the hot path.
- `min_rows_per_thread` dropped from 512 → 64: pool wakeup costs ~5–20 µs, not
  200 µs. wq/wo (896 rows) now gets all 12 threads instead of 2.
- `quantMatvecPar` signature: `n_threads: usize` → `pool: *ThreadPool`.
- `ThreadPool.init` takes `io: std.Io` (required by `std.Io.Mutex`/`std.Io.Condition`
  in Zig 0.16's async-cancellation-aware sync API).
- Pool created once in `cmdGenerate`, passed through `forwardOneModel`.

### Results (50 tokens, ReleaseFast)

| Threads | Wall time (mean ± σ) | tok/s | vs 1-thread |
|---------|----------------------|-------|-------------|
| 1       | 10.493 s ± 0.091 s   | 4.77  | —           |
| 4       | 4.661 s ± 0.168 s    | 10.7  | 2.25×       |
| 8       | 3.863 s ± 0.181 s    | 12.9  | 2.71×       |
| 12      | 3.546 s ± 0.065 s    | 14.1  | **2.96×**   |

### hyperfine (50 tokens, 5 runs each)

```
threads=1:   10.493 s ± 0.091 s
threads=4:    4.661 s ± 0.168 s
threads=8:    3.863 s ± 0.181 s
threads=12:   3.546 s ± 0.065 s
threads=12 is 2.96× faster than threads=1
```

### vs step 4 (12-thread)

Step 4 (raw spawn): 4.65 tok/s (12t was *slower* than 1t at 4.82)
Step 5 (pool):     14.1 tok/s — **3.0× faster at 12 threads**

### Analysis

**Spawn overhead removed**: Raw spawn was ~200 µs × ~650 spawns/token ≈ 130 ms
overhead, consuming nearly all the per-token budget. Pool wakeup (condition-variable
signal + futex) costs ~5–20 µs — 10–40× cheaper. The same 650 dispatches now add
< 13 ms total overhead.

**min_rows threshold**: Lowered from 512 → 64 because the amortisation criterion
changed from "is the work > 200 µs?" to "is the work > 20 µs?". At 64 rows ÷ 12
threads ≈ 6 rows/thread, a single wq (896×896) split is still 74 rows/thread —
well above threshold. wk/wv (128 rows) get 10 rows/thread; below the threshold for
small thread counts but submits correctly for 2 threads.

**Scaling ceiling** (4.77 → 14.1 = 2.96× for 12 threads): Amdahl's law; sequential
ops (rmsnorm, softmax, RoPE, residuals, sampling) are ~25% of wall time. The
theoretical max at 12 threads is 12/((1-0.75)+0.75/12) ≈ 4.4×. Achieving 2.96× at
12 threads means ~67% of the compute-bound work is parallelised effectively.

### Cumulative from Phase 4 baseline

| Step | tok/s (best config) | overall speedup |
|------|---------------------|----------------|
| Phase 4 serial scalar | 1.45 | — |
| 5.1 threading (raw spawn) | 2.23 (12t) | 1.54× |
| 5.2 SIMD dot + native target | 4.40 (12t) | 3.03× |
| 5.3 fused Q8_0 | 4.40 (12t) | 3.03× (Q4_K unchanged) |
| 5.4 fused Q5_0 | 4.82 (1t, spawn regressed MT) | 3.32× |
| **5.5 thread pool** | **14.1 (12t)** | **9.72×** |
