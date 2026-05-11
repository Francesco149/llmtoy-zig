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

### Compiler output (from `objdump -d -M intel`, ReleaseFast + native CPU)

Inner loop for the 32-element unrolled body. Each `ymm` register holds 8 × f32
(256 bits). `rdi` = pointer to `a[]`, `rsi` = pointer to `b[]`, `rax` = `i`.

```asm
; ── prologue: zero 4 accumulators ─────────────────────────────────────────
vxorps  ymm0, ymm0, ymm0       ; acc3 = 0
vxorps  ymm1, ymm1, ymm1       ; acc2 = 0
vxorps  ymm2, ymm2, ymm2       ; acc1 = 0
vxorps  ymm3, ymm3, ymm3       ; acc0 = 0

; ── 32-element inner loop (one iteration = 32 floats) ─────────────────────
.loop:
    ; load a[i+0..31] into ymm4-7  (4 × 256-bit unaligned load)
    vmovups ymm4, [rdi + rax*4]        ; a[i+ 0.. 7]
    vmovups ymm5, [rdi + rax*4 + 0x20] ; a[i+ 8..15]
    vmovups ymm6, [rdi + rax*4 + 0x40] ; a[i+16..23]
    vmovups ymm7, [rdi + rax*4 + 0x60] ; a[i+24..31]

    ; multiply each chunk by b[i+…]  (memory-source VMULPS = fused load+mul)
    vmulps  ymm4, ymm4, [rsi + rax*4]        ; a[i+0..7]  * b[i+0..7]
    vmulps  ymm8, ymm5, [rsi + rax*4 + 0x20] ; a[i+8..15] * b[i+8..15]
    vmulps  ymm6, ymm6, [rsi + rax*4 + 0x40] ; a[i+16..23]* b[i+16..23]
    vmulps  ymm5, ymm7, [rsi + rax*4 + 0x60] ; a[i+24..31]* b[i+24..31]

    add     rax, 0x20                         ; i += 32

    ; accumulate into acc0-3  (4 independent VADDPS, no stall)
    vaddps  ymm3, ymm3, ymm4
    vaddps  ymm2, ymm8, ymm2
    vaddps  ymm1, ymm1, ymm6
    vaddps  ymm0, ymm0, ymm5

    cmp  rcx, rdx
    jbe  .loop

; ── merge 4 accumulators → 1 ──────────────────────────────────────────────
vaddps  ymm2, ymm3, ymm2
vaddps  ymm0, ymm1, ymm0
vaddps  ymm0, ymm2, ymm0   ; ymm0 = final 8-lane sum

; ── horizontal reduction: 8 lanes → 1 scalar ──────────────────────────────
; (shuffle + add sequence; vextractf128 extracts hi 128 bits of ymm0)
vextractf128 xmm1, ymm0, 0x1   ; hi 4 lanes
; … several vshufpd / vaddss to fold down to xmm0[0]
vzeroupper                       ; AVX→SSE transition safety
ret
```

**Key points:**

- `vmulps ymm, ymm, [mem]` is a *memory-source* fused form: 1 µop that loads and
  multiplies in the same pipeline slot. This halves the instruction count for the
  multiply stage compared to a separate `vmovups` + `vmulps`.
- LLVM did **not** emit `vfmadd213ps`. Zen 2 has separate FP-MUL and FP-ADD
  execution units, so `vmulps` + `vaddps` can issue in parallel on the same cycle
  — the effective throughput matches FMA when both units are free.
- 4 independent accumulators (`ymm0`–`ymm3`) hide the 5-cycle ADD latency by
  keeping 4 independent dependency chains in flight simultaneously.
- The horizontal reduction (8 lanes → scalar) is done once per call, not per
  iteration, so its cost is amortised over the entire vector length.

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

## Step 3: Fused Q8_0 dequant + dot product

### The row_buf round-trip problem

The existing path for Q8_0:

```
dequantQ8_0(raw_bytes, row_buf)   ← writes 896 f32s = 3.5 KB to row_buf
dotf32(row_buf, vec)               ← reads back those 3.5 KB + 3.5 KB of vec
```

Each row touches 952 bytes of Q8_0 data + 3584 bytes of stores + 3584 bytes of
loads back = 8.1 KB total. Even with row_buf in L1D, the store→load round-trip
consumes store-buffer slots and load bandwidth.

### `dotQ8_0` — fused path

```zig
pub fn dotQ8_0(data: []const u8, vec: []const f32) f32 {
    for (0..n_blocks) |b| {
        const d  = f16_to_f32(blk[0..2]);
        const qs = blk[2..34];         // 32 × i8
        var acc: @Vector(8, f32) = @splat(0.0);
        inline for (0..4) |j| {        // unrolled at compile time
            const qi = @bitCast(@Vector(8, u8), qs[j*8..][0..8].*);
            const qf: @Vector(8, f32) = @floatFromInt(qi);  // i8 → f32
            acc += qf * vec[base + j*8..][0..8].*;
        }
        total += @reduce(.Add, acc) * d;
    }
}
```

Each row touches only 952 bytes (Q8_0 data) + 3584 bytes (vec read) = 4.5 KB,
and writes nothing to memory during the computation.

### Compiler output (from `objdump -d -M intel`, inner per-block loop)

```asm
; ── per-block loop (one iteration = 32 elements) ─────────────────────────
.block_loop:
    ; load f16 scale, convert to f32  (single instruction for f16→f32)
    vpinsrw     xmm2, xmm0, [rdi], 0x0     ; load 2-byte f16 scale
    vcvtph2ps   xmm2, xmm2                  ; f16 → f32

    ; sign-extend 4 × 8 i8 values → 4 × 8 i32  (vpmovsxbd: QWORD → YMM)
    vpmovsxbd   ymm3, [rdi+0x02]   ; qs[0..7]:   i8→i32, 8 bytes → 32 bytes
    vpmovsxbd   ymm4, [rdi+0x0a]   ; qs[8..15]
    vpmovsxbd   ymm5, [rdi+0x12]   ; qs[16..23]
    vpmovsxbd   ymm6, [rdi+0x1a]   ; qs[24..31]

    add         rdi, 0x22           ; advance to next block (34 bytes)

    ; convert i32→f32, multiply by vec, accumulate  (memory-source vmulps)
    vcvtdq2ps   ymm3, ymm3
    vmulps      ymm3, ymm3, [rax-0x60]    ; * vec[base+0..7]
    vcvtdq2ps   ymm4, ymm4
    vmulps      ymm4, ymm4, [rax-0x40]    ; * vec[base+8..15]
    vaddps      ymm3, ymm3, ymm4
    vcvtdq2ps   ymm5, ymm5
    vmulps      ymm4, ymm5, [rax-0x20]    ; * vec[base+16..23]
    vcvtdq2ps   ymm5, ymm6
    vmulps      ymm5, ymm5, [rax]         ; * vec[base+24..31]
    vaddps      ymm3, ymm3, ymm4
    vaddps      ymm3, ymm3, ymm5          ; ymm3 = block dot sum (before scale)

    ; horizontal reduce → scalar, multiply by scale d
    ; … (vshufpd / vextractf128 / vaddss sequence)
    vmulss      xmm2, xmm3, xmm2          ; block_sum × d
    vaddss      xmm0, xmm0, xmm2          ; total += block_sum

    dec         rcx
    jne         .block_loop
```

**Key instructions:**

- `vpmovsxbd ymm, [mem+offset]` — sign-extends 8 contiguous i8 values from
  memory directly into a 256-bit YMM register (8 × i32). No intermediate buffer
  write, no separate load instruction. One µop.
- `vcvtdq2ps ymm, ymm` — converts the 8 × i32 to 8 × f32. One µop.
- `vcvtph2ps xmm, xmm` — converts one f16 scale to f32. The compiler emits this
  because the block scale is stored in the GGUF file as f16.
- No `vmovups` store anywhere — the row_buf is gone entirely.

The `inline for (0..4)` is fully unrolled at compile time, producing 4 × (vpmovsxbd
+ vcvtdq2ps + vmulps + vaddps) in the loop body with no branch overhead.

### Results

| Config | tok/s (step 2) | tok/s (step 3) | gain |
|--------|---------------|---------------|------|
| Q8_0, 1 thread  | 6.40 | **14.06** | **2.20×** |
| Q8_0, 12 threads | 5.64 | 6.48 | 1.15× |
| Q4_K_M, 1 thread | 3.06 | 3.06 | — (unchanged) |
| Q4_K_M, 12 threads | 4.40 | 4.40 | — (unchanged) |

Q8_0 single-thread nearly doubles because the fused path reduces per-row memory
traffic from 8.1 KB to 4.5 KB and eliminates all intermediate stores.

Q8_0 multithreaded barely improves because the fused path makes per-row work
~2× cheaper, which means thread-spawn overhead (~200 µs × 636 spawns = 127 ms)
is now an even larger fraction of the ~71 ms token budget. This confirms that
a persistent thread pool is the correct next step for threading.

See `docs/benchmarks/phase5.md` for full numbers.

---

## Step 4: Fused Q5_0 dequant + dot product

### The real bottleneck

Inspecting the model with `llmtoy info` reveals that **Q5_0 is the dominant quant type**
(132 tensors) in `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`. Q4_K has only 12 tensors. The
step 3 assumption that Q4_K was the main target was wrong; Q5_0 comes first.

### Q5_0 block format

```
[0..1]   f16   d          — scale
[2..5]   u8[4] qh         — 1 high bit per element (32 bits total)
[6..21]  u8[16] qs        — 4 low bits per element, 2 per byte (nibble-packed)
```

Reconstruction: `val = d × ((nib | hi_bit<<4) − 16)`, range −16..15.

32 elements are interleaved: elements 0..15 use the **lo nibbles** of `qs[0..15]` with
`qh` bits 0..15; elements 16..31 use the **hi nibbles** of `qs[0..15]` with `qh` bits
16..31.

### `dotQ5_0` — fused path

Process 32 elements in 4 comptime-unrolled groups of 8 (`inline for (0..4) |g|`):

```zig
pub fn dotQ5_0(data: []const u8, vec: []const f32) f32 {
    const BIT_MASKS: @Vector(8, u32) = .{ 1, 2, 4, 8, 16, 32, 64, 128 };
    for (0..n_blocks) |b| {
        const d   = f16Bytes(blk[0..2]);
        const qh  = std.mem.readInt(u32, blk[2..6], .little);
        const qs  = blk[6..22];
        var acc: @Vector(8, f32) = @splat(0.0);
        inline for (0..4) |g| {
            // comptime: which half of qs, lo or hi nibble, which qh offset
            const nib: @Vector(8, u8) = ...;
            // isolate 8 consecutive qh bits via bitmask, normalise to 0/1
            const hb:  @Vector(8, u8) = @intCast(@min(qh_v & BIT_MASKS, splat(1)));
            const q5_u8 = nib | (hb << splat(4));       // 0..31
            const q5    = @as(@Vector(8, f32), @floatFromInt(q5_u8)) - splat(16.0);
            acc += q5 * vec[base + g*8..][0..8].*;
        }
        total += @reduce(.Add, acc) * d;
    }
}
```

### What the compiler did (annotated disassembly)

LLVM eliminated the `vsubps -16.0` entirely by encoding the subtraction into the
high-bit selection constant:

- Source: `nib | (hb << 4)` where hb ∈ {0, 1} → values 0..31 as u8, then `- 16.0`
- Compiler: replaces `0x10` (16) with `0xF0` (= −16 as i8), then uses `vpmovsxbd`
  (sign-extend i8→i32) instead of `vpmovzxbd`

For `hi=0`: blend selects `0xF0`, `nib | 0xF0` as i8 = `nib − 16` (−16..−1) ✓  
For `hi=1`: blend selects `0x00`, `nib | 0x00` as i8 = `nib` (0..15) ✓

The `vpcmpeqd ymm, ymm, zero` + `vpackssdw` + `vpacksswb` + `vpblendvb` sequence
performs the conditional selection; no branch in the inner loop. Full annotated asm
in `docs/benchmarks/phase5.md`.

### Results

| Config | tok/s (step 3) | tok/s (step 4) | gain |
|--------|----------------|----------------|------|
| Q4_K_M, 1 thread  | 3.06 | **4.82** | **1.57×** |
| Q4_K_M, 12 threads | 4.40 | 4.65 | 1.06× |
| Q8_0, 1 thread | 14.06 | 14.06 | — (no Q5_0 tensors) |

Single-thread is now faster than 12-thread. The per-row work is fast enough that
raw-spawn overhead dominates. A persistent thread pool is the correct next step.

See `docs/benchmarks/phase5.md` for full numbers and hyperfine data.

---

## Step 5: Persistent thread pool (`ThreadPool`)

### The problem with raw spawn

Every `quantMatvecPar` call spawned new OS threads. Linux `Thread.spawn` costs
~200 µs. With 24 layers × ~27 matmuls × a few threads each, this added ~130 ms
of pure spawn overhead per token — dwarfing the actual compute for fast quant types.

After step 4's fused Q5_0, each row is so fast that even `min_rows=512` could not
prevent spawn overhead from dominating. 12 threads ran *slower* than 1 thread.

### Design

```
ThreadPool:
  workers[N]  — threads blocked on work_ready condvar
  queue[]     — ring-buffer of (func, ctx) entries
  head/tail   — absolute counters; index via % cap
  pending     — jobs submitted but not yet complete

submit(func, ctx):   acquire mutex, enqueue, pending++, signal one worker
wait():              acquire mutex, sleep on all_done while pending > 0
workerLoop:          hold mutex, wait for work, release, execute, reacquire, pending--
```

Workers hold the mutex only when checking/modifying queue state, not during
execution. The lock is never held across the actual compute.

Zig 0.16 moved synchronisation primitives from `std.Thread.Mutex/Condition` to
`std.Io.Mutex/Condition`, which require an `io: std.Io` parameter on every call
(to support async cancellation). `ThreadPool` stores `io` at init time; all
`lockUncancelable`, `unlock`, `waitUncancelable`, `signal`, `broadcast` calls
pass `pool.io`.

`min_rows_per_thread` dropped 512 → 64: pool wakeup costs ~5–20 µs, so the
amortisation threshold is ~10× lower than for raw spawn.

### Results

| Threads | tok/s (step 4) | tok/s (step 5) | gain |
|---------|----------------|----------------|------|
| 1       | 4.82           | 4.77           | ~same |
| 4       | —              | 10.7           | —     |
| 8       | —              | 12.9           | —     |
| 12      | 4.65           | **14.1**       | **3.0×** |

12-thread pool (14.1 tok/s) is **9.72× faster than the Phase 4 serial baseline**
(1.45 tok/s). The multi-threading that was a regression at step 4 is now the
fastest configuration.

See `docs/benchmarks/phase5.md` for full hyperfine data and analysis.

---

## What's missing (Phase 6)

- **MoE routing**: Qwen3.6 and Gemma4 use Mixture-of-Experts FFN layers. The loader
  already maps `ffn_gate/up/down` but MoE models have dozens of expert FFNs and a
  learned router. Phase 6 implements MoE dispatch.
- **BF16 support**: some models store weights as BF16.
- **Batched prefill**: processing all prompt tokens in one batched matrix multiply
  rather than one-at-a-time would give much faster prefill.
