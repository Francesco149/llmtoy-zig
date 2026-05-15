# Phase 7 GPU — Path to Endgame

Where we are: Phase 7f/g landed. ~3.10–3.17 prefill / ~3.06–3.09 decode tok/s on Gemma4
26B A4B / RX 7800 XT (+37–56% vs 12-thread CPU). 5 GPU submits per layer; attention
(QK, softmax, V, RoPE), RMSNorms, residual adds still on CPU.

Where we want to land: **match or beat llama.cpp's Vulkan backend** for the same
model + GPU combo. llama.cpp's per-layer cost on this hardware is dominated by the
matmuls themselves, with everything else fused into a small handful of submits.
Concretely the targets are:

- 1 GPU submission per layer (record once, dispatch many)
- All compute on GPU; CPU only does the per-token sampler at the very end
- Matvec shaders that exploit subgroup-cooperative dot products (our current
  per-row shaders use 1 thread per output row, which leaves ~30× SIMD lane
  parallelism on the table for RDNA3)
- A passing **logit-equality** test against the CPU path — currently the GPU
  path is faster but produces slightly off tokens (see Phase 7h below).

This plan is structured so most of the work can be done by Sonnet. Hard
numerical debug and shader micro-optimization are flagged **[OPUS]**.

---

## Phase 7h — Correctness harness [SONNET]

**Status: blocker.** Confirmed regression: with `--temperature 0 --gpu` the model
emits glitched tokens (e.g. `"ever is the first addend"` where CPU emits `"2 is
the first addend"`). User noticed it starting around commit `10b478a` (expert
accum on GPU); could be older.

We cannot optimize further without a way to detect numerical regressions. Build
the harness first.

### 7h.1  `compare` CLI command

Add `llmtoy compare <model> <prompt> [--seed N] [--steps N]` that:
1. Runs one forward pass on CPU, captures logits per layer.
2. Runs the same pass on GPU, captures logits per layer.
3. Prints per-layer `max |Δ|` and `argmax_match` between the two.
4. At the final layer prints the top-5 tokens for both and flags any
   ranking difference.

Implementation hooks:
- Add an optional `[]f32` "tap" parameter on `forward()` that, if non-null,
  receives `x` (residual stream) at the end of each layer.
- Compare CPU vs GPU taps with `max(|a - b|)` and `max(|a - b| / max(|a|, eps))`.

This is the diagnostic surface we lack today. Once it exists every further
shader change is verified in one command.

### 7h.2  Per-tensor-type fuzz tests

For each shader (`matvec_q3_k`, `matvec_q4_k`, `matvec_q5_0`, `matvec_q5_1`,
`matvec_fused_gu_q3k`, `expert_accum`):
1. Generate random fp32 matrix.
2. Quantize on the CPU using our reference dequant code (round-trip).
3. Run the same matvec on CPU and GPU.
4. Assert `max |Δ| / max |out_cpu| < 1e-4` (Q-error dominates so loose
   tolerance is fine).

Half of these tests already exist but not all; sweep and fill gaps.

### 7h.3  Per-layer bisect mode

CLI flag `--gpu-layers L0:L1` that only runs GPU on layers L0..L1 inclusive
(others stay on CPU). With this + `compare` we can binary-search to find which
GPU layer first diverges, then which tensor within that layer.

### Deliverable
- `compare` command works end-to-end.
- A row of fuzz tests gates the GPU path in `zig build test`.
- Output of `llmtoy compare <model> "explain MoE"` clearly localizes the bug
  to a layer + tensor type.

---

## Phase 7i — Fix the numerical bug [OPUS]

Once 7h is in place, hand back to Opus to actually find the bug. Likely
suspects (in rough order of probability):

1. **`matvec_fused_gu_q3k.glsl`** — only Q3K shader doing two matrix reads per
   thread; subtle indexing or accumulation error wouldn't show up in the
   standalone Q3K shader test.
2. **`expert_accum.glsl`** — read-modify-write on a HOST_COHERENT buffer; if a
   barrier or memset is missed on some path the stale value leaks.
3. **Q5_0/Q5_1 dequant** in the matvec shader — many `ffn_down` tensors use
   these and any sign-bit / scale bug accumulates over 30 layers.
4. **GELU constant** — `0.7978845608` in shader vs `0.7978845608028654` in
   Zig. Round to the same f32 in theory; verify in practice.

Until 7h lands, this is unfalsifiable; don't try to fix blindly.

---

## Phase 7j — RMSNorm + residual on GPU [SONNET]

**Educational unit**: trivial GPU shaders (1 thread per element), but
demonstrates how to keep activations in VRAM across the layer boundary.

### Current waste

Every layer: rmsnorm → matmul (GPU) → download → CPU add residual → CPU rmsnorm
→ upload → matmul (GPU) → … . Each download/upload is a PCIe round-trip
(~10–50 µs) and there are ~6 per layer × 30 layers = ~180 trips per token.

### Step 1: `rmsnorm.glsl`
- One workgroup reduces `sum(x²)` via shared-memory reduction.
- Then each thread writes `x[i] * (weight[i] / sqrt(mean_sq + eps))`.
- Two bindings: `x` in/out + `weight` in. Push constants: `n`, `eps`.
- 1+ε weight handling (Gemma uses `1 + w` for RMSNorm).

### Step 2: `add.glsl` / `scale.glsl`
- Trivial elementwise. Single shader with a flag for `out = a + b` vs
  `out *= s`. Or two separate shaders — whichever reads cleaner.

### Step 3: residual buffer in VRAM
- Allocate device-local `x_buf` once at session start.
- After token embed, upload to `x_buf` once.
- All subsequent layer ops read/write `x_buf` in place.
- Only download `x_buf` after the final layer norm for the lm_head matmul.

### Win
Per-token PCIe traffic drops from ~180 round-trips to ~2 (token embed upload,
final logits download). Should be worth 10–20% on top of current numbers
purely from removed sync overhead.

---

## Phase 7k — Subgroup-cooperative matvec [OPUS + SONNET]

**This is where the biggest win is.** Our current matvec shaders are
`1 thread per output row` — for a 2816×2112 matrix that's 2816 threads doing
2112 multiplies each, sequentially. RDNA3 wavefronts are 32 lanes wide. We
should have 32 (or 64) threads cooperatively reduce the dot product for a
single row.

### Reference

llama.cpp's `mul_mat_vec_q3_k.comp` does exactly this. Key tricks:
- `BLOCK_SIZE = 32` (one subgroup per row).
- Threads in the subgroup each handle a contiguous chunk of `cols`.
- Partial sums combined via `subgroupAdd()`.
- Scales loaded once into shared memory per super-block.

### Estimated speedup

Our current Q3_K matvec is **memory-bound**: each thread reads one Q3K block
(110 bytes) per super-block and does the dequant arithmetic. With 32-way
subgroup reduction:
- Reads coalesce: 32 lanes read 32 contiguous bytes per cycle → effective
  bandwidth ~32× higher than scalar gather.
- Arithmetic stays the same but is hidden behind memory latency.
- Expected per-shader speedup: 4–10×.

Since matmuls dominate per-layer time, this should roughly double overall
tok/s — putting us in the 6–7 tok/s range, close to llama.cpp Vulkan.

### Plan

1. **[OPUS]** Port `mul_mat_vec_q3_k.comp` to our naming/binding conventions.
   This is fiddly because the GGML packing is subtle and any bug shows up as
   wrong logits. Use the harness from 7h to verify.
2. **[SONNET]** Same treatment for Q4_K, Q5_0, Q5_1, Q8_0 (mechanical port).
3. **[OPUS]** Same for `matvec_fused_gu_q3k.glsl` — this one's harder because
   two matrices are read per thread; the subgroup version has to alternate or
   ping-pong reads. May need a Phase 7g-style measurement of whether the fused
   path still wins after both shaders are subgroup-optimized.

### Tuning knobs to try

- BLOCK_SIZE 32 vs 64 (RDNA3 supports both via `VK_KHR_shader_subgroup_size_control`).
- One row per workgroup vs multiple rows per workgroup with separate accumulators.
- `coopmat` / FP16 accumulation (only if numerics check out at 7h tolerance).

---

## Phase 7l — Attention on GPU [SONNET + OPUS]

**Educational unit**: this is the meaty one. Lots of small steps.

Right now: 4 matmuls per layer go through the GPU (Q, K, V, O), but the
softmax(Q@K^T)·V step happens entirely on the CPU using the KV cache as
plain RAM.

### 7l.1  KV cache in VRAM [SONNET]
- Allocate device-local `k_cache`, `v_cache` per layer, sized
  `[max_seq × n_kv_heads × d_head]`.
- After `wk`/`wv` matmul writes to scratch buffer, follow with a tiny copy
  shader (or `vkCmdCopyBuffer`) that appends to the cache at the right
  position offset.
- Doesn't speed anything up yet, just unblocks 7l.2.

### 7l.2  Q·K^T + softmax shader [OPUS]
- One workgroup per attention head.
- Each thread computes one column of `qk` (= `q · K[t,:]`).
- Then subgroup-cooperative softmax (max → exp → sum → divide).
- Write softmax weights into shared memory or a small VRAM scratch.
- This is the most numerically-sensitive shader; FP16 accumulation is
  tempting but verify against the CPU path.

### 7l.3  attn_out = softmax_weights · V [SONNET]
- Standard matvec, but cols = current seq length (variable). Push constant.

### 7l.4  RoPE shader [SONNET]
- Gemma4 uses `rope_freqs.weight` (256 dims), not computed θ.
- Read `freqs` from a small device-local buffer.
- Apply per-element rotation to Q and K post-matmul, pre-cache-write.

### 7l.5  Sliding-window mask [OPUS]
- Gemma4 alternates local/global attention.
- For local layers, mask out tokens outside the sliding window.
- Simplest: bake the mask into the softmax shader via push constants
  (start_pos, end_pos). Cleanest: a small auxiliary "valid mask" buffer.

### Win
After 7l, the only CPU work per token is `topK` for the MoE router
(small — 128 floats → 8 indices), and the final sample. Per-layer is one
submit → many dispatches → one barrier sequence.

---

## Phase 7m — One submit per layer [SONNET]

By the time 7l lands we should be down to ~2 submits/layer (one for attention,
one for FFN+MoE). Collapse to 1 by recording everything in one command buffer
between the input-residual write and the output-residual read.

This is mechanical: re-order Phase 7f's batching to fold the residual/norm
shaders in. No new shaders.

---

## Phase 7n — Persistent command buffers [SONNET]

A command buffer that does the same dispatches every token can be recorded
once and replayed each forward step. The only per-token state is push
constants (current position, mask range).

Vulkan supports this via `VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT` and
re-bound descriptors. The win is removing host-side command-buffer construction
overhead — small (~50 µs/layer), but at 30 layers/token that's ~1.5 ms/token
which is non-trivial at 6–7 tok/s.

---

## Phase 7o — Last-mile micro-optimisation [OPUS]

By here we should be at parity with llama.cpp or close. Final
profile-driven tuning:

- Pipeline cache warm-up (avoid first-token jitter).
- `VK_EXT_shader_atomic_float` for cooperative reductions if available.
- Pre-sized descriptor pools (we allocate per-dispatch today; the pool grows).
- Single-buffer KV cache vs separate K/V (cache line behavior).
- `VK_KHR_pipeline_executable_properties` to inspect register pressure
  per shader and tune `local_size`.

---

## Sequencing summary

| Phase | Owner | Blocks | Estimated overall tok/s gain |
|-------|-------|--------|------------------------------|
| 7h    | Sonnet | everything | 0% (correctness only) |
| 7i    | Opus   | nothing (parallel to 7j) | 0% (correctness only) |
| 7j    | Sonnet | 7l        | +10–20%               |
| 7k    | Opus+Sonnet | 7m   | +80–100%              |
| 7l    | Sonnet+Opus | 7m   | +30–50%               |
| 7m    | Sonnet | 7n        | +5%                   |
| 7n    | Sonnet | 7o        | +5–10%                |
| 7o    | Opus   | —         | +5–10%                |

Multiplying optimistically: ~3.1 → ~9 tok/s. llama.cpp Vulkan on this combo
should be benchmarked early (in 7h) so we have a hard target.

---

## Working notes for Sonnet

- **Always use `nix develop --command zig build [test]`** — `zig` is not on PATH.
- **GPU runs require systemd-run wrap** for safety:
  `systemd-run --user --scope -p MemoryMax=40G --quiet -- nix develop --command ./zig-out/bin/llmtoy ...`
- **CLI argument order**: `generate <model> <prompt>` first, flags after.
- **Reference shaders**: `/opt/ai-lab/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/`.
  In particular `mul_mat_vec_q3_k.comp`, `mul_mat_vec_q4_k.comp`, `rope_*.comp`,
  `soft_max.comp`, `rms_norm.comp`.
- **Hand off to Opus** when: a numerical bug doesn't localize via the
  harness within an hour of bisecting, a shader port produces wrong values
  and the GGML packing doesn't visibly match, or a perf change regresses
  tok/s by >5% and the cause is non-obvious.
- **Don't optimize 7g style without 7h evidence**. The fused dense-FFN
  experiment is a cautionary tale — looked promising, was actually slower,
  required revert. Always measure before committing.

---

## Cross-references

- Current benchmark numbers: `docs/benchmarks/phase7_gpu.md` (Phase 7f baseline).
- GPU code: `src/gpu/{context,buffer,matvec}.zig`, `src/gpu/shaders/*.glsl`.
- Gemma4-specific GPU glue: `src/model/gemma4/{gpu_weights,forward}.zig`.
- llama.cpp reference: `/opt/ai-lab/llama.cpp` (per memory).
