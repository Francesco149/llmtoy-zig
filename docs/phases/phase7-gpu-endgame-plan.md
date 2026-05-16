# Phase 7 GPU — Path to Endgame

Where we are: Phase 7f/g landed. ~3.10–3.17 prefill / ~3.06–3.09 decode tok/s on
Gemma4 26B A4B / RX 7800 XT (+37–56% vs 12-thread CPU). 7h harness exposed two
real correctness bugs (Q5_1 nibble ordering, V silently skipped in batched QKV),
both fixed.

Where we want to land: **match or beat llama.cpp's Vulkan backend** for the same
model + GPU combo. llama.cpp Vulkan runs Gemma4 26B A4B at **80–100 tok/s** on
this hardware. We're at ~3 tok/s. The gap is not 2× — it's 30×. Closing it
requires a structural rewrite of the matmul + activation flow, not incremental
shader tuning.

The single biggest lever is **Q8_1 quantized activations + integer dot
product**. Everything else in this plan flows from that decision.

---

## Verification protocol (READ BEFORE ANY OPTIMIZATION)

Every GPU change — shader port, batching tweak, fused kernel, anything — must
clear this gate before commit. **No exceptions.** Phase 7g shipped a "fused"
shader that was actually slower; Phase 7e shipped a Q5_1 shader with wrong
nibble ordering that wasn't user-visible until Phase 7h's harness landed three
commits later. Both would have been caught at commit time by the protocol below.

### 1. Per-shader fuzz test (`zig build test`)

Every new shader needs a randomized fuzz test alongside the hand-crafted
single-block tests. Pattern is established in `src/gpu/matvec.zig`'s
`fuzzQuantMatvec` helper:

1. Generate random block bytes (clamp the f16 scale to ~0.012 to avoid
   NaN/Inf — the CPU dequant is the reference, so we don't care about
   reproducing real model weight statistics).
2. Run the existing CPU dequant + dot path (`math.quantMatvec` + the
   relevant `dq.dequantQ*`) as ground truth.
3. Run the GPU shader on the same bytes via `MatvecSession.runOwned`.
4. Assert `max |Δ| / max |out_cpu| < 1e-4`.

The current Q5_1 bug got past the existing single-block tests because they
only set `qs[0]=0x01` (element 0), where the buggy "interleaved" ordering and
the correct "low-then-high" ordering happen to agree. Random data exercises
every nibble pair and would have caught it on the first run.

### 2. Per-layer compare against CPU (`llmtoy compare`)

```sh
systemd-run --user --scope -p MemoryMax=40G --quiet -- nix develop --command \
  ./zig-out/bin/llmtoy compare <model> "explain MoE" --chat
```

Prints per-layer `max |Δ|` between the CPU and GPU residual streams plus the
final top-5 logit comparison. After the 7h fixes the baseline is:

- L0–L12: < 1.5% rel_err (numerical noise floor for f32 matvec without Q8_1
  activations; see Q8_1 milestone below)
- L13–L29: 1.5–55% rel_err drifting upward, **L28 argmax FAIL**, final token
  mismatch in some prompts

That drift is the unfixed-but-deferred amplification problem. It is **expected
to disappear** when we adopt Q8_1 activations (see milestone below); do not
spend more than an hour chasing it before then. What you ARE checking with
`compare` after every change is **regression**: does the change make any layer
that was passing before now fail, or push a per-layer rel_err meaningfully
higher than the baseline above?

### 3. Tensor-level bisect (`--gpu-layers L0:L1`)

When `compare` shows a regression, narrow it:

- `--gpu-layers L:L` runs only layer L on GPU; everything else CPU. Tells you
  whether the bug is in layer L specifically.
- For tensor-level bisect within a layer (which matmul or which expert), the
  cleanest pattern is to add a temporary `const DBG_NO_GPU_X = true` constant
  at the top of `forwardOne` and gate the relevant `mv()` call on it. Flip one
  toggle at a time, rebuild, re-run `compare`. **Remove all toggles before
  commit** — they're scratch debugging, not shipped configuration.

### 4. End-to-end generation parity

```sh
./llmtoy generate <model> "what is 1+1?" --chat --temperature 0 --max-tokens 30
./llmtoy generate <model> "what is 1+1?" --chat --temperature 0 --max-tokens 30 --gpu
```

At T=0 the two should produce **identical** output once Q8_1 activations land.
Until then, they should at minimum produce coherent (not glitched) text and
diverge only on the last few tokens of long generations.

### 5. Benchmark before / after

If a change is supposed to make things faster, prove it:

```sh
hyperfine --warmup 1 --runs 3 \
  'systemd-run … ./zig-out/bin/llmtoy generate <model> "<long prompt>" --chat --max-tokens 64 --gpu'
```

Record both prefill and decode tok/s. Store the delta in `docs/benchmarks/`.
If it regresses by >5% and the cause isn't obvious, revert and reopen.

---

## Phase 7h — Correctness harness [SONNET, DONE]

Delivered:

- `llmtoy compare <model> <prompt>` with per-layer residual diff + final top-5
- `--gpu-layers L0:L1` flag on `compare` and `generate` for per-layer bisect
- Optional `layer_taps: ?[][]f32` parameter on `forwardOne` for residual capture
- Randomized fuzz tests for every active shader (Q8_0, Q5_0, Q5_1, Q4_K, Q3_K)
  via `fuzzQuantMatvec` helper in `src/gpu/matvec.zig`. All pass at rel < 1e-4.

The harness immediately found two bugs (see 7i). Future agents extend this
infrastructure rather than rebuilding it.

---

## Phase 7i — Numerical bugs found by 7h [OPUS, PARTIALLY DONE]

### Fixed

1. **Q5_1 nibble ordering** (`matvec_q5_1.glsl`): shader walked `i = 0..31` and
   pulled `(qs[i/2] >> (4*(i%2))) & 0xF`, producing an interleaved
   `(low, high, low, high, …)` pairing. GGML packs Q5_1 as
   `(qs[0..15] low → elements 0..15; qs[0..15] high → elements 16..31)`,
   matching `dequantQ5_1`. Fixed by mirroring the (already correct) Q5_0
   shader's loop. Layer 0 rel_err: 98.7% → 0.82%. Commit `87a0f25`.

2. **V silently skipped in batched QKV** (`forward.zig`): `runLayerQKV` was
   entered whenever wq+wk had GPU sessions, but if `wv` existed and was
   CPU-only (e.g. Gemma4 L3 attn_v is Q5_K, not in `isGpuSupported`), the V
   dispatch was skipped without a CPU fallback and `v_cur` carried stale
   data from the previous layer. Tightened the batch condition so a CPU-only
   V drops to the per-call branch. Commit `e72aa4e`.

### Deferred to "Q8_1 milestone" below

Per-layer rel_err still drifts up to ~55% by L28 with a final-argmax mismatch
on some prompts. Bisect localizes this to **per-matmul precision noise (~1e-7
per output element) being amplified through `post_attention_norm`'s 1/RMS
division when the underlying activation magnitude is small at deep layers**.
Fuzz tests confirm every shader is correct to rel < 1e-4 in isolation, so
this is not a discrete shader bug.

The standard fix in the GGUF ecosystem is **Q8_1-quantized activations**:
quantize the input vector to Q8_1 once per matmul, do the dot product as
`int8 × int8 → int32` accumulation, convert to f32 at the end. This is
deterministic across CPU and GPU implementations (no floating-point reorder),
which is why llama.cpp Vulkan matches its CPU reference bit-for-bit. Adopting
it eliminates the drift as a side effect.

**Do not chase the residual amplification any further with our current f32
matvec stack.** Spend that time on the Q8_1 milestone instead.

---

## Phase 7k* — Q8_1 activation quantization + integer-dot matvec [DONE]

**Landed**: 4.28 prefill / 4.07 decode tok/s (vs 3.10/3.07 baseline). The
drift the original plan deferred ("expected to disappear" with Q8_1) did in
fact collapse once Q8_1 covered enough of the path — see notes below.

Reference: llama.cpp's `quantize_q8_1.comp`, `mul_mat_vecq.comp`,
`mul_mat_vecq_funcs.glsl`, `mul_mat_vec_base.glsl`.

### What changes

For every matmul (attention QKV/O, dense FFN gate/up/down, expert
gate_up_exps/down_exps, lm_head):

1. The activation vector stops being f32 in HOST_COHERENT and becomes Q8_1
   in device-local VRAM, written by a `quantize_q8_1` shader.
2. The matvec shader (`mul_mat_vecq_<weight_type>`) reads the weight in its
   GGUF block format AND the activation in Q8_1, does the dot via
   `subgroupAdd(int8_t × int8_t → int32_t)` with `GL_EXT_integer_dot_product`,
   and writes f32 output.
3. Each row's dot product is computed by a full subgroup (32 lanes on RDNA3)
   working cooperatively, not by a single thread sequentially looping over
   `cols`.

Both wins land at once: ~10× memory bandwidth (coalesced subgroup reads vs
scalar gather), ~4× ALU throughput (int dot vs scalar fma), AND CPU-matching
precision.

### Order of work

1. **[OPUS] `quantize_q8_1.glsl`**. Single elementwise shader: take an f32
   vector of length `cols`, write `cols / 32` Q8_1 blocks (each block: f16 d,
   f16 s, 32 × int8 qs). One workgroup per 32-element block, threads do the
   per-block max-abs reduction via `subgroupMax`. Verify with a fuzz test:
   `quantize_q8_1(x); dequantQ8_1(out)` should round-trip x to within Q8_1's
   ~1/127 error.

2. **[OPUS] Port `mul_mat_vec_q4_k.comp` first**. Q4_K is the most-used type
   in our model (all attention matmuls L0–L29). Adopt llama.cpp's binding
   conventions and `mul_mat_vec_base.glsl` framework as-is — don't try to
   shrink it. After porting:
   - Run the Q4_K fuzz test from `matvec.zig`. Must pass at rel < 1e-3
     (looser than f32 matvec because Q8_1 introduces real ~1/127 quant error
     on the activation).
   - Run `llmtoy compare`. Layer-by-layer rel_err should now be **flat at
     noise level** across all 30 layers (no more drift), and final argmax
     should match. This is the moment the deferred 7i drift dies.
   - Benchmark. Decode tok/s should roughly double immediately.

3. **[SONNET] Mechanical port for Q3_K, Q5_0, Q5_1, Q8_0**. Once Q4_K's
   pattern is established, the others are copy-paste of the corresponding
   llama.cpp `mul_mat_vec_q*_k.comp` with our binding layout. Each gets a
   fuzz test before commit.

4. **[OPUS] Decide the fate of `matvec_fused_gu_q3k.glsl` and
   `expert_accum.glsl`**. Both were performance hacks specific to the
   scalar-per-row matmul era. With subgroup matvecs the per-shader cost
   drops 4–10× and the fixed overhead of an extra dispatch per expert
   matters less. Likely outcome: delete the fused shader, keep the accum
   shader (since it removes a CPU readback regardless of matmul speed).
   Measure both ways, pick the winner.

### Estimated speedup

The full effect of points 1–4 is the bulk of the 30× gap to llama.cpp.
Realistic projection on this hardware: **3 → 30–50 tok/s** in one milestone.

### What this milestone makes obsolete

- Phase 7i's "remaining" amplification debug — replaced by Q8_1 determinism.
- The non-subgroup `matvec_q*_k.glsl` shader family — superseded by the
  `mul_mat_vecq_*` ports. Once those land and pass the fuzz tests, delete
  the old shaders and their pipeline structs in `gpu/matvec.zig`. Don't
  carry both.
- `FusedGateUpPipeline` — see point 4.

---

## Phase 7j — RMSNorm + residual + GPU GELU [INTEGRATED, SUBMIT 5 → 3]

**Status as of 2026-05-16 (integration session):**
- ✓ All shaders done: `rmsnorm.glsl`, `elem_add.glsl`, `elem_scale.glsl`,
  `gelu_mul.glsl`, `rope_neox_{table,theta}.glsl`. All fuzz-tested.
- ✓ Pipelines + per-layer norm-weight buffer uploads + x_vram/xb_vram/
  stage_buf in GpuWeights (commit `6f80133`).
- ✓ **`runLayerAttnQ8_1`** — fuses attn_norm + quantize + QKV in 1 submit.
- ✓ **`runLayerDenseFfnQ8_1`** — fuses ffn_norm + quantize + gate/up +
  gelu*up + w_down + post_ffw_norm_1 in 1 submit (kills 1 submit/layer).
- ✓ **`runLayerAttnResidualDenseFfnQ8_1`** — also folds wo + post_attn_norm +
  residual into the same submit (kills another 1/layer). Per-layer count
  drops from 5 → 3.
- ✗ RoPE / per-head Q/K/V norms on GPU — not integrated; per-head dispatches
  add their own submit cost and the CPU SIMD path is already cheap.
  Deferred indefinitely (better addressed in 7l with full attention on GPU).

**Actual measured gain:** ~0.5% on a 64-token generate
(28.60 s baseline → 28.44 s with full 3-submit chain — borderline 1σ).

**Plan estimate was wrong.** The +20–38% number assumed each saved submit
was ~3 ms of overhead. Measured on RX 7800 XT (Mesa RADV), per-submit
overhead is closer to **~100–200 µs**. The 5 → 3 reduction saves
~300–600 µs/token out of ~245 ms = 0.1–0.25%.

**What this means for the remaining gap to llama.cpp (4 → 80–100 tok/s):**
submit count was not the bottleneck. The bottleneck is **GPU matmul
throughput** — Q3_K/Q4_K integer-dot on this hardware/driver, plus the
inability to overlap CPU attention with GPU FFN. Phase 7l (attention on
GPU + KV-cache in VRAM) is the only remaining structural lever that
addresses this directly.

**Reordered to AFTER 7k\*.** Rationale: with Q8_1 matvecs, the CPU still has
to receive the f32 matvec output, do the norm, do the residual add, then
re-quantize to Q8_1 for the next matmul — that's 3 PCIe round-trips per
matmul × ~6 matmuls/layer × 30 layers = ~540 round-trips/token. Eliminating
those is what gets us from "matmul on GPU" to "compute on GPU".

After 7k\* lands, the activation already lives in VRAM in Q8_1 form. The
norms/residuals/RoPE need GPU shaders so the CPU never has to pull the
activation back to f32 between matmuls.

### Shaders to write

1. **`rmsnorm.glsl`**: workgroup-shared-memory reduction of `sum(x²)`, then
   per-element scale by `weight[i] / sqrt(mean_sq + eps)`. Handle Gemma's
   `(1 + w)` weight convention via a push-constant flag.
2. **`add.glsl`**: trivial elementwise `a += b`.
3. **`scale.glsl`**: trivial elementwise `x *= s` (used for embedding scale,
   layer_output_scale).
4. **`rope_neox.glsl`**: per-element rotation. Gemma4 reads
   `rope_freqs.weight` for global layers; SWA layers compute θ from
   `rope_theta_swa`. Push constant: `pos`. Two variants or one shader with a
   uniform flag.

### Residual stream lives in VRAM

After embed-lookup (still CPU), upload `x[d_model]` to a device-local
`x_buf` once. Every layer reads/writes `x_buf` in place via the shaders
above + 7k\*'s matvecs. Only download after the final out_norm for the
lm_head matmul.

Each shader gets a fuzz test (CPU reference vs GPU output, rel < 1e-5 since
these are exact rather than quantized).

---

## Phase 7l — Attention on GPU [SONNET + OPUS]

Same scope as the original plan, but now the matvecs and norms it needs
already exist (from 7k\* and 7j). What's left:

### 7l.1  KV cache in VRAM [OPUS, DONE]

Three sub-commits:

- **7l.1a** (`7255747`) — allocate device-local `k_vram[l]/v_vram[l]` per
  layer in `GpuWeights`, sized to match the existing `Gemma4KvCache`. Lazy
  init via `initKvVram(cfg, max_seq)`. Total ~560 MiB on Gemma4 26B A4B at
  `max_seq=4096`. Plus `GpuContext.copyBufferRegion` for offset-aware
  one-shot transfers, and slot upload/download helpers.

- **7l.1b** (`c317c42`) — `rmsnorm_perhead.glsl` + `RmsnormPerHeadPipeline`.
  One workgroup per head, 256 threads cooperate over `head_dim` via
  subgroup reduction. Handles Q-norm (1+w, with weight), K-norm
  (1+w, with weight), and V's rmsnormRaw (no weight at all) via push
  constants. Three fuzz tests, all rel < 1e-6.

- **7l.1c** (`b234300`) — `runLayerAttnQ8_1KvVram` extends the existing
  Q8_1 attention submit with: per-head Q/K rmsnorm + V rmsnormRaw + RoPE
  on Q,K + `vkCmdCopyBuffer` K/V into the per-layer cache, all in one
  command-buffer submit. `forward.zig` skips the CPU per-head norm + RoPE
  + `@memcpy` for the 24 SWA layers (which have `wv`); the 6 global layers
  (which share V from K) fall back to `runLayerAttnQ8_1` + the CPU chain.

**Verified**: per-layer rel_err identical to 7k\* baseline (layer 28
worst-case 4.83%), all 30 argmaxes match CPU, final argmax matches.

**Measured wall-clock** (3-run hyperfine, 64-token generate including
~5.7 s model setup): **28.121 ± 0.126 s** vs 28.445 ± 0.178 s baseline,
**−1.1% (within σ)**. As expected, 7l.1 alone is structural — the CPU
norm/rope savings are roughly cancelled by extra GPU dispatches in the
same submit. The VRAM-resident K/V cache is dead storage until 7l.2/3
wire it into the actual attention compute.

### 7l.2  Q·K^T + softmax shader [OPUS]
One workgroup per attention head. Each thread computes one column of `qk`,
then subgroup-cooperative softmax (max → exp → sum → divide). Reference
`soft_max.comp` from llama.cpp. **Most numerically sensitive shader in the
pipeline** — verify against CPU `attn.zig` before doing anything else.

### 7l.3  attn_out = softmax_weights · V [SONNET]
Standard matvec, cols = current seq length (variable, push constant).

### 7l.4  Sliding-window mask [OPUS]
Bake into the softmax shader via push constants `(start_pos, end_pos)`.
Gemma4 alternates SWA / global per layer.

After 7l, the only CPU work per token is MoE topK (128 floats → 8 indices —
small), and the final sample.

---

## Phase 7m — One submit per layer [SONNET]

By 7l end we should be at ~2 submits/layer. Collapse to 1 by recording
attention + FFN + MoE + residuals + norms in one command buffer between
layer-input-write and layer-output-read. Mechanical, no new shaders.

---

## Phase 7n — Persistent command buffers [SONNET]

The same command sequence runs every token; only push constants change
(position, mask range). Record once via `VK_COMMAND_BUFFER_USAGE_…` reuse
flags. Saves ~50 µs/layer × 30 = ~1.5 ms/token, ~5–10% at 30 tok/s.

---

## Phase 7o — Last-mile micro-optimisation [OPUS]

By here we should be at or near llama.cpp parity. Final profile-driven
tuning:

- Pipeline cache warm-up (avoid first-token jitter).
- `VK_KHR_cooperative_matrix` (coopmat) for fp16 GEMM in attention if RDNA3
  exposes it under our driver.
- Pre-sized descriptor pools (we re-allocate per dispatch today).
- Single-buffer KV cache vs separate K/V (cache line behavior).
- `VK_KHR_pipeline_executable_properties` to inspect register pressure and
  tune `local_size`.

---

## Sequencing summary (revised)

| Phase | Owner       | Status      | Achieved/Estimated gain                  |
|-------|-------------|-------------|------------------------------------------|
| 7h    | Sonnet      | DONE        | 0% (correctness only, two bugs found)    |
| 7i    | Opus        | PARTIAL     | 0% (two bugs fixed, drift deferred)      |
| 7k\*  | Opus+Sonnet | **DONE**    | **+38% prefill, +33% decode** (3.10 → 4.28 / 3.07 → 4.07 tok/s) — plus the drift collapsed (all argmaxes match CPU after enough Q8_1 coverage) |
| 7j    | Sonnet+Opus | **DONE**    | 5 → 3 submits/layer; +0.5% wall-clock (submit overhead was much smaller than estimated) |
| 7l.1  | Opus        | **DONE**    | KV cache in VRAM + GPU per-head norms + GPU RoPE; −1.1% wall-clock (within σ — structural, sets up 7l.2/3) |
| 7l.2+ | Sonnet+Opus | next        | +30–50% (Q·K^T+softmax, attn·V, SWA mask) |
| 7m    | Sonnet      | after 7l    | +5%                                      |
| 7n    | Sonnet      | after 7m    | +5–10%                                   |
| 7o    | Opus        | last        | +5–10%                                   |

Current standing: **4.28 prefill / 4.07 decode tok/s** vs llama.cpp Vulkan
~80–100 tok/s on the same hardware. Remaining gap is ~20×.

Notes from the 7k\* push:
- Q4_K covered 6/30 attention layers; Q3_K filled the other 24. Plan
  originally assumed all 30 were Q4_K — fix landed as the Q3_K shader.
- Integer-dot drift behavior turned out as predicted but only at the
  *full-path* level: Q8_1 attention alone or Q8_1 attention+MoE-gateup didn't
  fix the L13–L28 drift; once wo and dense FFN also went through Q8_1, the
  drift collapsed and final argmax matched CPU.
- Q5_0/Q5_1 shaders work at cols%32==0 (so they cover expert_down at
  cols=704 and dense_down at cols=2112 — both unaligned to 256). Note the
  packQ8_1_x4 helper had a partial-last-group bug only exposed on
  non-128-aligned activations; fixed.

Notes for 7j integration (when picked up):
- All shaders are committed and fuzz-tested (rel ~7e-8 for rmsnorm, ~1e-6
  for rope). Pipelines live in GpuWeights; per-layer norm weight buffers
  are already uploaded.
- The restructure is mostly forward.zig surgery: keep x in x_vram, use
  pl_rmsnorm/pl_elem_add/pl_rope_* dispatches between matvecs, and run
  each layer's matmul+norm+residual chain in fewer command-buffer
  submits (currently 5/layer; achievable 3/layer with CPU still owning
  attention compute + MoE routing).
- Estimated upside: ~20–38% from reduced submit count + eliminated
  per-matmul PCIe upload/download. Worth it but not catastrophic to leave
  on the table.

Benchmark llama.cpp Vulkan on this exact (model, GPU, driver, prompt)
combination once per milestone so we always have a hard reference number;
"close to llama.cpp" is a moving target as their backend evolves.

---

## Working notes for agents

- **Always use `nix develop --command zig build [test]`** — `zig` is not on PATH.
- **GPU runs require systemd-run wrap** for safety:
  `systemd-run --user --scope -p MemoryMax=40G --quiet -- nix develop --command ./zig-out/bin/llmtoy ...`
- **CLI argument order**: `generate <model> <prompt>` first, flags after.
- **Reference shaders**: `/opt/ai-lab/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/`.
  In particular `quantize_q8_1.comp`, `mul_mat_vecq.comp`, `mul_mat_vecq_funcs.glsl`,
  `mul_mat_vec_base.glsl`, `mul_mat_vec_q*_k.comp`, `rope_neox.comp`,
  `soft_max.comp`, `rms_norm.comp`.
- **The verification protocol at the top of this document is mandatory**, not
  aspirational. Two of the three commits before 7h would have failed it. If
  you ship a change that breaks it, revert.
- **Hand off to Opus** when: a numerical bug doesn't localize via the harness
  within an hour of bisecting, a shader port produces wrong values and the
  GGML packing doesn't visibly match, or a perf change regresses tok/s by
  >5% and the cause is non-obvious.
- **Don't optimize 7g style without 7h evidence**. The fused dense-FFN
  experiment is a cautionary tale — looked promising, was actually slower,
  required revert. Always measure before committing.

---

## Cross-references

- Current benchmark numbers: `docs/benchmarks/phase7_gpu.md` (Phase 7f baseline).
- GPU code: `src/gpu/{context,buffer,matvec}.zig`, `src/gpu/shaders/*.glsl`.
- Gemma4-specific GPU glue: `src/model/gemma4/{gpu_weights,forward}.zig`.
- llama.cpp reference: `/opt/ai-lab/llama.cpp` (per memory).
