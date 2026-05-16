# Phase 7 GPU Offload Benchmarks

Model: Gemma 4 26B A4B (APEX-I-Mini, GGUF)
Hardware: Ryzen 5900x, 64 GB RAM, AMD RX 7800 XT (16 GB VRAM)
Prompt: "What is 2+2?" (chat template, --seed 42)

## GPU upload

- Weights uploaded: attention + dense FFN only (MoE experts stay on CPU)
- VRAM used: 866 MiB (182 MiB baseline → 866 MiB after upload)
- GTT used: < 50 MiB peak
- Upload time: ~280 ms

## Prefill (20 tokens)

| Mode | tok/s |
|------|-------|
| CPU (12 threads) | 2.29 |
| GPU (attn + dense FFN) | 2.57 |
| Speedup | +12% |

## Decode (tok/s)

| Mode | tok/s |
|------|-------|
| CPU (12 threads) | 2.01 |
| GPU (attn + dense FFN) | 2.41 |
| Speedup | +20% |

## Phase 7b — Q5_1 + Q5_0 added (all w_down on GPU)

| Mode | tok/s prefill | tok/s decode |
|------|--------------|--------------|
| GPU (attn + full dense FFN) | 2.62 | 2.41 |
| Speedup vs CPU (+16% / +9%) | | |

- VRAM: 984 MiB (182 MiB baseline → 984 MiB after upload)
- Q5_0 adds 24 layers of ffn_down to GPU (48 tensors); Q5_1 adds layer 0

## Phase 7c–7d — MoE experts on GPU (batched dispatch + persistent mapping)

**NOTE**: All benchmarks before phase 7d were unknowingly run in Debug mode.
`standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseFast })` in Zig 0.16
only activates ReleaseFast when `--release` is passed; without it the binary is Debug.
Fixed in 7d by switching to `b.option(...) orelse .ReleaseFast`.

### Per-expert dispatch (abandoned — all benchmarks were debug mode)

Uploaded all 128 experts × 30 layers to VRAM (~9.6 GiB) and dispatched each
expert matmul individually (8 experts × 3 matmuls × 30 layers = 720 GPU submits/token).
Even in debug mode this was slower than the debug CPU path. Abandoned.

### Batched dispatch (2 submits/layer = 60 syncs/token)

All 8 active experts dispatched in 2 command buffers per layer:
1. Gate + up matmuls (16 dispatches, 1 submit)
2. CPU gelu*up via persistently-mapped HOST_COHERENT slices (0 vkMapMemory calls)
3. Down matmuls (8 dispatches, 1 submit)
4. CPU scale+accumulate from persistent output slices

Prompt: "Briefly explain the forward pass of a MoE model." (--chat, ReleaseFast)

| Mode | tok/s prefill | tok/s decode |
|------|--------------|--------------|
| CPU only (12 threads) | 2.27 | 1.98 |
| GPU (attn + dense FFN + all experts, batched) | 2.43 | **2.10** |
| Speedup | +7% | +6% |

GPU is faster. Persistent mapping (map I/O buffers once at init) eliminated
720 vkMapMemory/vkUnmapMemory calls per token with no measurable benefit vs
per-call mapping — the bottleneck is the 60 GPU sync points, not the maps.

### Fused gate-gelu-up shader (1 submit/layer = 30 syncs/token)

Single shader computes `gelu(gate@x) * (up@x)` in one dispatch per expert.
mid_bufs promoted to device-local VRAM (GPU-only, never CPU-mapped).

| Mode | tok/s prefill | tok/s decode |
|------|--------------|--------------|
| Fused (1 submit/layer) | 2.48–2.61 | **2.11** |
| vs batched (2 submits/layer) | — | 2.10 |
| vs CPU only | — | 1.98 |

Halving GPU syncs (60→30/token) gave only +0.01 tok/s improvement — submit
overhead is negligible vs actual compute. The 6% gain over CPU (1.98→2.11)
comes purely from VRAM bandwidth (432 GB/s vs ~50 GB/s system RAM).

Bug fixed: manual `exp(2t)/(exp(2t)+1)` tanh overflowed to NaN for large
activations; replaced with GLSL built-in `tanh()`.

### GPU expert accumulation (Phase 3 moved to GPU)

Perf stat comparison (CPU-only vs GPU path, 16 tokens):

| Metric | CPU-only | GPU (fused+accum) |
|--------|----------|-------------------|
| CPUs utilized | 7.6 | 3.3 |
| Instructions | 1063B | 570B |
| IPC | 1.9 | 1.9 |
| L1-dcache miss | 5.5% | 3.9% |
| Context switches | 603K | 180K |

Previously the 8 expert outputs (88 KiB total) were read from HOST_COHERENT
memory by CPU and accumulated with per-expert scales. Moved this to a new
`expert_accum.glsl` shader: reads flat device-local `expert_all_out_buf`
(8 × d_model f32), writes weighted sum to a small HOST_COHERENT `moe_gpu_buf`.
CPU then does a single 11 KiB copy instead of 8 × 11 KiB scattered reads + mul.

| Mode | tok/s prefill | tok/s decode |
|------|--------------|--------------|
| Fused only (CPU accum) | 2.45–2.61 | 2.41 |
| + GPU accum (this change) | **2.80** | **2.74** |
| vs CPU only | 2.27 | 1.98 |
| Speedup vs CPU | +23% | +38% |

13% improvement over the fused-only baseline. The bottleneck was PCIe reads of
8 × 11 KiB HOST_COHERENT memory; consolidating to 1 × 11 KiB removed the
pressure. Profile script now supports `GPU=1 ./scripts/profile_gemma4.sh stat`.

## Phase 7e baseline — after GPU accumulation (perf stat)

Prompt: "Briefly explain the full forward pass of a MoE model." (16 tokens, ReleaseFast)
2.79 tok/s prefill, 2.72 tok/s decode.

| Metric | value |
|--------|-------|
| Wall time (inference) | ~16s (+6s GPU setup) |
| CPUs utilized | 3.4 |
| Instructions | 571B |
| Cycles | 288B |
| IPC | 2.0 |
| L1-dcache miss | 4.0% |
| Branch miss | 3.8% |
| Context switches | 182K |

Per layer the GPU currently does 7 separate vkQueueWaitIdle calls:
wq → wk → wv → wo → w_gate → w_up → w_down, plus 1 for the expert batch.
Attention (RoPE, softmax, KV-cache gather) and RMSNorms still run on CPU.

Next: batch QKV into one submit, batch gate+up, then progressively move
norms/residual/RoPE/attention to GPU for truly minimal roundtrips.

## Phase 7f — Batched QKV and gate+up dispatch

wq+wk+wv batched into one command buffer (3→1 submit); w_gate+w_up likewise
(2→1). Each batch: upload xb once to shared_vec, dispatch N matmuls in
parallel reading the same input, download N separate outputs. Added
MatvecSession.recordMv() and GpuWeights.runLayerQKV/runLayerGateUp().

Submit count per layer: **8 → 5** (wq+wk+wv=1, wo=1, gate+up=1, w_down=1, experts=1).

| Mode | tok/s prefill | tok/s decode |
|------|--------------|--------------|
| Before (Phase 7e) | 2.79 | 2.72 |
| After batching | **3.10–3.17** | **3.06–3.09** |
| Speedup vs CPU | +37–40% | +54–56% |

Removing 3 submit cycles saves more than just submit overhead — each
eliminated round-trip also removes an upload and a download of xb/output.
(The previously reported 3.23/3.11 was a lucky single-run outlier; typical
variance across runs is ±5%.)

## Phase 7k* — Q8_1 activations + integer-dot matvec

Switched the matvec path from f32-activation × quantized-weight (scalar-per-row)
to **Q8_1-activation × quantized-weight, subgroup-cooperative integer-dot**,
following the structure llama.cpp Vulkan uses to close the gap to its CPU
reference.

Activations are quantized once to int8 with a Q8_1 block layout (32 elements,
f16 d, f16 d*sum, packed into x4 groups of 4 blocks for 16-byte loads). The
matmul reads i8 quants from both sides and computes the row dot via
`dotPacked4x8EXT` — one VOPD on RDNA3, `integerDotProduct4x8BitPackedSigned`
is hardware-accelerated on the RX 7800 XT.

Shaders added:
- `quantize_q8_1.glsl` — f32 → block_q8_1_x4
- `matvec_q4_k_q8_1.glsl` — Q4_K weights × Q8_1 acts
- `matvec_q3_k_q8_1.glsl` — Q3_K weights × Q8_1 acts
- `matvec_fused_gu_q3k_q8_1.glsl` — fused gate+gelu+up for Q3_K MoE experts
- `matvec_q5_0_q8_1.glsl` — Q5_0 weights × Q8_1 acts
- `matvec_q5_1_q8_1.glsl` — Q5_1 weights × Q8_1 acts

Coverage today: **every matmul that has a Q8_1 pipeline now uses it** —
attention QKV (Q3_K + Q4_K), wo (Q4_K), dense FFN gate+up (Q3_K, batched),
dense FFN down (Q5_0/Q5_1), MoE fused gate+up (Q3_K), MoE down (Q5_0/Q5_1).
The only matmul left on the f32-acts path is lm_head (Q5_K or similar —
no Q5_K shader yet).

Numerics, side-effect of the broader Q8_1 coverage: per-layer `compare`
rel_err finally collapses. Baseline L13–L28 ranged 1.5–55% with L28 argmax
FAIL; the full-Q8_1 path keeps L0–L29 below 2.05% (worst L28 at 5.6%) and
**every layer argmax matches CPU**, including the final argmax. This is the
bit-determinism property the plan predicted — Q8_1 alone doesn't deliver
it, but Q8_1 across enough of the path does, because there's less f32
reduction-order math left to diverge.

| Mode | tok/s prefill | tok/s decode |
|------|--------------|--------------|
| Phase 7f baseline (no Q8_1) | 3.10–3.17 | 3.06–3.09 |
| Q8_1 attention only (6 of 30 layers, Q4_K only) | ~3.11 | ~3.08 |
| Q8_1 attention all 30 layers (Q3_K + Q4_K) | ~3.21 | ~3.15 |
| Q8_1 attention + Q8_1 fused MoE gate+up | 3.71–3.78 | 3.66–3.68 |
| Q8_1 attention + full Q8_1 MoE (gate+up + down) | ~3.86 | ~3.66–3.70 |
| Full Q8_1 routing (attn + MoE + dense FFN + wo) | **~4.28** | **~4.07** |

Net **~+38% prefill / +33% decode** end-to-end vs Phase 7f baseline. The
biggest single contribution is the Q8_1 fused MoE gate+up; the dense FFN
gate+up gives a similar bump because Q3_K (cols=2816, rows=2112) is a
substantial matmul × 30 layers. Both `wo` and `w_down` see modest gains
since they're already small.

Numerics: `llmtoy compare` is essentially unchanged vs the pre-Q8_1 baseline
(per-layer rel_err matches within 0.01%; same pre-existing L28 argmax FAIL).
The Q8_1 path does not introduce drift; it does not fix the existing drift
either, because our CPU reference uses f32 dot product (not Q8_1), so the
bit-determinism property that lets llama.cpp Vulkan match its CPU reference
is structurally absent here. Generation output remains coherent.

Verification protocol (all from docs/phases/phase7-gpu-endgame-plan.md):
- Per-shader fuzz test rel < 1e-3 (Q4_K: 2.9e-5, Q3_K: 1.5e-7 model-sized)
- `llmtoy compare` argmax matches L0–L27 (same as baseline)
- End-to-end `generate --gpu --temperature 0` produces correct output

Hyperfine 3-run wall-clock for `generate "explain MoE in two sentences"
--chat --temperature 0 --max-tokens 64 --gpu` (includes ~5.8 s model
setup):

| Stage                                          | mean ± σ          |
|------------------------------------------------|-------------------|
| Pre-Q8_1 (Phase 7f baseline)                   | 32.17 ± 0.71 s    |
| Q8_1 attention + Q8_1 fused MoE gate+up        | 28.86 ± 0.37 s    |
| Q8_1 attention + full Q8_1 MoE (gate+up+down)  | 27.57 ± 0.41 s    |
| Full Q8_1 routing (attn + MoE + dense + wo)    | **25.43 ± 0.15 s**|

Compute-only delta (subtracting the ~5.8 s model load): 26.4 s → 19.6 s,
**+34% throughput**. Single-run tok/s reports 4.28 prefill / 4.07 decode —
matches the wall-clock drop.

## Phase 7j — Submit-fusion via VRAM intermediates (5 → 3 submits/layer)

Reuses the 7j scaffolding (rmsnorm + elem_add + gelu_mul + dense FFN VRAM
buffers + attn_in_buf/attn_vram) to collapse per-layer submits:

| Step | Per-layer submits | Description |
|------|---|-----------|
| `a6ce98c` (7k\* baseline)              | 5 | QKV, wo, gate+up, w_down, experts |
| `runLayerAttnQ8_1` + `runLayerFfnGateUpQ8_1` | 5 | Same submit count — only folds the two CPU rmsnorms into existing submits |
| `runLayerDenseFfnQ8_1`                       | 4 | Fuses gate+up + GPU GELU + w_down + post_ffw_norm_1 — eliminates the standalone w_down submit |
| `runLayerAttnResidualDenseFfnQ8_1`           | 3 | Also folds wo + post_attn_norm + residual into the same submit |

Hyperfine 5-run wall-clock for `generate "Write three lines about the
history of Rome." --chat --temperature 0 --max-tokens 64 --gpu` (includes
~5.7 s model setup):

| Build                                         | mean ± σ          |
|-----------------------------------------------|-------------------|
| `a6ce98c` baseline (5 submits/layer)          | 28.603 ± 0.125 s  |
| Fused dense FFN (4 submits/layer)             | 28.394 ± 0.191 s  |
| Full fused attn-residual + dense FFN (3 sub.) | 28.445 ± 0.178 s  |

## Phase 7m/Q6_K — Timestamp profiling exposes lm_head fallback

Prompt: `"explain Mixture of Experts in one sentence"` with `--chat
--temperature 0 --max-tokens 8 --threads 12 --gpu`.

Before Q6_K GPU support, `LLMTOY_GPU_PROFILE=1` showed only about 307 ms of
timestamped GPU dispatches across 28 forwarded tokens while wall-clock compute
was about 6.2 s. CPU `perf` showed the missing time: roughly 67% of samples in
`quant.dequant.dequantQ6K`. That made the first blocker a CPU fallback, not
one of the already-profiled Q3/Q4/Q5 GPU shaders.

After adding `matvec_q6_k_q8_1.glsl` and routing Q6_K through the Q8_1 path:

| Metric | Before | After |
|--------|--------|-------|
| Prefill | ~4.50 tok/s | 10.14 tok/s |
| Decode | ~4.52 tok/s | 10.16 tok/s |
| GPU upload/setup | ~5.8-6.2 s | ~6.0-6.3 s |

Hot timestamp rows after the fix:

| Label | Count | Total ms | Avg us | Share |
|-------|------:|---------:|-------:|------:|
| `dense_ffn.down` | 840 | 86.402 | 102.86 | 26.3% |
| `matvec_q8_1.single.262144x2816` | 28 | 47.367 | 1691.69 | 14.4% |
| `moe.fused_gate_up` | 4480 | 29.621 | 6.61 | 9.0% |
| `attention.qk_softmax` | 644 | 25.136 | 39.03 | 7.7% |
| `moe.down` | 4480 | 16.724 | 3.73 | 5.1% |

CPU `perf` after the fix no longer shows Q6_K. The remaining CPU-side samples
are mostly `ops.math.dequantRow` with Q3_K/Q5_K frames and
`quant.dequant.dotIQ4NL`, which should be mapped to exact tensors before adding
more shader formats.

The submit-count reductions buy ~150–300 µs/token in saved overhead. On
RX 7800 XT with our shaders the absolute per-submit cost (~150 µs) is
small relative to the 220 ms of GPU matmul work per token, so each saved
submit gives ~0.07% wall-clock — not the 1–2% the original plan estimated.
**Major remaining lever is matmul throughput**, which is Phase 7l's
target (attention compute on GPU, fp16 coopmat, persistent command
buffers).

All 30 layer argmaxes still match CPU; final lm_head argmax still
matches CPU on the "explain MoE" prompt. Per-layer rel\_err is comparable
to baseline (layer 28 worst case 4.83% vs baseline 5.59%).

## Phase 7l.1 — KV cache in VRAM + GPU per-head norms + GPU RoPE

Moves the entire attention front-end into the existing Q8_1 attention submit:
per-head Q/K rmsnorm + V rmsnormRaw + RoPE on Q,K + vkCmdCopyBuffer K/V into
per-layer device-local `k_vram[l]/v_vram[l]` cache. Per-token CPU work drops
by 30 layers × (n_heads + 2·n_kv) per-head rmsnorm + rope loops.

Path used: `runLayerAttnQ8_1KvVram` for the 24 SWA layers (which have `wv`);
the 6 global layers (which share V from K) fall through to `runLayerAttnQ8_1`
+ CPU per-head norms/rope.

Hyperfine 3-run wall-clock for `generate "explain Mixture of Experts in 64
words" --chat --temperature 0 --max-tokens 64 --gpu` (includes ~5.7 s model
setup):

| Build                                        | mean ± σ          |
|----------------------------------------------|-------------------|
| `b234300` 7l.1 KV-VRAM + GPU norms + RoPE    | 28.121 ± 0.126 s  |
| 7j baseline (3 submits/layer)                | 28.445 ± 0.178 s  |

That's −1.1% wall-clock — within 1σ of each other. As predicted by the
plan, 7l.1 alone is mostly a structural change: the CPU savings (~5 µs/layer
of per-head norm + rope work) are roughly cancelled by the additional GPU
dispatches per submit (5 extra: Q-norm, K-norm, V-norm, RoPE-Q, RoPE-K).
The VRAM-resident K/V cache is dead storage until 7l.2 (GPU Q·K^T+softmax)
and 7l.3 (GPU attn·V) wire it into the attention compute.

Verified via `llmtoy compare`: per-layer rel_err identical to 7k* baseline
(layer 28 worst-case 4.83%), all 30 argmaxes match CPU, final argmax matches
(1852 = 1852). End-to-end generate produces identical output to the
`runLayerAttnQ8_1` path at T=0.

VRAM cost: +560 MiB for the KV cache (30 layers × 2 × cap × nkv × 4 bytes,
with cap=512 for SWA and cap=4096 for global).

## Phase 7l.2/3 — GPU attention (Q·K^T + softmax + attn·V)

Replaces the per-head CPU `sdpAttn` loop with two GPU shaders dispatched
back-to-back in one submit, reading K/V directly from the 7l.1 VRAM cache.

**Shaders** (both at `src/gpu/shaders/`):
- `attn_qk_softmax.glsl`: one workgroup per Q head; 256 threads parallel
  across `win_len` positions; each thread sequentially computes the full
  head_dim dot product. 3-phase softmax (max → exp+sum → normalize) via
  subgroupMax/subgroupAdd + shared-memory cross-subgroup reduction.
  Circular-buffer slot indexing via `(seq - win_len + i) % cap` — SWA
  mask is implicit in `win_len`.
- `attn_av.glsl`: one workgroup per Q head; 256 threads parallel across
  head_dim output dimension; each thread accumulates one output element
  over all `win_len` positions. Coalesced reads of `v_cache[slot, kv_h, dd]`.

Five fuzz tests cover SWA (n_heads=16, n_kv=4, hd=256: win=1, 128, 1024)
and global (n_heads=4, n_kv=4, hd=512: win=64, 512) shapes. All pass at
rel < 2e-6 vs CPU `sdpAttn` reference.

Active on Gemma4's 24 SWA layers (which have `wv`); the 6 global layers
(share V from K) still use the CPU `sdpAttn` path. Attention output is
written into the existing `attn_in_buf` (which is `wo`'s input), and the
full-fused wo+dense-FFN submit skips its own upload — eliminating a PCIe
round-trip per layer.

Hyperfine 3-run wall-clock for `generate "explain Mixture of Experts in
64 words" --chat --temperature 0 --max-tokens 64 --gpu`:

| Build                                              | mean ± σ          |
|----------------------------------------------------|-------------------|
| `d04bfa7` 7l.2/3 + skip-download polish            | 27.355 ± 0.277 s  |
| `fbc6a91` 7l.2/3 (with attn_concat download)       | 27.508 ± 0.213 s  |
| `b234300` 7l.1c baseline                           | 28.121 ± 0.126 s  |
| `441b17e` 7j baseline (3 submits/layer)            | 28.445 ± 0.178 s  |

Cumulative 7l: **−3.8%** wall-clock vs 7j. Per-token: ~9 ms saved
(334.8 → 325.7 ms/token).

The win is smaller than the 30–50% the original plan estimated. Why:
- At typical decode `win_len` (≤ 84 in this bench), CPU `sdpAttn` was
  ~50 µs/layer × 30 = ~1.5 ms/token. Replacing it with GPU compute
  recovers ~1 ms/token (GPU isn't free either).
- 6 global layers still run CPU `sdpAttn` + per-head norms + RoPE.
- The dominant cost is **GPU matmul throughput** (Q3_K/Q4_K integer-dot
  on this hardware/driver) — the 220 ms/token of matmul compute is
  unchanged by 7l. Closing the remaining gap to llama.cpp needs better
  matmul shaders (subgroup matvec, fp16 coopmat) — Phase 7m+.

The structural value of 7l: K/V cache and attention all live in VRAM,
the activation no longer round-trips between CPU/GPU per matmul, and
follow-on optimisations (one-submit per layer, persistent command
buffers, V-from-K shader for global layers) all become possible.

Verified via `llmtoy compare`: all 30 layer argmaxes match CPU,
per-layer rel_err identical to 7l.1 baseline, final argmax matches
(1852 = 1852).

## Phase 7g — Fused dense FFN (experiment, reverted)

Attempted fusing gate-gelu-up + w_down into a single submit (4 submits/layer):
one dispatch computes `gelu(gate@x) * (up@x)` into a device-local mid_buf,
then a barrier + w_down dispatch reads mid_buf and writes the final output.
Dense FFN gate/up are Q3_K (same shader as experts).

| Mode | tok/s prefill | tok/s decode |
|------|--------------|--------------|
| Phase 7f (batched gate+up, separate down) | 3.10–3.17 | 3.06–3.09 |
| Fused gate-gelu-up + down (1 submit) | 3.07–3.08 | 2.96–3.00 |

The fused shader reads **two** Q3_K matrices (gate + up) per thread, doubling
register pressure and reducing GPU wavefront occupancy versus running two
separate 2112-row dispatches. The 1-submit saving (~0.1 ms/layer overhead) is
smaller than the per-element compute regression, so the net result is slower.

Contrast with the expert fused shader which _does_ help: expert matrices are
smaller (704 rows vs 2112) and 8 are dispatched per batch, giving more total
parallelism with less register pressure per wavefront.

**Reverted**: kept 5 submits/layer (wq+wk+wv=1, wo=1, gate+up=1, w_down=1,
experts=1). The fused Q3_K dense FFN shader is kept in
`src/gpu/shaders/matvec_fused_gu_q4k.glsl` for reference.

## Notes

- All benchmarks use `zig build` (defaults to ReleaseFast since phase 7d)
- `-Doptimize=Debug` required to build debug binaries
- systemd-run --scope -p MemoryMax=40G required for safe testing with GPU
- Stop token auto-detected from vocab (Gemma4 APEX: token 106 `<turn|>`)

## Bugs fixed during GPU bringup

1. `tensorBytes()` returned `mmap[offset..]` (full file tail) instead of
   `mmap[offset..offset+size]` — caused 13 GB staging allocations per tensor → OOM.
   Fixed by adding `TensorInfo.byteSize()` with correct GGUF block-size tables.

2. Double-free of `last_logits` when EOS token sampled in generation loop.
   Fixed by setting `last_logits = &.{}` on EOS break and guarding the outer free.
