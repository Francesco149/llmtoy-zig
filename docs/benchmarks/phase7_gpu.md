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

Coverage today: all 30 attention layers (wq+wk+wv, mixed Q3_K/Q4_K) + all MoE
expert matmuls (fused Q3_K gate+up AND Q5_0/Q5_1 down). Still on the
f32-activation path: wo (Q4_K, cols=4096) and dense FFN (Q3_K gate/up + Q5_1
down).

| Mode | tok/s prefill | tok/s decode |
|------|--------------|--------------|
| Phase 7f baseline (no Q8_1) | 3.10–3.17 | 3.06–3.09 |
| Q8_1 attention only (6 of 30 layers, Q4_K only) | ~3.11 | ~3.08 |
| Q8_1 attention all 30 layers (Q3_K + Q4_K) | ~3.21 | ~3.15 |
| Q8_1 attention + Q8_1 fused MoE gate+up | 3.71–3.78 | 3.66–3.68 |
| Q8_1 attention + full Q8_1 MoE (gate+up + down) | **~3.86** | **~3.66–3.70** |

Net **~+22% prefill / +20% decode** end-to-end vs Phase 7f baseline. The
biggest contribution is the Q8_1 fused MoE gate+up (+15%); the down-side
Q5_0/Q5_1 Q8_1 path adds another +2–4% on top (it's a smaller matmul but
already had the f32-acts batch-friendly down dispatch eliminating most of
the f32 overhead).

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
| Q8_1 attention + full Q8_1 MoE (gate+up+down)  | **27.57 ± 0.41 s**|

Compute-only delta (subtracting the ~5.8 s model load): 26.4 s → 21.8 s,
**+21% throughput** at the full-Q8_1-MoE stage. Single-run tok/s reports
3.86 prefill / 3.66–3.70 decode — matches the wall-clock drop.

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
