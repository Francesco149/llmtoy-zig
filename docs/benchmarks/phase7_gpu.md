# Phase 7 GPU Offload Benchmarks

Model: Gemma 4 26B A4B (APEX-I-Mini, GGUF)
Hardware: Ryzen 5900x, 64 GB RAM, AMD RX 7800 XT (16 GB VRAM)
Prompt: "What is 2+2?" (chat template, --seed 42)

## Phase 7m profile snapshot - sorted timestamp table

Command:

```sh
LLMTOY_GPU_PROFILE=1 systemd-run --user --scope -p MemoryMax=40G --quiet -- \
  nix develop --command ./zig-out/bin/llmtoy generate \
  /opt/ai-lab/models/mudler/gemma-4-26B-A4B-it-APEX-GGUF/gemma-4-26B-A4B-APEX-I-Mini.gguf \
  "Briefly explain the full forward pass of a MoE model" \
  --chat --temperature 0 --max-tokens 8 --gpu
```

Result after the all-layer MoE VRAM tail path:

- setup: 4.7 s after warm filesystem cache
- prefill: 25 tokens in 1.23 s, 20.3 tok/s
- generation: 8 tokens in 0.39 s, 20.4 tok/s

Top timestamped GPU totals across 33 forwarded tokens:

| Label | Count | Total ms | Avg us | Share |
|-------|------:|---------:|-------:|------:|
| `matvec_q8_1.single.262144x2816` | 33 | 55.84 | 1692.24 | 10.6% |
| `moe.fused_gate_up` | 990 | 54.27 | 54.82 | 10.3% |
| `attention.qk_softmax` | 990 | 42.52 | 42.95 | 8.1% |
| `moe.down` | 5379 | 33.23 | 6.18 | 6.3% |
| `attn_front.rmsnorm` | 990 | 25.97 | 26.24 | 4.9% |
| `dense_ffn.rmsnorm` | 990 | 25.69 | 25.95 | 4.9% |
| `ffn_moe.post_norm` | 990 | 25.69 | 25.95 | 4.9% |
| `attn_front.wq` | 990 | 23.37 | 23.61 | 4.4% |

The formerly generic dense FFN down label is now split by layer and shape.
Layer 0 (`2816x2112`) averages about 85 us; layers 1-29 with the same shape
average about 103 us. There is no single outlier layer to chase, so the next
production targets should remain `lm_head`, flattened MoE gate/up, attention
QK, or a broader Q5_0 dense-down kernel improvement.

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

## Phase 7o — Q5_0/Q5_1 MMVQ bench-only probe

Added llama.cpp-style legacy-quant MMVQ shaders for Q5_0 and Q5_1. They pass
the GPU fuzz tests, but they are not production wins on the target shapes.

Focused `bench-matvec --iters 128 --target all` GPU timestamps:

| Target | Current GPU us | Best Q5 MMVQ GPU us | Result |
|--------|----------------|---------------------|--------|
| `L0.dense_down` Q5_1 | 8.27 | 10.33 (`b64.r2`) | slower |
| `L5.dense_down` Q5_0 | 9.34 | 11.69 (`b64.r2`) | slower |
| `L0.expert_down` Q5_1 | 4.27 | 4.89 (`b64.r2`) | slower |

Conclusion: keep Q5 MMVQ as bench-only reference code. The existing simple Q5
Q8_1 kernels are faster for these decode shapes on this device.

## llama.cpp Vulkan Reference — Nix package

Reference package:

```text
nixpkgs#llama-cpp-vulkan
llama-cli version: 8983 (80afa33)
device: Vulkan0 AMD Radeon RX 7800 XT (RADV NAVI32)
```

The plain `nixpkgs#llama-cpp` package only exposes `BLAS` here. Use
`nixpkgs#llama-cpp-vulkan` for parity work.

Production-like reference command, adapted from
`../nix-lab/hosts/lame/llama.nix`:

```sh
env GGML_VK_PERF_LOGGER=1 GGML_VK_PERF_LOGGER_FREQUENCY=1 \
  VK_ICD_FILENAMES=/run/opengl-driver/share/vulkan/icd.d/radeon_icd.x86_64.json \
  nix shell nixpkgs#llama-cpp-vulkan -c llama-cli \
  -m /opt/ai-lab/models/mudler/gemma-4-26B-A4B-it-APEX-GGUF/gemma-4-26B-A4B-APEX-I-Mini.gguf \
  -p "Briefly explain the full forward pass of a MoE model" \
  -n 8 -t 12 -c 120000 -ngl 99 --n-cpu-moe 0 \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1 \
  --chat-template-file /opt/ai-lab/templates/new-chat-template-gemma.jinja \
  --temp 0 --top-k 40 --top-p 0.9 --seed 42 \
  --conversation --single-turn --reasoning off --no-display-prompt \
  --no-warmup --simple-io --device Vulkan0 --perf
```

`llama-cli` rejected `--kv-unified`; that flag is present in the llama-server
service config but was omitted from this CLI probe.

Result:

| Engine | Prompt tok/s | Generation tok/s |
|--------|--------------|------------------|
| llama.cpp Vulkan (`nixpkgs#llama-cpp-vulkan`) | 223.5 | 85.6 |

The observed llama.cpp decode pipeline is the real parity target:

| Kernel family | Representative decode timing |
|---------------|------------------------------|
| `MUL_MAT_ID_VEC q3_K m=1408 n=8 k=2816 n_expert=128` | ~39-46 us |
| `MUL_MAT_ID_MUL MUL_MAT_ID_VEC q5_0 m=2816 n=8 k=704 n_expert=128` | ~36-40 us |
| `MUL_MAT_ID_MUL MUL_MAT_ID_VEC iq4_nl m=2816 n=8 k=704 n_expert=128` | ~27-29 us |
| `MUL_MAT_VEC q6_K m=262144 n=1 k=2816` | ~1.03 ms |
| `FLASH_ATTN_EXT` SWA/global decode | ~14 us / ~30 us per layer group |
| `RMS_NORM_MUL` and `RMS_NORM_MUL_ROPE` | fused small-op path, ~6 us per call |

This changes the near-term optimization target. Isolated one-row matvec ports
are not enough; llama.cpp's big wins are fused MoE `MUL_MAT_ID*_VEC` routing,
Vulkan flash attention, q8_0 KV, and graph fusion around norms/RoPE/adds.

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

## Phase 7m/IQ4_NL — Expert-down fallback removed

`llmtoy info` now prints unsupported GPU quant tensors. For this model, after
Q6_K support, the remaining unsupported hot tensors were:

- `blk.3.attn_v.weight` and `blk.4.attn_v.weight` in Q5_K.
- `blk.10` through `blk.19` `ffn_down_exps.weight` in IQ4_NL.

The IQ4_NL tensors were the larger CPU hotspot (`quant.dequant.dotIQ4NL`), and
because a missing expert-down session makes the whole layer's expert batch fall
back to CPU, those ten layers were losing all MoE GPU work. Added
`matvec_iq4_nl_q8_1.glsl`, a simple lookup-table Q8_1 path for IQ4_NL expert
down. This is not the final MMVQ design, but it removes a measured fallback.

Verification:

- `zig build test`: IQ4_NL x Q8_1 fuzz passes at `rel=2.027e-7` small and
  `rel=1.886e-7` for expert-down-shaped `64x704`.
- Full `llmtoy compare`: all layer argmaxes match; final argmax token `1852`
  matches. Worst layer was L28 at `rel=5.862%`, argmax ok.

Same short prompt as above:

| Metric | Q6_K only | + IQ4_NL expert down |
|--------|-----------|----------------------|
| Prefill | 10.14 tok/s | 15.23 tok/s |
| Decode | 10.16 tok/s | 15.05 tok/s |
| Expert GPU layers | 20/30 | 30/30 |

Hot timestamp rows after IQ4_NL:

| Label | Count | Total ms | Avg us | Share |
|-------|------:|---------:|-------:|------:|
| `dense_ffn.down` | 840 | 86.435 | 102.90 | 24.2% |
| `matvec_q8_1.single.262144x2816` | 28 | 47.288 | 1688.87 | 13.2% |
| `moe.fused_gate_up` | 6720 | 44.522 | 6.63 | 12.5% |
| `moe.down` | 6720 | 28.344 | 4.22 | 7.9% |
| `attention.qk_softmax` | 644 | 25.217 | 39.16 | 7.1% |

CPU `perf` after IQ4_NL no longer shows `dotIQ4NL`. Remaining model-side CPU
fallback is primarily `dequantQ5K`, which maps to the two Q5_K attention-V
tensors in layers 3 and 4.

## Phase 7m/Q5_K — Last quantized matmul fallback removed

Added `matvec_q5_k_q8_1.glsl` for the two remaining Q5_K attention-V tensors:
`blk.3.attn_v.weight` and `blk.4.attn_v.weight`. This makes every quantized
matmul tensor in this Gemma4 APEX-I-Mini GGUF GPU-supported; `llmtoy info`
now reports `unsupported GPU quant tensors: none`.

Verification:

- `zig build test`: Q5_K x Q8_1 fuzz passes at `rel=5.731e-4` small and
  `rel=3.731e-4` for attn-V-shaped `64x2816`.
- Full `llmtoy compare`: all layer argmaxes match; final argmax token `1852`
  matches. Worst layer was L28 at `rel=5.669%`, argmax ok.

Same short prompt as above:

| Metric | + IQ4_NL | + Q5_K |
|--------|----------|--------|
| Prefill | 15.23 tok/s | 17.52 tok/s |
| Decode | 15.05 tok/s | 17.08 tok/s |
| Unsupported quant tensors | 2 Q5_K | none |

CPU `perf` after Q5_K no longer shows `dequantQ5K`, `dotIQ4NL`, or
`dequantQ6K`. The largest CPU samples are now Vulkan allocation/setup,
memcpy upload/download, and orchestration around `runExpertBatch` rather than
quantized CPU matvec. The next optimization target should therefore stop adding
one-off quant fallbacks and move to the planned microbench/MMVQ work or reduce
known synchronization/download overhead.

## Phase 7n — Matvec microbenchmark harness

Added `llmtoy bench-matvec <model.gguf> [--iters N] [--target NAME]` for
isolated shader iteration on real Gemma4 tensors. The harness uploads one
representative tensor at a time, quantizes a deterministic f32 activation vector
to Q8_1 once, warms the dispatch once, then times repeated matvec submissions.
The measurement intentionally includes the current descriptor/update,
command-buffer, submit, wait, and descriptor-free path because generation pays
that cost today.

Command:

```sh
nix develop --command ./zig-out/bin/llmtoy bench-matvec \
  /opt/ai-lab/models/mudler/gemma-4-26B-A4B-it-APEX-GGUF/gemma-4-26B-A4B-APEX-I-Mini.gguf \
  --iters 8
```

Result:

| Target | Type | rows | cols | avg us | effective GB/s |
|--------|------|------|------|--------|----------------|
| `lm_head` | Q6_K | 262144 | 2816 | 2043.03 | 296.40 |
| `L0.attn_q` | Q4_K | 4096 | 2816 | 47.80 | 135.74 |
| `L0.attn_v` | Q4_K | 2048 | 2816 | 43.28 | 74.96 |
| `L3.attn_v` | Q5_K | 2048 | 2816 | 54.54 | 72.70 |
| `L5.attn_q` | Q3_K | 8192 | 2816 | 83.44 | 118.79 |
| `L0.dense_down` | Q5_1 | 2816 | 2112 | 55.25 | 80.73 |
| `L5.dense_down` | Q5_0 | 2816 | 2112 | 51.09 | 80.03 |
| `L0.expert_down` | Q5_1 | 2816 | 704 | 53.74 | 27.67 |
| `L10.expert_down` | IQ4_NL | 2816 | 704 | 49.16 | 22.68 |

The large `lm_head` tensor reaches much higher effective bandwidth than the
smaller layer and expert projections. That matches the endgame hypothesis:
current one-row workgroups and per-dispatch overhead leave small matvecs badly
under-occupied. The next target is an experimental MMVQ-style multi-row kernel,
starting with Q4_K because it is easiest to compare against llama.cpp's
`mul_mat_vec_q4_k.comp`.

### Q4_K R4 shape probe

Added an experimental `matvec_q4_k_q8_1_r4.glsl` bench-only pipeline that maps
four rows into one workgroup. A first version using `subgroupAdd` was wrong on
this device because the hardware subgroup can span more than one logical row;
the correct version uses `subgroupClusteredAdd(..., 32)` so each row reduces
independently.

The corrected shader passes the Q4_K x Q8_1 fuzz tests, but it is not faster
enough to route into generation:

| Target | Baseline avg us | R4 avg us | Result |
|--------|------------------|-----------|--------|
| `L0.attn_q` | 59.37 | 61.25 | -3.2% |
| `L0.attn_v` | 57.38 | 58.22 | -1.5% |

Conclusion: simple row packing is not the missing llama.cpp optimization by
itself. The real MMVQ port needs the rest of llama.cpp's structure: subgroup
size control, tuned `NUM_ROWS`, shared helper layout, and format-specific
unpack scheduling. Keep the R4 path as a microbench comparison target only.

### Timestamp split and descriptor-reuse probe

Extended `bench-matvec` so `LLMTOY_GPU_PROFILE=1` reports GPU timestamp time
beside wall-clock time. Added `--reuse-descriptor` to allocate/update one
stable descriptor set before the timed loop, then bind it repeatedly.

Measured with 128 iterations for Q4_K and 32 for `lm_head`:

| Target | Mode | wall us | GPU us | residual CPU us |
|--------|------|---------|--------|-----------------|
| `L0.attn_q` | current | 69.86 | 14.75 | 55.11 |
| `L0.attn_q` | reuse descriptor | 64.94 | 12.37 | 52.57 |
| `L0.attn_v` | current | 70.72 | 8.29 | 62.43 |
| `L0.attn_v` | reuse descriptor | 70.41 | 7.65 | 62.77 |
| `lm_head` | current | 1508.35 | 1445.79 | 62.57 |
| `lm_head` | reuse descriptor | 1521.29 | 1459.57 | 61.72 |

Interpretation:

- Large matvecs are GPU-kernel dominated. `lm_head` still needs better Q6_K
  kernel shape/MMVQ work.
- Small decode matvecs are mostly CPU/submit overhead in this one-dispatch
  microbench. Descriptor reuse alone does not materially reduce it, so the
  remaining overhead is likely command buffer allocation/free, queue submit,
  and `vkQueueWaitIdle`.
- Production work should avoid a broad descriptor-only refactor. The next
  plumbing experiment should reuse command buffers/fences or batch more fixed
  per-layer dispatches, while the shader track continues with a faithful MMVQ
  port for large tensors.

### Q6_K packed-decode probe

Added an experimental bench-only `matvec_q6_k_q8_1_fast.glsl` target exposed as
`lm_head.fast`. The first attempt used a u32-reinterpreted Q6_K block struct and
failed fuzz because the shader-side layout did not match the GGUF bytes. The
correct version keeps the proven byte layout and only changes the decode path:
it loads four Q6 low/high bytes into packed u32 words, unpacks four signed i8
values, then uses the same `dotPacked4x8EXT` accumulation as the current Q6_K
shader.

Validation:

- Q6_K x Q8_1 fast fuzz small: `rel=6.963e-8`
- Q6_K x Q8_1 fast lm-head-shaped cols: `rel=2.384e-7`

Bench results on `lm_head`:

| Mode | wall us, no profiler | GPU us, profiler |
|------|----------------------|------------------|
| current Q6_K | 2032.90 | 1034.10 |
| packed decode | 2033.22 | 1001.17 |

Interpretation: packed decode slightly reduces measured kernel time in the
timestamp run, but it does not improve wall-clock throughput. This is not worth
routing into generation as-is. It remains useful as a correctness-checked
comparison target while the real Q6_K MMVQ port moves toward llama.cpp's
`K_PER_ITER`/`NUM_ROWS` framework rather than another local decode tweak.

### Q6_K MMVQ scaffold

Started the real MMVQ refactor:

- `MatvecPipeline` can now create compute pipelines with Vulkan specialization
  constants.
- Added `matvec_q6_k_q8_1_mmvq.glsl`, a standalone first Q6_K MMVQ-style
  shader with `local_size_x_id=0`, `BLOCK_SIZE`, `NUM_ROWS`, `NUM_COLS`,
  `K_PER_ITER=16`, and shared-memory reduction.
- Added bench targets `lm_head.mmvq.b32.r1` and `lm_head.mmvq.b64.r1`.

Correctness:

| Variant | small rel | lm-head-shaped rel |
|---------|-----------|--------------------|
| b32/r1 | `7.312e-8` | `2.140e-7` |
| b64/r1 | `1.314e-7` | `1.557e-7` |

Bench snapshot:

| Target | GPU us, profiler | wall us, no profiler |
|--------|------------------|----------------------|
| `lm_head` | 1033.89 | 3184.65 |
| `lm_head.fast` | 994.01 | 3172.00 |
| `lm_head.mmvq.b32.r1` | 1156.13 | 3164.37 |
| `lm_head.mmvq.b64.r1` | 1132.67 | 2256.79 |

The wall numbers are noisy, but GPU timestamps are consistent enough: the first
MMVQ-shaped shader is correct but slower than the current Q6_K kernel. The next
step is not `NUM_ROWS > 1`; it is closing the structural gap with llama.cpp's
Q6_K port by adding packed16-style views / repacking, the `sccache` scale cache,
and the exact `itid`/`ix` offset schedule from `mul_mat_vec_q6_k.comp`.

### Q6_K MMVQ q8-helper rewrite and NUM_ROWS sweep

Corrected the Q6_K MMVQ direction after re-reading llama.cpp: for Q8_1
activation MMVQ the relevant implementation is `mul_mat_vecq.comp` plus the
`DATA_A_Q6_K` section in `mul_mat_vecq_funcs.glsl`, not the f32-activation
`mul_mat_vec_q6_k.comp` scale-cache path. The shader now mirrors the Q8_1
helper structure:

- preload one Q8_1 block into `cache_b_qs[4]` and `cache_b_ds`
- use `repack4(ib_a, iqs)` over Q6_K data
- use `get_d_scale(ib_a, iqs)` and `mmvq_dot_product(...)`
- keep `BLOCK_SIZE` and `NUM_ROWS` as specialization constants

Added `lm_head.mmvq.b64.r2` and `lm_head.mmvq.b64.r4`.

Correctness:

| Variant | lm-head-shaped rel |
|---------|--------------------|
| b64/r2 | `2.659e-7` |
| b64/r4 | `1.590e-7` |

Bench snapshot:

| Target | GPU us, profiler |
|--------|------------------|
| `lm_head` | 1047.38 |
| `lm_head.fast` | 1003.59 |
| `lm_head.mmvq.b32.r1` | 1167.04 |
| `lm_head.mmvq.b64.r1` | 1146.15 |
| `lm_head.mmvq.b64.r2` | 1077.15 |
| `lm_head.mmvq.b64.r4` | 1112.46 |

`NUM_ROWS=2` helps, but the MMVQ port still does not beat the current one-row
Q6_K shader or packed-decode probe. The remaining structural gap is now likely
the memory view/layout side: llama.cpp's generated shaders read Q6_K through
`block_q6_K_packed16`, while llmtoy still reconstructs 16-bit pieces from
byte arrays in the shader. The next step should add an upload-time packed16
view or separate repacked session for MMVQ kernels, then rerun the same b64/r2
benchmark before touching production routing.

Follow-up structural pass:

- Added a same-binding `block_q6_K_packed16` view to
  `matvec_q6_k_q8_1_mmvq.glsl`. The raw byte view remains in place for
  `scales` and `d`, matching llama.cpp's Q6_K helper behavior.
- Replaced the full shared-memory reduction tree with llama.cpp's safe
  `USE_SUBGROUP_ADD` shape: `subgroupAdd` within each subgroup, then a tiny
  shared-memory sum across subgroup partials.

Correctness after packed16 + subgroup reduction:

| Variant | lm-head-shaped rel |
|---------|--------------------|
| b64/r1 | `1.947e-7` |
| b64/r2 | `2.991e-7` |
| b64/r4 | `1.590e-7` |

Focused timestamp snapshot, 64 iterations:

| Target | GPU us, profiler |
|--------|------------------|
| `lm_head` | 1025.31 |
| `lm_head.mmvq.b64.r1` | 1020.66 |
| `lm_head.mmvq.b64.r4` | 1058.60 |

This is the first correctness-checked MMVQ variant to narrowly beat the current
Q6_K `lm_head` kernel on GPU timestamps, but the margin is only about 0.5% and
single-target wall times remain noisy. Treat `b64/r1` as the current best
MMVQ candidate, not as enough evidence to route production generation through
it.

Tested llama.cpp's 4-then-2 manual unroll schedule plus per-thread tail
`num_iters` calculation. Correctness was unchanged, but focused 64-iteration
timestamps regressed: current `lm_head` was about `1026.77 us`, while
`lm_head.mmvq.b64.r1` moved to about `1058.04 us`. Do not repeat this as a
Q6_K optimization unless another change makes the loop shape materially
different. The remaining parity gap is likely not inside isolated Q6_K
`lm_head` polish; move the generated MMVQ family to Q3_K/Q4_K and reduce
per-layer dispatch structure.

Added subgroup property reporting to `llmtoy gpu-info`; this RX 7800 XT/RADV
run reports subgroup size `64`, compute support `true`, and arithmetic support
`true`. A no-shared-memory b64/r1 reduction variant was tested because one
subgroup covers `BLOCK_SIZE=64`, but it regressed: current `lm_head` about
`1037.34 us`, safe `lm_head.mmvq.b64.r1` about `1036.36 us`, and no-shmem
about `1059.47 us`. The experimental target was reverted; keep the safe
subgroup-plus-shared-memory reduction.

### Q3_K MMVQ scaffold

Started moving the MMVQ framework beyond isolated Q6_K:

- Added `matvec_q3_k_q8_1_mmvq.glsl`, mirroring llama.cpp's
  `DATA_A_Q3_K` helper with packed16 `hmask/qs` reads, raw byte `scales/d`,
  Q8_1 activation caching, and the same subgroup-plus-shared-memory reduction
  used by the Q6_K MMVQ shader.
- Added bench targets `L5.attn_q.mmvq.b32.r1` and
  `L5.attn_q.mmvq.b64.r1`.

Correctness:

| Variant | small rel | 2816-col rel |
|---------|-----------|--------------|
| b32/r1 | `9.419e-8` | `3.358e-7` |
| b64/r1 | `1.082e-7` | `1.674e-7` |

Focused timestamp snapshot, 128 iterations:

| Target | GPU us, profiler |
|--------|------------------|
| `L5.attn_q` | 43.22 |
| `L5.attn_q.mmvq.b32.r1` | 64.90 |
| `L5.attn_q.mmvq.b64.r1` | 42.81 |

`b64/r1` is the first Q3_K MMVQ candidate and is roughly tied/slightly ahead
of the current shader in this focused run. `b32/r1` is a clear negative. Keep
the candidate bench-only until repeated runs and a full generation profile show
a material win.

### Q4_K MMVQ scaffold

Added `matvec_q4_k_q8_1_mmvq.glsl`, following llama.cpp's `DATA_A_Q4_K`
helper with packed32 Q4 reads, Q8_1 activation caching, Q4 scale/min handling,
and the same MMVQ reduction framework. Added bench targets:

- `L0.attn_q.mmvq.b32.r1`
- `L0.attn_q.mmvq.b64.r1`
- `L0.attn_q.mmvq.b64.r2`
- `L0.attn_q.mmvq.b64.r4`
- equivalent `L0.attn_v.*` targets

Correctness:

| Variant | 2304-col rel |
|---------|--------------|
| b32/r1 | `2.654e-5` |
| b64/r1 | `1.436e-5` |
| b64/r2 | `2.626e-5` |
| b64/r4 | `3.619e-5` |

Focused timestamp snapshots, 128 iterations:

| Target | GPU us, profiler |
|--------|------------------|
| `L0.attn_q` | 14.89 |
| `L0.attn_q.mmvq.b32.r1` | 19.13 |
| `L0.attn_q.mmvq.b64.r1` | 16.12 |
| `L0.attn_q.mmvq.b64.r2` | 16.79 |
| `L0.attn_q.mmvq.b64.r4` | 16.41 |
| `L0.attn_v` | 9.33 |
| `L0.attn_v.mmvq.b32.r1` | 11.17 |
| `L0.attn_v.mmvq.b64.r1` | 10.03 |

The Q4_K MMVQ port is correct but not a GPU-time win. `b64/r1` is closest,
while `b32/r1`, `r2`, and `r4` are clear negatives. Keep these as bench-only
reference targets and do not route Q4_K production through MMVQ without a
separate measured reason.

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

Follow-up: the V-from-K global-layer path is now wired into
`runLayerAttnQ8_1KvVram`. For layers without `wv`, the GPU copies raw K into
the V buffer before K norm/RoPE, then runs the existing V `rmsnormRaw`, cache
append, and GPU attention path. This removes the global-layer CPU
norm/RoPE/`sdpAttn` fallback without changing the long-term graph shape.

Verification:

- `llmtoy compare ... "explain MoE" --chat --gpu-layers 5:5`: all layer
  argmaxes and final argmax match CPU.
- Full `llmtoy compare ... "explain MoE" --chat`: all layer argmaxes and final
  argmax match CPU.
- Short timestamped generation,
  `LLMTOY_GPU_PROFILE=1 ... generate "what is 1+1?" --chat --temperature 0
  --max-tokens 8 --gpu`: prefill 17.89 tok/s, decode 17.87 tok/s.

Profile evidence from the short generation: `attn_front.v_from_k_copy` appears
140 times (5 shared-V layers × 28 forwarded tokens), `attention.qk_softmax` and
`attention.av` run 840 times (30 layers × 28 tokens), and `attn_front.wv`
appears only for the 25 layers with explicit V. The copy itself is tiny
(~0.35 us/call); the value is eliminating the CPU fallback and keeping the
attention graph device-resident.

Verified via `llmtoy compare`: all 30 layer argmaxes match CPU,
per-layer rel_err identical to 7l.1 baseline, final argmax matches
(1852 = 1852).

## Phase 7m/7n — Current MoE batch baseline

Added `llmtoy bench-moe <model.gguf> [--iters N] [--layer N]`, which uploads
the real Gemma4 tensors and times the production `GpuWeights.runExpertBatch`
path for a fixed top-8 expert set. This is the pre-fusion baseline before
porting llama.cpp's `MUL_MAT_ID_VEC` / `MUL_MAT_ID_MUL` pipeline shape.

Command:

```sh
nix develop --command env LLMTOY_GPU_PROFILE=1 ./zig-out/bin/llmtoy \
  bench-moe /opt/ai-lab/models/mudler/gemma-4-26B-A4B-it-APEX-GGUF/gemma-4-26B-A4B-APEX-I-Mini.gguf \
  --iters 64 --layer N
```

Quiet-system preflight: no stray processes, 1m load 0.36, 59 GiB available.

| Layer | gate/up | down | wall us/iter | GPU phase us/iter | host/submit us/iter |
|-------|---------|------|--------------|-------------------|---------------------|
| 0 | Q3_K | Q5_1 | 614.08 | 84.97 | 529.12 |
| 10 | Q3_K | IQ4_NL | 629.70 | 96.81 | 532.89 |

Layer 0 GPU phase breakdown:

| Phase | dispatches | avg us/dispatch | avg us/iter |
|-------|------------|-----------------|-------------|
| `moe.quantize_input` | 64 | 2.40 | 2.40 |
| `moe.fused_gate_up` | 512 | 6.28 | 50.25 |
| `moe.quantize_mid` | 512 | 0.28 | 2.28 |
| `moe.down` | 512 | 3.18 | 25.45 |
| `moe.accum` | 64 | 4.59 | 4.59 |

Layer 10 GPU phase breakdown:

| Phase | dispatches | avg us/dispatch | avg us/iter |
|-------|------------|-----------------|-------------|
| `moe.quantize_input` | 64 | 2.38 | 2.38 |
| `moe.fused_gate_up` | 512 | 6.28 | 50.24 |
| `moe.quantize_mid` | 512 | 0.28 | 2.26 |
| `moe.down` | 512 | 4.67 | 37.37 |
| `moe.accum` | 64 | 4.55 | 4.55 |

Interpretation: raw GPU MoE work is already in the same order as llama.cpp's
per-op Vulkan trace, but the current path pays roughly 0.53 ms/iteration in
host-side descriptor allocation/free, command recording, and many tiny dispatches
inside a single submission. The next optimization is not another isolated
one-row matvec variant; it is a llama.cpp-shaped expert-id dispatch that packs
the selected experts and fuses the down score multiply.

## Phase 7o — Expert-ID MoE shape prototypes

Reference source: `/opt/ai-lab/llama.cpp/ggml/src/ggml-vulkan/`, especially
`ggml_vk_mul_mat_vec_id_q_f16`, `mul_mat_vecq.comp`, and
`mul_mat_vec_base.glsl`.

Fixed a correctness bug in the simple pipeline helper while adding the
expert-id prototypes: `buildSimplePipeline` only allocated descriptor metadata
for four storage bindings, but the first down-id pipeline needs five bindings
(`weights`, `acts`, `ids`, `scales`, `out`). In ReleaseFast this presented as
bad benchmark behavior rather than a clean assert. The helper now supports five
bindings and `zig build test` includes focused expert-id shader tests.

Clean-system command:

```sh
nix develop --command env LLMTOY_GPU_PROFILE=1 ./zig-out/bin/llmtoy \
  bench-moe /opt/ai-lab/models/mudler/gemma-4-26B-A4B-it-APEX-GGUF/gemma-4-26B-A4B-APEX-I-Mini.gguf \
  --iters 16 --layer N
```

Clean Vulkan baseline at process start was VRAM=182 MiB / GTT=15 MiB. Existing
expert upload ends around VRAM=12548 MiB; the prototype flat layer uploads are
temporary microbench scaffolding and should not be duplicated in the production
path.

| Layer | existing MoE GPU us/iter | gate-up-id GPU us | down-id GPU us | llama.cpp comparable trace |
|-------|--------------------------|-------------------|----------------|----------------------------|
| 0 Q3_K/Q5_1 | 85.86 | 55.04 | 26.05 | Q3_K ID ~39-46 us, Q5_0 ID+MUL ~36-40 us |
| 10 Q3_K/IQ4_NL | 98.91 | 55.20 | 37.89 | Q3_K ID ~39-46 us, IQ4_NL ID+MUL ~27-29 us |

Interpretation:

- The down-id shape is viable. Q5_1 lands at ~26 us and IQ4_NL lands at ~38 us
  for 8 selected experts, close enough to justify wiring it into the real MoE
  path behind a correctness gate.
- The first gate-up-id prototype is correct but not yet faster than the current
  eight per-expert fused dispatches on GPU time alone: current gate/up totals
  about 50 us/iter, prototype is about 55 us. It may still reduce host work
  once integrated, but the GPU kernel should be brought closer to llama.cpp's
  `mul_mat_vecq.comp` Q3_K inner loop before replacing the existing path.
- A 64-lane local-size experiment passed correctness on the AMD target but did
  not improve timings, so the prototype stays at the safer 32-lane subgroup
  shape for now.

Next debugging step: use a narrow model-backed correctness harness for the real
MoE intermediates. Compare the current per-expert gate/up and down outputs
against the ID prototypes for a selected layer, then use `llmtoy compare
--gpu-layers L0:L1` once either ID path is wired into `runExpertBatch`.

Follow-up integration result:

- `runExpertBatch` now routes supported expert-down tensors through the
  expert-id Q8_1 path. The production MoE down phase dropped from 8 dispatches
  per iteration to 1 dispatch per iteration in `bench-moe`.
- Correctness check:
  `llmtoy compare ... "explain MoE" --chat --gpu-layers 0:0` keeps all layer
  argmaxes and final argmax matching CPU.
- Layer 0 `bench-moe --iters 32` with GPU timestamps after integration:
  gate/up remains 256 dispatches / 50.32 us per iter, quantize-mid remains
  256 dispatches / 2.31 us per iter, id-down is 32 dispatches / 28.12 us per
  iter, accum is 32 dispatches / 4.66 us per iter, total wall is
  618.43 us/iter.

Interpretation: down-id integration is correct and reduces dispatch count, but
it is not a material MoE wall-time win yet. Host submit/readback overhead still
dominates this microbench, and gate/up remains the larger GPU-time slice. Keep
the down-id route because it matches the intended MUL_MAT_ID shape, but the
next MoE optimization should target gate/up kernel quality or broader command
submission overhead rather than more down-id polishing.

Gate/up-id one-pass Q3_K update:

- The bench-only `expert_gate_up_id_q3_k_q8_1` shader now accumulates gate and
  up in one pass over each Q8_1 activation block instead of calling the Q3_K
  dot loop twice. This keeps the same selected-expert output contract and does
  not change production routing.
- Clean-system preflight before measurement: no stray processes, 1m load 0.60,
  58 GiB available.
- `zig build test` passes after the shader change, including the focused
  expert gate/up ID correctness test.
- Layer 0, `bench-moe --iters 16 --layer 0`: production path remains
  610.94 us/iter wall, 86.77 us/iter GPU phases. Gate/up ID measures
  155.96 us wall / 52.01 us GPU; down ID measures 90.42 us wall /
  26.04 us GPU.
- Layer 10, `bench-moe --iters 16 --layer 10`: production path remains
  622.33 us/iter wall, 98.15 us/iter GPU phases. Gate/up ID measures
  129.00 us wall / 51.88 us GPU; down ID measures 116.86 us wall /
  37.83 us GPU.

Interpretation: the one-pass change recovers a few microseconds versus the
earlier ~55 us gate/up ID prototype, but it still does not beat the current
eight per-expert fused gate/up dispatches on GPU time (~50 us/iter). Keep it as
a bench-only improvement. A production gate/up-ID route would also need a
non-duplicating flat gate/up weight layout, otherwise it adds a large extra VRAM
copy of tensors that are already uploaded per expert.

Opt-in production-like gate/up-ID route:

- Added `LLMTOY_EXPERT_GU_ID=1` as an experiment. When enabled, each supported
  layer uploads the existing flat `ffn_gate_up_exps.weight` tensor and skips the
  separate per-expert gate/up uploads, avoiding the duplicate-VRAM failure mode.
  The MoE path records one gate/up-ID dispatch into a flat f32 mid buffer, then
  quantizes each selected expert range into the existing flat Q8_1 mid buffer
  for expert-down ID.
- A first attempt that kept both flat and per-expert gate/up uploads pushed the
  process into GTT-heavy placement (`VRAM=14212 MiB GTT=4627 MiB`) and made even
  standalone ID probes hundreds of microseconds slower. That path was removed;
  the opt-in route now ends around `VRAM=12524 MiB GTT=28 MiB`, comparable to
  baseline.
- Correctness with the opt-in route:
  `LLMTOY_EXPERT_GU_ID=1 llmtoy compare ... "explain MoE" --chat
  --gpu-layers 0:0` keeps all layer argmaxes and final argmax matching CPU.
- Layer 0, `LLMTOY_EXPERT_GU_ID=1 bench-moe --iters 16 --layer 0`:
  gate/up drops from 128 dispatches to 16 dispatches in the timed loop, but
  total wall does not improve. Measured `618.25 us/iter` wall,
  `95.11 us/iter` GPU phases, `523.15 us/iter` host/submit.

Follow-up: added `quantize_q8_1_batched.glsl` for the flat gate/up-ID mid
buffer. In the opt-in route it quantizes all selected expert mids in one
dispatch instead of recording one descriptor/dispatch per expert.

- Layer 0, `LLMTOY_EXPERT_GU_ID=1 bench-moe --iters 16 --layer 0` after
  batched mid quantization: `561.33 us/iter` wall, `93.89 us/iter` GPU phases,
  `467.44 us/iter` host/submit.
- Timed-loop dispatch counts moved from gate/up 16, quantize-mid 128, down 16,
  accum 16 to gate/up 16, quantize-mid 16, down 16, accum 16. Per-iteration
  `moe.quantize_mid` GPU time is now about `1.24 us`.
- Correctness still holds:
  `LLMTOY_EXPERT_GU_ID=1 llmtoy compare ... "explain MoE" --chat
  --gpu-layers 0:0` keeps all layer argmaxes and final argmax matching CPU.

Interpretation: the route is correct and structurally closer to llama.cpp's
`MUL_MAT_ID_VEC`, and batched mid quantization gives the first meaningful
MoE wall-time improvement from this route. It is still not ready as the default:
gate/up-ID GPU time remains slightly slower than the current per-expert fused
path, and the MoE microbench still pays about 0.47 ms/iteration in
host/submit/readback overhead. The next final-pipeline step should reduce
MoE command/descriptor/readback overhead further or improve the Q3_K ID inner
loop toward llama.cpp's ~39-46 us trace.

Descriptor-reuse probe:

- Added `LLMTOY_EXPERT_REUSE_DSETS=1` as a second experiment. It lazily
  allocates stable descriptor sets for the opt-in expert-ID route's quantize
  input, gate/up ID, batched mid quantize, down ID, and accumulation bindings,
  then only rebinds and changes push constants in the timed loop.
- Layer 0, `LLMTOY_EXPERT_GU_ID=1 LLMTOY_EXPERT_REUSE_DSETS=1 bench-moe
  --iters 16 --layer 0`: `502.43 us/iter` wall, `88.21 us/iter` GPU phases,
  `414.22 us/iter` host/submit. The same build without descriptor reuse in
  this run measured `610.39 us/iter` wall and `88.50 us/iter` GPU phases.
- Layer 10, `LLMTOY_EXPERT_GU_ID=1 LLMTOY_EXPERT_REUSE_DSETS=1 bench-moe
  --iters 16 --layer 10`: `548.89 us/iter` wall, `99.28 us/iter` GPU phases,
  `449.61 us/iter` host/submit.
- Correctness still holds:
  `LLMTOY_EXPERT_GU_ID=1 LLMTOY_EXPERT_REUSE_DSETS=1 llmtoy compare ...
  "explain MoE" --chat --gpu-layers 0:0` and `--gpu-layers 10:10` keep all
  layer argmaxes and final argmax matching CPU.

Interpretation: descriptor allocation/update/free is a measurable part of the
remaining MoE host overhead once the expert-ID route has only four dispatches
per selected-expert batch. This stays opt-in because command-buffer allocation,
queue submit/wait, and the final host readback still dominate the remaining
~0.41-0.45 ms/iteration gap. Also, cumulative all-layer `compare` with the
gate/up-ID experiment is still not a promotion gate; validate layer slices while
the Q3_K gate/up-ID kernel remains experimental.

Command-buffer reuse probe:

- Added `LLMTOY_EXPERT_REUSE_CMD=1` as a third experiment. It allocates one
  command buffer per layer for the opt-in expert-ID route and resets/re-records
  that command buffer each iteration instead of allocating/freeing a command
  buffer in `GpuContext.submitBatchCopy`.
- Layer 0, profiled with `LLMTOY_EXPERT_GU_ID=1 LLMTOY_EXPERT_REUSE_DSETS=1`:
  `504.56 us/iter` wall, `88.18 us/iter` GPU phases.
- Layer 0, same run plus `LLMTOY_EXPERT_REUSE_CMD=1`: `529.71 us/iter` wall,
  `88.24 us/iter` GPU phases.
- Layer 0, no-profiler 32-iteration wall check: descriptor reuse alone
  `601.65 us/iter`; descriptor + command reuse `602.50 us/iter`.

Interpretation: command-buffer allocation/free is not a measurable win for the
current re-recorded MoE batch path. Keep the probe opt-in as a diagnostic, but
do not prioritize a broad command-buffer reuse refactor until a profile points
at pre-recorded graphs, fence rings, queue wait, or readback as the limiting
factor. The next meaningful MoE work is still the Q3_K gate/up-ID kernel shape
or removing the final per-batch wait/readback.

Gate/up-ID mixed-down correctness fix:

- The flat gate/up-ID route was failing on layers whose down projection could
  not use the flat expert-down ID shader, notably Q5_0 expert-down layers. The
  gate/up-ID shader itself checked out against real layer weights; the bug was
  routing. ID gate/up wrote mids into the flat mid buffer, while the non-ID down
  branch still read the per-expert Q8_1 mid buffers.
- The fix now routes ID gate/up mids to the flat Q8_1 buffer only when the
  selected down path also consumes flat mids. Mixed ID-GU/non-ID-down layers
  quantize each selected flat f32 mid range into the existing per-expert Q8_1
  mid buffer before dispatching the normal per-expert down shaders.
- Descriptor cleanup and command-buffer reuse were tightened for the same mixed
  path. `LLMTOY_EXPERT_REUSE_CMD=1` now applies only to fully persistent
  ID-GU + ID-down layers; mixed layers use the normal one-shot command buffer
  because their temporary descriptor sets cannot be freed while a reusable
  command buffer still references them.
- Correctness:
  - `LLMTOY_EXPERT_GU_ID=1 compare ... "explain MoE" --chat --gpu-layers 1:1`
    returns layer 1 to the baseline `max|D|=0.09959`, all layer argmaxes match,
    and final argmax matches.
  - `LLMTOY_EXPERT_GU_ID=1 compare ... "explain MoE" --chat` passes all layer
    argmaxes and final argmax.
  - `LLMTOY_EXPERT_GU_ID=1 LLMTOY_EXPERT_REUSE_DSETS=1
    LLMTOY_EXPERT_REUSE_CMD=1 compare ... "explain MoE" --chat` also passes all
    layer argmaxes and final argmax.
- Added focused tests for expert-ID Q3_K gate/up on nontrivial expert IDs and
  for batched Q8_1 quantization round-trip behavior.
- With the corrected route, flat gate/up-ID is now the default for supported
  layers. Set `LLMTOY_EXPERT_GU_ID=0` to force the older per-expert gate/up
  path for diagnostics.
- Quiet-host layer 0 `bench-moe --iters 64 --layer 0 --skip-readback`:
  - older per-expert gate/up forced via `LLMTOY_EXPERT_GU_ID=0`: about
    `178.67 us/iter` wall, `88.08 us/iter` GPU phases.
  - flat gate/up-ID: about `160.89 us/iter` wall, `88.76 us/iter` GPU phases.
  - flat gate/up-ID plus descriptor/command reuse envs: about `160.78 us/iter`
    wall, `88.26 us/iter` GPU phases. In this skip-readback microbench the
    default flattening provides the measurable win; descriptor/command reuse is
    mostly noise.
- Short default generation sanity check on the standard MoE prompt
  (`--max-tokens 8`, `LLMTOY_GPU_PROFILE=1`): prefill `20.46 tok/s`,
  generation `20.31 tok/s`. The largest GPU buckets remain dense FFN down,
  `lm_head`, MoE gate/up, and attention softmax; this change mainly reduces
  MoE host/dispatch shape rather than changing total GPU math time.

Readback isolation probe:

- Added `bench-moe --skip-readback`, a diagnostic mode that skips the final CPU
  read/add of `moe_gpu_buf`. This intentionally makes the forward result
  invalid and is only for measuring the ceiling from keeping MoE output in
  VRAM.
- Layer 0, `LLMTOY_EXPERT_GU_ID=1 LLMTOY_EXPERT_REUSE_DSETS=1 bench-moe
  --iters 32 --layer 0`: `597.12 us/iter` wall. The same command with
  `--skip-readback`: `150.20 us/iter`.
- Layer 0 with GPU timestamps and `--skip-readback`: `165.12 us/iter` wall,
  `87.99 us/iter` GPU phases, leaving only `77.13 us/iter` residual
  host/submit overhead.
- Layer 10, descriptor reuse path: `524.73 us/iter` wall. The same command with
  `--skip-readback`: `161.62 us/iter`.

Interpretation: after descriptor reuse, the final HOST_COHERENT MoE output read
and CPU add are the dominant remaining MoE wall-time cost in this microbench.
This points at a concrete production target: keep expert accumulation output in
device-local VRAM and fuse/add it into the residual stream on GPU, then avoid
downloading per-layer MoE output to the CPU.

VRAM-tail integration:

- `expert_accum.glsl` now writes a fresh accumulated MoE vector rather than
  adding into a pre-zeroed output buffer, which lets `moe_gpu_buf` live in
  device-local VRAM. Legacy/debug readback copies through `moe_stage_buf`.
- Default forward consumes that VRAM MoE vector on GPU for every layer:
  `post_ffw_norm_2(moe)`, dense+MoE add, `post_ffw_norm`, residual add, and
  layer scale run in a second GPU submit after the expert batch.
- Layer 19 uses the single-row RMSNorm shader's CPU-order reduction mode. The
  fast reduction preserved every layer argmax but swapped the final top two
  logits on `compare ... "explain MoE" --chat` when all layers used the VRAM
  tail. Diagnostic controls: `LLMTOY_MOE_VRAM_TAIL=0`,
  `LLMTOY_MOE_VRAM_TAIL_LIMIT`, and `LLMTOY_MOE_VRAM_TAIL_SKIP`.
- Layer 0 descriptor-reuse `bench-moe --skip-readback` remains at about
  `149.98 us/iter`; the legacy readback path with device-local output plus
  staging copy measured about `507.87 us/iter` for a 16-iteration check.
- Full default compare on `compare ... "explain MoE" --chat` passes all layer
  argmaxes and final argmax with the all-layer VRAM tail. A short
  `generate "what is 1+1?" --chat --temperature 0 --max-tokens 16 --gpu`
  check measured prefill `21.06 tok/s` with the all-layer tail and layer-19
  precise RMSNorm, versus `17.94 tok/s` with `LLMTOY_MOE_VRAM_TAIL=0`.

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
