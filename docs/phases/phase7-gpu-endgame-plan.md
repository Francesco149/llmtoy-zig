# Phase 7 GPU Endgame Plan

Current status as of 2026-05-16: the Vulkan path is correct enough to optimize,
but still well behind llama.cpp Vulkan on the same Gemma4 26B A4B / RX 7800 XT
class setup. The earlier plan correctly pushed us toward Q8_1 activations and
more GPU-resident state, but the code has moved on: Q8_1 matvecs, GPU
RMSNorm/RoPE, VRAM KV cache, decode attention, GPU timestamp profiling, and
Q8_1 paths for every quantized matmul format used by the target GGUF now exist.

The ground truth now is simple:

- End-to-end decode was about 4 tok/s before the Q6_K fallback fix, about
  10 tok/s after Q6_K, about 15 tok/s after IQ4_NL expert-down coverage, and
  about 17 tok/s after Q5_K attention-V coverage. llama.cpp is reported around
  80-100 tok/s on this hardware, but Phase 7t must turn that into a local
  apples-to-apples reference run.
- Submit count and CPU attention were secondary effects before profiling.
- Initial timestamp profiling showed only about 307 ms of measured GPU dispatch
  time across a 28-token short run while wall time was about 6.2 s. CPU `perf`
  then showed most samples in `quant.dequant.dequantQ6K`, meaning an unprofiled
  CPU Q6_K fallback, not the existing GPU shaders, was the first blocker.
- The repo now has early opt-in GPU timestamp profiling. It is sufficient to
  catch large blind spots, but still needs per-layer/per-shape detail before
  shader work can move fast.
- After Q6_K, IQ4_NL, and Q5_K fallback removal, the expected dominant
  bottleneck is cumulative GPU layer work, synchronization around many small
  dispatch batches, and remaining CPU orchestration/download overhead.

The target is not "make one more fused shader." The target is to either port
llama.cpp's actual Vulkan MMVQ/MMQ strategy faithfully enough to match it, or
prove with per-dispatch timings why this model/hardware path needs a different
specialized kernel.

---

## Verification Gate

Every GPU change must clear this gate before commit.

1. Run shader/unit tests:

```sh
nix develop --command zig build test
```

Every new shader needs a fuzz test against the CPU reference. For quantized
matvecs, keep the existing pattern in `src/gpu/matvec.zig`: random block bytes,
CPU dequant/dot reference, GPU dispatch, and relative-error assertion. Do not
trust hand-authored single-block cases alone.

2. Run per-layer CPU/GPU compare:

```sh
systemd-run --user --scope -p MemoryMax=40G --quiet -- \
  nix develop --command ./zig-out/bin/llmtoy compare <model> "explain MoE" --chat
```

Expected current behavior: all layer argmaxes and final argmax should match on
the standard prompts. Small rel_err differences are expected because the GPU
path uses Q8_1 activations where the CPU reference still uses f32 dots.

3. Bisect regressions with `--gpu-layers L0:L1`.

If a regression appears, first isolate the layer. For tensor-level isolation,
temporary local toggles inside `forwardOne` are acceptable, but remove them
before commit.

4. Check deterministic generation:

```sh
./zig-out/bin/llmtoy generate <model> "what is 1+1?" --chat --temperature 0 --max-tokens 30
./zig-out/bin/llmtoy generate <model> "what is 1+1?" --chat --temperature 0 --max-tokens 30 --gpu
```

5. Benchmark only on a quiet host:

```sh
scripts/check_benchmark_noise.sh
hyperfine --warmup 1 --runs 3 \
  'systemd-run --user --scope -p MemoryMax=40G --quiet -- nix develop --command ./zig-out/bin/llmtoy generate <model> "<prompt>" --chat --temperature 0 --max-tokens 64 --gpu'
```

Store meaningful benchmark results in `docs/benchmarks/phase7_gpu.md`. If a
performance patch regresses by more than 5% and the cause is not understood,
revert or park it behind an explicit experiment branch.

---

## Current GPU Path, From Code

Relevant files:

- `src/model/gemma4/forward.zig`: layer orchestration and CPU/GPU fallbacks.
- `src/model/gemma4/gpu_weights.zig`: GPU-resident weights, dispatch chains,
  KV VRAM, Q8_1 paths, expert batch path.
- `src/gpu/matvec.zig`: Vulkan pipeline wrappers and fuzz tests.
- `src/gpu/context.zig`: command buffers, submits, barriers, buffer copies.
- `src/gpu/shaders/*.glsl`: actual kernels.

What is already done:

- Q8_1 activation quantization via `quantize_q8_1.glsl`.
- Q3_K/Q4_K/Q5_0/Q5_1 Q8_1 matvec shaders.
- Q8_1 attention QKV, output projection, dense FFN gate/up, dense FFN down
  where supported, MoE gate/up/down.
- GPU RMSNorm, element add/scale, GELU multiply, per-head norms, RoPE.
- VRAM KV cache and GPU decode attention for the 24 SWA layers with explicit V.
- Dense FFN and attention/FFN submit fusion down to roughly 3 submits per layer
  on the fast path.

Important remaining fallbacks and caveats:

- `lm_head` now has a Q6_K x Q8_1 GPU path in progress. Q5_K/IQ variants still
  need coverage if future models use them.
- The 6 global layers that share V from K still use the CPU attention path.
- Dense `w_down` has comments that still mention f32-activation fallback for
  non-256-aligned widths; current Q5_0/Q5_1 Q8_1 coverage exists and should be
  verified with profiling rather than assumed.
- Descriptor sets and command buffers are allocated/freed at dispatch frequency.
  This is probably not the 20x gap, but it will matter once matmul is faster.
- Blocking submits now wait a context-owned fence, but there is no broad fence
  ring or async graph execution yet.

---

## llama.cpp Vulkan Delta

Reference tree: `/opt/ai-lab/llama.cpp/ggml/src/ggml-vulkan/`.

The current llmtoy Q8_1 matvec shaders are not faithful ports of llama.cpp's
fast path. Example: `src/gpu/shaders/matvec_q4_k_q8_1.glsl` uses:

- `local_size_x = 32`
- one subgroup/workgroup per output row
- each lane strides over 32-element sub-blocks
- one `subgroupAdd`

llama.cpp's `mul_mat_vec_q4_k.comp` plus `mul_mat_vec_base.glsl` uses a more
general MMVQ framework:

- specialization constants for `BLOCK_SIZE`, `NUM_ROWS`, `NUM_COLS`
- multiple rows per workgroup
- optional multiple columns/batch vectors per workgroup
- subgroup size control in pipeline creation
- different reductions depending on subgroup/shared-memory capabilities
- shared code across quant types with tuned packing/repack helpers

llama.cpp also has a separate `mul_mmq.comp` tiled matrix path for larger
batch/prompt work, and device feature logic for integer dot, subgroup control,
cooperative matrix, pipeline executable stats, async behavior, and profiling.

The likely core mistake in our current roadmap was treating "Q8_1 + integer
dot + subgroup" as equivalent to llama.cpp. It is only the first layer. Kernel
shape and occupancy still matter, and our one-row workgroups leave too much
performance on the table.

---

## Endgame Pipeline Picture

Do not re-optimize the current path as if it were the final architecture. The
final decode pipeline should look much closer to llama.cpp's Vulkan graph:

- Weights are uploaded once and stay in VRAM. Supported quant formats include
  all formats the target model actually uses on hot tensors: Q3_K, Q4_K,
  Q5_0/Q5_1, Q5_K, Q6_K, and IQ4_NL.
- The residual stream, normalized activations, Q/K/V, KV cache, MoE
  intermediates, and logits scratch stay device-resident across the token. CPU
  should not see intermediate vectors except for sampling/logits and optional
  debug checks.
- Decode matmuls use MMVQ-style quantized weight x quantized activation
  kernels: multiple rows per workgroup, tuned subgroup size, shared reduction
  variants where useful, and quant-format helpers ported from llama.cpp before
  local invention.
- Prefill is a separate MMQ-style matrix-matrix path. Repeated batch-1 decode is
  acceptable for correctness bring-up, not for parity.
- A token should be recorded as a small number of command buffers or a graph-like
  sequence with persistent descriptors and fence reuse. `vkQueueWaitIdle` after
  every mini-batch is a bring-up crutch.
- GPU profiling is always on for optimization experiments: timestamped dispatch
  spans, CPU perf, and periodic llama.cpp reference runs are the source of truth.

This means near-term fixes are only worth doing if they either remove a current
CPU fallback or become one of the final MMVQ/MMQ kernels. Q6_K, IQ4_NL, and
Q5_K x Q8_1 all qualified because they removed observed fallbacks and cover
formats used by this target model.

---

## Phase 7m - Profiling Infrastructure First

Status: STARTED. `LLMTOY_GPU_PROFILE=1` now enables Vulkan timestamp queries
and prints aggregate dispatch timings at GPU shutdown. Coverage currently
includes the standalone matvec runner plus the main Gemma4 Q8_1 attention,
attention compute, dense FFN, and MoE GPU dispatch chains.

Initial measurement, standard short prompt, 28 forwarded tokens:

- Prefill/generation wall time was still about 4.5 tok/s.
- Timestamped GPU dispatches summed to only about 307 ms total, around
  11 ms/token.
- CPU `perf` showed about two thirds of samples in `quant.dequant.dequantQ6K`.
- Conclusion: the first optimization target is CPU fallback discovery/removal;
  do not start MMVQ tuning until the profile table accounts for the full token.

After adding Q6_K x Q8_1 routing:

- Prefill/generation rose to about 10 tok/s on the same short prompt.
- `matvec_q8_1.single.262144x2816` appears in the timestamp table 28 times,
  averaging about 1.69 ms per call.
- `dequantQ6K` disappeared from CPU `perf`. Remaining CPU samples are mostly
  `dequantRow` for Q3_K/Q5_K-style fallbacks and `dotIQ4NL`, plus Vulkan memory
  allocation/free during setup/teardown.
- The next profiling task is to map those CPU fallbacks to exact tensors and
  decide whether they belong in the final GPU path or should be eliminated by
  graph restructuring.

After adding IQ4_NL x Q8_1 expert-down routing:

- Prefill/generation rose again to about 15 tok/s on the same short prompt.
- `dotIQ4NL` disappeared from CPU `perf`.
- `llmtoy info` maps the remaining unsupported model tensors to two Q5_K
  attention-V matrices in layers 3 and 4. That is now the only known quantized
  matmul CPU fallback in this model.

After adding Q5_K x Q8_1 attention-V routing:

- Prefill/generation rose again to about 17 tok/s on the same short prompt.
- `dequantQ5K` disappeared from CPU `perf`.
- `llmtoy info` now reports no unsupported GPU quant tensors for this target
  GGUF. Future work should stop chasing one-off quant fallbacks and move to
  MMVQ/MMQ kernel shape, synchronization, and command/descriptor overhead.

Add a GPU profiler that can answer: where does one token spend GPU time, by
layer and by dispatch type?

Minimum implementation:

- Extend `GpuContext` with a timestamp query pool, timestamp period, and a
  disabled-by-default profiler mode controlled by an env var such as
  `LLMTOY_GPU_PROFILE=1`.
- Add helpers:
  - `beginProfiledBatch(label)`
  - `writeTimestamp(cmd, label)` or scoped dispatch labels
  - `endProfiledBatch`
  - `collectProfileResults`
- Timestamp boundaries around every meaningful dispatch in the Gemma4 GPU path:
  `rmsnorm`, `quantize_q8_1`, each matvec type, `gelu_mul`, expert fused GU,
  expert down, expert accum, attention QK, attention AV, copies/downloads.
- Print a compact table aggregated by `(kernel, quant_type, shape)`:
  count, total ms, avg us, min/max, and percent of GPU time.
- Also emit per-layer totals so slow global/SWA layers stand out.

Use llama.cpp's `GGML_VK_PERF_LOGGER` implementation as the design reference:
`ggml-vulkan.cpp` has a timestamp query pool and writes timestamps around graph
nodes/fusions.

Remaining deliverable:

- `LLMTOY_GPU_PROFILE=1 ... generate --gpu --max-tokens 8` produces a stable
  profile table.
- `docs/benchmarks/phase7_gpu.md` gets one profile snapshot for the standard
  prompt.
- Extend shape/format labels beyond standalone `runQ8_1Mv` to all matvec spans
  so dense FFN, MoE, and attention projections can be compared by shape.

Expected result:

- Matvecs should dominate. If they do not, update this plan with the measured
  blocker before touching shaders.

---

## Phase 7n - Matvec Microbenchmark Harness

Status: STARTED. `llmtoy bench-matvec` now loads representative real tensors
from the Gemma4 target GGUF, uploads one tensor at a time, quantizes a
deterministic f32 activation vector to Q8_1 once, then repeatedly dispatches
the current Q8_1 matvec path with the same descriptor/command/submit/wait
overhead used by generation.

Usage:

```sh
nix develop --command ./zig-out/bin/llmtoy bench-matvec <model.gguf> \
  --iters 64 --target all
```

To isolate one shape:

```sh
nix develop --command ./zig-out/bin/llmtoy bench-matvec <model.gguf> \
  --iters 256 --target L0.attn_q
```

Current targets are `lm_head`, `lm_head.fast`, `lm_head.mmvq.b32.r1`,
`lm_head.mmvq.b64.r1`, `lm_head.mmvq.b64.r2`, `lm_head.mmvq.b64.r4`,
`L0.attn_q`, `L0.attn_q.r4`, `L0.attn_v`, `L0.attn_v.r4`, `L3.attn_v`,
`L5.attn_q`, `L5.attn_q.mmvq.b32.r1`, `L5.attn_q.mmvq.b64.r1`,
`L0.attn_q.mmvq.b32.r1`, `L0.attn_q.mmvq.b64.r1`,
`L0.attn_q.mmvq.b64.r2`, `L0.attn_q.mmvq.b64.r4`,
`L0.attn_v.mmvq.b32.r1`, `L0.attn_v.mmvq.b64.r1`,
`L0.attn_v.mmvq.b64.r2`, `L0.attn_v.mmvq.b64.r4`, `L0.dense_down`,
`L0.dense_down.mmvq.b64.r1`, `L0.dense_down.mmvq.b64.r2`,
`L0.dense_down.mmvq.b64.r4`, `L5.dense_down`,
`L5.dense_down.mmvq.b64.r1`, `L5.dense_down.mmvq.b64.r2`,
`L5.dense_down.mmvq.b64.r4`, `L0.expert_down`,
`L0.expert_down.mmvq.b64.r1`, `L0.expert_down.mmvq.b64.r2`,
`L0.expert_down.mmvq.b64.r4`, and `L10.expert_down`. They cover Q3_K,
Q4_K, Q5_0, Q5_1, Q5_K, Q6_K, and IQ4_NL with the target model's actual
row/column sizes. The `.r4` targets are experimental Q4_K row-batching probes,
`lm_head.fast` is an experimental Q6_K packed-decode probe, `lm_head.mmvq.*`
targets are early Q6_K MMVQ ports, `L5.attn_q.mmvq.*` targets are early
Q3_K MMVQ ports, `L0.attn_q/v.mmvq.*` targets are early Q4_K MMVQ ports,
and `dense_down/expert_down.mmvq.*` targets are early Q5_0/Q5_1 MMVQ ports.
None is a production route until it beats the current path.

End-to-end tok/s is too noisy for shader iteration. Add a command or test-only
binary that runs one GPU kernel shape thousands of times with fixed buffers.

Required shapes for Gemma4 26B A4B:

- Attention Q/K/V/O: Q3_K and Q4_K, `cols = d_model`, rows matching the actual
  tensors.
- Dense FFN gate/up: Q3_K, `cols = d_model`, `rows = d_ffn`.
- Dense/expert down: Q5_0/Q5_1 where present, `cols = d_ffn` or expert width,
  `rows = d_model`.
- Expert gate/up/down: actual expert matrix shapes.
- lm_head: Q6_K, `rows = vocab_size`, `cols = d_model`.

The harness should report:

- GPU timestamp average per dispatch
- effective weight bandwidth GB/s
- effective integer dot operations/s where meaningful
- CPU-side dispatch overhead separately from GPU elapsed time

The first landed version reports wall-clock average per current matvec call and
effective weight bandwidth. This intentionally includes the current
descriptor/update/submit/wait path because that is what decode pays today.
With `LLMTOY_GPU_PROFILE=1`, it now also reports GPU timestamp elapsed and the
residual CPU/submit overhead. Next increments should add integer-dot throughput
where the quant format makes the count useful.

Use `--reuse-descriptor` to test a persistent descriptor-set ceiling for one
stable binding. Current measurements show descriptor reuse alone does not
materially reduce the 50-60 us residual CPU overhead on small matvecs, so broad
production descriptor churn is lower priority than command-buffer/fence reuse
and submit batching.

This harness is where shader variants compete. Do not use full generation to
decide between two matvec kernels unless the microbench result is inconclusive.

First measurement with `--iters 8` on the target model:

| Target | Type | rows x cols | avg us | effective GB/s |
|--------|------|-------------|--------|----------------|
| `lm_head` | Q6_K | 262144 x 2816 | 2043.03 | 296.40 |
| `L0.attn_q` | Q4_K | 4096 x 2816 | 47.80 | 135.74 |
| `L0.attn_v` | Q4_K | 2048 x 2816 | 43.28 | 74.96 |
| `L3.attn_v` | Q5_K | 2048 x 2816 | 54.54 | 72.70 |
| `L5.attn_q` | Q3_K | 8192 x 2816 | 83.44 | 118.79 |
| `L0.dense_down` | Q5_1 | 2816 x 2112 | 55.25 | 80.73 |
| `L5.dense_down` | Q5_0 | 2816 x 2112 | 51.09 | 80.03 |
| `L0.expert_down` | Q5_1 | 2816 x 704 | 53.74 | 27.67 |
| `L10.expert_down` | IQ4_NL | 2816 x 704 | 49.16 | 22.68 |

Interpretation: the small expert-down shapes are dominated by launch/descriptor
overhead and poor occupancy, while the large lm_head shape reaches much higher
bandwidth. This points directly at Phase 7o: MMVQ-style multi-row workgroups
and better shader shape should come before command plumbing, except where the
microbench proves a shape is launch-bound.

First Q4_K row-batching probe:

- `matvec_q4_k_q8_1_r4.glsl` maps four output rows into one workgroup.
- Correctness requires row-local reduction. Plain `subgroupAdd` is wrong when a
  hardware subgroup spans multiple logical rows; the current shader uses
  `subgroupClusteredAdd(..., 32)`.
- Bench result with 128 iterations: `L0.attn_q` baseline 59.37 us vs `.r4`
  61.25 us; `L0.attn_v` baseline 57.38 us vs `.r4` 58.22 us.
- Conclusion: naive row packing is not enough. Keep this as a comparison
  target, but do not route generation through it. The next MMVQ attempt should
  port llama.cpp's framework more faithfully rather than only changing
  `local_size_y`.

First Q6_K packed-decode probe:

- `matvec_q6_k_q8_1_fast.glsl` keeps the current one-row dispatch shape but
  decodes four Q6 values at a time from packed byte words.
- A u32-reinterpreted Q6_K struct failed fuzz; keep the byte layout unless a
  dedicated repack step is added.
- Correct byte-layout version passes Q6_K x Q8_1 fuzz with `rel=6.963e-8`
  small and `rel=2.384e-7` lm-head-shaped.
- Bench result: current `lm_head` 2032.90 us wall / 1034.10 us GPU; packed
  decode 2033.22 us wall / 1001.17 us GPU. This is not a production win.
- Conclusion: local decode cleanup is not enough. The next Q6_K attempt should
  port llama.cpp's `mul_mat_vecq.comp`/`mul_mat_vec_q6_k.comp` loop structure
  more directly, including `K_PER_ITER`, `BLOCK_SIZE`, and specialization
  variants.

First Q6_K MMVQ port slice:

- Host-side `MatvecPipeline` now supports Vulkan specialization constants.
- `matvec_q6_k_q8_1_mmvq.glsl` uses `local_size_x_id = 0`, `BLOCK_SIZE`,
  `NUM_ROWS`, and `NUM_COLS` specialization constants. The first variants are
  `lm_head.mmvq.b32.r1` and `lm_head.mmvq.b64.r1`.
- The shader uses the first llama-shaped loop: `K_PER_ITER=16`, per-invocation
  `temp[NUM_ROWS]`, and shared-memory reduction across `BLOCK_SIZE`. It still
  uses llmtoy's byte-backed Q6_K layout rather than a repacked
  `data_a_packed16` view.
- Correctness:
  - b32/r1 small `rel=7.312e-8`, lm-head-shaped `rel=2.140e-7`
  - b64/r1 small `rel=1.314e-7`, lm-head-shaped `rel=1.557e-7`
- Bench result with GPU timestamps:
  - current `lm_head`: about 1034 us GPU
  - `lm_head.fast`: about 994 us GPU
  - `lm_head.mmvq.b32.r1`: about 1156 us GPU
  - `lm_head.mmvq.b64.r1`: about 1133 us GPU
- Conclusion: the scaffolding is correct, but this first direct shape is slower
  because it has not yet ported llama.cpp's packed16 views, scale cache, and
  exact Q6_K per-thread offsets. Next work should close that gap before trying
  `NUM_ROWS > 1`.

Q6_K MMVQ q8-helper rewrite:

- Re-reading llama.cpp showed the Q8_1 activation path should mirror
  `mul_mat_vecq.comp` and `mul_mat_vecq_funcs.glsl`, not the f32-activation
  `mul_mat_vec_q6_k.comp` `sccache` path.
- The MMVQ shader now uses `cache_b_qs[4]`, `cache_b_ds`, `repack4`,
  `get_d_scale`, and `mmvq_dot_product` in the same structure as llama.cpp's
  `DATA_A_Q6_K` helper.
- Added `lm_head.mmvq.b64.r2` and `lm_head.mmvq.b64.r4`.
- Correctness: b64/r2 `rel=2.659e-7`, b64/r4 `rel=1.590e-7` on lm-head-shaped
  fuzz.
- GPU timestamps: b64/r2 improves to about 1077 us, b64/r4 about 1112 us, but
  both still trail the current Q6_K shader and `lm_head.fast`.
- Added llama.cpp-style packed16 aliasing and subgroup reduction:
  `matvec_q6_k_q8_1_mmvq.glsl` now declares the same binding as both raw
  `block_q6_K` and `block_q6_K_packed16`, reads `ql/qh` through packed16, and
  still reads `scales/d` through the raw byte view. Reduction now follows
  llama.cpp's safe `USE_SUBGROUP_ADD` path: subgroup reduction first, shared
  memory only across subgroup partials.
- Correctness after this pass: b64/r1 `rel=1.947e-7`, b64/r2 `rel=2.991e-7`,
  b64/r4 `rel=1.590e-7` on lm-head-shaped fuzz.
- Focused GPU timestamps after this pass: current `lm_head` about 1025 us,
  `lm_head.mmvq.b64.r1` about 1021 us, and `lm_head.mmvq.b64.r4` about
  1059 us. This is the first MMVQ variant to narrowly beat the current Q6_K
  path, but the win is too small to promote into generation yet.
- The llama.cpp 4-then-2 manual iteration unroll plus per-thread tail
  `num_iters` was tested and reverted. Correctness was unchanged, but focused
  timestamps regressed: current `lm_head` about 1027 us, `b64.r1` about
  1058 us. Do not repeat that unroll as a standalone Q6_K optimization.
- `llmtoy gpu-info` now reports subgroup properties. RX 7800 XT/RADV reports
  subgroup size 64, compute support true, and arithmetic support true.
- A no-shared-memory b64/r1 reduction variant was tested and reverted. It was
  valid on this device because subgroup size covers `BLOCK_SIZE=64`, but it
  regressed: current `lm_head` about 1037 us, safe `b64.r1` about 1036 us,
  no-shmem about 1059 us. Keep the safe subgroup-plus-shared-memory reduction.
- Updated conclusion: for Q6_K, the obvious llama.cpp structural pieces are now
  in place for the isolated MMVQ target. Stop polishing Q6_K in isolation and
  port the same MMVQ family to Q3_K/Q4_K, where the model spends much more
  decode time.

---

## Phase 7o - Faithful llama.cpp MMVQ Port

Status: STARTED. This is the main performance project. The early Q4_K R4 and
Q6_K packed-decode probes were useful negative results: naive row packing and
local decode cleanup are not enough. The next work should port the llama.cpp
framework shape, not keep inventing isolated one-row shaders.

Goal: replace the current one-row Q8_1 matvec shaders with a close port of
llama.cpp's MMVQ structure, then tune for RX 7800 XT.

Start with Q4_K because it is easiest to compare against
`mul_mat_vec_q4_k.comp`. Then port Q3_K, Q5_0, Q5_1, Q6_K, Q5_K, and IQ4_NL.

Implementation plan:

1. Add a new experimental pipeline family rather than deleting the existing
   shaders immediately. Keep a runtime or compile-time switch so one command
   can compare old vs new.

   Host-side requirement: `MatvecPipeline` must support Vulkan specialization
   constants. llama.cpp uses `local_size_x_id = 0` plus constants:

   - constant `0`: `BLOCK_SIZE`, also the compute workgroup X size
   - constant `1`: `NUM_ROWS`
   - constant `2`: `NUM_COLS`

   Add experimental init methods such as `initQ6KQ8_1Mmvq(ctx, .{ ... })` and
   bench-only targets such as `lm_head.mmvq.b64.r1`. Do not replace the
   current production pipeline until a variant wins in `bench-matvec` and
   passes fuzz.

2. Port the shared framework shape:
   - `BLOCK_SIZE` specialization constant, initially 32 and 64. llama.cpp's
     generated shader defaults to 32; RDNA3 may prefer 64 when full subgroup
     control is available, but measure both.
   - `NUM_ROWS` specialization constant, test 1, 2, 4, 8. Avoid the previous
     `local_size_y` R4 mistake: llama.cpp keeps `local_size_y = 1` and stores
     per-row accumulators in `temp[NUM_COLS][NUM_ROWS]` inside each invocation.
   - `NUM_COLS` specialization constant, test 1 first; use >1 only for batched
     prefill or grouped expert work where the input has multiple vectors.
   - subgroup size control if exposed; compare subgroup 32 and 64 on RDNA3.
   - shared-memory vs no-shared-memory reduction variants.

   Reduction variants matter:

   - `USE_SUBGROUP_ADD_NO_SHMEM`: fastest only when one subgroup covers the
     whole workgroup; unsafe for multiple subgroups unless full support is
     guaranteed.
   - `USE_SUBGROUP_ADD`: subgroup reduction plus shared memory across
     subgroups. This is the safe first port for `BLOCK_SIZE=64`.
   - plain shared-memory tree: fallback/reference variant.

3. Port the Q8_1 activation loop from `mul_mat_vecq.comp`, not just the
   quant-specific decode:

   - define `MMQ` and `B_TYPE block_q8_1_x4`
   - use `K_PER_ITER = 16` for K-quants (`DATA_A_QUANT_K`)
   - preload Q8_1 activation words into `cache_b_qs[K_PER_ITER / 4]`
   - use `b_block_idx_outer = b_block_idx / 4` and `b_block_idx_inner = b_block_idx % 4`
     to match the x4 activation packing already used by llmtoy
   - iterate `num_iters = ceil(ncols / (K_PER_ITER * BLOCK_SIZE))`
   - preserve llama.cpp's 4-then-2 manual unroll structure until a measured
     local change beats it

4. Port one quant-specific file at a time. For Q6_K, the relevant llama.cpp
   decode structure is `mul_mat_vec_q6_k.comp`:

   - `shared FLOAT_TYPE sccache[2][BLOCK_SIZE/16][16]`
   - `itid = tid % 16`, `ix = tid / 16`, so each 16-thread group handles a
     256-element Q6_K block lane
   - `v_im = itid / 8`, `v_in = itid - 8*v_im`
   - `ql_offset = 64*v_im + 4*v_in`
   - `qh_offset = 32*v_im + 4*v_in`
   - `s_offset = 8*v_im + v_in/4`
   - load Q6 low/high pieces with the same `data_a_packed16` pattern before
     translating to this repo's byte-backed structs
   - carry `all_threads` handling for tails even if Gemma dimensions are
     usually friendly; fuzz should include small/tail shapes if the shader
     supports them

5. Match llama.cpp's Q4_K repacking and scale/min handling first. Do not invent
   a new layout unless the faithful port is already measured.

6. Extend the port to Q3_K. This is likely the highest-impact quant type for
   this model because many attention and FFN gate/up matrices are Q3_K.

7. Extend to Q5_0/Q5_1 for down projections and experts.

8. Add Q6_K to the MMVQ family early for `lm_head`, because timestamped
   `bench-matvec` shows this large tensor is GPU-kernel dominated. It should
   use the same framework as Q3/Q4 rather than remain a bespoke one-row shader.

9. Add IQ4_NL to the MMVQ family after the fallback-removal shader is stable.
   Its current lookup-table shader is useful, but it is not a faithful
   llama.cpp MMVQ-style kernel.

10. Add Q5_K to the MMVQ family after the fallback-removal shader is stable.
   Correctness first; K-quants are easy to get subtly wrong.

Completed implementation slices:

1. Add specialization support to `MatvecPipeline.initFromSpv`.
2. Add a Q6_K MMVQ shader target with `BLOCK_SIZE`, `NUM_ROWS`, and `NUM_COLS`
   specialization constants.
3. Align the Q6_K Q8_1 path with `mul_mat_vecq.comp` /
   `mul_mat_vecq_funcs.glsl`, including `cache_b_qs`, `cache_b_ds`, `repack4`,
   `get_d_scale`, and `mmvq_dot_product`.
4. Add packed16 Q6_K `ql/qh` reads by aliasing the same weight binding as both
   raw and `block_q6_K_packed16`. No upload-time repack was needed for Q6_K
   because the packed16 view preserves the same 210-byte block layout.
5. Replace the plain shared-memory tree with subgroup reduction plus
   cross-subgroup shared-memory partials.
6. Sweep `lm_head.mmvq.b32.r1`, `b64.r1`, `b64.r2`, and `b64.r4`; current best
   candidate is `b64.r1`.

Next implementation slice:

1. Repeat the Q3_K b64/r1 measurement once more after the next shader change;
   keep it bench-only unless it stays ahead in stable runs.
2. Move to the next non-Q4/Q6 bottleneck from the profile. Likely candidates:
   Q5_0/Q5_1 down projections, Q5_K attention-V, or IQ4_NL expert down. Keep
   each as a bench-only MMVQ target until it beats the current shader.
3. Keep current production routing on the existing kernels until a full
   generate profile shows a material token/s improvement, not just a narrow
   isolated `lm_head` win.

Q3_K MMVQ first slice:

- Added `matvec_q3_k_q8_1_mmvq.glsl`, using the same generated-style framework
  as the Q6_K MMVQ shader.
- The shader follows llama.cpp's `DATA_A_Q3_K` helper: packed16 `hmask/qs`
  view, raw byte `scales/d` view, Q8_1 cache, and subgroup-plus-shared-memory
  reduction.
- Added bench targets `L5.attn_q.mmvq.b32.r1` and
  `L5.attn_q.mmvq.b64.r1`.
- Correctness: b32/r1 `rel=3.358e-7`, b64/r1 `rel=1.674e-7` on 2816-column
  fuzz.
- Focused GPU timestamps: current `L5.attn_q` about 43.22 us, b32/r1 about
  64.90 us, b64/r1 about 42.81 us.
- Updated conclusion: b64/r1 is a bench-only Q3_K candidate; b32/r1 is a
  negative. Repeat/tune b64 only if it remains ahead in stable runs, then port
  the same framework to Q4_K.

Q4_K MMVQ first slice:

- Added `matvec_q4_k_q8_1_mmvq.glsl`, using llama.cpp's `DATA_A_Q4_K`
  structure: packed32 Q4 reads, Q8_1 cache, Q4 scale/min decode, and the safe
  MMVQ reduction.
- Added bench targets for `L0.attn_q` and `L0.attn_v`: b32/r1, b64/r1,
  b64/r2, and b64/r4.
- Correctness: b32/r1 `rel=2.654e-5`, b64/r1 `rel=1.436e-5`,
  b64/r2 `rel=2.626e-5`, b64/r4 `rel=3.619e-5` on 2304-column fuzz.
- Focused GPU timestamps: current `L0.attn_q` about 14.89 us, b32/r1 about
  19.13 us, b64/r1 about 16.12 us, b64/r2 about 16.79 us, b64/r4 about
  16.41 us. Current `L0.attn_v` about 9.33 us, b32/r1 about 11.17 us,
  b64/r1 about 10.03 us.
- Updated conclusion: Q4_K MMVQ is correct but not a win. Keep it bench-only
  as a reference. The current Q4_K shader is already strong for these small
  attention projections; do not spend another session on Q4_K MMVQ unless a
  full profile shows a different Q4_K shape dominates.

Q5_0/Q5_1 MMVQ first slice:

- Added `matvec_q5_0_q8_1_mmvq.glsl` and
  `matvec_q5_1_q8_1_mmvq.glsl`, following llama.cpp's legacy-quant
  `DATA_A_Q5_0`/`DATA_A_Q5_1` MMVQ path: `K_PER_ITER=8`, packed16 quant
  reads, packed32 Q5_1 scale/min reads, Q8_1 cache, and the safe
  subgroup-plus-shared-memory reduction.
- Added bench-only targets for dense down and one expert-down slice:
  `L0.dense_down.mmvq.b64.r{1,2,4}`,
  `L5.dense_down.mmvq.b64.r{1,2,4}`, and
  `L0.expert_down.mmvq.b64.r{1,2,4}`.
- Correctness: all new Q5_0/Q5_1 MMVQ fuzz tests pass at the existing
  `rel < 1e-3` tolerance on 704-column model-shaped cases.
- Focused GPU timestamps with `--iters 128 --target all`: current
  `L0.dense_down` about 8.27 us vs MMVQ best about 10.33 us; current
  `L5.dense_down` about 9.34 us vs MMVQ best about 11.69 us; current
  `L0.expert_down` about 4.27 us vs MMVQ best about 4.89 us.
- Updated conclusion: Q5_0/Q5_1 MMVQ is correct but not a win. Keep these as
  bench-only reference targets. Do not route generation through them; the next
  optimization should move to a different measured bottleneck, likely Q5_K or
  IQ4_NL MMVQ, global-layer CPU attention, or command/fence overhead if the
  next full profile points there.

Acceptance criteria:

- Every new quant shader passes fuzz tests at the existing tolerances.
- Microbench shows a large per-dispatch improvement over the current shader.
  A faithful MMVQ port should aim for multi-x speedup, not single-digit percent.
- End-to-end generation improves materially. If microbench improves but tok/s
  does not, inspect synchronization, downloads, and fallback paths with the
  profiler before changing the shader again.

Notes:

- The current shaders use `dotPacked4x8EXT`, so the gap is not "missing integer
  dot." The gap is occupancy, memory coalescing, row batching, subgroup sizing,
  and possibly poor Q3_K/Q4_K unpack scheduling.
- Check device properties, not assumptions. llama.cpp only enables integer dot
  when `integerDotProduct4x8BitPackedSignedAccelerated` is true.

---

## Phase 7p - Prefill Is Matrix-Matrix, Not Repeated Decode

Status: TODO after MMVQ profiling.

The current public path calls `forwardOne` for every prompt token. That means
prefill is effectively repeated batch-1 decode. llama.cpp uses graph execution
and can hit matrix-matrix kernels when batch size is large enough.

Plan:

- Add profiling that separates prefill and decode dispatch shapes.
- Decide whether to implement a real batched prefill path for Gemma4:
  - batch several prompt tokens through attention/FFN matmuls,
  - use llama.cpp's `mul_mmq.comp` strategy for quantized weight x Q8_1
    activations where it applies,
  - keep decode on MMVQ.
- Do not block decode parity on this. Decode is the visible 4 tok/s problem,
  but prefill parity needs MMQ eventually.

Acceptance criteria:

- Prefill benchmark improves independently of decode.
- Decode path remains clean and does not regress to accommodate prefill.

---

## Phase 7q - Remove CPU Round Trips That Profiling Proves Matter

Status: TODO, lower priority than matvec.

Known remaining CPU/GPU synchronization points:

- `runLayerAttnResidualDenseFfnQ8_1` downloads both `x` and `ffn_out` after its
  submit. The CPU needs `x` for later orchestration today, but the long-term
  target is a VRAM-resident residual stream across the whole layer.
- `runExpertBatch` writes MoE input/scales from CPU and reads accumulated MoE
  output back to CPU.
- MoE top-k routing is still CPU-side.
- Global layers without explicit V still use CPU attention.
- Command submission still blocks after most batches, now via a context-owned
  fence rather than `vkQueueWaitIdle`.

Do not attack these blindly. Use Phase 7m profile data. If matvec is still 90%
of GPU time and CPU wait is just waiting for matvec, round-trip removal will not
close a 20x gap.

High-value cleanup once matvec is faster:

- Keep `x`, dense FFN output, and MoE output in VRAM until final logits.
- Move router top-k to GPU or at least avoid readback of intermediate vectors.
- Add a V-from-K shader/path for the 6 global layers so they use GPU attention.
- Expand the first-pass submit fences into a real fence ring once profiling
  shows queue wait as a bottleneck.

---

## Phase 7r - Descriptor, Command, and Pipeline Overhead

Status: STARTED after VRAM-tail work exposed host overhead. `GpuContext` now
owns one Vulkan pipeline cache handle, all compute pipeline builders route
through it, and cache data is serialized under `$XDG_CACHE_HOME/llmtoy` (or
`$HOME/.cache/llmtoy`) by default. `LLMTOY_PIPELINE_CACHE=/path/file` overrides
the location and `LLMTOY_PIPELINE_CACHE=0` disables disk persistence. The
synchronous submit helpers now wait on a context-owned fence instead of
`vkQueueWaitIdle`, and the narrow async MoE path recycles a tiny fence pool.

The current wrappers allocate/update/free descriptor sets for nearly every
dispatch and allocate/free command buffers for every submit. That overhead is
visible in code (`MatvecPipeline.record`, many `vkFreeDescriptorSets`, and
`GpuContext.submitBatchCopy`), but earlier measurements show submit overhead is
not the primary bottleneck while matvecs take hundreds of ms/token.

Roadmap:

- Use descriptor buffers or persistent descriptor sets for stable bindings.
- Pre-allocate per-layer/per-dispatch descriptor sets where buffers are stable.
- Reuse command buffers where the sequence is fixed and only push constants
  change.
- Expand the current one-fence synchronous submit path into a measured fence
  ring only if CPU profiles show queue wait as material.
- Measure warm-start pipeline creation after cache serialization.

Acceptance criteria:

- CPU-side profile shows measurable dispatch overhead before this work starts.
- Afterward, CPU time per token drops measurably without GPU timestamp regressions.

---

## Phase 7s - Device Feature and Shader Introspection

Status: STARTED, useful alongside 7o.

`llmtoy gpu-info --verbose` now prints the first-pass device facts needed for
shader tuning:

- subgroup size and supported subgroup operations
- integer dot product feature and acceleration booleans
- subgroup size control support
- timestamp period
- memory heaps/types relevant to VRAM/GTT
- pipeline executable properties support
- cooperative matrix extension support
- cooperative matrix supported shapes when the extension function is available

On the RX 7800 XT/RADV target, the verbose probe reports subgroup size 64,
subgroup size control with min 32/max 64 for compute, accelerated packed 4x8
integer dot product, `VK_KHR_pipeline_executable_properties`, and
`VK_KHR_cooperative_matrix`. The cooperative-matrix shape query reports 14
subgroup-scoped 16x16x16 shapes covering int8/u8 integer accumulation and f16
variants. This explains why 32-lane shader assumptions need explicit handling
on this driver and gives a concrete target shape if cooperative-matrix kernels
are prototyped later.

If `VK_KHR_pipeline_executable_properties` is available, add an opt-in debug
path to dump register usage/occupancy-like stats for the hot matvec pipelines.
llama.cpp already has plumbing for pipeline stats; use it as the reference.

This is not a substitute for timestamps, but it explains why one shader variant
wins or loses on RADV/RDNA3.

---

## Phase 7t - llama.cpp Reference Benchmark Discipline

Status: STARTED. A Nix Vulkan reference is now recorded.

Important: `nixpkgs#llama-cpp` only exposes `BLAS` in this environment. Use
`nixpkgs#llama-cpp-vulkan`; it reports `Vulkan0: AMD Radeon RX 7800 XT (RADV
NAVI32)`.

The Gemma4 service flags live in `../nix-lab/hosts/lame/llama.nix`:

- package: `pkgs.llama-cpp-vulkan`
- `-c 120000`, `-ngl 99`, `--n-cpu-moe 0`, `--kv-unified`
- `--flash-attn on`
- `--cache-type-k q8_0`, `--cache-type-v q8_0`
- `--parallel 1`
- Gemma chat template: `/opt/ai-lab/templates/new-chat-template-gemma.jinja`

`llama-cli` version `8983 (80afa33)` rejects `--kv-unified`, but the rest of
the production-like flags work for a CLI probe. The short reference run
reported about `223.5` prompt tok/s and `85.6` generation tok/s.

Most important finding: llama.cpp's actual decode pipeline for this model is
not just "better one-row MMVQ." The hot path uses:

- `MUL_MAT_ID_VEC` for Q3_K expert gate/up
- `MUL_MAT_ID_MUL MUL_MAT_ID_VEC` for expert down with fused scale multiply
- `FLASH_ATTN_EXT` for attention
- q8_0 KV cache in the production config
- fused small ops such as `RMS_NORM_MUL` and `RMS_NORM_MUL_ROPE`

Representative decode timings from the reference:

- `MUL_MAT_ID_VEC q3_K m=1408 n=8 k=2816 n_expert=128`: about 39-46 us
- `MUL_MAT_ID_MUL MUL_MAT_ID_VEC q5_0 m=2816 n=8 k=704 n_expert=128`:
  about 36-40 us
- `MUL_MAT_ID_MUL MUL_MAT_ID_VEC iq4_nl m=2816 n=8 k=704 n_expert=128`:
  about 27-29 us
- `MUL_MAT_VEC q6_K m=262144 n=1 k=2816`: about 1.03 ms
- `FLASH_ATTN_EXT` decode: about 14 us for SWA layer groups and about 30 us
  for global layer groups

Once per milestone, benchmark the local llama.cpp clone with the same model,
prompt, max token count, GPU, and driver. Record:

- llama.cpp command line
- git revision
- Vulkan device line
- prefill/decode tok/s
- whether MMVQ/MMQ/integer-dot/coopmat paths are enabled if visible
- any relevant env vars such as `GGML_VK_DISABLE_MMVQ`,
  `GGML_VK_FORCE_MMVQ`, `GGML_VK_PERF_LOGGER`

The reported 80-100 tok/s target may be real, stale, or measured with different
batching/model flags. Future agents need a hard local reference, not folklore.

Updated next target after local llama.cpp Vulkan trace:

The key reference result is that llama.cpp's `MUL_MAT_VEC q6_K
m=262144 n=1 k=2816` is about 1.03 ms, essentially matching llmtoy's
`lm_head` Q6_K timing. The 4x end-to-end gap is therefore not explained by the
largest logits matvec. Do not spend the next session on broad isolated MMVQ
variants unless a same-shape llama.cpp timing proves that tensor is materially
faster there.

Next-session plan:

1. Fix the async/graph prerequisites from the reverted attempt: give
   `GpuContext` a stable address and add deferred descriptor-set free lists so
   descriptor sets are released only after the owning fence signals. Current
   status: `runExpertBatch` and the no-readback GPU attention path attach
   transient descriptor sets to a generic pending async fence and pass
   `compare` without descriptor reuse enabled. Dense FFN, post-attention, and
   one-shot helper paths still need the same conversion before they can be
   allowed in flight.
2. Build a steady decode command-buffer path that records a larger per-token or
   per-layer graph and blocks at real host boundaries, not after every local
   phase. Keep current synchronous code as the fallback while proving this.
3. Port llama.cpp's selected-expert decode shape before more single-matrix
   shader tuning: `MUL_MAT_ID_VEC` for Q3_K gate/up and fused
   `MUL_MAT_ID_MUL`-style score multiply for Q5_0/Q5_1/IQ4_NL down.
4. Port the small-op fusions that are clearly visible in the llama.cpp trace:
   `RMS_NORM_MUL`, `RMS_NORM_MUL_ROPE`, and where practical
   `RMS_NORM_MUL_ROPE_VIEW_SET_ROWS`.
5. Replace or substantially rework the current attention fused-small shader
   toward llama.cpp `FLASH_ATTN_EXT`, including q8_0 KV cache compatibility.
6. Keep top-k/router fusion (`TOPK_MOE_EARLY_SOFTMAX_NORM`) as a follow-up once
   the selected-expert matvec and graph submission shape are closer to the
   reference.

Reference commands and kernel timings are recorded in
`docs/benchmarks/phase7_gpu.md` under the llama.cpp Vulkan reference section.

---

## Sequencing Summary

| Phase | Status | Purpose | Expected impact |
|-------|--------|---------|-----------------|
| 7h-7l | DONE | Correctness, Q8_1, GPU norms/RoPE/KV/attention | 3 tok/s -> ~4 tok/s |
| 7m | STARTED | GPU timestamp profiler | Blocks informed work |
| 7n | STARTED | Matvec + MoE microbench harness | Blocks shader iteration |
| 7o | TODO | Faithful llama.cpp expert-id matvecs | Main MoE lever |
| 7p | TODO | Real batched prefill / MMQ | Prefill parity |
| 7q | TODO | Remove proven CPU round trips | Secondary after graph shape |
| 7r | STARTED | Descriptor/command/fence reuse | Immediate next unblocker |
| 7s | TODO | Feature/stats introspection | Supports shader tuning |
| 7t | STARTED | Repeatable llama.cpp reference | Keeps target honest |

Priority has changed after the llama.cpp trace: **graph/submit lifetime first,
then selected-expert fusion, then attention/norm fusion, then more MMVQ**. If
an agent wants to tune another isolated quant matvec first, require a
same-shape llama.cpp timing that shows llmtoy is behind on that exact kernel.

---

## Working Notes for Agents

- Always use `nix develop --command zig build [test]`; plain `zig` may not be
  on PATH.
- GPU runs should use the memory-capped wrapper:

```sh
systemd-run --user --scope -p MemoryMax=40G --quiet -- \
  nix develop --command ./zig-out/bin/llmtoy ...
```

- CLI argument order is `generate <model> <prompt>` first, flags after.
- `bench-moe` baseline on this host:
  - layer 0 Q3_K/Q5_1: 614.08 us wall, 84.97 us GPU phases per top-8 MoE batch
  - layer 10 Q3_K/IQ4_NL: 629.70 us wall, 96.81 us GPU phases per top-8 MoE batch
  - the ~0.53 ms gap is host/descriptor/recording overhead from the current
    many-dispatch path; see `docs/benchmarks/phase7_gpu.md`
- Expert-ID prototype status:
  - `expert_down_id_q5_0_q8_1`, `expert_down_id_q5_1_q8_1`, and
    `expert_down_id_iq4_nl_q8_1` pass focused shader tests. Current focused
    shape timings are about 26.5 us for Q5_0 layer 1, 26 us for Q5_1 layer 0,
    and 38 us for IQ4_NL layer 10, each for 8 selected experts.
  - `expert_gate_up_id_q3_k_q8_1` passes focused shader tests. The current
    one-pass gate/up variant is about 52 us for 8 selected experts versus
    about 50 us for the current eight per-expert fused dispatches. Do not wire
    it into production until the Q3_K inner loop is closer to llama.cpp's
    `mul_mat_vecq.comp`, or until a production-like integration proves that
    lower host overhead outweighs the slightly slower GPU phase without adding
    a duplicate flat gate/up upload.
  - Flat gate/up-ID is now the default for supported Q3_K expert gate/up
    layers. Set `LLMTOY_EXPERT_GU_ID=0` to force the older per-expert gate/up
    route. The mixed path for layers with non-ID down projections now quantizes
    flat f32 mids back into per-expert Q8_1 mid buffers before dispatching the
    normal down shaders; this fixed the previous layer-1/Q5_0 correctness
    failure. Full `compare ... "explain MoE" --chat` passes with default
    gate/up-ID and with the descriptor/command reuse experiments enabled.
  - `LLMTOY_EXPERT_REUSE_DSETS=1` is a follow-up experiment for the flattened
    expert-ID route. It reuses stable descriptor sets for the flattened
    four-dispatch MoE sequence. Layer 0 improved again to about 502 us/iter
    wall with the same ~88 us GPU phase time; layer 10 measured about
    549 us/iter wall with ~99 us GPU phases. This confirms descriptor churn
    matters once the route is flattened, but command-buffer allocation,
    submit/wait, and final readback remain the larger gap.
  - `LLMTOY_EXPERT_REUSE_CMD=1` is a narrower command-buffer reuse probe for
    fully persistent expert-ID layers. It is intentionally disabled for mixed
    ID-GU/non-ID-down layers because those still use temporary descriptor sets,
    and reusable command buffers keep descriptor references alive until reset.
    Treat command-buffer allocation/free as a low-priority issue unless future
    profiling isolates pre-recorded graphs, fence rings, queue wait, or readback.
  - `bench-moe --skip-readback` is a diagnostic, not a correct forward path.
    It skips the final CPU read/add of the accumulated MoE output. With
    descriptor reuse, layer 0 dropped from about 597 us/iter to 150 us/iter and
    layer 10 dropped from about 525 us/iter to 162 us/iter. A timestamped layer
    0 run with readback skipped measured about 88 us GPU phases and only about
    77 us residual host/submit overhead. This makes the next production target
    explicit: keep MoE accumulation output device-resident and add it to the
    residual stream on GPU instead of downloading per layer.
  - The production path now implements most of that target: `expert_accum`
    writes a fresh MoE vector to device-local VRAM, then the layer tail runs
    MoE post-norm, dense+MoE combine, final FFN norm, residual add, and layer
    scale on GPU for every layer. Layer 19 uses the single-row RMSNorm shader's
    CPU-order reduction mode to avoid the final top-2 logit swap previously
    seen on the "explain MoE" compare prompt. `LLMTOY_MOE_VRAM_TAIL=0`
    restores the old readback/combine path, and `LLMTOY_MOE_VRAM_TAIL_LIMIT` /
    `LLMTOY_MOE_VRAM_TAIL_SKIP` remain diagnostic controls.
  - Before enabling any ID path in `runExpertBatch`, add or use a model-backed
    correctness check that compares real selected-expert intermediates against
    the existing per-expert path, then run `llmtoy compare --gpu-layers`.
- The old `scripts/regression_compare.py` workflow is obsolete for GPU work.
  Use `llmtoy compare`, shader fuzz tests, GPU profiles, and direct llama.cpp
  milestone benchmarks.
- Reference llama.cpp shaders:
  - `vulkan-shaders/quantize_q8_1.comp`
  - `vulkan-shaders/mul_mat_vecq.comp`
  - `vulkan-shaders/mul_mat_vecq_funcs.glsl`
  - `vulkan-shaders/mul_mat_vec_base.glsl`
  - `vulkan-shaders/mul_mat_vec_q*_k.comp`
  - `vulkan-shaders/mul_mmq.comp`
  - `vulkan-shaders/mul_mmq_funcs.glsl`
  - `vulkan-shaders/flash_attn*.comp`
  - `ggml-vulkan.cpp` for pipeline creation, feature gates, profiling, and
    MMVQ/MMQ selection heuristics.

---

## Cross-References

- Current benchmark history: `docs/benchmarks/phase7_gpu.md`.
- General profiling notes: `docs/profiling.md`.
- GPU code: `src/gpu/{context,buffer,matvec}.zig`, `src/gpu/shaders/*.glsl`.
- Gemma4 GPU glue: `src/model/gemma4/{gpu_weights,forward}.zig`.
- llama.cpp reference: `/opt/ai-lab/llama.cpp`.
