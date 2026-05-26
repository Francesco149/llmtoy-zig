# Phase 7 - Vulkan GPU Path

This is the single planning and implementation note for the Vulkan backend.
Older GPU notes described the bring-up path; this document describes the
current code, the target endgame, and the optimization rules.

## Goal

Match llama.cpp Vulkan performance on the target Gemma4 26B A4B APEX-I-Mini
GGUF on RX 7800 XT/RADV while keeping the code small enough to inspect.

The current GPU path is correct enough to optimize and covers the target
model's hot quant formats, but it is still well behind llama.cpp. Recent local
measurements show llmtoy around tens of tok/s depending on prompt and build,
while a local `llama-cli` Vulkan reference run reported about 85 tok/s decode
on the same model/hardware class. Treat that llama.cpp run as the parity target,
not an abstract "one more fused shader" goal.

## Reference Files

- `src/gpu/context.zig`: Vulkan instance/device/queue, command buffers, fences,
  pipeline cache, barriers, timestamp profiler.
- `src/gpu/buffer.zig`: host-coherent, staging, and device-local buffers.
- `src/gpu/matvec.zig`: pipeline wrappers, descriptor recording helpers, GPU
  fuzz tests, benchmark-only MMVQ variants.
- `src/gpu/shaders/*.glsl`: compute kernels.
- `src/model/gemma4/gpu_weights.zig`: uploaded weights, dispatch chains, KV
  VRAM, dense/attention/MoE/final-logits GPU paths.
- `src/model/gemma4/forward.zig`: layer orchestration and CPU/GPU fallbacks.
- `docs/benchmarks/phase7_gpu.md`: historical measurements and negative
  results. It is a lab notebook, not the architecture source of truth.

Reference implementation:

- `/opt/ai-lab/llama.cpp/ggml/src/ggml-vulkan/`

Important llama.cpp areas:

- command-buffer pooling and reuse in `vk_command_pool`
- graph fusion around `MUL_MAT_ID`, `ADD_ID`, `MUL`, top-k MoE, RMSNorm/RoPE
- MMVQ/MMQ matmul shaders under `ggml-vulkan/vulkan-shaders/`
- Vulkan perf logger and timestamp query handling

## Vulkan Model

The backend uses a single compute queue. Weights and long-lived scratch buffers
live in device-local VRAM; small CPU-written inputs and readback buffers use
host-coherent memory.

Core Vulkan objects:

```text
Instance -> PhysicalDevice -> Device -> Queue
CommandPool / CommandBuffer
Buffer + DeviceMemory
DescriptorSetLayout / DescriptorSet
PipelineLayout / Pipeline
ShaderModule
```

Vulkan handles come from one shared `@cImport` module, `src/gpu/vk.zig`, so
opaque C handle types unify across files. Embedded SPIR-V arrays are declared
with `align(4)` because `VkShaderModuleCreateInfo.pCode` requires 4-byte
alignment.

The build compiles GLSL to SPIR-V with `glslc --target-env=vulkan1.3` and
embeds the resulting bytes in the Zig binary. Runtime shader loading is not
used.

## Current Production Path

The production GPU route is Gemma4-specific and optimized for one-token decode.

What is already on GPU:

- Q8_1 activation quantization.
- Q3_K, Q4_K, Q5_0, Q5_1, Q5_K, Q6_K, and IQ4_NL Q8_1 matvec coverage for the
  target GGUF's hot tensors.
- Attention Q/K/V, attention output projection, dense FFN gate/up/down, MoE
  gate/up/down, and final lm_head.
- GPU RMSNorm, add/add-scale, GELU multiply, per-head norms, RoPE, and final
  logit softcap.
- KV cache in VRAM, including the global-layer shared-V case.
- Flattened expert gate/up and down tensors for selected-expert ID dispatches.
- GPU MoE accumulation and the VRAM MoE tail, so the dense-FFN + MoE residual
  tail stays GPU-side for the normal path.
- Persistent descriptor sets for the fixed-buffer dense, attention-front,
  MoE-tail, and final-logits chains where bindings are stable.
- Per-layer reusable command buffers for synchronous MoE-tail submits.
- GPU timestamp profiling with dispatch and in-batch gap summaries.

Known production shape:

- The CPU still performs routing/top-k selection and sampling.
- The CPU still writes per-token activations, selected expert IDs, and expert
  scales for MoE.
- Most submits are still synchronous fence waits. Some non-tail MoE work can
  run async with deferred descriptor frees, but there is no full graph executor.
- Prefill still calls `forwardOne` per prompt token. There is no MMQ/batched
  prefill path yet.

## Current Performance Picture

Do not rely on old Phase 7a-7k numbers for current decisions. They are useful
history, but many of their bottlenecks are gone.

The current bottlenecks to measure before changing code:

- final lm_head Q6_K matvec remains a large single dispatch
- MoE selected-expert gate/up and down dispatches
- attention QK/softmax and fused-small attention variants
- RMSNorm/add-RMSNorm latency floors on small single-row reductions
- remaining host orchestration and synchronous submit/fence cost

Recent profiler work showed that in-batch GPU idle gaps were small compared to
dispatch time. That means command cleanup matters, but kernel quality and graph
shape are still the main levers unless a fresh profile says otherwise.

## llama.cpp Delta

The major gap is no longer missing quant formats. The gap is pipeline shape.

llama.cpp's decode path uses graph-level fusion and tuned kernel families:

- `MUL_MAT_ID*_VEC` for selected-expert MoE work, not many unrelated per-expert
  matvecs.
- MMVQ-style matvec kernels with specialization constants for `BLOCK_SIZE`,
  `NUM_ROWS`, and `NUM_COLS`, subgroup-aware reductions, and quant-specific
  packed helpers.
- FLASH_ATTN_EXT-style attention rather than separate small bespoke passes.
- fused small ops such as RMSNorm+mul/RoPE/add where graph edges allow it.
- command-buffer pooling and deferred resource lifetime tied to submitted work.

llmtoy already has bench-only MMVQ ports for several formats. Most are correct
but not production wins on the measured shapes. Keep them as references; promote
only when a focused microbench and a full generation profile both show material
improvement.

## Endgame Pipeline

The intended final decode pipeline:

- weights remain uploaded in VRAM for all hot quant formats used by the target
  model
- residual stream, normalized activations, Q/K/V, KV cache, MoE intermediates,
  and final-logits scratch remain device-resident across the token
- CPU sees only routing inputs that are not yet GPU-side, final logits for
  sampling, and optional debug readbacks
- decode matmuls use proven MMVQ-style kernels or simpler kernels only where
  profiling proves the simpler kernel is faster for the actual shape
- prefill uses a separate batched MMQ-style path rather than repeated decode
- descriptor sets and command buffers are persistent or deferred by submit
  lifetime; no GPU-visible resource is freed before its fence signals
- benchmarking always includes a local llama.cpp Vulkan reference run for
  parity checks

## Verification Gate

Every GPU behavior change must clear the relevant parts of this gate before
commit.

Shader/unit tests:

```sh
nix develop --command zig build test
```

Every new quant shader needs fuzz coverage against the CPU reference. Keep the
existing pattern in `src/gpu/matvec.zig`: randomized quant bytes, CPU
dequant/dot reference, GPU dispatch, and relative-error assertion.

CPU/GPU compare:

```sh
systemd-run --user --scope -p MemoryMax=40G --quiet -- \
  nix develop --command ./zig-out/bin/llmtoy compare <model> "explain MoE" --chat
```

If a regression appears, isolate with:

```sh
./zig-out/bin/llmtoy compare <model> "explain MoE" --chat --gpu-layers L0:L1
```

Deterministic smoke:

```sh
./zig-out/bin/llmtoy generate <model> "what is 1+1?" --chat --temperature 0 --max-tokens 30
./zig-out/bin/llmtoy generate <model> "what is 1+1?" --chat --temperature 0 --max-tokens 30 --gpu
```

Benchmark only after checking host noise:

```sh
scripts/check_benchmark_noise.sh
```

Store meaningful benchmark/profile snapshots in
`docs/benchmarks/phase7_gpu.md`.

## Profiling Commands

End-to-end GPU profile:

```sh
LLMTOY_GPU_PROFILE=1 systemd-run --user --scope -p MemoryMax=40G --quiet -- \
  nix develop --command ./zig-out/bin/llmtoy generate <model> \
  "Briefly explain the full forward pass of a MoE model" \
  --chat --temperature 0 --max-tokens 8 --gpu
```

Matvec microbench:

```sh
nix develop --command ./zig-out/bin/llmtoy bench-matvec <model> \
  --iters 128 --target all
```

MoE microbench:

```sh
nix develop --command ./zig-out/bin/llmtoy bench-moe <model> \
  --iters 64 --layer 10 --skip-readback
```

llama.cpp Vulkan reference:

```sh
env GGML_VK_PERF_LOGGER=1 GGML_VK_PERF_LOGGER_FREQUENCY=1 \
  VK_ICD_FILENAMES=/run/opengl-driver/share/vulkan/icd.d/radeon_icd.x86_64.json \
  nix shell nixpkgs#llama-cpp-vulkan -c llama-cli \
  -m <model> \
  -p "Briefly explain the full forward pass of a MoE model" \
  -n 8 -t 12 -c 120000 -ngl 99 --n-cpu-moe 0 \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1 \
  --chat-template-file /opt/ai-lab/templates/new-chat-template-gemma.jinja \
  --temp 0 --top-k 40 --top-p 0.9 --seed 42 \
  --conversation --single-turn --reasoning off --no-display-prompt \
  --no-warmup --simple-io --device Vulkan0 --perf
```

## Active Optimization Queue

Work on these in order unless a fresh profile changes the evidence.

1. Keep removing production-only overhead that is already structurally aligned
   with llama.cpp: persistent descriptors, reusable command buffers, deferred
   descriptor frees, and fence reuse. These should be small, measurable commits.
2. Improve selected-expert MoE shape. The target is closer to llama.cpp
   `MUL_MAT_ID*_VEC`: fewer dispatches, better occupancy for `[8 selected
   experts x rows]`, and fused scale/mul/add where it pays.
3. Improve attention shape. The fused-small shader is structurally right but
   only a small win; compare against llama.cpp FLASH_ATTN_EXT timings before
   adding another local variant.
4. Revisit RMSNorm/add-RMSNorm small-op fusions. The single-row reduction floor
   is visible; wins should come from fusing with neighboring ops, not polishing
   the standalone kernel indefinitely.
5. Promote MMVQ variants only when they beat current production kernels on
   target shapes and improve end-to-end generation. Existing Q4/Q5/Q5_K MMVQ
   ports are correct but measured negative or too small to route.
6. Add real batched prefill/MMQ after decode parity work is better understood.

## Negative Results To Preserve

Do not repeat these without new evidence:

- naive Q4_K row packing via `local_size_y` did not beat the current shader
- Q6_K packed-decode cleanup without the broader llama.cpp shape was not a win
- Q6_K MMVQ b64 variants were at best a narrow isolated lm_head win, not enough
  to route production
- Q4_K, Q5_0/Q5_1, and Q5_K MMVQ ports were correct but slower on measured
  target decode shapes
- IQ4_NL expert-down b16 and integer-accumulation variants were slower than the
  default expert-down shader
- a broad async submit attempt needs stable `GpuContext` lifetime plus
  per-submit deferred descriptor-set frees before it can be safe

## Status

Phase 7 is in the optimization/parity phase, not bring-up. The infrastructure
is present, the target model runs on GPU, and the remaining work is mostly
kernel/graph shape plus careful lifetime and submission cleanup.
