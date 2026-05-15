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

Next: profiling to identify actual bottleneck (CPU accumulation? attention path? PCIe?)

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
