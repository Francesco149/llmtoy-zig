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

## Phase 7c — MoE experts on GPU (per-expert dispatch, abandoned)

Uploaded all 128 experts × 30 layers to VRAM (~9.6 GiB) and dispatched each
expert matmul individually (8 experts × 3 matmuls × 30 layers = 720 GPU submits/token).

| Mode | tok/s prefill | tok/s decode |
|------|--------------|--------------|
| GPU (attn + dense FFN, experts on CPU) | 2.58 | 2.44 |
| GPU (attn + dense FFN + all experts)   | 1.61 | 1.52 |

Per-expert dispatch is **slower**: 720 sequential `vkQueueSubmit+vkQueueWaitIdle`
per token dominate over the compute savings. Expert upload reverted.
Next: batched dispatch — all 8 active experts in one command buffer per layer.

## Notes

- Modest speedup because MoE expert compute (128 experts, 8 active/token) still runs on CPU
- GPU dispatch overhead for small-ish matrices limits gains
- systemd-run --scope -p MemoryMax=40G required for safe testing
- All benchmarks use `zig build -Doptimize=ReleaseFast` (default since phase 7c)

## Bugs fixed during GPU bringup

1. `tensorBytes()` returned `mmap[offset..]` (full file tail) instead of
   `mmap[offset..offset+size]` — caused 13 GB staging allocations per tensor → OOM.
   Fixed by adding `TensorInfo.byteSize()` with correct GGUF block-size tables.

2. Double-free of `last_logits` when EOS token sampled in generation loop.
   Fixed by setting `last_logits = &.{}` on EOS break and guarding the outer free.
