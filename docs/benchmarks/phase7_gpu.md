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

## Notes

- Modest speedup because MoE expert compute (128 experts, 8 active/token) still runs on CPU
- GPU dispatch overhead for small-ish matrices limits gains
- w_down (Q5_1) not GPU-supported, falls back to CPU
- systemd-run --scope -p MemoryMax=40G required for safe testing

## Bugs fixed during GPU bringup

1. `tensorBytes()` returned `mmap[offset..]` (full file tail) instead of
   `mmap[offset..offset+size]` — caused 13 GB staging allocations per tensor → OOM.
   Fixed by adding `TensorInfo.byteSize()` with correct GGUF block-size tables.

2. Double-free of `last_logits` when EOS token sampled in generation loop.
   Fixed by setting `last_logits = &.{}` on EOS break and guarding the outer free.
