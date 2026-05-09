# Roadmap

Each phase is a self-contained educational unit: runnable, tested, and documented in `docs/phases/`.

## Phase 0 — Foundation (current)
- [x] Nix dev shell (zig, zls, perf, gdb, hyperfine)
- [x] Git repo, conventions, CLAUDE.md
- [ ] `scripts/build.sh` convenience wrapper

## Phase 1 — GGUF Parsing ✓
- Parse GGUF v3 header + metadata
- Memory-map model weights (zero-copy)
- CLI: `llmtoy info <model.gguf>` — print model metadata

## Phase 2 — Tokenization
- Load BPE vocab from GGUF
- Encode / decode text
- CLI: `llmtoy tokenize <model.gguf> "hello world"`

## Phase 3 — Naive CPU Inference
- Naive (no SIMD) matmul, RMSNorm, softmax, SiLU
- Single-head attention + FFN forward pass
- Greedy sampling
- Synthetic tiny model tests to validate numerics

## Phase 4 — Full CPU Forward Pass
- Multi-head / grouped-query attention
- KV cache
- Temperature / top-k / top-p sampling
- End-to-end text generation on a real (small) model

## Phase 5 — CPU Optimizations
- AVX2 SGEMM kernel
- Multi-threading (std.Thread.Pool)
- Q4_K / Q8_0 dequant in hot path
- Benchmark each optimization vs phase 4 baseline (hyperfine)

## Phase 6 — MoE Architecture
- Expert routing layer
- Sparse dispatch (top-k experts)
- Target: Qwen3.6 35B A3B or Gemma4 26B A4B

## Phase 7 — GPU Path
- Evaluate Vulkan vs ROCm complexity, pick simpler
- Compute shader for matmul
- Partial GPU offload (attention on GPU, rest CPU)
- Benchmark GPU vs CPU-only

## Phase 8 — Multimodal (stretch)
- SigLIP vision encoder (Gemma4 integration)
- Image token injection into forward pass

---

Target models:
- `/opt/ai-lab/models/mudler/gemma-4-26B-A4B-it-APEX-GGUF/gemma-4-26B-A4B-APEX-I-Mini.gguf`
- `/opt/ai-lab/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-I-Mini.gguf`
