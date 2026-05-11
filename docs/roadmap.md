# Roadmap

Each phase is a self-contained educational unit: runnable, tested, and documented in `docs/phases/`.

## Phase 0 — Foundation ✓
- [x] Nix dev shell (zig, zls, perf, gdb, hyperfine)
- [x] Git repo, conventions, CLAUDE.md
- [ ] `scripts/build.sh` convenience wrapper

## Phase 1 — GGUF Parsing ✓
- Parse GGUF v3 header + metadata
- Memory-map model weights (zero-copy)
- CLI: `llmtoy info <model.gguf>` — print model metadata

## Phase 2 — Tokenization ✓
- Load BPE vocab from GGUF
- Encode / decode text
- CLI: `llmtoy tokenize <model.gguf> "hello world"`

## Phase 3 — Naive CPU Inference ✓
- Naive (no SIMD) matmul, RMSNorm, softmax, SiLU
- Single-head attention + FFN forward pass
- Greedy sampling
- Synthetic tiny model tests to validate numerics

## Phase 4 — Full CPU Forward Pass ✓
- [x] Multi-head / grouped-query attention (GQA)
- [x] RoPE positional embeddings
- [x] KV cache (incremental decoding)
- [x] Temperature / top-k / top-p sampling
- [x] Q5_0 / Q6_K / Q8_0 / Q4_K dequantisation (on-the-fly, one row at a time)
- [x] GGUF weight loader → Config + ModelWeights (Qwen2 attention biases)
- [x] `generate` CLI command with per-token timing
- [x] End-to-end validation: Qwen2.5-0.5B `def add(a,b): return` → `a + b`

## Phase 5 — CPU Optimizations ✓
- [x] AVX2 SIMD dot product (`@Vector(8, f32)`, 4 accumulators) + native CPU target
- [x] Multi-threading: raw Thread.spawn → persistent thread pool (`std.Io.Mutex/Condition`)
- [x] Fused dequant+dot for Q8_0 and Q5_0 (eliminate row_buf round-trip)
- [x] Benchmark each optimization vs phase 4 baseline (hyperfine)

Completed steps (Qwen2.5-0.5B Q4_K_M, Ryzen 3600):
- Step 1: raw Thread.spawn → 1.54× (1.45 → 2.23 tok/s)
- Step 2: AVX2 SIMD + native target → 3.03× (4.40 tok/s 12t)
- Step 3: fused Q8_0 dequant+dot → 14.06 tok/s on Q8_0
- Step 4: fused Q5_0 dequant+dot → 3.32× (4.82 tok/s 1t)
- Step 5: persistent thread pool → **9.72× overall** (14.1 tok/s 12t)

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
