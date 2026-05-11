# llmtoy-zig

An educational LLM inference engine written in [Zig](https://ziglang.org/), built incrementally as a learning project. The goal is to understand both Zig and LLM inference from the ground up — starting from the simplest possible implementation and progressing toward a performant, hardware-aware inference engine targeting Gemma 4 and Qwen3.6 MoE models on Linux x86_64.

> **AI-assisted project**: development is done in collaboration with [Claude](https://claude.ai). Commits are co-authored accordingly.

## Goals

- Learn Zig and low-level performance programming hands-on
- Understand LLM inference architecture by building it step by step
- Each phase is a standalone educational unit: runnable, tested, documented
- Target: run quantized MoE models (Gemma 4 26B A4B, Qwen3.6 35B A3B) locally on consumer hardware
- CPU path first, then Vulkan GPU path

## Quickstart

```sh
nix develop          # enter the dev shell (zig, zls, perf, gdb, hyperfine)
zig build            # compile
zig build test       # run all unit tests
```

> `zig` is only on PATH inside the Nix dev shell. First run downloads packages; subsequent runs use the local store.

## Usage

All commands live in `zig-out/bin/llmtoy`. Pass a GGUF model file as the first argument.

```sh
# Inspect a model's architecture and quantization breakdown
llmtoy info model.gguf

# Tokenize text and show token IDs + decoded pieces
llmtoy tokenize model.gguf "hello, world"

# Generate text (greedy, 20 tokens)
llmtoy generate model.gguf "The capital of France is" \
    --max-tokens 20 --temperature 0.0

# Generate with sampling (temperature + nucleus)
llmtoy generate model.gguf "def fibonacci(n):" \
    --max-tokens 60 --temperature 0.8 --top-p 0.9 --top-k 40

# Wrap the prompt in the model's minimal chat template before generation
llmtoy generate model.gguf "What is the capital of France?" \
    --chat --max-tokens 8 --temperature 0.1 --threads 12

# Flags:  --chat   --max-tokens N   --temperature T   --top-p P   --top-k K   --seed S   --threads N
```

The `generate` command writes the completed text to stdout and progress/timing to stderr:

```
loading model.gguf...
  layers=24 heads=14/2 d_model=896 d_ffn=4864 vocab=151936
prefilling 5 prompt tokens...
  prefill: 20314 ms
The capital of France is Paris.
  generated: 4 tokens in 16341 ms (0 tok/s)
```

## Roadmap

See [docs/roadmap.md](docs/roadmap.md) for the phase-by-phase plan.

| Phase | Topic | Status |
|-------|-------|--------|
| 0 | Foundation — Nix, project structure | ✓ |
| 1 | GGUF parsing | ✓ |
| 2 | Tokenization | ✓ |
| 3 | Naive CPU inference | ✓ |
| 4 | Full CPU forward pass + sampling | ✓ |
| 5 | CPU optimizations (threading + AVX2) | current (threading done) |
| 6 | MoE architecture (Gemma4 / Qwen3.6) | planned |
| 7 | GPU path (Vulkan or ROCm) | planned |
| 8 | Multimodal (stretch) | planned |

## Hardware targets

- Ryzen 5600 / 5900x, 64 GB RAM
- GPU inference via Vulkan (on the 3600 machine)

## License

MIT
