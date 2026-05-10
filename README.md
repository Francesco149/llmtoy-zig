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
zig build test       # run tests
zig build run        # run the binary
```

> First `nix develop` downloads packages; subsequent invocations use the local store — no network hit.

## Roadmap

See [docs/roadmap.md](docs/roadmap.md) for the phase-by-phase plan.

| Phase | Topic | Status |
|-------|-------|--------|
| 0 | Foundation — Nix, project structure | ✓ |
| 1 | GGUF parsing | ✓ |
| 2 | Tokenization | ✓ |
| 3 | Naive CPU inference | ✓ |
| 4 | Full CPU forward pass + sampling | current |
| 5 | CPU optimizations (SIMD, threading, quant) | planned |
| 6 | MoE architecture (Gemma4 / Qwen3.6) | planned |
| 7 | GPU path (Vulkan or ROCm) | planned |
| 8 | Multimodal (stretch) | planned |

## Hardware targets

- Ryzen 5600 / 5900x, 64 GB RAM
- GPU inference via Vulkan (on the 3600 machine)

## License

MIT
