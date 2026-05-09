# llmtoy-zig — Agent Instructions

Educational LLM inference engine built incrementally in Zig.

## What this is

Step-by-step inference implementation targeting:
- **Models**: Gemma 4 26B A4B (APEX-I-Mini) and Qwen3.6 35B A3B (APEX-I-Mini), both GGUF format
- **Hardware**: Ryzen 3600 / 5900x, 64 GB RAM, consumer GPU (Vulkan, on the 3600 machine)
- **OS**: Linux x86_64

Each phase is an independent educational unit — old paths stay runnable and tested forever.

## Project layout

```
src/              Zig source (keep files small and decoupled)
docs/roadmap.md   Phase-by-phase plan — check here for current phase
docs/phases/      Per-phase write-ups and notes
docs/benchmarks/  Benchmark results per optimization step
scripts/          Build wrappers, benchmark runners
```

Root stays clean — no extra files at the top level.

## Build & test

```sh
nix develop          # enter dev shell (once per terminal)
zig build            # compile
zig build run        # run
zig build test       # unit tests
```

All CI-equivalent checks must pass before committing: `zig build test`.

## Conventions

- **Files**: single-purpose, small. Split early rather than growing large files.
- **Comments**: only write one when the WHY is non-obvious. No what/how narration.
- **Tests**: write regression tests for every path as we add it. Old paths get integration tests so we can verify they still work as we restructure.
- **Benchmarks**: document with hyperfine. Store results in `docs/benchmarks/` so we can track the progression.
- **Commits**: co-author with Claude (`Co-Authored-By: Claude <noreply@anthropic.com>`).
- **Phases**: each phase has its own doc in `docs/phases/phaseN-*.md` explaining the concepts and implementation choices.

## Philosophy

1. Simplest working implementation first, then optimize.
2. Each optimization is its own educational unit with measured before/after.
3. Modularity first — we swap backends in and out. CPU naive → CPU SIMD → GPU.
4. Fast compilation is a feature. Keep compile times visible and fast.
5. No premature abstractions. Three similar lines beats a bad abstraction.

## Current phase

**Phase 0 — Foundation.** See `docs/roadmap.md` for the full plan.

Next: Phase 1 — GGUF parsing (`src/gguf/`).

## Model paths

```
/opt/ai-lab/models/mudler/gemma-4-26B-A4B-it-APEX-GGUF/gemma-4-26B-A4B-APEX-I-Mini.gguf
/opt/ai-lab/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-I-Mini.gguf
```
