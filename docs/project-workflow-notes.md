# Project Workflow Notes

Persistent notes for future work on this repo.

## Purpose

`llmtoy-zig` is an educational Zig inference engine. Correctness and readability
matter more than peak speed unless a phase is explicitly about optimization.

Dense Qwen-style models are the baseline path. Gemma4/MoE lives in
`src/model/gemma4/` and should stay isolated unless a primitive is genuinely
shared.

## Standard Commands

Use the flake/dev shell.

```sh
nix develop --command zig build -Doptimize=ReleaseFast
nix develop --command zig build test
```

Run expensive generation checks with 12 threads unless debugging threading:

```sh
./zig-out/bin/llmtoy generate <model.gguf> "<prompt>" --max-tokens 8 --threads 12 --temperature 0.1
```

The Gemma4 smoke test that should produce `Paris.`:

```sh
./zig-out/bin/llmtoy generate \
  /opt/ai-lab/models/mudler/gemma-4-26B-A4B-it-APEX-GGUF/gemma-4-26B-A4B-APEX-I-Mini.gguf \
  $'<|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>The capital of France is' \
  --max-tokens 8 --threads 12 --temperature 0.1
```

The same test using the minimal chat-template helper:

```sh
./zig-out/bin/llmtoy generate \
  /opt/ai-lab/models/mudler/gemma-4-26B-A4B-it-APEX-GGUF/gemma-4-26B-A4B-APEX-I-Mini.gguf \
  "What is the capital of France?" \
  --chat --max-tokens 8 --threads 12 --temperature 0.1
```

## Model Paths

Primary Gemma4 target:

```text
/opt/ai-lab/models/mudler/gemma-4-26B-A4B-it-APEX-GGUF/gemma-4-26B-A4B-APEX-I-Mini.gguf
```

Quick dense smoke tests:

```text
/opt/ai-lab/models/lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF/Qwen2.5-0.5B-Instruct-Q8_0.gguf
/opt/ai-lab/models/lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
```

Large dense model:

```text
/opt/ai-lab/models/lmstudio-community/Qwen2.5-Coder-14B-Instruct-GGUF/Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf
```

MoE follow-up target:

```text
/opt/ai-lab/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-I-Mini.gguf
```

Known-good reference source:

```text
/opt/ai-lab/llama.cpp
```

## Gemma4 Facts To Remember

- This GGUF uses `tokenizer.ggml.model = gemma4`.
- Gemma4 tokenizer is raw UTF-8 BPE with spaces normalized to `▁`, not GPT-2
  byte-level `Ġ` tokenization.
- This GGUF's chat template uses `<|turn>` and `<turn|>` control tokens.
- BOS is token 2; EOS is token 1.
- `gemma4.embedding_length_per_layer_input = 0` for the current target, so the
  optional per-layer embedding path in llama.cpp is inactive.
- `gemma4.attention.shared_kv_layers = 0`, so every layer owns KV.
- llama.cpp confirms Gemma4 attention scale is `1.0`.
- V gets raw RMSNorm after projection. For global layers without `attn_v.weight`,
  copy raw K into V before applying K's learned norm.

## Debugging Discipline

1. Check tokenization before changing forward-pass math.
2. Cross-reference llama.cpp for model-family wiring.
3. Keep debug prints out of hot paths once the issue is isolated.
4. Prefer focused smoke tests with `--max-tokens 8 --threads 12 --temperature 0.1`.
5. Dense-model regressions should use the small Qwen2.5 0.5B files first.
6. Run only one model/engine at a time for reference comparisons. CPU inference
   is already slow and memory hungry; parallel model runs can distort timings or
   exhaust RAM.

## Known Follow-Ups

- Add stop-string handling for `<turn|>` and ChatML-style end markers.
- Improve Gemma4 tokenization toward a 1:1 llama.cpp match for punctuation,
  bytes/fallback tokens, and edge-case whitespace.
- Add direct known-good regression comparisons against llama.cpp and eventually
  transformers.
- Add Q3_K dequantization fixtures against ggml output.
- Once correctness regressions are stable, resume CPU/GPU optimization work.
