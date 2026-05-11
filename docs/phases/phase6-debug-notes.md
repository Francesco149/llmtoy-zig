# Phase 6 Debug Notes — Gemma4 Forward Pass

## Status

Gemma4 now generates meaningful output for the target smoke test.

Command:

```sh
./zig-out/bin/llmtoy generate \
  /opt/ai-lab/models/mudler/gemma-4-26B-A4B-it-APEX-GGUF/gemma-4-26B-A4B-APEX-I-Mini.gguf \
  $'<|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>The capital of France is' \
  --max-tokens 8 --threads 12 --temperature 0.1
```

Observed output:

```text
The capital of France is Paris.<turn|>L's capital of
```

The important part is that the first generated content is now `Paris.`. The
trailing text is just the model continuing after the answer because we ask for a
fixed number of tokens and do not yet implement stop-string handling for
`<turn|>`.

## Root Cause

The bad output was primarily a tokenizer/chat-template mismatch, not an MoE or
attention-scale bug.

Gemma4 GGUF metadata says `tokenizer.ggml.model = gemma4`. llama.cpp handles this
as SentencePiece-style BPE:

- spaces are normalized to U+2581 `▁`
- BPE runs over raw UTF-8 text, not GPT-2 byte-encoded text
- text is not split into whitespace-delimited words before BPE; only newlines
  are split specially
- chat turns use `<|turn>` and `<turn|>`, not Gemma 2 style
  `<start_of_turn>` / `<end_of_turn>`

Before the fix, `The capital of France is` tokenized as:

```text
{ 2, 818, 245237, 41626, 245237, 1340, 245237, 31756, 245237, 511 }
tokens: <bos> The Ġ capital Ġ of Ġ France Ġ is
```

After the fix, it tokenizes as:

```text
{ 2, 818, 5279, 529, 7001, 563 }
tokens: <bos> The ▁capital ▁of ▁France ▁is
```

The correct chat-form prompt tokenizes with the model's control tokens:

```text
<|turn> = 105
<turn|> = 106
newline = 107
```

The old documented prompt using `<start_of_turn>` was split as ordinary text and
was not a valid Gemma4 chat template for this GGUF.

## Confirmed Architecture Notes

- `n_layers=30, n_heads=16, d_model=2816, d_ffn=2112`
- MoE: `d_expert=704, n_experts=128, n_experts_used=8`
- SWA layers: 0-4, 6-10, 12-16, 18-22, 24-28
- Global layers: 5, 11, 17, 23, 29
- SWA: `head_dim=256`, `n_kv_heads=8`, `rope_theta=10000`, window=1024
- Global: `head_dim=512`, `n_kv_heads=2`, explicit `rope_freqs.weight`
- `gemma4.attention.shared_kv_layers = 0` for this GGUF, so every layer has its
  own KV cache.
- `gemma4.embedding_length_per_layer_input = 0`, so the optional per-layer
  embedding path in llama.cpp is inactive for this GGUF.
- `logit_softcap = 30.0`
- BOS token = 2, EOS token = 1

llama.cpp cross-checks:

- attention scale is intentionally `1.0` for Gemma4
- V is unweighted RMS-normalized after projection
- global layers omit `attn_v.weight`; llama.cpp uses `Kcur` as `Vcur`, then
  applies unweighted V RMSNorm
- router input is `rmsnormRaw(attn_out) * 1/sqrt(d_model) * ffn_gate_inp.scale`
- dense and MoE FFN outputs are each post-normalized, added, post-normalized
  again, then added to the residual

## Current Implementation Notes

- `src/tokenizer/bpe.zig` keeps the old GPT-style path for Qwen/dense models and
  adds a Gemma4-specific raw UTF-8 BPE path.
- Gemma4 decode maps `▁` back to a literal space.
- Generation now stops on `vocab.eos_token_id` instead of a hard-coded token id.
- `llmtoy info` prints the embedded chat template and key Gemma4 metadata
  (`embedding_length_per_layer_input`, `shared_kv_layers`, BOS/EOS).
- Diagnostic prints were removed from the Gemma4 forward pass hot path after the
  issue was isolated.

## Verification

Passed:

```sh
nix develop --command zig build -Doptimize=ReleaseFast
nix develop --command zig build test
```

Gemma4 smoke test:

```text
prefill: 12338 ms
generated: 8 tokens in 4282 ms (1 tok/s)
output includes: Paris.
```

Dense-model tokenizer smoke test:

```sh
./zig-out/bin/llmtoy tokenize \
  /opt/ai-lab/models/lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF/Qwen2.5-0.5B-Instruct-Q8_0.gguf \
  "The capital of France is"
```

Qwen still uses GPT-style `Ġ` tokens:

```text
{ 785, 6722, 315, 9625, 374 }
tokens: The Ġcapital Ġof ĠFrance Ġis
```

## Follow-Ups

- Add stop-token/stop-string handling for Gemma4 turn delimiters such as
  `<turn|>`.
- Add a direct Q3_K dequantization regression test against ggml output. The
  current model behavior suggests Q3_K is not grossly broken, but it is still a
  high-value test because expert weights use Q3_K heavily.
- Consider adding a minimal chat-template helper instead of requiring users to
  hand-write `<|turn>` prompts.
