# Phase 6 — Gemma4, MoE, and Architecture-Specific Forward Passes

Phase 3 gave us the smallest useful transformer:

```text
x = x + Attention(RMSNorm(x))
x = x + FFN(RMSNorm(x))
```

Phase 4 turned that into a real decoder-only forward pass: RoPE, multi-head
attention, GQA, a KV cache, quantized matrix-vector multiplies, and sampling.

Phase 6 is where the "LLaMA-like" shorthand stops being enough. Gemma4 still has
the same broad skeleton, but many details are different enough that a generic
forward pass produces plausible-looking numbers and bad text. The model does not
explode, and the logits are finite, but the learned computation is wired
incorrectly.

The lesson: once the basic transformer pieces work, inference becomes a problem
of faithfully reproducing a model family's exact wiring.

## The High-Level Shape

Gemma4 keeps the same residual-stream idea from earlier phases. The residual
stream `x` is still the main highway through the network. Each layer reads it,
computes a delta, and adds the delta back.

What changes is that both the attention block and FFN block are more specialised:

```text
token id
  │
  ▼
token embedding × sqrt(d_model)
  │
  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ repeat for each Gemma4 layer                                        │
│                                                                     │
│   x                                                                 │
│   │                                                                 │
│   ├─ attn_norm ── Q/K/V ── Q/K/V norms ── RoPE ── attention ── Wo ─┐│
│   │                                                               ││
│   └────────────────────── + post_attention_norm(attn_out) ◀───────┘│
│   │                                                                 │
│   ▼                                                                 │
│   attn_residual                                                     │
│   │                                                                 │
│   ├─ dense FFN path ────────────────────────┐                       │
│   │   ffn_norm → GELU gate × up → down      │                       │
│   │   → post_ffw_norm_1                     │                       │
│   │                                         │                       │
│   ├─ MoE path ──────────────────────────────┤                       │
│   │   router chooses top-k experts          │                       │
│   │   pre_ffw_norm_2 → selected experts     │                       │
│   │   → weighted sum → post_ffw_norm_2      │                       │
│   │                                         │                       │
│   └─ add dense + MoE → post_ffw_norm ───────┘                       │
│   │                                                                 │
│   └─ residual add, then layer_output_scale                          │
└─────────────────────────────────────────────────────────────────────┘
  │
  ▼
output_norm → lm_head → logit softcap → sampling
```

The important difference from Phase 4 is that "the FFN" is no longer one block.
Gemma4 has a shared dense FFN and a sparse expert FFN in every layer. Both read
the same post-attention residual stream. Both produce model-width vectors. The
model normalizes them separately, adds them, normalizes the combination, then
adds that back to the residual stream.

## Gemma4 Configuration From GGUF

For the target model:

```text
d_model       = 2816
n_layers      = 30
n_heads       = 16
d_ffn         = 2112      shared dense FFN hidden size
n_experts     = 128
n_used        = 8         selected experts per token
d_expert      = 704       each expert hidden size
vocab         = 262144
```

Gemma4 has two attention layer types:

```text
SWA layers:     0-4, 6-10, 12-16, 18-22, 24-28
global layers:  5, 11, 17, 23, 29
```

SWA means sliding-window attention. Instead of attending to the full prefix,
these layers attend only to the most recent window:

```text
current pos = 1500, window = 1024

visible positions:
477 ........................................ 1500
^                                            ^
oldest visible                              current
```

Global layers attend to the whole prefix. A useful intuition is that SWA layers
handle local texture and syntax, while the occasional global layers let
long-range information move across the sequence.

This is why [kv_cache.zig](/opt/ai-lab/llmtoy-zig/src/model/gemma4/kv_cache.zig)
stores a per-layer capacity:

```zig
const cap = if (cfg.is_swa[l])
    @min(max_seq, cfg.sliding_window)
else
    max_seq;
```

For SWA, the cache is a circular buffer. Once the window is full, position
`pos + 1024` reuses the slot from `pos`. That is fine because the older token is
no longer visible to that layer.

## Attention Differences

Phase 4 attention looked like this:

```text
RMSNorm(x)
  ├─ Wq → Q → RoPE
  ├─ Wk → K → RoPE → KV cache
  └─ Wv → V        → KV cache

attention(Q, cached K, cached V) → Wo → residual add
```

Gemma4 adds more normalization around Q/K/V and after attention:

```text
attn_norm(x)
  ├─ Wq → per-head RMSNorm(q_norm) → RoPE
  ├─ Wk → per-head RMSNorm(k_norm) → RoPE → KV cache
  └─ Wv → raw RMSNorm(no weight)          → KV cache

attention scale = 1.0
attention output → Wo → post_attention_norm → residual add
```

The new primitive is **raw RMSNorm**:

```text
out[i] = x[i] / rms(x)
```

This is the same as RMSNorm from Phase 3, but without a learned weight vector.
It preserves direction and fixes magnitude. For V, that means every value head
contributes at a controlled scale before attention blends values together.

Gemma4 also uses attention scale `1.0`, not `1/sqrt(head_dim)`. In Phase 3 we
explained the usual scaling as a way to keep dot products from saturating
softmax. Gemma4's per-head Q/K RMSNorm changes that calibration. The learned
`attn_q_norm.weight` and `attn_k_norm.weight` become part of the score-scale
control, so the architecture uses no extra pre-softmax division.

### Global Attention Oddity: V Can Be K

Some Gemma4 global layers omit `attn_v.weight`. In that case llama.cpp sets
`Vcur = Kcur` before K normalization. The sequence is:

```text
Kraw = Wk · x
Vraw = Kraw
K    = rmsnorm(Kraw, attn_k_norm)
V    = rmsnormRaw(Vraw)
```

This is easy to get subtly wrong. If you copy K after applying `attn_k_norm`,
then V gets the learned K scale too, which the model did not train with.

## RoPE: Two Flavors In One Model

SWA layers use computed RoPE frequencies with `rope_theta_swa = 10000`.
Global layers use explicit `rope_freqs.weight` from the GGUF. Both layer types
use GPT-NeoX-style pairing, which rotates across the two halves of a head:

```text
standard pair layout:  (0,1), (2,3), (4,5), ...
NeoX pair layout:      (0,h), (1,h+1), (2,h+2), ... where h = head_dim / 2
```

This pairing is not a cosmetic detail. It changes which coordinates are mixed
by the positional rotation. A model trained with NeoX pairing will still produce
finite activations if consecutive-pair RoPE is used, but attention positions are
wrong and generation can drift into repetition.

Phase 4's RoPE function computes:

```text
freq_i = theta ^ (-2i / head_dim)
```

Gemma4 global layers can instead load `freq_i` directly:

```zig
rope_mod.applyRopeFreqsNeox(q_head, w.rope_freqs, pos);
```

Conceptually this is the same rotation math. The difference is where the
frequency table comes from. For an educational engine, supporting both is useful:
one path teaches the formula; the other path teaches that real model files
sometimes store architecture-specific constants directly.

## The FFN Becomes Two FFNs

The Phase 3/4 FFN was a SwiGLU block:

```text
ffn_norm(x)
  ├─ Wgate → activation
  ├─ Wup   ──────────── multiply elementwise
  └─ Wdown ← hidden vector
```

Gemma4's dense path is similar but uses GELU as the gate activation:

```text
dense_in = rmsnorm(attn_residual, ffn_norm)
gate     = Wgate · dense_in
up       = Wup   · dense_in
hidden   = GELU(gate) * up
dense    = Wdown · hidden
dense    = rmsnorm(dense, post_ffw_norm_1)
```

GELU is a smoother gate than ReLU and a different shape than SiLU:

```text
GELU(x) ≈ x * Φ(x)
```

where `Φ(x)` is the normal distribution CDF. Intuitively, GELU says "let strong
positive matches through, suppress negative matches, and keep a soft transition
around zero." In code we use the common tanh approximation in
[math.zig](/opt/ai-lab/llmtoy-zig/src/ops/math.zig).

## MoE: Conditional FFN Compute

Mixture-of-Experts is easiest to understand as replacing one very wide FFN with
a library of smaller FFNs plus a router:

```text
token state
   │
   ├─ router: "which experts should handle this token?"
   │
   └─ selected experts run their own FFNs
```

In a dense FFN, every hidden neuron gets evaluated for every token. In MoE, the
model has many expert FFNs, but only a few run for each token. Gemma4 has 128
experts and uses the top 8 for each token.

Database analogy: attention is a soft lookup over previous tokens. MoE is a soft
dispatch to specialist functions. A token that looks like code may route to
different experts than a token that looks like geography or dialogue formatting.
The router is learned, so the specialties are not named by humans, but the
computation is still conditional.

### Router Computation

Gemma4 routes on the post-attention residual stream, not on the already
expert-normalized input:

```text
router_in = rmsnormRaw(attn_residual)
router_in *= 1 / sqrt(d_model)
router_in *= ffn_gate_inp.scale

router_logits = ffn_gate_inp.weight · router_in
router_probs  = softmax(router_logits)
top8          = largest router_probs
```

The `1/sqrt(d_model)` factor is the same kind of variance control we saw in
attention. The router weight matrix has `n_experts` rows, one score per expert.
Softmax turns those scores into mixture weights.

### Expert Computation

Each expert is another gated FFN, but with smaller hidden dimension:

```text
moe_in = rmsnorm(attn_residual, pre_ffw_norm_2)

for each expert e in top8:
    gate_e   = Wgate_e · moe_in
    up_e     = Wup_e   · moe_in
    hidden_e = GELU(gate_e) * up_e
    out_e    = Wdown_e · hidden_e

    moe_out += router_probs[e] * down_exps.scale[e] * out_e

moe_out = rmsnorm(moe_out, post_ffw_norm_2)
```

The expert weights are packed as 3-D GGUF tensors:

```text
ffn_gate_up_exps.weight: [d_model, 2*d_expert, n_experts]
ffn_down_exps.weight:    [d_expert, d_model, n_experts]
```

The loader exposes those as a flat matrix. The forward pass computes the byte
offset for expert `e` and then runs normal quantized matvecs on that slice.

```zig
const gu_per_expert = 2 * cfg.d_expert * gu_row_bytes;
const gate_data = lw.gate_up_exps.data[eidx * gu_per_expert ..];
const up_data = gate_data[cfg.d_expert * gu_row_bytes ..];
```

This keeps the implementation educational: no sparse batching yet, just "pick
the expert's rows and multiply."

## Combining Dense And MoE Paths

Gemma4 does not choose between the dense FFN and MoE. It uses both:

```text
dense = shared dense FFN(attn_residual)
moe   = selected experts(attn_residual)

combined = dense + moe
combined = rmsnorm(combined, post_ffw_norm)
x        = attn_residual + combined
x        = x * layer_output_scale
```

The dense FFN is like a shared baseline computation every token gets. The MoE
path is extra conditional capacity. The final norm makes their sum a controlled
delta before it re-enters the residual stream.

`layer_output_scale` is a learned scalar per layer. A residual network is a long
chain of additions; small scalar gates help keep the whole chain numerically
well-behaved. Think of it as a volume knob on each layer's final output.

## Logit Soft-Capping

After the final `lm_head`, Gemma4 applies:

```text
logit = tanh(logit / cap) * cap
```

with `cap = 30`.

This is not a sampling temperature. It happens before sampling and changes the
model's logits. The purpose is to prevent extremely large logits from dominating
too hard. Values near zero are almost unchanged because `tanh(x) ≈ x` for small
`x`; very large positive or negative values asymptotically approach `±cap`.

```text
raw logit:      -100   -30   -5    0    5    30    100
soft-capped:     -30   -23   -5    0    5    23     30
```

It is a smooth limiter, not a hard clamp.

## Tokenization Was The Actual Bug

The forward pass can be mostly correct and still fail if the token IDs are wrong.
That is what happened here.

Gemma4 uses a BPE tokenizer, but not GPT-2 byte-level BPE. It is
SentencePiece-style BPE:

```text
"The capital of France is"
```

Wrong GPT-style tokenization:

```text
<bos> The Ġ capital Ġ of Ġ France Ġ is
```

Correct Gemma4 tokenization:

```text
<bos> The ▁capital ▁of ▁France ▁is
```

The visual difference looks minor. It is not minor to the model. Embedding rows
are the model's input vocabulary. Feeding `Ġ` plus `capital` is like giving the
model two different words than the one it trained on.

The same applied to chat formatting. This GGUF uses:

```text
<|turn>user
What is the capital of France?<turn|>
<|turn>model
<|channel>thought
<channel|>
```

not:

```text
<start_of_turn>user
...
<end_of_turn>
```

After the tokenizer path and prompt format matched Gemma4, the model generated
`Paris.` for the target factual smoke test.

The standalone template at `/opt/ai-lab/templates/new-chat-template-gemma.jinja`
matches the relevant simple-chat behavior from the GGUF template:

- user messages render as `<|turn>user\n{content}<turn|>\n`
- assistant generation starts with `<|turn>model\n`
- when thinking is disabled, an empty thought channel is inserted:
  `<|channel>thought\n<channel|>`

That empty thought channel is not an answer prefix; it tells the model the
reasoning section is already closed, so visible answer text should begin after
`<channel|>`.

## What Phase 6 Adds To The Mental Model

Phase 3 taught the universal transformer primitives:

```text
matvec, RMSNorm, softmax, attention, FFN
```

Phase 4 taught the practical decoder machinery:

```text
RoPE, multi-head attention, GQA, KV cache, quantized matvec
```

Phase 6 adds architecture fidelity:

```text
per-head Q/K/V normalization
sliding-window plus global attention
explicit RoPE frequency tensors
dense FFN plus sparse MoE FFN
router softmax plus top-k expert dispatch
layer output scales
logit soft-capping
model-specific tokenization and chat templates
```

The core inference loop is still recognisable. The hard part is no longer
understanding what attention is; it is making every small model-specific choice
line up with training.

## Practical Debugging Lessons

Finite activations are not enough. The broken run had reasonable RMS values and
no NaNs, but generated induction-like output because the prompt token IDs were
wrong.

Cross-check architecture details against a known-good engine before changing
math. In this case llama.cpp confirmed that Gemma4 really does use attention
scale `1.0` and raw V RMSNorm, so those suspicious-looking details were correct.

Always verify tokenization early. A one-line `tokenize` command can save hours
of forward-pass debugging.

Keep architecture-specific code isolated. The Gemma4 path lives under
[src/model/gemma4](/opt/ai-lab/llmtoy-zig/src/model/gemma4/), while the dense
Qwen path stays simple. That keeps the educational version readable and makes it
clear which ideas are universal and which are model-family details.

Gemma4-specific CPU optimization notes live in
[phase6-gemma4-cpu-optimizations.md](/opt/ai-lab/llmtoy-zig/docs/phases/phase6-gemma4-cpu-optimizations.md).
