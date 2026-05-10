# Phase 3 — Naive CPU Inference

## Architecture target

Both Qwen3 and Gemma4 are decoder-only transformers of the LLaMA family. Each layer has the same two-part structure:

```
x = x + Attention(RMSNorm(x))
x = x + FFN(RMSNorm(x))
```

The **residual stream** `x` (shape `[d_model]`) carries information from the embedding through every layer to the final output. Every sub-layer reads from it, computes a delta, and adds it back.

## Primitive ops (`src/ops/`)

### `matvec(out, mat, vec, rows, cols)`

The most-called operation in inference. Weight matrices are stored row-major; row `i` is the weight vector for output neuron `i`.

```
out[i] = Σ_j  mat[i,j] * vec[j]
```

All matmuls in a transformer can be decomposed into `matvec` calls (one per output position). SIMD optimisation in Phase 5 targets this loop.

### `rmsnorm(out, x, weight, eps)`

Root-mean-square normalisation — a cheaper alternative to LayerNorm that omits the mean subtraction. Used at the input of every attention and FFN sub-layer.

```
rms  = sqrt(mean(x²) + eps)
out[i] = x[i] / rms * weight[i]
```

The `weight` vector is a learned per-dimension scale stored in the model. `eps` (typically 1e-5 or 1e-6) prevents division by zero.

### `softmax(x)`

In-place, numerically stable: subtract `max(x)` before `exp` so no value overflows to `inf`.

```
x[i] = exp(x[i] − max) / Σ exp(x[j] − max)
```

### `silu(x)`

Sigmoid linear unit, used as the gating activation in SwiGLU:

```
silu(x) = x · sigmoid(x) = x / (1 + exp(−x))
```

## Single-head attention (`src/ops/attn.zig`)

Scaled dot-product attention for a single head over a causal prefix of length `seq_len`:

```
scores[i] = dot(q, k[i]) / sqrt(head_dim)     for i = 0..seq_len-1
weights    = softmax(scores)
out        = Σ_i  weights[i] · v[i]
```

The `1/sqrt(head_dim)` scale prevents scores from growing large when `head_dim` is large, which would saturate softmax and kill the gradients. In inference it's just an empirical detail of how the model was trained.

Causal masking — preventing position `t` from attending to future positions — is enforced by only passing `k[0..t+1]` and `v[0..t+1]` to the function. The caller controls the visible prefix; no explicit `-inf` masking needed.

### Why the failing test was wrong

An early test tried to show attention "concentrating on the best-matching key" using `head_dim=2` and values `v[0]=[99,99]`, `v[1]=[7,7]`. The score difference was only ~0.7 nats, giving attention weights ≈ (0.33, 0.67) — still blending heavily with the large `v[0]`. The fix: `head_dim=1` with `q=10, k=[−10, 10]` gives scores `(−100, 100)`, softmax collapses to `(~0, ~1)`, and the output is within 1e-3 of `v[1]=7`. A good reminder that attention concentration depends on the *ratio* of scores, not their absolute values.

## SwiGLU FFN

Both target models use SwiGLU (Swish-gated linear unit) for the FFN block:

```
gate   = silu(Wgate · x)
up     = Wup · x
hidden = gate ⊙ up          (element-wise product)
out    = Wdown · hidden
```

Three weight matrices instead of two. The gating mechanism lets the network suppress irrelevant dimensions before the expensive `Wdown` projection.

## Forward pass (`src/model/forward.zig`)

Full causal forward pass over a sequence of `T` tokens:

1. **Embedding lookup**: `x[t] = token_emb[token[t]]` — copy the embedding row into the residual stream for each position.
2. **Per-layer loop** (L layers):
   - Precompute `K[t]` and `V[t]` for all positions from the current `x[t]`.
   - For each position `t`, compute `Q[t]`, run `sdpAttn` over `K[0..t+1]`, project the output through `Wo`, add to `x[t]`.
   - Apply SwiGLU FFN to each `x[t]`, add result to residual.
3. **Output**: apply `out_norm` to `x[T−1]`, then `matvec(lm_head, ...)` to get logits. Pass to `greedy()`.

This recomputes all K and V on every call — O(T² · L) attention work. Correct but impractical for long sequences. Phase 4 adds a KV cache that reduces this to O(T · L) per new token.

## Greedy sampling (`src/model/sample.zig`)

Argmax over the logit vector. Deterministic, zero temperature. Phase 4 adds temperature, top-k, and top-p.

## Synthetic test model

Tests use `Config{ vocab_size=16, d_model=8, n_heads=1, n_layers=1, d_ffn=16 }` with all weights initialised to zero except where explicitly set. `setNormWeightsOne()` sets all RMSNorm weight vectors to 1 (identity scale), removing the learned component from normalisation so we can reason about intermediate values.

A forward pass on `tokens=[1, 3]` with `lm_head[5,0]=10` and `token_emb[3,0]=0.1` should produce token 5 as the greedy prediction — the embedding of token 3 activates the large lm_head weight at row 5 through the residual stream.

## What's missing (Phase 4)

- **RoPE**: rotary positional embeddings are needed for real Qwen3/Gemma4 inference. Without them, the model has no positional information.
- **Multi-head / GQA attention**: our single-head implementation doesn't scale to real models (Qwen3 has 16 Q heads and 2 KV heads).
- **KV cache**: required for any practical generation; without it every token requires O(T) recomputation of all previous keys and values.
- **Dequantisation**: real weights are Q4_K, Q3_K, etc. — the forward pass currently requires f32.
- **Sampling**: temperature, top-k, top-p.
