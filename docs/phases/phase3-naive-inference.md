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

Root-mean-square normalisation — used at the input of every attention and FFN sub-layer.

```
rms     = sqrt(mean(x²) + eps)
out[i]  = x[i] / rms * weight[i]
```

The `weight` vector is a learned per-dimension scale stored in the model. `eps` (typically 1e-5 or 1e-6) prevents division by zero.

**Relationship to L2 normalisation.** Plain L2 normalisation divides by the vector's magnitude to produce a unit vector:

```
||x||  = sqrt(Σ xᵢ²)
out[i] = x[i] / ||x||          → magnitude always = 1
```

RMS is just the magnitude divided by `sqrt(n)`:

```
rms(x) = sqrt(Σ xᵢ² / n) = ||x|| / sqrt(n)
```

So with all-ones weights, `out[i] = x[i] / rms = x[i] * sqrt(n) / ||x||` — the same *direction* as L2-normalised, scaled by `sqrt(n)`. With `x = [1, 2, 3, 4]` and `w = [1, 1, 1, 1]`:

```
||x||   = sqrt(30) ≈ 5.477
rms(x)  = sqrt(30/4) ≈ 2.739

L2-norm:  [0.183, 0.365, 0.548, 0.730]   magnitude = 1.0
RMSNorm:  [0.365, 0.730, 1.095, 1.461]   magnitude = sqrt(4) = 2.0
          — same direction, exactly sqrt(n) larger
```

At Qwen3's `d_model = 2048`, the RMSNorm output has magnitude `sqrt(2048) ≈ 45`. The learned `weight` vector then re-scales each dimension independently; after training it encodes which dimensions should be amplified or suppressed.

**Relationship to LayerNorm.** LayerNorm additionally subtracts the mean before normalising:

```
LayerNorm: out[i] = (x[i] − mean(x)) / std(x) * w[i] + b[i]
```

The mean subtraction matters when the residual stream accumulates a large constant offset across dimensions. With `x = [100, 101, 102]`:

```
L2-norm / RMSNorm:  ≈ [0.570, 0.576, 0.581]  — constant offset dominates, tiny spread
LayerNorm:            [−1.22, 0.0,  +1.22]   — mean stripped, differences preserved
```

RMSNorm's authors found empirically that the mean subtraction rarely matters for language models (the residual stream doesn't accumulate large constant shifts in practice), so they dropped it — one fewer operation, no bias parameter `b`, same downstream quality.

### `softmax(x)`

Converts an arbitrary vector of real numbers into a probability distribution: all values become positive and sum to 1.

```
softmax(x)[i] = exp(x[i]) / Σ exp(x[j])
```

**Why exp causes extreme skew.** Linear differences become exponential ratios. If two logits differ by `d`, their softmax weights differ by a factor of `exp(d)`:

```
x = [1, 2, 3]
exp(x) = [2.72, 7.39, 20.09]   — each step is e≈2.72× larger
softmax = [0.090, 0.245, 0.665] — values 1 apart linearly, 7× apart in probability
```

Add one outlier and the rest effectively vanish:

```
x = [1, 2, 10]
exp(x) = [2.72, 7.39, 22026]
softmax ≈ [0.00012, 0.00033, 0.99955]
```

Equal inputs give a uniform distribution — the only case where softmax is not skewed:

```
x = [3, 3, 3]
softmax = [0.333, 0.333, 0.333]
```

**Numerical stability.** `exp(1000)` overflows to `inf`. The fix: subtract `max(x)` before applying `exp`. This doesn't change the output because the constant cancels between numerator and denominator:

```
exp(x[i] − c) / Σ exp(x[j] − c)  =  exp(x[i]) / Σ exp(x[j])   for any c
```

Setting `c = max(x)` guarantees the largest input to `exp` is 0, so `exp(0) = 1` is the ceiling.

**Connection to greedy decoding.** Since `exp` is monotone, `argmax(softmax(x)) = argmax(x)` — the greedy token is just the highest logit. Softmax itself isn't needed for greedy; it becomes necessary in Phase 4 for nucleus (top-p) sampling, where we accumulate probability mass rather than just taking the peak.

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
