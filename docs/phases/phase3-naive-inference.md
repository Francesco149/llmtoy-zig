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

### The problem attention solves

Each token in a sequence needs to gather information from other tokens to understand its context. "Bank" means something different in "river bank" vs "savings bank" — the surrounding words resolve the ambiguity. Attention is the mechanism by which a token reaches back into the sequence and pulls in relevant context.

The naive solution would be a fixed weighted average of all previous tokens. Attention makes those weights *content-dependent*: each token decides dynamically, based on what it's currently representing, which past tokens are worth attending to.

### Soft lookup: the database analogy

Imagine a key-value database with fuzzy matching:

```
keys:   ["name",     "age", "city" ]
values: ["Alice",    "30",  "Paris"]
query:  "location"
```

A hard lookup fails — "location" doesn't exactly match any key. A soft lookup computes a similarity score between the query and every key, converts the scores to weights via softmax, and returns the weighted sum of values. "location" is most similar to "city", so the output is mostly "Paris" with a small contribution from the others.

Attention is exactly this, operating on continuous vectors instead of strings. The query, keys, and values are all dense vectors in `ℝ^head_dim`. Dot product measures similarity: `dot(q, k)` is large when the vectors point in similar directions and small (or negative) when they don't.

### Why dot product measures meaning

The dimensions of these vectors are not individually interpretable — you can't point at dimension 847 and say "this one means royalty." Meaning is *distributed*: a concept is encoded as a direction in the full space, spread across many dimensions simultaneously. Individual dimensions look like noise if you inspect them in isolation.

What training produces is a geometry: things that appear in similar contexts get pushed toward similar directions. The dot product then measures how much two vectors overlap directionally — large and positive means pointing the same way (similar meaning/context), near zero means perpendicular (unrelated), negative means opposing.

The classic demonstration is from Word2Vec:

```
vec("king") − vec("man") + vec("woman") ≈ vec("queen")
```

No dimension was designed to mean "royalty" or "gender." The model arranged the space so that the *relationship* between concepts is a consistent direction, and vector arithmetic reflects real-world structure.

A consequence of distributing meaning across dimensions: you can fit far more than `d` distinct concepts into `d` dimensions by using *nearly* orthogonal directions. In 2048 dimensions there are vastly more than 2048 distinguishable directions — the model exploits this, encoding more concepts than it has dimensions by overlapping them at slight angles. Research calls this the superposition hypothesis.

The technically clean similarity measure is cosine similarity — dot product divided by both magnitudes — which measures pure direction regardless of scale. Scaled dot-product attention uses `dot(q,k) / sqrt(head_dim)` instead, a cheaper approximation that works because `Wq` and `Wk` learn to keep vectors at roughly consistent scale during training.

### Q, K, V projections

The residual stream `x` for each token is a single vector that carries everything the model knows about that token so far. Before the dot-product lookup, three separate linear projections transform it into three distinct roles:

```
Q = Wq · x    "what am I looking for?"
K = Wk · x    "how do I present myself as a lookup target?"
V = Wv · x    "what will I contribute if selected?"
```

Separating these three roles is the key design choice. A token can be highly *queryable* (a good key to match against) without having anything useful to *contribute* (its value is not informative), and vice versa. The three weight matrices are learned independently, so the model can specialise each role.

In "The cat sat on the mat", when processing "sat":
- Q encodes "I'm a verb, looking for my subject"
- K for "cat" encodes "I'm a noun that could be a subject"
- V for "cat" encodes the full semantic content of "cat"
- The high Q·K score pulls "cat"'s V into "sat"'s context

### Full computation, traced

With `head_dim = 2` and a 2-token prefix, say positions 0 and 1:

```
q    = [1.0, 0.0]          ← query for the current position
k[0] = [0.9, 0.1]          ← key at position 0 (similar to q)
k[1] = [−0.5, 0.8]         ← key at position 1 (dissimilar)

scores[0] = dot(q, k[0]) / sqrt(2) = 0.90 / 1.41 ≈  0.64
scores[1] = dot(q, k[1]) / sqrt(2) = −0.50 / 1.41 ≈ −0.35

weights = softmax([0.64, −0.35]) ≈ [0.73, 0.27]

v[0] = [1.0, 2.0]
v[1] = [3.0, 4.0]

out = 0.73 · [1.0, 2.0] + 0.27 · [3.0, 4.0]
    = [0.73, 1.46] + [0.81, 1.08]
    = [1.54, 2.54]
```

Position 0 contributes 73% of the output because its key matched the query; position 1 still contributes 27% — attention always produces a weighted blend, never a hard selection (unless scores are extreme).

### Why divide by `sqrt(head_dim)`

For random unit vectors in `ℝ^d`, the expected dot product is 0 but the standard deviation is `sqrt(d)`. With `head_dim = 64` (common in practice), random scores have std ≈ 8. Fed into softmax, scores like `[−12, 3, 15]` produce a near-zero/one distribution with no gradient signal — the model can't learn because softmax is already saturated.

Dividing by `sqrt(head_dim)` brings the scores back to O(1) variance regardless of dimension, keeping softmax in its informative, gradient-friendly range.

### Causal masking

A language model generates tokens left to right, so position `t` must not be able to see positions `t+1, t+2, …` — that would be cheating. In our implementation, the caller enforces this by passing only `k[0..t+1]` and `v[0..t+1]` to `sdpAttn`. No explicit `-inf` masking is needed; future positions simply aren't visible.

In training (processing a full sequence in parallel) the mask is usually applied by adding `-inf` to future positions before softmax. `exp(−inf) = 0` zeroes them out cleanly.

### What the weights learn

After training, the three matrices `Wq`, `Wk`, `Wv` in each head encode a learned relationship type. Different heads in a multi-head model tend to specialise: one head might learn syntactic subject-verb agreement, another might learn coreference (matching pronouns to their referents), another might track positional proximity. No single head is designed for this explicitly — it emerges from training.

The output projection `Wo` then mixes the contributions from all heads back into the residual stream dimension, letting the model combine the different relationship types it has discovered.

### Why the failing test was wrong

An early test tried to show attention "concentrating on the best-matching key" using `head_dim=2` and values `v[0]=[99,99]`, `v[1]=[7,7]`. The score difference was only ~0.7, giving weights ≈ (0.33, 0.67) — still blending heavily with the large `v[0]`. The fix: `head_dim=1` with `q=10, k=[−10, 10]` gives scores `(−100, 100)`, softmax collapses to `(~0, ~1)`, and the output is within 1e-3 of `v[1]=7`. The lesson: concentration depends on the *ratio* of scores (through exp), not their absolute values — and `v[0]=99` being large is irrelevant once its attention weight is near zero.

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
