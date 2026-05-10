# Phase 4 — Full CPU Forward Pass

## What changed from Phase 3

Phase 3 built a correct but minimal transformer: single-head attention, no positional information, full sequence recomputation on every call. Phase 4 fills in the four missing pieces that real models need:

1. **RoPE** — positional encoding that bakes position into Q and K before attention
2. **Multi-head / GQA** — `n_heads` independent attention heads sharing a single residual stream; grouped-query attention reduces KV head count below Q head count
3. **KV cache** — stores K and V from previous positions so each new token costs O(1) attention work, not O(T²)
4. **Dequantisation** — on-the-fly Q4_K and Q8_0 decode during matmul so model weights stay in their compressed GGUF form

Sampling (temperature, top-k, top-p) is also added, rounding out the full generation loop.

---

## RoPE (`src/ops/rope.zig`)

### Why positional encoding at all

Attention is order-agnostic: the dot product `Q · Kᵀ` is identical whether token A is 3 positions before token B or 300. Without positional information the model can't distinguish "cat sat on mat" from "mat sat on cat."

The classic fix (Transformer 2017) adds position-dependent sine/cosine vectors to the embedding before the first layer. The vectors shift across layers and can't be undone. RoPE takes a cleaner approach: encode position as a *rotation* of Q and K, applied fresh before every attention computation.

### What RoPE does

Q and K are split into consecutive pairs `(x₀, x₁), (x₂, x₃), …`. Each pair is rotated by a position-dependent angle:

```
freq_i  = θ ^ (−2i / head_dim)     — slower for high-index pairs
angle   = pos × freq_i
[x₀, x₁] ← [x₀·cos − x₁·sin, x₀·sin + x₁·cos]
```

`θ` (rope_theta) is typically 500k–1M for modern models (higher = slower frequency decay = longer effective context). Low-index pairs rotate fast (change a lot between adjacent tokens); high-index pairs rotate slowly (encode coarser positional structure).

### Why this gives relative position awareness

When you compute `Qₜ · Kₛ`, the rotation at position `t` and `s` interact inside the dot product:

```
dot(R(t)·q, R(s)·k) = dot(q, R(s−t)·k)
```

The score depends only on `t − s` (the relative distance), not on absolute positions. In principle this should work at any context length. In practice it breaks down — and understanding why explains a lot about modern context-length engineering.

### Why longer contexts degrade quality

Each dimension pair has a characteristic wavelength — the relative distance at which it completes one full rotation cycle:

```
λᵢ = 2π × rope_theta^(2i/d)
```

For `rope_theta=10000` and `d=128`: `λ₀ ≈ 6` tokens, `λ₆₃ ≈ 62,000` tokens. Low-index pairs rotate fast (short wavelength); high-index pairs rotate slowly (long wavelength).

During training with max context length `L`, dimension pair `i` sweeps through angles in the range `[0, 2π × L/λᵢ]`. For fast pairs (`L >> λᵢ`) the model sees many complete cycles and learns the full rotational pattern. For slow pairs (`L << λᵢ`) the model only observes a small arc of the cycle — those dimensions are undertrained.

When inference extends beyond `L`, the slow dimensions start producing angles never seen during training. Attention heads that learned to use those dimensions for coarse positional structure produce garbage, and output quality degrades.

The failure is not that long-range pairs have *smaller* angle deltas per step (they do, by design). It's that the dimensions designed for long-range encoding never complete enough of their cycle during a finite training run to be reliable at inference time.

### Why `rope_theta=1,000,000` fixes this

Raising the base stretches all wavelengths by the same factor. With `rope_theta=1,000,000` the slow end reaches `λ₆₃ ≈ 6.2 billion` tokens, so for a 128K-token training run even the slowest dimension sweeps through `128000 / 6.2B ≈ 0.002%` of its cycle. That sounds worse — but the key is that the *fast* dimensions still rotate rapidly and handle local structure correctly, and the model simply doesn't need the slow dimensions to work at normal context lengths.

The practical fix: train at context length `L`, set `rope_theta` large enough that `λ_{max} >> L`. The slow dimensions contribute almost no rotational signal within the training window (angles stay near zero), but they don't hurt either. At inference beyond `L`, they're still near-zero — the model is still in-distribution.

Fancier approaches (YaRN, NTK-aware scaling) apply *different* stretch factors to different frequency bands, since uniform scaling also slows down the fast dimensions in ways that can hurt short-range attention quality.

### Implementation

`applyRope(vec, pos, theta)` modifies a single head's vector in-place. Called on every Q and every K just after the W_q / W_k projection, before the attention dot product.

---

## Multi-head & GQA

### Multi-head attention

Each attention layer projects the residual stream into `n_heads` separate Q/K/V triples, runs `sdpAttn` independently per head, concatenates the outputs, and projects back through `Wo`:

```
Q_h  = Wq[h] · x                        (shape: [head_dim])
K_h  = Wk[h] · x
V_h  = Wv[h] · x
a_h  = sdpAttn(Q_h, K[0..t], V[0..t])   (shape: [head_dim])
out  = Wo · concat(a_0, …, a_{H-1})     (shape: [d_model])
```

In practice `Wq` is stored as a single matrix `[n_heads × head_dim, d_model]`, so one `matvec` call produces all heads' Q vectors at once.

Different heads specialise. One might learn syntactic agreement (verb ↔ subject), another coreference (pronoun ↔ referent), another local proximity. No head is designed for this; it emerges from training.

### Grouped-query attention (GQA)

Full multi-head attention has the same number of K/V heads as Q heads (`n_kv_heads = n_heads`). With 32 Q heads and a 4096-token context, the KV cache alone is 32 × 4096 × head_dim × 2 × L floats. For large models this exceeds practical memory.

GQA reduces this by having fewer KV heads than Q heads. A group of Q heads (size `n_heads / n_kv_heads`) shares one pair of K/V projections:

```
kv_h = h / (n_heads / n_kv_heads)   — Q head h uses KV head kv_h
```

Qwen3 and Gemma4 both use GQA. Qwen3-0.6B: 16 Q heads, 8 KV heads (group size 2). The KV cache is halved.

### Weight dimensions (updated)

The key dimension change from Phase 3 (where only single-head was correct):

| Matrix | Rows | Cols |
|--------|------|------|
| `wq[l]` | `n_heads × head_dim` | `d_model` |
| `wk[l]` | `n_kv_heads × head_dim` | `d_model` |
| `wv[l]` | `n_kv_heads × head_dim` | `d_model` |
| `wo[l]` | `d_model` | `n_heads × head_dim` |

---

## KV Cache (`src/model/kv_cache.zig`)

### The O(T²) problem

Phase 3 recomputed K and V for all positions on every forward call. Generating a 1000-token response required 1+2+3+…+1000 ≈ 500,000 attention operations instead of 1000. For a 30-layer model this is a 500× slowdown relative to optimal.

### The O(1) per-token solution

The KV cache stores K and V from all previous positions. At position `t`:

1. Compute new K and V for `x[t]`, write them into `cache[l][t]`
2. Run attention using `cache[l][0..t+1]` — no recomputation
3. Each new token costs O(1) attention work (one new row, then O(t) dot products)

Total work for generating T tokens is O(T × t_max) ≈ O(T²) but with a much smaller constant — the prefill of the prompt still pays O(T²) but all generation is O(T) per token.

### Layout

```
k[l]:  [max_seq_len × n_kv_heads × head_dim] f32
v[l]:  [max_seq_len × n_kv_heads × head_dim] f32
```

Access pattern: `k[l][pos * n_kv_heads * head_dim + kv_h * head_dim ..]`

The KV heads for a given layer at all positions are interleaved in memory. When computing attention for Q head `h`, we gather its KV head's rows into a temporary contiguous buffer and pass them to `sdpAttn`.

### Phase 4 generation loop

```
kv = KvCache.init(cfg)

// Prefill: process all prompt tokens, filling the cache
for (prompt_ids, pos) -> logits:
    forwardOneModel(token, pos, &kv, ...)

// Autoregressive generation
while not EOS:
    next = sample(logits, params)
    logits = forwardOneModel(next, pos, &kv, ...)
    pos += 1
    emit(decode(next))
```

---

## On-the-fly Dequantisation

### The memory problem with eager f32 decoding

The target models have billions of parameters. Even at Q4_K (0.5 bytes/param), Qwen3.6-35B is ~17 GB on disk. Decoded to f32 (4 bytes/param) it would be ~140 GB — beyond our 64 GB RAM.

The solution: never fully materialise f32 weights. Instead, dequantize one row of each matrix immediately before computing its dot product with the input vector.

### `quantMatvec` (`src/ops/math.zig`)

```
for each output row i:
    decode row i from raw bytes → row_buf[cols]
    out[i] = dot(row_buf, vec)
```

Peak extra memory: `O(max(cols))` — one decoded row at a time, immediately consumed. The raw quantized data lives in the mmap'd GGUF file; `row_buf` is a stack-sized scratch allocated once per `forwardOneModel` call.

### Q8_0 format (34 bytes per 32 elements)

```
[f16 scale] [i8 × 32]
val[i] = scale × qs[i]
```

Simple and fast to decode. Used for embeddings and output projections in some models.

### Q4_K format (144 bytes per 256 elements)

```
[f16 d] [f16 dmin] [u8 × 12 scales] [u8 × 128 qs]
```

256-element super-block divided into 8 sub-blocks of 32. Each sub-block has a 6-bit scale `sc` and 6-bit min `mn` packed into the 12-byte `scales` array:

```
sub < 4: sc = scales[sub] & 0x3F,  mn = scales[sub+4] & 0x3F
sub ≥ 4: sc = (scales[sub+4] & 0xF) | ((scales[sub-4] >> 6) << 4)
          mn = (scales[sub+4] >> 4)  | ((scales[sub  ] >> 6) << 4)
```

4-bit values are packed two per byte, interleaved by 64-element chunk:
- `qs[j/2 + k]` lo nibble → value at `j + k` (first 32 of the 64-element chunk)
- `qs[j/2 + k]` hi nibble → value at `j + k + 32` (second 32)

Dequant: `val = d × sc × q4 − dmin × mn`

The k-quants format was designed so that sub-block scales can represent both positive (normal) and negative (min-shifted) quantization grids, significantly improving quality at low bit widths.

---

## Sampling (`src/model/sample.zig`)

Three independent filters applied before sampling:

### Temperature

Divides logits by `T` before softmax. `T < 1` sharpens the distribution (more deterministic, closer to greedy). `T > 1` flattens it (more random, more diverse). `T = 0` is treated as greedy (argmax).

### Top-k

After softmax, zero out all tokens except the k highest-probability ones. Prevents pathological low-probability tokens from ever being sampled. `k = 40` is a common default.

### Top-p (nucleus sampling)

Sort tokens descending by probability. Keep adding tokens to the "nucleus" until the cumulative probability reaches `p`. Zero out all tokens outside the nucleus. Adapts the effective k to the shape of the distribution: when the model is confident (few high-prob tokens), the nucleus is small; when uncertain (flat distribution), the nucleus grows.

The three filters compose: temperature → softmax → top-k → top-p → renormalize → multinomial sample.

---

## ModelWeights and the GGUF Loader

### Design

Norm weights (d_model elements per layer, trivially small) are dequantised to f32 at load time. Matrix weights (`wq`, `wk`, `wv`, `wo`, `w_gate`, `w_up`, `w_down`, `lm_head`, `token_emb`) are held as `RawMatrix` — a raw byte slice pointing directly into the mmap'd GGUF file plus the ggml type tag. No copy, no decode at load time.

### Tensor naming convention (GGUF standard)

```
token_embd.weight           — token embeddings
output_norm.weight          — final RMSNorm
output.weight               — LM head (may be tied to token_embd)
blk.{l}.attn_norm.weight    — attention RMSNorm
blk.{l}.attn_q.weight       — Q projection
blk.{l}.attn_k.weight       — K projection
blk.{l}.attn_v.weight       — V projection
blk.{l}.attn_output.weight  — output projection (Wo)
blk.{l}.ffn_norm.weight     — FFN RMSNorm
blk.{l}.ffn_gate.weight     — SwiGLU gate
blk.{l}.ffn_up.weight       — SwiGLU up
blk.{l}.ffn_down.weight     — SwiGLU down
```

### configFromGguf

Reads the architecture-prefixed metadata keys (`{arch}.embedding_length`, `.block_count`, `.attention.head_count`, `.attention.head_count_kv`, `.feed_forward_length`, `.rope.freq_base`, etc.) to build the Config without any hard-coded model knowledge.

---

## What's missing (Phase 5)

- **MoE routing**: Qwen3.6 and Gemma4 use Mixture-of-Experts FFN layers. The current loader maps `ffn_gate/up/down` but MoE models have dozens of expert FFNs and a learned router. Phase 6 implements this.
- **SIMD matmul**: `quantMatvec` is naive scalar. Phase 5 introduces AVX2 kernels that process 8 f32s per instruction, giving ~4–8× throughput on the matmul-dominated attention and FFN paths.
- **Multi-threading**: each layer's token can be processed in parallel using `std.Thread.Pool`. Phase 5 adds this.
- **Batched prefill**: `forwardOneModel` processes one token at a time even during prefill, which is suboptimal. Phase 5 will add a batched prefill path.
- **BF16 support**: some models store weights as BF16. The dequant module needs a BF16 → f32 converter.
