const std      = @import("std");
const Config   = @import("config.zig").Config;
const Weights  = @import("weights.zig").Weights;
const KvCache  = @import("kv_cache.zig").KvCache;
const mw       = @import("model_weights.zig");
const math     = @import("../ops/math.zig");
const attn_mod = @import("../ops/attn.zig");
const rope     = @import("../ops/rope.zig");

// ── batch forward (no KV cache) — used for tests ─────────────────────────────

/// Full causal forward pass over a token sequence.
///
/// Supports any n_heads / n_kv_heads (GQA when n_kv_heads < n_heads).
/// Recomputes K/V for every layer from scratch — O(T² · L) work.
/// Suitable for testing; forwardOne with a KvCache is used for generation.
///
/// Returns the logit vector for the next token (caller owns the slice).
pub fn forward(
    tokens: []const u32,
    w: *const Weights,
    cfg: Config,
    allocator: std.mem.Allocator,
) ![]f32 {
    const T   = tokens.len;
    const d   = cfg.d_model;
    const hd  = cfg.headDim();
    const nq  = cfg.n_heads    * hd;
    const nkv = cfg.n_kv_heads * hd;
    const ff  = cfg.d_ffn;

    const x = try allocator.alloc(f32, T * d);
    defer allocator.free(x);

    const xb      = try allocator.alloc(f32, d);
    defer allocator.free(xb);
    const q_all   = try allocator.alloc(f32, T * nq);
    defer allocator.free(q_all);
    const ks      = try allocator.alloc(f32, T * nkv);
    defer allocator.free(ks);
    const vs      = try allocator.alloc(f32, T * nkv);
    defer allocator.free(vs);
    const attn_concat = try allocator.alloc(f32, nq);
    defer allocator.free(attn_concat);
    const ks_head = try allocator.alloc(f32, T * hd);
    defer allocator.free(ks_head);
    const vs_head = try allocator.alloc(f32, T * hd);
    defer allocator.free(vs_head);
    const attn_proj = try allocator.alloc(f32, d);
    defer allocator.free(attn_proj);
    const gate_buf  = try allocator.alloc(f32, ff);
    defer allocator.free(gate_buf);
    const up_buf    = try allocator.alloc(f32, ff);
    defer allocator.free(up_buf);
    const ffn_out   = try allocator.alloc(f32, d);
    defer allocator.free(ffn_out);

    // Embedding lookup
    for (tokens, 0..) |tok, t| {
        @memcpy(x[t * d ..][0..d], w.token_emb[tok * d ..][0..d]);
    }

    for (0..cfg.n_layers) |l| {
        // Compute Q, K, V for all positions; apply RoPE in-place.
        for (0..T) |t| {
            math.rmsnorm(xb, x[t * d ..][0..d], w.attn_norm[l], cfg.eps);
            math.matvec(q_all[t * nq ..][0..nq],   w.wq[l], xb, nq,  d);
            math.matvec(ks[t * nkv ..][0..nkv], w.wk[l], xb, nkv, d);
            math.matvec(vs[t * nkv ..][0..nkv], w.wv[l], xb, nkv, d);

            for (0..cfg.n_heads)    |h| rope.applyRope(q_all[t * nq  + h * hd ..][0..hd], t, cfg.rope_theta);
            for (0..cfg.n_kv_heads) |h| rope.applyRope(ks[t * nkv + h * hd ..][0..hd],   t, cfg.rope_theta);
        }

        // Attention per position.
        for (0..T) |t| {
            const seq = t + 1;

            for (0..cfg.n_heads) |h| {
                const kv_h = h / cfg.kvGroupSize();

                // Gather contiguous K/V for this KV head across positions 0..t.
                for (0..seq) |s| {
                    @memcpy(ks_head[s * hd ..][0..hd], ks[s * nkv + kv_h * hd ..][0..hd]);
                    @memcpy(vs_head[s * hd ..][0..hd], vs[s * nkv + kv_h * hd ..][0..hd]);
                }

                try attn_mod.sdpAttn(
                    attn_concat[h * hd ..][0..hd],
                    q_all[t * nq + h * hd ..][0..hd],
                    ks_head[0..seq * hd],
                    vs_head[0..seq * hd],
                    seq, hd, allocator,
                );
            }

            // Wo projects concatenated head outputs → d_model; residual add.
            math.matvec(attn_proj, w.wo[l], attn_concat, d, nq);
            for (x[t * d ..][0..d], attn_proj) |*xi, delta| xi.* += delta;
        }

        // FFN (SwiGLU).
        for (0..T) |t| {
            math.rmsnorm(xb, x[t * d ..][0..d], w.ffn_norm[l], cfg.eps);
            math.matvec(gate_buf, w.w_gate[l], xb, ff, d);
            math.matvec(up_buf,   w.w_up[l],   xb, ff, d);
            for (gate_buf, up_buf) |*g, u| g.* = math.silu(g.*) * u;
            math.matvec(ffn_out, w.w_down[l], gate_buf, d, ff);
            for (x[t * d ..][0..d], ffn_out) |*xi, delta| xi.* += delta;
        }
    }

    const logits = try allocator.alloc(f32, cfg.vocab_size);
    math.rmsnorm(xb, x[(T - 1) * d ..][0..d], w.out_norm, cfg.eps);
    math.matvec(logits, w.lm_head, xb, cfg.vocab_size, d);
    return logits;
}

// ── incremental forward (KV cache) — used for generation ─────────────────────

/// Process one token at sequence position `pos`, using cached K/V for all
/// previous positions.
///
/// Call this for every prompt token (prefill) and then for each generated
/// token. On each call it writes new K/V into `kv` at position `pos` before
/// running attention, so subsequent calls will see this token's context.
///
/// Returns the logit slice for the next token (caller owns it).
pub fn forwardOne(
    token: u32,
    pos: usize,
    kv: *KvCache,
    w: *const Weights,
    cfg: Config,
    allocator: std.mem.Allocator,
) ![]f32 {
    const seq = pos + 1;
    const d   = cfg.d_model;
    const hd  = cfg.headDim();
    const nq  = cfg.n_heads    * hd;
    const nkv = cfg.n_kv_heads * hd;
    const ff  = cfg.d_ffn;

    const x = try allocator.alloc(f32, d);
    defer allocator.free(x);
    const xb        = try allocator.alloc(f32, d);
    defer allocator.free(xb);
    const q_all     = try allocator.alloc(f32, nq);
    defer allocator.free(q_all);
    const k_new     = try allocator.alloc(f32, nkv);
    defer allocator.free(k_new);
    const v_new     = try allocator.alloc(f32, nkv);
    defer allocator.free(v_new);
    const attn_concat = try allocator.alloc(f32, nq);
    defer allocator.free(attn_concat);
    const ks_head   = try allocator.alloc(f32, seq * hd);
    defer allocator.free(ks_head);
    const vs_head   = try allocator.alloc(f32, seq * hd);
    defer allocator.free(vs_head);
    const attn_proj = try allocator.alloc(f32, d);
    defer allocator.free(attn_proj);
    const gate_buf  = try allocator.alloc(f32, ff);
    defer allocator.free(gate_buf);
    const up_buf    = try allocator.alloc(f32, ff);
    defer allocator.free(up_buf);
    const ffn_out   = try allocator.alloc(f32, d);
    defer allocator.free(ffn_out);

    @memcpy(x, w.token_emb[token * d ..][0..d]);

    for (0..cfg.n_layers) |l| {
        math.rmsnorm(xb, x, w.attn_norm[l], cfg.eps);
        math.matvec(q_all, w.wq[l], xb, nq,  d);
        math.matvec(k_new, w.wk[l], xb, nkv, d);
        math.matvec(v_new, w.wv[l], xb, nkv, d);

        for (0..cfg.n_heads)    |h| rope.applyRope(q_all[h * hd ..][0..hd], pos, cfg.rope_theta);
        for (0..cfg.n_kv_heads) |h| rope.applyRope(k_new[h * hd ..][0..hd], pos, cfg.rope_theta);

        // Write new K/V into cache at this position.
        @memcpy(kv.k[l][pos * nkv ..][0..nkv], k_new);
        @memcpy(kv.v[l][pos * nkv ..][0..nkv], v_new);

        for (0..cfg.n_heads) |h| {
            const kv_h = h / cfg.kvGroupSize();

            for (0..seq) |s| {
                @memcpy(ks_head[s * hd ..][0..hd], kv.k[l][s * nkv + kv_h * hd ..][0..hd]);
                @memcpy(vs_head[s * hd ..][0..hd], kv.v[l][s * nkv + kv_h * hd ..][0..hd]);
            }

            try attn_mod.sdpAttn(
                attn_concat[h * hd ..][0..hd],
                q_all[h * hd ..][0..hd],
                ks_head[0..seq * hd],
                vs_head[0..seq * hd],
                seq, hd, allocator,
            );
        }

        math.matvec(attn_proj, w.wo[l], attn_concat, d, nq);
        for (x, attn_proj) |*xi, delta| xi.* += delta;

        math.rmsnorm(xb, x, w.ffn_norm[l], cfg.eps);
        math.matvec(gate_buf, w.w_gate[l], xb, ff, d);
        math.matvec(up_buf,   w.w_up[l],   xb, ff, d);
        for (gate_buf, up_buf) |*g, u| g.* = math.silu(g.*) * u;
        math.matvec(ffn_out, w.w_down[l], gate_buf, d, ff);
        for (x, ffn_out) |*xi, delta| xi.* += delta;
    }

    const logits = try allocator.alloc(f32, cfg.vocab_size);
    math.rmsnorm(xb, x, w.out_norm, cfg.eps);
    math.matvec(logits, w.lm_head, xb, cfg.vocab_size, d);
    return logits;
}

// ── real-model forward (quantized weights, KV cache) ─────────────────────────

/// Identical to forwardOne but operates on ModelWeights (quantized matrices).
///
/// Allocates a row-sized scratch buffer for on-the-fly dequantization; every
/// large matmul dequantizes one row at a time into this buffer then computes
/// the dot product, keeping peak extra memory at O(d_model).
pub fn forwardOneModel(
    token: u32,
    pos: usize,
    kv: *KvCache,
    w: *const mw.ModelWeights,
    cfg: Config,
    pool: *math.ThreadPool,
    allocator: std.mem.Allocator,
) ![]f32 {
    const seq = pos + 1;
    const d   = cfg.d_model;
    const hd  = cfg.headDim();
    const nq  = cfg.n_heads    * hd;
    const nkv = cfg.n_kv_heads * hd;
    const ff  = cfg.d_ffn;

    // scratch: per-thread dequant buffers; one col-sized buffer per worker.
    const n_threads = pool.threads.len;
    const max_cols = @max(@max(d, ff), @max(nq, nkv));
    const scratch = try allocator.alloc(f32, n_threads * max_cols);
    defer allocator.free(scratch);

    const x         = try allocator.alloc(f32, d);  defer allocator.free(x);
    const xb        = try allocator.alloc(f32, d);  defer allocator.free(xb);
    const q_all     = try allocator.alloc(f32, nq); defer allocator.free(q_all);
    const k_new     = try allocator.alloc(f32, nkv);defer allocator.free(k_new);
    const v_new     = try allocator.alloc(f32, nkv);defer allocator.free(v_new);
    const attn_concat = try allocator.alloc(f32, nq);defer allocator.free(attn_concat);
    const ks_head   = try allocator.alloc(f32, seq * hd);defer allocator.free(ks_head);
    const vs_head   = try allocator.alloc(f32, seq * hd);defer allocator.free(vs_head);
    const attn_proj = try allocator.alloc(f32, d);  defer allocator.free(attn_proj);
    const gate_buf  = try allocator.alloc(f32, ff); defer allocator.free(gate_buf);
    const up_buf    = try allocator.alloc(f32, ff); defer allocator.free(up_buf);
    const ffn_out   = try allocator.alloc(f32, d);  defer allocator.free(ffn_out);

    embedLookup(x, w.token_emb, token, d, scratch[0..max_cols]);

    for (0..cfg.n_layers) |l| {
        const lw = &w.layers[l];

        math.rmsnorm(xb, x, lw.attn_norm, cfg.eps);
        math.quantMatvecPar(q_all, lw.wq.data, lw.wq.type_, xb, nq,  d, scratch, pool);
        math.quantMatvecPar(k_new, lw.wk.data, lw.wk.type_, xb, nkv, d, scratch, pool);
        math.quantMatvecPar(v_new, lw.wv.data, lw.wv.type_, xb, nkv, d, scratch, pool);

        // Add optional per-layer Q/K/V biases (Qwen2 style).
        if (lw.q_bias) |b| { for (q_all, b) |*q, bv| q.* += bv; }
        if (lw.k_bias) |b| { for (k_new, b) |*k, bv| k.* += bv; }
        if (lw.v_bias) |b| { for (v_new, b) |*v, bv| v.* += bv; }

        for (0..cfg.n_heads)    |h| rope.applyRope(q_all[h * hd ..][0..hd], pos, cfg.rope_theta);
        for (0..cfg.n_kv_heads) |h| rope.applyRope(k_new[h * hd ..][0..hd], pos, cfg.rope_theta);

        @memcpy(kv.k[l][pos * nkv ..][0..nkv], k_new);
        @memcpy(kv.v[l][pos * nkv ..][0..nkv], v_new);

        for (0..cfg.n_heads) |h| {
            const kv_h = h / cfg.kvGroupSize();
            for (0..seq) |s| {
                @memcpy(ks_head[s * hd ..][0..hd], kv.k[l][s * nkv + kv_h * hd ..][0..hd]);
                @memcpy(vs_head[s * hd ..][0..hd], kv.v[l][s * nkv + kv_h * hd ..][0..hd]);
            }
            try attn_mod.sdpAttn(
                attn_concat[h * hd ..][0..hd],
                q_all[h * hd ..][0..hd],
                ks_head[0..seq * hd], vs_head[0..seq * hd],
                seq, hd, allocator,
            );
        }

        math.quantMatvecPar(attn_proj, lw.wo.data, lw.wo.type_, attn_concat, d, nq, scratch, pool);
        for (x, attn_proj) |*xi, delta| xi.* += delta;

        math.rmsnorm(xb, x, lw.ffn_norm, cfg.eps);
        math.quantMatvecPar(gate_buf, lw.w_gate.data, lw.w_gate.type_, xb, ff, d, scratch, pool);
        math.quantMatvecPar(up_buf,   lw.w_up.data,   lw.w_up.type_,   xb, ff, d, scratch, pool);
        for (gate_buf, up_buf) |*g, u| g.* = math.silu(g.*) * u;
        math.quantMatvecPar(ffn_out, lw.w_down.data, lw.w_down.type_, gate_buf, d, ff, scratch, pool);
        for (x, ffn_out) |*xi, delta| xi.* += delta;
    }

    const logits = try allocator.alloc(f32, cfg.vocab_size);
    math.rmsnorm(xb, x, w.out_norm, cfg.eps);
    math.quantMatvecPar(logits, w.lm_head.data, w.lm_head.type_, xb, cfg.vocab_size, d, scratch, pool);
    return logits;
}

/// Dequantize a single row from a RawMatrix into `out`.
fn embedLookup(out: []f32, mat: mw.RawMatrix, row: u32, cols: usize, row_buf: []f32) void {
    _ = row_buf; // unused when type is already f32 for large embeddings
    const dq = @import("../quant/dequant.zig");
    const row_bytes = switch (mat.type_) {
        .f32  => cols * 4,
        .f16  => cols * 2,
        .q5_0 => (cols / dq.Q5_0_BLOCK_ELEMS) * dq.Q5_0_BLOCK_BYTES,
        .q8_0 => (cols / dq.Q8_0_BLOCK_ELEMS) * dq.Q8_0_BLOCK_BYTES,
        .q4_k => (cols / dq.QK_K) * dq.Q4_K_BLOCK_BYTES,
        .q6_k => (cols / dq.QK_K) * dq.Q6_K_BLOCK_BYTES,
        else  => @panic("unsupported embedding quant type"),
    };
    const src = mat.data[@as(usize, row) * row_bytes ..][0..row_bytes];
    switch (mat.type_) {
        .f32  => dq.dequantF32(src, out),
        .f16  => dq.dequantF16(src, out),
        .q5_0 => dq.dequantQ5_0(src, out),
        .q8_0 => dq.dequantQ8_0(src, out),
        .q4_k => dq.dequantQ4K(src, out),
        .q6_k => dq.dequantQ6K(src, out),
        else  => unreachable,
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

test "forward: output is a valid token ID" {
    const cfg = Config{
        .vocab_size = 16, .d_model = 8, .n_layers = 1,
        .n_heads = 1, .n_kv_heads = 1, .d_ffn = 16,
    };
    var w = try Weights.initZero(cfg, std.testing.allocator);
    defer w.deinit();
    w.setNormWeightsOne();
    for (0..cfg.d_model) |i| w.token_emb[3 * cfg.d_model + i] = @as(f32, @floatFromInt(i)) * 0.1;
    w.lm_head[5 * cfg.d_model + 0] = 10.0;

    const tokens = [_]u32{ 1, 3 };
    const logits = try forward(&tokens, &w, cfg, std.testing.allocator);
    defer std.testing.allocator.free(logits);

    try std.testing.expectEqual(logits.len, cfg.vocab_size);
    const next = @import("sample.zig").greedy(logits);
    try std.testing.expect(next < cfg.vocab_size);
}

test "forward: deterministic — same input gives same output" {
    const cfg = Config{
        .vocab_size = 8, .d_model = 4, .n_layers = 1,
        .n_heads = 1, .n_kv_heads = 1, .d_ffn = 8,
    };
    var w = try Weights.initZero(cfg, std.testing.allocator);
    defer w.deinit();
    w.setNormWeightsOne();
    w.wq[0][0] = 0.5; w.wq[0][5] = -0.3;
    w.wk[0][1] = 0.7; w.wv[0][2] = 0.4;
    w.lm_head[0] = 1.0; w.lm_head[5] = 2.0;

    const tokens = [_]u32{ 0, 1, 2 };
    const a = try forward(&tokens, &w, cfg, std.testing.allocator);
    defer std.testing.allocator.free(a);
    const b = try forward(&tokens, &w, cfg, std.testing.allocator);
    defer std.testing.allocator.free(b);
    for (a, b) |av, bv| try std.testing.expectEqual(av, bv);
}

test "forward vs forwardOne: incremental matches batch for single-head" {
    // forwardOne called for each prompt token should produce identical logits
    // to the batch forward function for the last position.
    const cfg = Config{
        .vocab_size = 8, .d_model = 4, .n_layers = 1,
        .n_heads = 1, .n_kv_heads = 1, .d_ffn = 8,
    };
    var w = try Weights.initZero(cfg, std.testing.allocator);
    defer w.deinit();
    w.setNormWeightsOne();
    w.wq[0][0] = 0.5;  w.wq[0][5] = -0.3;
    w.wk[0][1] = 0.7;  w.wv[0][2] = 0.4;
    w.lm_head[0] = 1.0; w.lm_head[5] = 2.0;

    const tokens = [_]u32{ 0, 1, 2 };

    // Batch forward.
    const batch = try forward(&tokens, &w, cfg, std.testing.allocator);
    defer std.testing.allocator.free(batch);

    // Incremental forward — call forwardOne for each position.
    var kv = try KvCache.init(cfg, std.testing.allocator);
    defer kv.deinit();

    var incr: []f32 = undefined;
    for (tokens, 0..) |tok, pos| {
        if (pos > 0) std.testing.allocator.free(incr);
        incr = try forwardOne(tok, pos, &kv, &w, cfg, std.testing.allocator);
    }
    defer std.testing.allocator.free(incr);

    // Both should give the same logits for the final position.
    for (batch, incr) |bv, iv| try std.testing.expectApproxEqAbs(bv, iv, 1e-4);
}

test "forward: multi-head (n_heads=2, n_kv_heads=1 GQA)" {
    // Smoke test with 2 Q heads sharing 1 KV head.
    const cfg = Config{
        .vocab_size = 8, .d_model = 4, .n_layers = 1,
        .n_heads = 2, .n_kv_heads = 1, .d_ffn = 8,
    };
    var w = try Weights.initZero(cfg, std.testing.allocator);
    defer w.deinit();
    w.setNormWeightsOne();
    // With all-zero weights except norms the residual stream carries the
    // embedding through; just check we get a valid vocab token back.
    w.lm_head[0] = 1.0;

    const tokens = [_]u32{ 0, 1 };
    const logits = try forward(&tokens, &w, cfg, std.testing.allocator);
    defer std.testing.allocator.free(logits);
    try std.testing.expectEqual(logits.len, cfg.vocab_size);
    const next = @import("sample.zig").greedy(logits);
    try std.testing.expect(next < cfg.vocab_size);
}
