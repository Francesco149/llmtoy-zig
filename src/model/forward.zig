const std = @import("std");
const Config = @import("config.zig").Config;
const Weights = @import("weights.zig").Weights;
const math = @import("../ops/math.zig");
const attn_mod = @import("../ops/attn.zig");

/// Full causal forward pass over a token sequence.
///
/// Processes all positions through all layers (O(T² · L) attention).
/// No KV cache — every call recomputes from scratch. Correct but slow;
/// Phase 4 adds caching for practical generation speed.
///
/// Returns the logit vector for the next token (caller owns the slice).
pub fn forward(
    tokens: []const u32,
    w: *const Weights,
    cfg: Config,
    allocator: std.mem.Allocator,
) ![]f32 {
    const T = tokens.len;
    const d = cfg.d_model;
    const hd = cfg.headDim();
    const ff = cfg.d_ffn;

    // Hidden states for all positions: x[t] is the residual stream at position t.
    const x = try allocator.alloc(f32, T * d);
    defer allocator.free(x);

    // Scratch buffers reused each layer.
    const xb = try allocator.alloc(f32, d);   // normed input
    defer allocator.free(xb);
    const q = try allocator.alloc(f32, hd);
    defer allocator.free(q);
    const attn_out = try allocator.alloc(f32, hd);
    defer allocator.free(attn_out);
    const gate_buf = try allocator.alloc(f32, ff);
    defer allocator.free(gate_buf);
    const up_buf = try allocator.alloc(f32, ff);
    defer allocator.free(up_buf);
    const ffn_out = try allocator.alloc(f32, d);
    defer allocator.free(ffn_out);

    // KV cache for current layer: rebuilt each layer from the current x[t] values.
    const ks = try allocator.alloc(f32, T * hd);
    defer allocator.free(ks);
    const vs = try allocator.alloc(f32, T * hd);
    defer allocator.free(vs);

    // Embedding lookup: x[t] = token_emb[token[t]]
    for (tokens, 0..) |tok, t| {
        @memcpy(x[t * d ..][0..d], w.token_emb[tok * d ..][0..d]);
    }

    for (0..cfg.n_layers) |l| {
        // Precompute K and V for all positions at this layer.
        for (0..T) |t| {
            math.rmsnorm(xb, x[t * d ..][0..d], w.attn_norm[l], cfg.eps);
            math.matvec(ks[t * hd ..][0..hd], w.wk[l], xb, hd, d);
            math.matvec(vs[t * hd ..][0..hd], w.wv[l], xb, hd, d);
        }

        // Attention: for each position, attend to all preceding positions (causal).
        for (0..T) |t| {
            math.rmsnorm(xb, x[t * d ..][0..d], w.attn_norm[l], cfg.eps);
            math.matvec(q, w.wq[l], xb, hd, d);

            try attn_mod.sdpAttn(attn_out, q, ks[0..(t + 1) * hd], vs[0..(t + 1) * hd], t + 1, hd, allocator);

            // Project attention output back to d_model and add to residual.
            math.matvec(ffn_out, w.wo[l], attn_out, d, hd);
            for (x[t * d ..][0..d], ffn_out) |*xi, delta| xi.* += delta;
        }

        // FFN (SwiGLU): gate(x) = silu(Wgate·x) ⊙ (Wup·x); out = Wdown·gate(x)
        for (0..T) |t| {
            math.rmsnorm(xb, x[t * d ..][0..d], w.ffn_norm[l], cfg.eps);
            math.matvec(gate_buf, w.w_gate[l], xb, ff, d);
            math.matvec(up_buf, w.w_up[l], xb, ff, d);
            for (gate_buf, up_buf) |*g, u| g.* = math.silu(g.*) * u;
            math.matvec(ffn_out, w.w_down[l], gate_buf, d, ff);
            for (x[t * d ..][0..d], ffn_out) |*xi, delta| xi.* += delta;
        }
    }

    // Output norm + lm_head on the final position only.
    const logits = try allocator.alloc(f32, cfg.vocab_size);
    math.rmsnorm(xb, x[(T - 1) * d ..][0..d], w.out_norm, cfg.eps);
    math.matvec(logits, w.lm_head, xb, cfg.vocab_size, d);
    return logits;
}

// ── tests ─────────────────────────────────────────────────────────────────────

test "forward: output is a valid token ID" {
    const cfg = Config{ .vocab_size = 16, .d_model = 8, .n_layers = 1, .n_heads = 1, .d_ffn = 16 };

    var w = try Weights.initZero(cfg, std.testing.allocator);
    defer w.deinit();

    // Identity-ish weights: norm weights = 1, lm_head = identity (first 8 rows),
    // rest zero. Result will be token 0 (zero logits → greedy picks first).
    w.setNormWeightsOne();
    // Give token 3 a distinctive embedding so we can track it through the pass.
    for (0..cfg.d_model) |i| w.token_emb[3 * cfg.d_model + i] = @as(f32, @floatFromInt(i)) * 0.1;
    // lm_head row 5 has a large weight on dim 0 so token 3's embedding activates it.
    w.lm_head[5 * cfg.d_model + 0] = 10.0;

    const tokens = [_]u32{ 1, 3 };
    const logits = try forward(&tokens, &w, cfg, std.testing.allocator);
    defer std.testing.allocator.free(logits);

    try std.testing.expectEqual(logits.len, cfg.vocab_size);
    // Verify output is a valid token ID (basic sanity).
    const next = @import("sample.zig").greedy(logits);
    try std.testing.expect(next < cfg.vocab_size);
}

test "forward: deterministic — same input gives same output" {
    const cfg = Config{ .vocab_size = 8, .d_model = 4, .n_layers = 1, .n_heads = 1, .d_ffn = 8 };

    var w = try Weights.initZero(cfg, std.testing.allocator);
    defer w.deinit();
    w.setNormWeightsOne();
    // Sprinkle some non-zero weights to make the pass non-trivial.
    w.wq[0][0] = 0.5; w.wq[0][5] = -0.3;
    w.wk[0][1] = 0.7; w.wv[0][2] = 0.4;
    w.lm_head[0] = 1.0; w.lm_head[5] = 2.0;

    const tokens = [_]u32{ 0, 1, 2 };

    const logits_a = try forward(&tokens, &w, cfg, std.testing.allocator);
    defer std.testing.allocator.free(logits_a);
    const logits_b = try forward(&tokens, &w, cfg, std.testing.allocator);
    defer std.testing.allocator.free(logits_b);

    for (logits_a, logits_b) |a, b| try std.testing.expectEqual(a, b);
}
