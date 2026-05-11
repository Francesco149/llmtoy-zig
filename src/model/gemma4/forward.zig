/// Gemma4 incremental forward pass (one token at a time, KV cache).
///
/// Layer order (per block l):
///   1.  attn_norm(x) → Q/K/V projections → per-head Q/K norms → RoPE
///   2.  SDP attention (SWA or global, limited window for SWA)
///   3.  post_attention_norm(attn_out) + residual → attn_residual
///   4a. Dense FFN: ffn_norm(attn_residual) → gate*up (GELU) → down
///       → post_ffw_norm_1
///   4b. MoE FFN:  pre_ffw_norm_2(attn_residual) → top-8 experts
///       → post_ffw_norm_2
///   5.  post_ffw_norm(dense_out + moe_out) + residual (attn_residual)
///   6.  × layer_output_scale

const std       = @import("std");
const cfg_      = @import("config.zig");
const wt_       = @import("weights.zig");
const kv_       = @import("kv_cache.zig");
const math      = @import("../../ops/math.zig");
const rope_mod  = @import("../../ops/rope.zig");
const attn_mod  = @import("../../ops/attn.zig");
const moe_mod   = @import("../../ops/moe.zig");
const dq        = @import("../../quant/dequant.zig");

pub const Gemma4Config   = cfg_.Gemma4Config;
pub const Gemma4Weights  = wt_.Gemma4Weights;
pub const Gemma4KvCache  = kv_.Gemma4KvCache;

/// Process one token at `pos`, writing new K/V into `kv`.
/// Returns a caller-owned logit slice.
pub fn forwardOne(
    token: u32,
    pos: usize,
    kv: *Gemma4KvCache,
    w: *const Gemma4Weights,
    cfg: Gemma4Config,
    pool: *math.ThreadPool,
    allocator: std.mem.Allocator,
) ![]f32 {
    const d         = cfg.d_model;
    const n_threads = pool.threads.len;
    const max_nq    = cfg.n_heads * cfg.head_dim_global;  // global has larger heads
    const max_nkv   = blk: {
        var m: usize = 0;
        for (0..cfg.n_layers) |l| if (cfg.nkv(l) > m) { m = cfg.nkv(l); };
        break :blk m;
    };
    const max_cols  = @max(@max(d, max_nq), max_nkv);

    // Per-thread dequant scratch (largest possible dimension).
    const scratch = try allocator.alloc(f32, n_threads * max_cols);
    defer allocator.free(scratch);

    const x          = try allocator.alloc(f32, d); defer allocator.free(x);
    const xb         = try allocator.alloc(f32, d); defer allocator.free(xb);
    const attn_buf   = try allocator.alloc(f32, d); defer allocator.free(attn_buf);
    const ffn_buf    = try allocator.alloc(f32, d); defer allocator.free(ffn_buf);

    // Q/K/V buffers: sized for the largest layer (global).
    const q_buf = try allocator.alloc(f32, max_nq);  defer allocator.free(q_buf);
    const k_buf = try allocator.alloc(f32, max_nkv); defer allocator.free(k_buf);
    const v_buf = try allocator.alloc(f32, max_nkv); defer allocator.free(v_buf);
    const attn_concat = try allocator.alloc(f32, max_nq); defer allocator.free(attn_concat);

    // KV gather buffers: sized for global worst case.
    const max_win    = @max(cfg.sliding_window, pos + 1);
    const ks_head    = try allocator.alloc(f32, max_win * cfg.head_dim_global);
    defer allocator.free(ks_head);
    const vs_head    = try allocator.alloc(f32, max_win * cfg.head_dim_global);
    defer allocator.free(vs_head);

    // Embedding lookup.  Gemma scales embeddings by sqrt(d_model).
    embedLookup(x, w.token_emb, token, d, scratch[0..max_cols]);
    const emb_scale = @sqrt(@as(f32, @floatFromInt(d)));
    for (x) |*v| v.* *= emb_scale;

    for (0..cfg.n_layers) |l| {
        const lw      = &w.layers[l];
        const is_swa  = cfg.is_swa[l];
        const hd      = cfg.headDim(l);
        const nq_l    = cfg.nq(l);
        const nkv_l   = cfg.nkv(l);
        const n_kv_l  = cfg.n_kv_heads[l];
        const q       = q_buf[0..nq_l];
        const k_cur   = k_buf[0..nkv_l];
        const v_cur   = v_buf[0..nkv_l];

        // ── Attention ─────────────────────────────────────────────────────────

        math.rmsnorm(xb, x, lw.attn_norm, cfg.eps);
        math.quantMatvecPar(q,     lw.wq.data, lw.wq.type_, xb, nq_l,  d, scratch, pool);
        math.quantMatvecPar(k_cur, lw.wk.data, lw.wk.type_, xb, nkv_l, d, scratch, pool);

        // V = Wv*x if present, else V = pre-norm K (global layers share K/V projection)
        if (lw.wv) |wv| {
            math.quantMatvecPar(v_cur, wv.data, wv.type_, xb, nkv_l, d, scratch, pool);
        } else {
            @memcpy(v_cur, k_cur); // copy raw K before K-norm is applied
        }

        // Per-head Q and K norms (weighted RMSNorm).
        for (0..cfg.n_heads) |h| math.rmsnorm(
            q[h * hd ..][0..hd], q[h * hd ..][0..hd], lw.q_norm, cfg.eps);
        for (0..n_kv_l) |h| math.rmsnorm(
            k_cur[h * hd ..][0..hd], k_cur[h * hd ..][0..hd], lw.k_norm, cfg.eps);

        // Plain (unweighted) RMSNorm on V — Gemma4 applies this after V projection.
        for (0..n_kv_l) |h| math.rmsnormRaw(
            v_cur[h * hd ..][0..hd], v_cur[h * hd ..][0..hd], cfg.eps);

        // RoPE on Q heads.
        if (is_swa) {
            for (0..cfg.n_heads) |h|
                rope_mod.applyRopeNeox(q[h * hd ..][0..hd], pos, cfg.rope_theta_swa);
            for (0..n_kv_l) |h|
                rope_mod.applyRopeNeox(k_cur[h * hd ..][0..hd], pos, cfg.rope_theta_swa);
        } else {
            for (0..cfg.n_heads) |h|
                rope_mod.applyRopeFreqsNeox(q[h * hd ..][0..hd], w.rope_freqs, pos);
            for (0..n_kv_l) |h|
                rope_mod.applyRopeFreqsNeox(k_cur[h * hd ..][0..hd], w.rope_freqs, pos);
        }

        // Write K, V into KV cache.
        const kv_nkv  = nkv_l;
        const kv_cap  = kv.cap[l];
        const kv_slot = pos % kv_cap; // circular buffer
        @memcpy(kv.k[l][kv_slot * kv_nkv ..][0..kv_nkv], k_cur);
        @memcpy(kv.v[l][kv_slot * kv_nkv ..][0..kv_nkv], v_cur);

        // Determine attended positions.
        const seq     = pos + 1;
        const win_len = if (is_swa) @min(seq, cfg.sliding_window) else seq;
        // For SWA: attend positions [pos - win_len + 1 .. pos].
        const start   = if (seq > win_len) seq - win_len else 0;
        _ = start; // used implicitly via kv_slot math below

        // Compute attention for each Q head.
        @memset(attn_concat[0..nq_l], 0.0);
        for (0..cfg.n_heads) |h| {
            const kv_h = h / (cfg.n_heads / n_kv_l);

            // Gather K and V for this head from the circular KV cache.
            const ks = ks_head[0..win_len * hd];
            const vs = vs_head[0..win_len * hd];
            for (0..win_len) |wi| {
                // wi=0 is the oldest position, wi=win_len-1 is current (pos).
                const abs_pos = (seq - win_len) + wi;
                const slot    = abs_pos % kv_cap;
                @memcpy(ks[wi * hd ..][0..hd], kv.k[l][slot * kv_nkv + kv_h * hd ..][0..hd]);
                @memcpy(vs[wi * hd ..][0..hd], kv.v[l][slot * kv_nkv + kv_h * hd ..][0..hd]);
            }

            try attn_mod.sdpAttn(
                attn_concat[h * hd ..][0..hd],
                q[h * hd ..][0..hd],
                ks, vs,
                win_len, hd,
                1.0, // Gemma4 uses scale=1.0 (per-head Q/K norms control magnitude)
                allocator,
            );
        }

        // Output projection → post-attention norm → first residual.
        math.quantMatvecPar(attn_buf, lw.wo.data, lw.wo.type_, attn_concat[0..nq_l], d, nq_l, scratch, pool);
        math.rmsnorm(attn_buf, attn_buf, lw.post_attention_norm, cfg.eps);
        for (x, attn_buf) |*xi, a| xi.* += a;

        // ── Dense FFN path ────────────────────────────────────────────────────

        math.rmsnorm(xb, x, lw.ffn_norm, cfg.eps);
        const gate_buf  = try allocator.alloc(f32, cfg.d_ffn); defer allocator.free(gate_buf);
        const up_buf    = try allocator.alloc(f32, cfg.d_ffn); defer allocator.free(up_buf);
        math.quantMatvecPar(gate_buf, lw.w_gate.data, lw.w_gate.type_, xb, cfg.d_ffn, d, scratch, pool);
        math.quantMatvecPar(up_buf,   lw.w_up.data,   lw.w_up.type_,   xb, cfg.d_ffn, d, scratch, pool);
        for (gate_buf, up_buf) |*g, u| g.* = math.gelu(g.*) * u;
        math.quantMatvecPar(ffn_buf, lw.w_down.data, lw.w_down.type_, gate_buf, d, cfg.d_ffn, scratch, pool);
        math.rmsnorm(ffn_buf, ffn_buf, lw.post_ffw_norm_1, cfg.eps);

        // ── MoE FFN path ──────────────────────────────────────────────────────

        // Expert input (named norm).
        const moe_in = try allocator.alloc(f32, d); defer allocator.free(moe_in);
        math.rmsnorm(moe_in, x, lw.pre_ffw_norm_2, cfg.eps);

        // Router input: raw rmsnorm(x) * (1/sqrt(d)) * router_scale.
        const router_in = try allocator.alloc(f32, d); defer allocator.free(router_in);
        math.rmsnormRaw(router_in, x, cfg.eps);
        const inv_sqrt_d = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));
        for (router_in, lw.router_scale) |*r, s| r.* *= inv_sqrt_d * s;

        // Router logits → softmax → top-k indices.
        const router_out = try allocator.alloc(f32, cfg.n_experts); defer allocator.free(router_out);
        math.quantMatvec(router_out, lw.router_w.data, lw.router_w.type_,
            router_in, cfg.n_experts, d, scratch[0..d]);
        math.softmax(router_out);

        const top_idx = try allocator.alloc(usize, cfg.n_experts_used);
        defer allocator.free(top_idx);
        topK(router_out, top_idx);

        // Run each selected expert; accumulate into moe_buf.
        const moe_buf = try allocator.alloc(f32, d); defer allocator.free(moe_buf);
        @memset(moe_buf, 0.0);

        const gu_row_bytes   = math.rowBytes(lw.gate_up_exps.type_, d);
        const gu_per_expert  = 2 * cfg.d_expert * gu_row_bytes;
        const dn_row_bytes   = math.rowBytes(lw.down_exps.type_, cfg.d_expert);
        const dn_per_expert  = d * dn_row_bytes;
        const expert_scratch = try allocator.alloc(f32, n_threads * @max(d, cfg.d_expert));
        defer allocator.free(expert_scratch);
        const eg = try allocator.alloc(f32, cfg.d_expert); defer allocator.free(eg);
        const eu = try allocator.alloc(f32, cfg.d_expert); defer allocator.free(eu);
        const ed = try allocator.alloc(f32, d);            defer allocator.free(ed);

        for (top_idx) |eidx| {
            const w_score = router_out[eidx];

            const gate_data = lw.gate_up_exps.data[eidx * gu_per_expert ..];
            math.quantMatvecPar(eg, gate_data, lw.gate_up_exps.type_,
                moe_in, cfg.d_expert, d, expert_scratch, pool);

            const up_data = gate_data[cfg.d_expert * gu_row_bytes ..];
            math.quantMatvecPar(eu, up_data, lw.gate_up_exps.type_,
                moe_in, cfg.d_expert, d, expert_scratch, pool);

            for (eg, eu) |*g, u| g.* = math.gelu(g.*) * u;

            const dn_data = lw.down_exps.data[eidx * dn_per_expert ..];
            math.quantMatvecPar(ed, dn_data, lw.down_exps.type_,
                eg, d, cfg.d_expert, expert_scratch, pool);

            // Scale by per-expert output scale and router weight.
            const expert_scale = lw.down_exps_scale[eidx] * w_score;
            for (moe_buf, ed) |*m, ev| m.* += expert_scale * ev;
        }

        math.rmsnorm(moe_buf, moe_buf, lw.post_ffw_norm_2, cfg.eps);

        // ── Combine and second residual ───────────────────────────────────────

        // combined = ffn_buf + moe_buf; then post_ffw_norm; then + x
        for (ffn_buf, moe_buf) |*a, b| a.* += b;
        math.rmsnorm(ffn_buf, ffn_buf, lw.post_ffw_norm, cfg.eps);
        for (x, ffn_buf) |*xi, f| xi.* += f;

        // Layer output scale (scalar).
        if (lw.layer_output_scale != 1.0) {
            for (x) |*xi| xi.* *= lw.layer_output_scale;
        }

    }

    // ── Final logits ─────────────────────────────────────────────────────────

    math.rmsnorm(xb, x, w.out_norm, cfg.eps);
    const logits = try allocator.alloc(f32, cfg.vocab_size);
    math.quantMatvecPar(logits, w.lm_head.data, w.lm_head.type_,
        xb, cfg.vocab_size, d, scratch, pool);

    // Soft-capping: tanh(logits / cap) * cap
    if (cfg.logit_softcap != 0.0) {
        for (logits) |*v| {
            v.* = std.math.tanh(v.* / cfg.logit_softcap) * cfg.logit_softcap;
        }
    }

    return logits;
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn embedLookup(out: []f32, mat: wt_.RawMatrix, row: u32, cols: usize, row_buf: []f32) void {
    const row_bytes = math.rowBytes(mat.type_, cols);
    const src = mat.data[@as(usize, row) * row_bytes ..][0..row_bytes];
    math.dequantRow(src, out[0..cols], mat.type_);
    _ = row_buf;
}

/// In-place top-k selection by descending score. O(n·k), k ≤ 16.
fn topK(scores: []const f32, out: []usize) void {
    for (0..out.len) |i| {
        var best: usize = 0;
        var best_val: f32 = -std.math.inf(f32);
        for (scores, 0..) |s, j| {
            var dup = false;
            for (out[0..i]) |p| if (p == j) { dup = true; break; };
            if (!dup and s > best_val) { best_val = s; best = j; }
        }
        out[i] = best;
    }
}
