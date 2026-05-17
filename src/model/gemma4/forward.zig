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
const std = @import("std");
const cfg_ = @import("config.zig");
const wt_ = @import("weights.zig");
const kv_ = @import("kv_cache.zig");
const math = @import("../../ops/math.zig");
const rope_mod = @import("../../ops/rope.zig");
const attn_mod = @import("../../ops/attn.zig");
const moe_mod = @import("../../ops/moe.zig");
const dq = @import("../../quant/dequant.zig");
const gpu_w_ = @import("gpu_weights.zig");

pub const Gemma4Config = cfg_.Gemma4Config;
pub const Gemma4Weights = wt_.Gemma4Weights;
pub const Gemma4KvCache = kv_.Gemma4KvCache;
pub const GpuWeights = gpu_w_.GpuWeights;
pub const GpuLayerWeights = gpu_w_.GpuLayerWeights;
const MatvecPipeline = @import("../../gpu/matvec.zig").MatvecPipeline;

/// Process one token at `pos`, writing new K/V into `kv`.
/// Returns a caller-owned logit slice.
/// Pass gpu != null to offload attention + dense-FFN matmuls to the GPU.
/// layer_taps: if non-null, slice of length n_layers; after each layer l, x is
///   copied into layer_taps[l] for comparison. Caller must pre-allocate each slice.
/// gpu_layer_range: if non-null, GPU is used only for layers [r[0]..=r[1]].
pub fn forwardOne(
    token: u32,
    pos: usize,
    kv: *Gemma4KvCache,
    w: *const Gemma4Weights,
    cfg: Gemma4Config,
    pool: *math.ThreadPool,
    allocator: std.mem.Allocator,
    gpu: ?*const GpuWeights,
    layer_taps: ?[][]f32,
    gpu_layer_range: ?[2]usize,
) ![]f32 {
    const d = cfg.d_model;
    const n_threads = pool.threads.len;
    const max_nq = cfg.n_heads * cfg.head_dim_global; // global has larger heads
    const max_nkv = blk: {
        var m: usize = 0;
        for (0..cfg.n_layers) |l| if (cfg.nkv(l) > m) {
            m = cfg.nkv(l);
        };
        break :blk m;
    };
    const max_cols = @max(@max(d, max_nq), max_nkv);

    // Per-thread dequant scratch (largest possible dimension).
    const scratch = try allocator.alloc(f32, n_threads * max_cols);
    defer allocator.free(scratch);

    const x = try allocator.alloc(f32, d);
    defer allocator.free(x);
    const xb = try allocator.alloc(f32, d);
    defer allocator.free(xb);
    const attn_buf = try allocator.alloc(f32, d);
    defer allocator.free(attn_buf);
    const ffn_buf = try allocator.alloc(f32, d);
    defer allocator.free(ffn_buf);
    const moe_buf = try allocator.alloc(f32, d);
    defer allocator.free(moe_buf);
    const moe_in = try allocator.alloc(f32, d);
    defer allocator.free(moe_in);
    const router_in = try allocator.alloc(f32, d);
    defer allocator.free(router_in);
    const router_out = try allocator.alloc(f32, cfg.n_experts);
    defer allocator.free(router_out);
    const top_idx = try allocator.alloc(usize, cfg.n_experts_used);
    defer allocator.free(top_idx);

    // Q/K/V buffers: sized for the largest layer (global).
    const q_buf = try allocator.alloc(f32, max_nq);
    defer allocator.free(q_buf);
    const k_buf = try allocator.alloc(f32, max_nkv);
    defer allocator.free(k_buf);
    const v_buf = try allocator.alloc(f32, max_nkv);
    defer allocator.free(v_buf);
    const attn_concat = try allocator.alloc(f32, max_nq);
    defer allocator.free(attn_concat);

    // KV gather buffers: sized for global worst case.
    const max_win = @max(cfg.sliding_window, pos + 1);
    const ks_head = try allocator.alloc(f32, max_win * cfg.head_dim_global);
    defer allocator.free(ks_head);
    const vs_head = try allocator.alloc(f32, max_win * cfg.head_dim_global);
    defer allocator.free(vs_head);

    // FFN and expert temporaries are reused for every layer. These used to be
    // allocated inside the layer loop, which made the allocator part of the hot
    // path for every generated token.
    const gate_buf = try allocator.alloc(f32, cfg.d_ffn);
    defer allocator.free(gate_buf);
    const up_buf = try allocator.alloc(f32, cfg.d_ffn);
    defer allocator.free(up_buf);
    const eg = try allocator.alloc(f32, cfg.d_expert);
    defer allocator.free(eg);
    const eu = try allocator.alloc(f32, cfg.d_expert);
    defer allocator.free(eu);
    const ed = try allocator.alloc(f32, d);
    defer allocator.free(ed);

    // Embedding lookup.  Gemma scales embeddings by sqrt(d_model).
    embedLookup(x, w.token_emb, token, d, scratch[0..max_cols]);
    const emb_scale = @sqrt(@as(f32, @floatFromInt(d)));
    for (x) |*v| v.* *= emb_scale;

    for (0..cfg.n_layers) |l| {
        const lw = &w.layers[l];
        const is_swa = cfg.is_swa[l];
        const hd = cfg.headDim(l);
        const nq_l = cfg.nq(l);
        const nkv_l = cfg.nkv(l);
        const n_kv_l = cfg.n_kv_heads[l];
        const q = q_buf[0..nq_l];
        const k_cur = k_buf[0..nkv_l];
        const v_cur = v_buf[0..nkv_l];

        // GPU layer weights for this layer (null ⇒ CPU fallback for every call).
        const gpu_here = if (gpu_layer_range) |r| (l >= r[0] and l <= r[1]) else true;
        const glw: ?*const GpuLayerWeights = if (gpu != null and gpu_here) &gpu.?.layers[l] else null;

        // Decide all per-layer GPU dispatch paths up front.  The wo + post-
        // attention residual stage is part of the dense FFN submit when
        // can_full_fused is true, so we skip it in the attention block below.
        const wv_present = lw.wv != null;
        const wv_on_gpu = wv_present and (if (glw) |g| g.wv != null else false);
        const wq_q8_1 = gpu != null and glw != null and glw.?.wq != null and gpu.?.q8_1PipelineFor(lw.wq.type_) != null;
        const wk_q8_1 = gpu != null and glw != null and glw.?.wk != null and gpu.?.q8_1PipelineFor(lw.wk.type_) != null;
        const wv_q8_1 = wv_on_gpu and gpu.?.q8_1PipelineFor(lw.wv.?.type_) != null;
        const can_q8_1_qkv = gpu != null and wq_q8_1 and wk_q8_1 and (!wv_present or wv_q8_1);
        const can_batch_qkv = !can_q8_1_qkv and gpu != null and glw != null and
            glw.?.wq != null and glw.?.wk != null and (!wv_present or wv_on_gpu);

        const wo_q8_pl: ?*const MatvecPipeline = if (gpu) |g|
            (if (glw != null and glw.?.wo != null) g.q8_1PipelineFor(lw.wo.type_) else null)
        else
            null;
        const gate_q8_pl: ?*const MatvecPipeline = if (gpu) |g|
            (if (glw != null and glw.?.w_gate != null) g.q8_1PipelineFor(lw.w_gate.type_) else null)
        else
            null;
        const up_q8_pl: ?*const MatvecPipeline = if (gpu) |g|
            (if (glw != null and glw.?.w_up != null) g.q8_1PipelineFor(lw.w_up.type_) else null)
        else
            null;
        const can_q8_1_ffn = gate_q8_pl != null and up_q8_pl != null and (lw.w_gate.type_ != .q3_k or d % 256 == 0) and (lw.w_up.type_ != .q3_k or d % 256 == 0);
        const w_down_on_gpu = gpu != null and glw != null and glw.?.w_down != null;
        const can_fused_dense = can_q8_1_ffn and w_down_on_gpu;
        // Full fused path also needs wo on Q8_1.  wo cols = nq_l: 256-aligned
        // (n_heads × head_dim ∈ {1024, 2048} for Gemma4) so Q3_K Q8_1 works.
        const wo_q8_1_ok = wo_q8_pl != null and (lw.wo.type_ != .q3_k or nq_l % 256 == 0);
        const can_full_fused = can_fused_dense and wo_q8_1_ok;

        // ── Attention ─────────────────────────────────────────────────────────
        //
        // QKV dispatch paths, in priority order:
        //   0. 7l.1 KV-VRAM path: Q8_1 QKV + GPU per-head Q/K/V norms + GPU
        //      RoPE + GPU KV cache append in ONE submit. Requires wv present
        //      (global SWA layers without wv fall through to path 1).
        //   1. Q8_1 path:  wq+wk+(wv) all on GPU with Q8_1 pipelines →
        //      runLayerAttnQ8_1 fuses rmsnorm + quantize + matvecs.
        //   2. f32 batch:  some session lacks a Q8_1 pipeline (e.g. Q5_K wv) →
        //      CPU rmsnorm + parallel f32-activation matvecs.
        //   3. Per-call:   any session missing / CPU-only → CPU rmsnorm + mv().
        const kv_cap_l = kv.cap[l];
        const kv_slot_l = pos % kv_cap_l;
        const can_q8_1_kv_vram = can_q8_1_qkv and (if (gpu) |g| g.k_vram != null else false);

        var gpu_did_norms_and_rope = false;
        if (can_q8_1_kv_vram) {
            const g = gpu.?;
            const wq_q8_pl = g.q8_1PipelineFor(lw.wq.type_).?;
            const wk_q8_pl = g.q8_1PipelineFor(lw.wk.type_).?;
            const wv_q8_pl: ?*const MatvecPipeline = if (wv_present)
                g.q8_1PipelineFor(lw.wv.?.type_)
            else
                null;
            try g.runLayerAttnQ8_1KvVram(l, cfg.eps, wq_q8_pl, wk_q8_pl, wv_q8_pl, @intCast(cfg.n_heads), @intCast(n_kv_l), @intCast(hd), @intCast(pos), is_swa, cfg.rope_theta_swa, @intCast(kv_slot_l), x, q[0..nq_l], k_cur, v_cur);
            gpu_did_norms_and_rope = true;
        } else if (can_q8_1_qkv) {
            const g = gpu.?;
            const wq_q8_pl = g.q8_1PipelineFor(lw.wq.type_).?;
            const wk_q8_pl = g.q8_1PipelineFor(lw.wk.type_).?;
            const wv_q8_pl: ?*const MatvecPipeline = if (wv_present)
                g.q8_1PipelineFor(lw.wv.?.type_)
            else
                null;
            try g.runLayerAttnQ8_1(l, cfg.eps, wq_q8_pl, wk_q8_pl, wv_q8_pl, x, q[0..nq_l], k_cur, if (wv_present) v_cur else null);
            if (!wv_present) @memcpy(v_cur, k_cur);
        } else if (can_batch_qkv) {
            math.rmsnorm(xb, x, lw.attn_norm, cfg.eps);
            const g = gpu.?;
            const wv_pl: ?*const MatvecPipeline = if (wv_on_gpu)
                g.pipelineFor(lw.wv.?.type_)
            else
                null;
            try g.runLayerQKV(l, g.pipelineFor(lw.wq.type_), g.pipelineFor(lw.wk.type_), wv_pl, xb[0..d], q[0..nq_l], k_cur, v_cur);
            if (!wv_present) @memcpy(v_cur, k_cur);
        } else {
            math.rmsnorm(xb, x, lw.attn_norm, cfg.eps);
            try mv(q[0..nq_l], lw.wq, xb[0..d], scratch, pool, if (glw) |g| g.wq else null, gpu);
            try mv(k_cur, lw.wk, xb[0..d], scratch, pool, if (glw) |g| g.wk else null, gpu);
            if (lw.wv) |wv| {
                try mv(v_cur, wv, xb[0..d], scratch, pool, if (glw) |g| g.wv else null, gpu);
            } else {
                @memcpy(v_cur, k_cur);
            }
        }

        // Per-head Q/K/V norms + RoPE on Q,K. Done on GPU in 7l.1 path; CPU
        // here is the fallback for paths 1–3.
        if (!gpu_did_norms_and_rope) {
            for (0..cfg.n_heads) |h| math.rmsnorm(q[h * hd ..][0..hd], q[h * hd ..][0..hd], lw.q_norm, cfg.eps);
            for (0..n_kv_l) |h| math.rmsnorm(k_cur[h * hd ..][0..hd], k_cur[h * hd ..][0..hd], lw.k_norm, cfg.eps);
            for (0..n_kv_l) |h| math.rmsnormRaw(v_cur[h * hd ..][0..hd], v_cur[h * hd ..][0..hd], cfg.eps);

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
        }

        // Write K, V into CPU KV cache. (GPU 7l.1 path also wrote into
        // k_vram[l]/v_vram[l]; the CPU sdpAttn still reads from kv.k[l]/kv.v[l]
        // until 7l.2/3 puts the attention compute itself on the GPU.)
        const kv_nkv = nkv_l;
        @memcpy(kv.k[l][kv_slot_l * kv_nkv ..][0..kv_nkv], k_cur);
        @memcpy(kv.v[l][kv_slot_l * kv_nkv ..][0..kv_nkv], v_cur);

        // Determine attended positions.
        const seq = pos + 1;
        const win_len = if (is_swa) @min(seq, cfg.sliding_window) else seq;
        // For SWA: attend positions [pos - win_len + 1 .. pos].
        const start = if (seq > win_len) seq - win_len else 0;
        _ = start; // used implicitly via kv_slot math below

        // Compute attention. 7l.2/3 GPU path is taken when the 7l.1c QKV+norm
        // chain ran (so Q is in q_out_buf and K/V live in k_vram/v_vram).
        // Attention output is written to attn_in_buf on the GPU side; the
        // full-fused wo+dense-FFN path reads it directly. Only download to
        // attn_concat when the non-full-fused wo fallback needs it on CPU.
        const use_gpu_attn = gpu_did_norms_and_rope;
        if (use_gpu_attn) {
            const dst: ?[]f32 = if (can_full_fused) null else attn_concat[0..nq_l];
            try gpu.?.runLayerAttention(l, @intCast(cfg.n_heads), @intCast(n_kv_l), @intCast(hd), @intCast(pos), @intCast(win_len), @intCast(kv_cap_l), 1.0, dst);
        } else {
            @memset(attn_concat[0..nq_l], 0.0);
            for (0..cfg.n_heads) |h| {
                const kv_h = h / (cfg.n_heads / n_kv_l);

                const ks = ks_head[0 .. win_len * hd];
                const vs = vs_head[0 .. win_len * hd];
                for (0..win_len) |wi| {
                    const abs_pos = (seq - win_len) + wi;
                    const slot = abs_pos % kv_cap_l;
                    @memcpy(ks[wi * hd ..][0..hd], kv.k[l][slot * kv_nkv + kv_h * hd ..][0..hd]);
                    @memcpy(vs[wi * hd ..][0..hd], kv.v[l][slot * kv_nkv + kv_h * hd ..][0..hd]);
                }

                try attn_mod.sdpAttn(
                    attn_concat[h * hd ..][0..hd],
                    q[h * hd ..][0..hd],
                    ks,
                    vs,
                    win_len,
                    hd,
                    1.0, // Gemma4 uses scale=1.0 (per-head Q/K norms control magnitude)
                    allocator,
                );
            }
        }

        // Output projection → post-attention norm → first residual.
        // can_full_fused: skipped here; runLayerAttnResidualDenseFfnQ8_1 below
        // does wo + post_attn_norm + residual on the GPU in one submit.
        if (!can_full_fused) {
            try mv(attn_buf, lw.wo, attn_concat[0..nq_l], scratch, pool, if (glw) |g| g.wo else null, gpu);
            math.rmsnorm(attn_buf, attn_buf, lw.post_attention_norm, cfg.eps);
            for (x, attn_buf) |*xi, a| xi.* += a;
        }

        // ── Dense FFN path ────────────────────────────────────────────────────
        //
        // FFN dispatch paths, in priority order:
        //   -1. Full fused chain: also folds wo + post_attn_norm + residual
        //       in, so the entire post-attention block runs in ONE submit
        //       (runLayerAttnResidualDenseFfnQ8_1). 3 submits/layer total.
        //    0. Fused dense FFN only (runLayerDenseFfnQ8_1): rmsnorm + gate +
        //       up + gelu*up + w_down + post_ffw_norm_1 in one submit, but wo
        //       still goes through its own submit. 4 submits/layer.
        //    1. Q8_1 gate+up only: w_down not on GPU.
        //    2. f32 batch.
        //    3. Per-call.
        const can_batch_ffn = !can_q8_1_ffn and gpu != null and glw != null and
            glw.?.w_gate != null and glw.?.w_up != null;

        if (can_full_fused) {
            const g = gpu.?;
            try g.runLayerAttnResidualDenseFfnQ8_1(l, cfg.eps, wo_q8_pl.?, gate_q8_pl.?, up_q8_pl.?, g.pipelineFor(lw.w_down.type_), x, attn_concat[0..nq_l], ffn_buf, use_gpu_attn);
            // x updated with post-attn residual; ffn_buf has post_ffw_norm_1.
        } else if (can_fused_dense) {
            const g = gpu.?;
            try g.runLayerDenseFfnQ8_1(l, cfg.eps, gate_q8_pl.?, up_q8_pl.?, g.pipelineFor(lw.w_down.type_), x, ffn_buf);
        } else {
            if (can_q8_1_ffn) {
                try gpu.?.runLayerFfnGateUpQ8_1(l, cfg.eps, gate_q8_pl.?, up_q8_pl.?, x, gate_buf, up_buf);
            } else if (can_batch_ffn) {
                math.rmsnorm(xb, x, lw.ffn_norm, cfg.eps);
                const g = gpu.?;
                try g.runLayerGateUp(l, g.pipelineFor(lw.w_gate.type_), g.pipelineFor(lw.w_up.type_), xb[0..d], gate_buf, up_buf);
            } else {
                math.rmsnorm(xb, x, lw.ffn_norm, cfg.eps);
                try mv(gate_buf, lw.w_gate, xb[0..d], scratch, pool, if (glw) |g| g.w_gate else null, gpu);
                try mv(up_buf, lw.w_up, xb[0..d], scratch, pool, if (glw) |g| g.w_up else null, gpu);
            }
            for (gate_buf, up_buf) |*g, u| g.* = math.gelu(g.*) * u;
            try mv(ffn_buf, lw.w_down, gate_buf, scratch, pool, if (glw) |g| g.w_down else null, gpu);
            math.rmsnorm(ffn_buf, ffn_buf, lw.post_ffw_norm_1, cfg.eps);
        }

        // ── MoE FFN path ──────────────────────────────────────────────────────

        // Expert input (named norm).
        math.rmsnorm(moe_in, x, lw.pre_ffw_norm_2, cfg.eps);

        // Router input: raw rmsnorm(x) * (1/sqrt(d)) * router_scale.
        math.rmsnormRaw(router_in, x, cfg.eps);
        const inv_sqrt_d = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));
        for (router_in, lw.router_scale) |*r, s| r.* *= inv_sqrt_d * s;

        // Router logits → softmax → top-k indices.
        math.quantMatvec(router_out, lw.router_w.data, lw.router_w.type_, router_in, cfg.n_experts, d, scratch[0..d]);
        math.softmax(router_out);

        topK(router_out, top_idx);

        // Run each selected expert; accumulate into moe_buf.
        @memset(moe_buf, 0.0);

        // Batched GPU path: 2 submits per layer (n gate+up dispatches, then n down).
        // Falls back to per-expert CPU path if experts aren't on GPU.
        const expert_gpu_ok = if (gpu != null and gpu_here)
            gpu.?.runExpertBatch(l, top_idx, lw.gate_up_exps.type_, lw.down_exps.type_, lw.down_exps_scale, moe_in, router_out, moe_buf)
        else
            error.ExpertNotOnGpu;

        if (expert_gpu_ok) |_| {
            // GPU path completed; moe_buf already accumulated.
        } else |_| {
            // CPU fallback: one expert at a time with thread pool.
            const gu_row_bytes = math.rowBytes(lw.gate_up_exps.type_, d);
            const gu_per_expert = 2 * cfg.d_expert * gu_row_bytes;
            const dn_row_bytes = math.rowBytes(lw.down_exps.type_, cfg.d_expert);
            const dn_per_expert = d * dn_row_bytes;
            for (top_idx) |eidx| {
                const w_score = router_out[eidx];
                const gate_data = lw.gate_up_exps.data[eidx * gu_per_expert ..];
                const up_data = gate_data[cfg.d_expert * gu_row_bytes ..];
                const dn_data = lw.down_exps.data[eidx * dn_per_expert ..];
                math.quantMatvecPar(eg, gate_data, lw.gate_up_exps.type_, moe_in, cfg.d_expert, d, scratch, pool);
                math.quantMatvecPar(eu, up_data, lw.gate_up_exps.type_, moe_in, cfg.d_expert, d, scratch, pool);
                for (eg, eu) |*g, u| g.* = math.gelu(g.*) * u;
                math.quantMatvecPar(ed, dn_data, lw.down_exps.type_, eg, d, cfg.d_expert, scratch, pool);
                const expert_scale = lw.down_exps_scale[eidx] * w_score;
                for (moe_buf, ed) |*m, ev| m.* += expert_scale * ev;
            }
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

        if (layer_taps) |taps| @memcpy(taps[l], x);
    }

    // ── Final logits ─────────────────────────────────────────────────────────

    math.rmsnorm(xb, x, w.out_norm, cfg.eps);
    const logits = try allocator.alloc(f32, cfg.vocab_size);
    try mv(logits, w.lm_head, xb[0..d], scratch, pool, if (gpu) |g| g.lm_head else null, gpu);

    // Soft-capping: tanh(logits / cap) * cap
    if (cfg.logit_softcap != 0.0) {
        for (logits) |*v| {
            v.* = std.math.tanh(v.* / cfg.logit_softcap) * cfg.logit_softcap;
        }
    }

    return logits;
}

// ── helpers ───────────────────────────────────────────────────────────────────

// Run one matmul on the GPU (if session != null) or the CPU thread pool.
// sess_opt is passed by value (cheap handle copy); no deinit is called here.
//
// Routing priority:
//   1. GPU with Q8_1 pipeline + aligned cols  → runQ8_1Mv (integer-dot)
//   2. GPU session present, no Q8_1 pipeline   → sess.run (f32-acts matvec)
//   3. No GPU session                          → CPU thread pool
fn mv(
    out: []f32,
    mat: wt_.RawMatrix,
    vec: []const f32,
    scratch: []f32,
    pool: *math.ThreadPool,
    sess_opt: ?@import("../../gpu/matvec.zig").MatvecSession,
    gw: ?*const GpuWeights,
) !void {
    if (sess_opt) |sess| {
        const gw_ = gw.?;
        if (gw_.q8_1PipelineFor(mat.type_)) |q8_1_pl| {
            // cols alignment: K-quants need %256, Q5_0/Q5_1 need %32. The
            // q8_1PipelineFor returns null for unsupported types so we only
            // hit this for supported quant families.
            const k_aligned = switch (mat.type_) {
                .q3_k, .q4_k, .q5_k, .q6_k => sess.cols % 256 == 0,
                .q5_0, .q5_1 => sess.cols % 32 == 0,
                else => false,
            };
            if (k_aligned) {
                try gw_.runQ8_1Mv(q8_1_pl, &sess, vec, out);
                return;
            }
        }
        try sess.run(&gw_.ctx, gw_.pipelineFor(mat.type_), &gw_.shared_vec.?, &gw_.shared_out.?, vec, out);
    } else {
        math.quantMatvecPar(out, mat.data, mat.type_, vec, out.len, vec.len, scratch, pool);
    }
}

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
            for (out[0..i]) |p| if (p == j) {
                dup = true;
                break;
            };
            if (!dup and s > best_val) {
                best_val = s;
                best = j;
            }
        }
        out[i] = best;
    }
}
