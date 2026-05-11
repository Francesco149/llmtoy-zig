/// Mixture-of-Experts FFN dispatch.
///
/// Implements sparse top-k routing over expert FFNs where each expert uses a
/// fused SwiGLU (gate+up packed together → down). Experts are stored back-to-back
/// in a single 3D weight tensor; we compute the byte offset for each selected expert.
///
/// Gemma4 expert layout:
///   gate_up_exps dims [d_model, 2*d_expert, n_experts]:
///     per expert: 2*d_expert rows × d_model cols; first half = gate, second = up
///   down_exps dims [d_expert, d_model, n_experts]:
///     per expert: d_model rows × d_expert cols

const std   = @import("std");
const mw    = @import("../model/model_weights.zig");
const math  = @import("math.zig");

/// Run the MoE FFN for a single token, accumulate into `out` (must be pre-zeroed).
pub fn moeForward(
    out:            []f32,           // [d_model], caller zeros before call
    router_w:       mw.RawMatrix,    // [n_experts, d_model] F32 router weights
    gate_up:        mw.RawMatrix,    // 3D: [d_model, 2*d_expert, n_experts]
    down:           mw.RawMatrix,    // 3D: [d_expert, d_model, n_experts]
    x:              []const f32,     // [d_model] input
    n_experts_used: usize,
    d_expert:       usize,
    pool:           *math.ThreadPool,
    allocator:      std.mem.Allocator,
) !void {
    const n_experts = router_w.rows;
    const d_model   = x.len;
    const n_threads = pool.threads.len;

    // Per-expert buffers.
    const gate_buf   = try allocator.alloc(f32, d_expert);
    defer allocator.free(gate_buf);
    const up_buf     = try allocator.alloc(f32, d_expert);
    defer allocator.free(up_buf);
    const down_buf   = try allocator.alloc(f32, d_model);
    defer allocator.free(down_buf);
    const max_cols   = @max(d_model, d_expert);
    const par_scratch = try allocator.alloc(f32, n_threads * max_cols);
    defer allocator.free(par_scratch);

    // Router scores (F32, simple matvec).
    const scores = try allocator.alloc(f32, n_experts);
    defer allocator.free(scores);
    math.quantMatvec(scores, router_w.data, .f32, x, n_experts, d_model, par_scratch[0..d_model]);
    math.softmax(scores);

    // Top-k selection (k is small, usually 2-4).
    const top_idx = try allocator.alloc(usize, n_experts_used);
    defer allocator.free(top_idx);
    topK(scores, top_idx);

    const gu_row_bytes  = math.rowBytes(gate_up.type_, d_model);
    const gu_per_expert = 2 * d_expert * gu_row_bytes;

    const dn_row_bytes  = math.rowBytes(down.type_, d_expert);
    const dn_per_expert = d_model * dn_row_bytes;

    for (top_idx) |eidx| {
        const w = scores[eidx];

        const gate_data = gate_up.data[eidx * gu_per_expert ..];
        math.quantMatvecPar(gate_buf, gate_data, gate_up.type_, x, d_expert, d_model, par_scratch, pool);

        const up_data = gate_data[d_expert * gu_row_bytes ..];
        math.quantMatvecPar(up_buf, up_data, gate_up.type_, x, d_expert, d_model, par_scratch, pool);

        for (gate_buf, up_buf) |*g, u| g.* = math.silu(g.*) * u;

        const dn_data = down.data[eidx * dn_per_expert ..];
        math.quantMatvecPar(down_buf, dn_data, down.type_, gate_buf, d_model, d_expert, par_scratch, pool);

        for (out, down_buf) |*o, v| o.* += w * v;
    }
}

/// In-place top-k by value (descending). O(n·k) — k is always ≤ 8.
fn topK(scores: []const f32, out: []usize) void {
    for (0..out.len) |i| {
        var best: usize = 0;
        var best_val: f32 = -std.math.inf(f32);
        for (scores, 0..) |s, j| {
            var already = false;
            for (out[0..i]) |prev| if (prev == j) { already = true; break; };
            if (!already and s > best_val) { best_val = s; best = j; }
        }
        out[i] = best;
    }
}
