const std = @import("std");
const math = @import("math.zig");

/// Single-head scaled dot-product attention (causal).
///
/// q:   [head_dim]               query for the current (last) position
/// ks:  [seq_len × head_dim]     key vectors, row-major
/// vs:  [seq_len × head_dim]     value vectors, row-major
/// out: [head_dim]               weighted sum of value rows
///
/// All seq_len positions are attended to (caller is responsible for passing
/// only the causally-valid prefix).
pub fn sdpAttn(
    out: []f32,
    q: []const f32,
    ks: []const f32,
    vs: []const f32,
    seq_len: usize,
    head_dim: usize,
    scale: f32,
    allocator: std.mem.Allocator,
) !void {
    std.debug.assert(q.len == head_dim);
    std.debug.assert(ks.len == seq_len * head_dim);
    std.debug.assert(vs.len == seq_len * head_dim);
    std.debug.assert(out.len == head_dim);

    const scores = try allocator.alloc(f32, seq_len);
    defer allocator.free(scores);

    for (0..seq_len) |i| {
        var dot: f32 = 0.0;
        for (q, ks[i * head_dim ..][0..head_dim]) |qv, kv| dot += qv * kv;
        scores[i] = dot * scale;
    }
    math.softmax(scores);

    @memset(out, 0.0);
    for (0..seq_len) |i| {
        for (out, vs[i * head_dim ..][0..head_dim]) |*o, vv| o.* += scores[i] * vv;
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

test "sdpAttn: seq_len=1 → output equals value" {
    // With one position, softmax([score]) = [1], so out = v[0] exactly.
    const hd = 4;
    const q = [_]f32{ 1, 0, 0, 0 };
    const k = [_]f32{ 1, 0, 0, 0 };
    const v = [_]f32{ 3, 1, 4, 1 };
    var out: [hd]f32 = undefined;
    try sdpAttn(&out, &q, &k, &v, 1, hd, 1.0, std.testing.allocator);
    for (out, v) |o, vv| try std.testing.expectApproxEqAbs(vv, o, 1e-5);
}

test "sdpAttn: equal keys → average of values" {
    // Two identical key rows → equal attention weights (0.5 each).
    const hd = 2;
    const q = [_]f32{ 1, 0 };
    const ks = [_]f32{ 1, 0, 1, 0 }; // k[0] == k[1]
    const vs = [_]f32{ 0, 0, 2, 4 }; // v[0]=[0,0], v[1]=[2,4]
    var out: [hd]f32 = undefined;
    try sdpAttn(&out, &q, &ks, &vs, 2, hd, 1.0, std.testing.allocator);
    // 0.5*[0,0] + 0.5*[2,4] = [1,2]
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out[1], 1e-5);
}

test "sdpAttn: extreme score difference → output ≈ winning value" {
    // head_dim=1: score = raw dot product (scale=1).
    // q=10 vs k[0]=-10 → score -100; vs k[1]=10 → score 100.
    // softmax(-100, 100) ≈ (0, 1) — attention collapses to position 1.
    const hd = 1;
    const q = [_]f32{10};
    const ks = [_]f32{ -10, 10 }; // k[0] anti-parallel, k[1] parallel
    const vs = [_]f32{ 99, 7 };
    var out: [hd]f32 = undefined;
    try sdpAttn(&out, &q, &ks, &vs, 2, hd, 1.0, std.testing.allocator);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), out[0], 1e-3);
}
