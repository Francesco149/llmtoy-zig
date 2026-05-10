const std = @import("std");
const math = @import("../ops/math.zig");

pub const SampleParams = struct {
    temperature: f32 = 1.0,  // 0 = greedy; >0 = stochastic
    top_k: u32 = 0,           // 0 = disabled
    top_p: f32 = 1.0,         // 1.0 = disabled (nucleus sampling)
};

/// Greedy decoding: return the index of the highest logit.
pub fn greedy(logits: []const f32) u32 {
    var best: u32 = 0;
    for (logits[1..], 1..) |v, i| {
        if (v > logits[best]) best = @intCast(i);
    }
    return best;
}

/// Sample the next token from logits.
///
/// With temperature=0 (or temperature extremely small) falls back to greedy.
/// Otherwise: divide by temperature → softmax → optional top-k filter →
/// optional top-p (nucleus) filter → multinomial sample.
///
/// Modifies `logits` in-place (temperature + softmax applied directly).
/// Requires a scratch allocation of `logits.len * @sizeOf(usize)` bytes.
pub fn sample(
    logits: []f32,
    params: SampleParams,
    rng: std.Random,
    allocator: std.mem.Allocator,
) !u32 {
    if (params.temperature == 0.0) return greedy(logits);

    // Temperature scale.
    if (params.temperature != 1.0) {
        const inv_t = 1.0 / params.temperature;
        for (logits) |*l| l.* *= inv_t;
    }

    math.softmax(logits);

    // Top-k filtering: zero out all but the k highest probabilities.
    if (params.top_k > 0 and params.top_k < logits.len) {
        try applyTopK(logits, params.top_k, allocator);
    }

    // Top-p (nucleus) filtering.
    if (params.top_p < 1.0) {
        try applyTopP(logits, params.top_p, allocator);
    }

    // Renormalize after any filtering.
    renormalize(logits);

    // Multinomial sample.
    const r = rng.float(f32);
    var cum: f32 = 0.0;
    for (logits, 0..) |p, i| {
        cum += p;
        if (r <= cum) return @intCast(i);
    }
    return @intCast(logits.len - 1);
}

fn applyTopK(probs: []f32, k: u32, allocator: std.mem.Allocator) !void {
    // Copy and sort descending to find the k-th largest threshold.
    const tmp = try allocator.dupe(f32, probs);
    defer allocator.free(tmp);
    std.mem.sortUnstable(f32, tmp, {}, std.sort.desc(f32));
    const threshold = tmp[@min(k, tmp.len) - 1];
    for (probs) |*p| if (p.* < threshold) { p.* = 0.0; };
}

fn applyTopP(probs: []f32, top_p: f32, allocator: std.mem.Allocator) !void {
    // Build sorted index array descending by probability.
    const indices = try allocator.alloc(usize, probs.len);
    defer allocator.free(indices);
    for (indices, 0..) |*idx, i| idx.* = i;
    std.mem.sortUnstable(usize, indices, probs, struct {
        fn lt(ps: []f32, a: usize, b: usize) bool {
            return ps[a] > ps[b]; // descending
        }
    }.lt);

    // Keep tokens until cumulative probability reaches top_p; zero the rest.
    // Accumulate BEFORE checking so the token that pushes us over the threshold
    // is always kept (handles top_p=0 correctly — only the top-1 token survives).
    var cum: f32 = 0.0;
    var cutoff = false;
    for (indices) |idx| {
        if (cutoff) {
            probs[idx] = 0.0;
        } else {
            cum += probs[idx];
            if (cum >= top_p) cutoff = true;
        }
    }
}

fn renormalize(probs: []f32) void {
    var sum: f32 = 0.0;
    for (probs) |p| sum += p;
    if (sum > 0.0) {
        const inv = 1.0 / sum;
        for (probs) |*p| p.* *= inv;
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

test "greedy: returns argmax" {
    const logits = [_]f32{ 0.1, 5.0, 2.0, 3.0 };
    try std.testing.expectEqual(@as(u32, 1), greedy(&logits));
}

test "sample: temperature=0 behaves like greedy" {
    var logits = [_]f32{ 1.0, 3.0, 2.0 };
    var prng = std.Random.DefaultPrng.init(42);
    const tok = try sample(&logits, .{ .temperature = 0.0 }, prng.random(), std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 1), tok);
}

test "sample: stays within vocab range" {
    const n = 100;
    var logits: [n]f32 = undefined;
    for (&logits, 0..) |*l, i| l.* = @as(f32, @floatFromInt(i));
    var prng = std.Random.DefaultPrng.init(0);
    const tok = try sample(&logits, .{ .temperature = 1.0, .top_k = 10 }, prng.random(), std.testing.allocator);
    try std.testing.expect(tok < n);
}

test "sample: top_p=0.0 always picks highest-prob token" {
    var logits = [_]f32{ 0.5, 2.0, 0.1 };
    var prng = std.Random.DefaultPrng.init(7);
    // top_p=0 means only the single top token survives nucleus filtering.
    const tok = try sample(&logits, .{ .temperature = 1.0, .top_p = 0.0 }, prng.random(), std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 1), tok);
}
