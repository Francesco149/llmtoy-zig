const std = @import("std");

/// Apply Rotary Position Embedding (RoPE) in-place to one head's vector.
///
/// Rotates consecutive pairs of dimensions by a position-dependent angle:
///
///   freq_i    = theta ^ (-2i / head_dim)
///   angle     = pos * freq_i
///   [x₂ᵢ, x₂ᵢ₊₁] ← [x₂ᵢ·cos − x₂ᵢ₊₁·sin, x₂ᵢ·sin + x₂ᵢ₊₁·cos]
///
/// Applied independently to every head in Q and K before computing attention
/// scores. Makes dot products sensitive to relative token distance — a key
/// at position 5 "knows" it's 3 steps away from a query at position 8.
pub fn applyRope(vec: []f32, pos: usize, theta: f32) void {
    std.debug.assert(vec.len % 2 == 0);
    const half = vec.len / 2;
    for (0..half) |i| {
        const freq = 1.0 / std.math.pow(
            f32,
            theta,
            @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(vec.len)),
        );
        const angle = @as(f32, @floatFromInt(pos)) * freq;
        const cos_a = @cos(angle);
        const sin_a = @sin(angle);
        const x0 = vec[2 * i];
        const x1 = vec[2 * i + 1];
        vec[2 * i]     = x0 * cos_a - x1 * sin_a;
        vec[2 * i + 1] = x0 * sin_a + x1 * cos_a;
    }
}

/// Apply RoPE using an explicit pre-computed frequency table.
///
/// freqs[i] holds the inverse frequency for the i-th pair of dimensions.
/// freqs.len must equal vec.len / 2.  Used for Gemma4 global attention where
/// frequencies are stored in the GGUF as rope_freqs.weight.
pub fn applyRopeFreqs(vec: []f32, freqs: []const f32, pos: usize) void {
    std.debug.assert(vec.len == freqs.len * 2);
    const fpos: f32 = @floatFromInt(pos);
    for (0..freqs.len) |i| {
        const angle = fpos * freqs[i];
        const cos_a = @cos(angle);
        const sin_a = @sin(angle);
        const x0 = vec[2 * i];
        const x1 = vec[2 * i + 1];
        vec[2 * i]     = x0 * cos_a - x1 * sin_a;
        vec[2 * i + 1] = x0 * sin_a + x1 * cos_a;
    }
}

/// Apply GPT-NeoX style RoPE pairing.
///
/// Instead of rotating consecutive pairs `(0,1), (2,3), ...`, NeoX-style RoPE
/// rotates dimensions across the two halves of the head:
/// `(0, half), (1, half+1), ...`. llama.cpp uses this RoPE layout for Gemma4.
pub fn applyRopeNeox(vec: []f32, pos: usize, theta: f32) void {
    std.debug.assert(vec.len % 2 == 0);
    const half = vec.len / 2;
    for (0..half) |i| {
        const freq = 1.0 / std.math.pow(
            f32,
            theta,
            @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(vec.len)),
        );
        const angle = @as(f32, @floatFromInt(pos)) * freq;
        const cos_a = @cos(angle);
        const sin_a = @sin(angle);
        const x0 = vec[i];
        const x1 = vec[i + half];
        vec[i]        = x0 * cos_a - x1 * sin_a;
        vec[i + half] = x0 * sin_a + x1 * cos_a;
    }
}

/// Apply GPT-NeoX style RoPE using an explicit pre-computed frequency table.
pub fn applyRopeFreqsNeox(vec: []f32, freqs: []const f32, pos: usize) void {
    std.debug.assert(vec.len == freqs.len * 2);
    const half = vec.len / 2;
    const fpos: f32 = @floatFromInt(pos);
    for (0..half) |i| {
        const angle = fpos * freqs[i];
        const cos_a = @cos(angle);
        const sin_a = @sin(angle);
        const x0 = vec[i];
        const x1 = vec[i + half];
        vec[i]        = x0 * cos_a - x1 * sin_a;
        vec[i + half] = x0 * sin_a + x1 * cos_a;
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

test "applyRope: pos=0 is identity" {
    // At position 0 all angles are 0 → cos=1, sin=0 → no rotation.
    var v = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    applyRope(&v, 0, 10000.0);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 1.0), v[0], 1e-6);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 2.0), v[1], 1e-6);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 3.0), v[2], 1e-6);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 4.0), v[3], 1e-6);
}

test "applyRope: preserves vector magnitude" {
    // Rotation preserves L2 norm of each pair.
    var v = [_]f32{ 3.0, 4.0, 1.0, 0.0 };
    const mag0_before = v[0] * v[0] + v[1] * v[1];
    const mag1_before = v[2] * v[2] + v[3] * v[3];
    applyRope(&v, 7, 10000.0);
    const mag0_after = v[0] * v[0] + v[1] * v[1];
    const mag1_after = v[2] * v[2] + v[3] * v[3];
    try @import("std").testing.expectApproxEqAbs(mag0_before, mag0_after, 1e-5);
    try @import("std").testing.expectApproxEqAbs(mag1_before, mag1_after, 1e-5);
}

test "applyRope: same pos gives same rotation" {
    var v1 = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    var v2 = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    applyRope(&v1, 5, 10000.0);
    applyRope(&v2, 5, 10000.0);
    for (v1, v2) |a, b| try @import("std").testing.expectEqual(a, b);
}

test "applyRopeNeox: rotates across head halves" {
    var v = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    applyRopeNeox(&v, 0, 10000.0);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 1.0), v[0], 1e-6);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 2.0), v[1], 1e-6);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 3.0), v[2], 1e-6);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 4.0), v[3], 1e-6);

    applyRopeNeox(&v, 1, 10000.0);
    // Pair (0,2) changes under the first frequency while dimensions 1 and 3
    // use the much slower second frequency.
    try @import("std").testing.expect(v[0] != 1.0);
    try @import("std").testing.expect(v[2] != 3.0);
}
