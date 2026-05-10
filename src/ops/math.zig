const std    = @import("std");
const GgmlType = @import("../gguf/types.zig").GgmlType;
const dq       = @import("../quant/dequant.zig");

/// Matrix-vector multiply: out[rows] = mat[rows×cols] · vec[cols]
/// mat is row-major: row i starts at mat[i*cols].
pub fn matvec(out: []f32, mat: []const f32, vec: []const f32, rows: usize, cols: usize) void {
    std.debug.assert(mat.len == rows * cols);
    std.debug.assert(vec.len == cols);
    std.debug.assert(out.len == rows);
    for (0..rows) |i| {
        var sum: f32 = 0.0;
        for (mat[i * cols ..][0..cols], vec) |w, x| sum += w * x;
        out[i] = sum;
    }
}

/// RMS normalization: out[i] = (x[i] / rms(x)) * weight[i]
pub fn rmsnorm(out: []f32, x: []const f32, weight: []const f32, eps: f32) void {
    std.debug.assert(x.len == weight.len and out.len == x.len);
    var ss: f32 = 0.0;
    for (x) |v| ss += v * v;
    const rms_inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(x.len)) + eps);
    for (out, x, weight) |*o, v, w| o.* = v * rms_inv * w;
}

/// In-place softmax. Numerically stable (subtract max before exp).
pub fn softmax(x: []f32) void {
    var max = x[0];
    for (x[1..]) |v| if (v > max) { max = v; };
    var sum: f32 = 0.0;
    for (x) |*v| { v.* = @exp(v.* - max); sum += v.*; }
    for (x) |*v| v.* /= sum;
}

/// SiLU activation: x * sigmoid(x)
pub fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

/// Matrix-vector multiply with quantized weight matrix.
///
/// Dequantizes one row at a time into a stack-allocated f32 buffer, computing
/// the dot product with `vec` before moving to the next row. This keeps peak
/// memory usage at O(cols) rather than O(rows × cols) — critical for large
/// weight matrices that would overflow RAM when fully materialized as f32.
pub fn quantMatvec(
    out: []f32,
    mat_data: []const u8,
    mat_type: GgmlType,
    vec: []const f32,
    rows: usize,
    cols: usize,
    row_buf: []f32, // caller-provided scratch of length `cols`
) void {
    std.debug.assert(row_buf.len >= cols);
    std.debug.assert(out.len == rows);

    const row_bytes = switch (mat_type) {
        .f32   => cols * 4,
        .f16   => cols * 2,
        .q5_0  => (cols / dq.Q5_0_BLOCK_ELEMS) * dq.Q5_0_BLOCK_BYTES,
        .q8_0  => (cols / dq.Q8_0_BLOCK_ELEMS) * dq.Q8_0_BLOCK_BYTES,
        .q4_k  => (cols / dq.QK_K) * dq.Q4_K_BLOCK_BYTES,
        .q6_k  => (cols / dq.QK_K) * dq.Q6_K_BLOCK_BYTES,
        else   => @panic("unsupported quant type in quantMatvec"),
    };

    for (0..rows) |i| {
        const row_data = mat_data[i * row_bytes ..][0..row_bytes];
        const row = row_buf[0..cols];
        switch (mat_type) {
            .f32  => dq.dequantF32(row_data, row),
            .f16  => dq.dequantF16(row_data, row),
            .q5_0 => dq.dequantQ5_0(row_data, row),
            .q8_0 => dq.dequantQ8_0(row_data, row),
            .q4_k => dq.dequantQ4K(row_data, row),
            .q6_k => dq.dequantQ6K(row_data, row),
            else  => unreachable,
        }
        var sum: f32 = 0.0;
        for (row, vec) |w, x| sum += w * x;
        out[i] = sum;
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

test "matvec: 3×3 identity" {
    const mat = [_]f32{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    const vec = [_]f32{ 1, 2, 3 };
    var out: [3]f32 = undefined;
    matvec(&out, &mat, &vec, 3, 3);
    try std.testing.expectApproxEqAbs(@as(f32, 1), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2), out[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3), out[2], 1e-6);
}

test "matvec: scaling matrix" {
    // [[2,0],[0,3]] * [4,5] = [8,15]
    const mat = [_]f32{ 2, 0, 0, 3 };
    const vec = [_]f32{ 4, 5 };
    var out: [2]f32 = undefined;
    matvec(&out, &mat, &vec, 2, 2);
    try std.testing.expectApproxEqAbs(@as(f32, 8), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 15), out[1], 1e-6);
}

test "rmsnorm: all-ones weight is pure normalisation" {
    // x=[1,2,3,4], w=[1,1,1,1]
    // rms = sqrt((1+4+9+16)/4) = sqrt(7.5)
    const x = [_]f32{ 1, 2, 3, 4 };
    const w = [_]f32{ 1, 1, 1, 1 };
    var out: [4]f32 = undefined;
    rmsnorm(&out, &x, &w, 1e-8);
    const rms = @sqrt((1.0 + 4.0 + 9.0 + 16.0) / 4.0);
    try std.testing.expectApproxEqAbs(x[0] / rms, out[0], 1e-5);
    try std.testing.expectApproxEqAbs(x[3] / rms, out[3], 1e-5);
}

test "softmax: sums to 1 and preserves order" {
    var x = [_]f32{ 1.0, 2.0, 3.0 };
    softmax(&x);
    var sum: f32 = 0;
    for (x) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-6);
    try std.testing.expect(x[2] > x[1] and x[1] > x[0]);
}

test "softmax: stable for large values" {
    var x = [_]f32{ 1000.0, 1001.0, 1002.0 };
    softmax(&x);
    var sum: f32 = 0;
    for (x) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
}

test "silu: zero → zero, positive → less than input" {
    try std.testing.expectApproxEqAbs(@as(f32, 0), silu(0), 1e-6);
    // silu(x) < x for x > 0 (sigmoid < 1)
    try std.testing.expect(silu(1.0) > 0 and silu(1.0) < 1.0);
}
