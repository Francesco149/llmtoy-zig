const std    = @import("std");
const GgmlType = @import("../gguf/types.zig").GgmlType;
const dq       = @import("../quant/dequant.zig");
pub const ThreadPool = @import("thread_pool.zig").ThreadPool;

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

/// RMS normalize without a learned weight vector: out[i] = x[i] / rms(x)
pub fn rmsnormRaw(out: []f32, x: []const f32, eps: f32) void {
    var ss: f32 = 0.0;
    for (x) |v| ss += v * v;
    const rms_inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(x.len)) + eps);
    for (out, x) |*o, v| o.* = v * rms_inv;
}

/// GELU activation (approximate, tanh variant): x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715*x³)))
pub fn gelu(x: f32) f32 {
    const c: f32 = 0.7978845608028654; // sqrt(2/π)
    return 0.5 * x * (1.0 + std.math.tanh(c * (x + 0.044715 * x * x * x)));
}

/// Dot product using 8-wide f32 SIMD with 4 independent accumulators.
///
/// 4 accumulators amortise the 5-cycle FMA latency on Zen 2 (2 FMA units,
/// each with 5-cycle latency → need ~10 in-flight ops; 4 × 8 = 32 elements
/// per iteration keeps both units busy). For AVX2 each iteration is 4 ×
/// VFMADD256, processing 32 floats per cycle at peak.
pub inline fn dotf32(a: []const f32, b: []const f32) f32 {
    const n = a.len;
    var acc0: @Vector(8, f32) = @splat(0.0);
    var acc1: @Vector(8, f32) = @splat(0.0);
    var acc2: @Vector(8, f32) = @splat(0.0);
    var acc3: @Vector(8, f32) = @splat(0.0);
    var i: usize = 0;
    while (i + 32 <= n) : (i += 32) {
        acc0 += @as(@Vector(8, f32), a[i +  0 ..][0..8].*) * @as(@Vector(8, f32), b[i +  0 ..][0..8].*);
        acc1 += @as(@Vector(8, f32), a[i +  8 ..][0..8].*) * @as(@Vector(8, f32), b[i +  8 ..][0..8].*);
        acc2 += @as(@Vector(8, f32), a[i + 16 ..][0..8].*) * @as(@Vector(8, f32), b[i + 16 ..][0..8].*);
        acc3 += @as(@Vector(8, f32), a[i + 24 ..][0..8].*) * @as(@Vector(8, f32), b[i + 24 ..][0..8].*);
    }
    var acc = (acc0 + acc1) + (acc2 + acc3);
    while (i + 8 <= n) : (i += 8) {
        acc += @as(@Vector(8, f32), a[i..][0..8].*) * @as(@Vector(8, f32), b[i..][0..8].*);
    }
    var sum = @reduce(.Add, acc);
    while (i < n) : (i += 1) sum += a[i] * b[i];
    return sum;
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

    const row_bytes = rowBytes(mat_type, cols);

    for (0..rows) |i| {
        const row_data = mat_data[i * row_bytes ..][0..row_bytes];
        if (mat_type == .q8_0) {
            out[i] = dq.dotQ8_0(row_data, vec);
            continue;
        }
        if (mat_type == .q5_0) {
            out[i] = dq.dotQ5_0(row_data, vec);
            continue;
        }
        if (mat_type == .iq4_nl) {
            out[i] = dq.dotIQ4NL(row_data, vec);
            continue;
        }
        const row = row_buf[0..cols];
        dequantRow(row_data, row, mat_type);
        out[i] = dotf32(row, vec);
    }
}

/// Bytes occupied by one row of a quantized matrix.
pub fn rowBytes(mat_type: GgmlType, cols: usize) usize {
    return switch (mat_type) {
        .f32    => cols * 4,
        .f16    => cols * 2,
        .q5_0   => (cols / dq.Q5_0_BLOCK_ELEMS) * dq.Q5_0_BLOCK_BYTES,
        .q5_1   => (cols / dq.Q5_1_BLOCK_ELEMS) * dq.Q5_1_BLOCK_BYTES,
        .q8_0   => (cols / dq.Q8_0_BLOCK_ELEMS) * dq.Q8_0_BLOCK_BYTES,
        .q4_k   => (cols / dq.QK_K) * dq.Q4_K_BLOCK_BYTES,
        .q5_k   => (cols / dq.QK_K) * dq.Q5_K_BLOCK_BYTES,
        .q6_k   => (cols / dq.QK_K) * dq.Q6_K_BLOCK_BYTES,
        .q3_k   => (cols / dq.QK_K) * dq.Q3_K_BLOCK_BYTES,
        .iq4_nl => (cols / dq.IQ4_NL_BLOCK_ELEMS) * dq.IQ4_NL_BLOCK_BYTES,
        else    => std.debug.panic("unsupported quant type in rowBytes: {s}", .{mat_type.label()}),
    };
}

/// Dequantize one row of data into a pre-allocated f32 buffer.
pub fn dequantRow(data: []const u8, out: []f32, mat_type: GgmlType) void {
    switch (mat_type) {
        .f32    => dq.dequantF32(data, out),
        .f16    => dq.dequantF16(data, out),
        .q5_0   => dq.dequantQ5_0(data, out),
        .q5_1   => dq.dequantQ5_1(data, out),
        .q8_0   => dq.dequantQ8_0(data, out),
        .q4_k   => dq.dequantQ4K(data, out),
        .q5_k   => dq.dequantQ5K(data, out),
        .q6_k   => dq.dequantQ6K(data, out),
        .q3_k   => dq.dequantQ3K(data, out),
        .iq4_nl => dq.dequantIQ4NL(data, out),
        else    => std.debug.panic("unsupported quant type in dequantRow: {s}", .{mat_type.label()}),
    }
}

/// Context passed to each worker thread for a parallel quantMatvec slice.
const RowJob = struct {
    mat_data:  []const u8,
    mat_type:  GgmlType,
    vec:       []const f32,
    out:       []f32,       // sub-slice of the caller's `out`; length == n_rows
    row_buf:   []f32,       // per-thread scratch; length == cols
    row_start: usize,       // first global row (for mat_data addressing)
    n_rows:    usize,
    cols:      usize,

    fn poolRun(ctx: *anyopaque) void {
        const job: *RowJob = @ptrCast(@alignCast(ctx));
        job.run();
    }

    fn run(job: *RowJob) void {
        const row_bytes = rowBytes(job.mat_type, job.cols);
        for (0..job.n_rows) |i| {
            const row_data = job.mat_data[(job.row_start + i) * row_bytes ..][0..row_bytes];
            if (job.mat_type == .q8_0) {
                job.out[i] = dq.dotQ8_0(row_data, job.vec);
                continue;
            }
            if (job.mat_type == .q5_0) {
                job.out[i] = dq.dotQ5_0(row_data, job.vec);
                continue;
            }
            if (job.mat_type == .iq4_nl) {
                job.out[i] = dq.dotIQ4NL(row_data, job.vec);
                continue;
            }
            const row = job.row_buf[0..job.cols];
            dequantRow(row_data, row, job.mat_type);
            job.out[i] = dotf32(row, job.vec);
        }
    }
};

/// Parallel version of quantMatvec using a persistent thread pool.
///
/// Splits output rows across workers. Each worker gets its own row_buf
/// from `scratch` (caller must supply `pool.threads.len × cols` floats).
/// Falls back to the serial path when the matrix is too small to benefit.
pub fn quantMatvecPar(
    out: []f32,
    mat_data: []const u8,
    mat_type: GgmlType,
    vec: []const f32,
    rows: usize,
    cols: usize,
    scratch: []f32,   // length >= pool.threads.len * cols
    pool: *ThreadPool,
) void {
    // With a persistent pool there is no spawn cost, so the threshold can be
    // much lower than the old 512. 64 rows-per-thread still amortises the
    // condition-variable wakeup (a few µs) without leaving parallelism on the
    // table for mid-sized matrices like wq/wo (896 rows).
    const min_rows: usize = 64;
    const n = pool.threads.len;
    const nt = @min(n, (rows + min_rows - 1) / min_rows);

    if (nt <= 1) {
        quantMatvec(out, mat_data, mat_type, vec, rows, cols, scratch[0..cols]);
        return;
    }

    var jobs: [64]RowJob = undefined;
    const base = rows / nt;
    const extra = rows % nt;
    var row: usize = 0;
    for (0..nt) |t| {
        const n_rows = base + if (t < extra) @as(usize, 1) else @as(usize, 0);
        jobs[t] = .{
            .mat_data  = mat_data,
            .mat_type  = mat_type,
            .vec       = vec,
            .out       = out[row..][0..n_rows],
            .row_buf   = scratch[t * cols ..][0..cols],
            .row_start = row,
            .n_rows    = n_rows,
            .cols      = cols,
        };
        pool.submit(RowJob.poolRun, &jobs[t]);
        row += n_rows;
    }
    pool.wait();
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

test "dotf32: matches scalar for lengths 1, 7, 8, 32, 33" {
    inline for ([_]usize{ 1, 7, 8, 32, 33 }) |n| {
        var a: [n]f32 = undefined;
        var b: [n]f32 = undefined;
        var expected: f32 = 0;
        for (0..n) |i| {
            a[i] = @as(f32, @floatFromInt(i + 1)) * 0.5;
            b[i] = @as(f32, @floatFromInt(n - i)) * 0.3;
            expected += a[i] * b[i];
        }
        try std.testing.expectApproxEqAbs(expected, dotf32(&a, &b), 1e-4);
    }
}
