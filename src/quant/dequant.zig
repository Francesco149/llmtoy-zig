/// Dequantization for the GGUF quantization types used in our target models.
///
/// All functions take a raw byte slice (as it comes from the mmap'd GGUF file)
/// and a pre-allocated f32 output slice whose length must equal the number of
/// elements to decode.
///
/// Types supported:
///   F32  — memcpy, no conversion (passthrough)
///   F16  — IEEE half-precision → f32
///   Q8_0 — 8-bit symmetric, 32-element blocks
///   Q4_K — 4-bit k-quants, 256-element super-blocks (Q4_K_S / Q4_K_M share the same block layout)

const std = @import("std");

// ── F32 ───────────────────────────────────────────────────────────────────────

pub fn dequantF32(data: []const u8, out: []f32) void {
    const bytes = std.mem.sliceAsBytes(out);
    @memcpy(bytes, data[0..bytes.len]);
}

// ── F16 ───────────────────────────────────────────────────────────────────────

pub fn dequantF16(data: []const u8, out: []f32) void {
    for (out, 0..) |*o, i| {
        const raw = std.mem.readInt(u16, data[i * 2 ..][0..2], .little);
        o.* = @floatCast(@as(f16, @bitCast(raw)));
    }
}

// ── Q8_0 ──────────────────────────────────────────────────────────────────────
//
// Block layout (34 bytes for 32 elements):
//   [0..1]  f16  scale `d`
//   [2..33] i8[32]  quantized values
//
// Dequant: val = d * qs[i]

pub const Q8_0_BLOCK_ELEMS = 32;
pub const Q8_0_BLOCK_BYTES = 2 + Q8_0_BLOCK_ELEMS; // 34

pub fn dequantQ8_0(data: []const u8, out: []f32) void {
    const n_blocks = out.len / Q8_0_BLOCK_ELEMS;
    for (0..n_blocks) |b| {
        const blk = data[b * Q8_0_BLOCK_BYTES ..][0..Q8_0_BLOCK_BYTES];
        const d = f16Bytes(blk[0..2]);
        for (0..Q8_0_BLOCK_ELEMS) |i| {
            const q: i8 = @bitCast(blk[2 + i]);
            out[b * Q8_0_BLOCK_ELEMS + i] = d * @as(f32, @floatFromInt(q));
        }
    }
}

// ── Q4_K ──────────────────────────────────────────────────────────────────────
//
// Super-block layout (144 bytes for 256 elements):
//   [0..1]   f16  `d`     super-scale for sub-block scales
//   [2..3]   f16  `dmin`  super-scale for sub-block mins
//   [4..15]  u8[12] packed 6-bit sub-block scales (8 scales + 8 mins)
//   [16..143] u8[128] 4-bit values packed as: qs[k] lo-nibble = value[64*(k/32) + k%32],
//                                              qs[k] hi-nibble = value[64*(k/32) + k%32 + 32]
//
// The 256 elements are divided into 4 chunks of 64, each chunk further split into
// two 32-element sub-blocks (8 sub-blocks total, indices 0-7).
//
// Sub-block scale extraction (get_scale_min_k4 from ggml-quants.c):
//   j < 4:  sc = scales[j] & 0x3F,       mn = scales[j+4] & 0x3F
//   j >= 4: sc = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
//           mn = (scales[j+4] >> 4)   | ((scales[j  ] >> 6) << 4)

pub const QK_K = 256;
pub const K_SCALE_SIZE = 12;
pub const Q4_K_BLOCK_BYTES = 2 + 2 + K_SCALE_SIZE + QK_K / 2; // 144

pub fn dequantQ4K(data: []const u8, out: []f32) void {
    const n_blocks = out.len / QK_K;
    for (0..n_blocks) |b| {
        const blk = data[b * Q4_K_BLOCK_BYTES ..][0..Q4_K_BLOCK_BYTES];
        const d    = f16Bytes(blk[0..2]);
        const dmin = f16Bytes(blk[2..4]);
        const scales = blk[4..16];
        const qs     = blk[16..144];

        var sub: usize = 0;
        var chunk: usize = 0;
        while (chunk < QK_K) : (chunk += 64) {
            const sm1 = scaleMin(sub,     scales);
            const sm2 = scaleMin(sub + 1, scales);
            sub += 2;

            const d1 = d * @as(f32, @floatFromInt(sm1.sc));
            const m1 = dmin * @as(f32, @floatFromInt(sm1.mn));
            const d2 = d * @as(f32, @floatFromInt(sm2.sc));
            const m2 = dmin * @as(f32, @floatFromInt(sm2.mn));

            for (0..32) |i| {
                const byte = qs[chunk / 2 + i];
                const q_lo: u8 = byte & 0x0F;
                const q_hi: u8 = byte >> 4;
                out[b * QK_K + chunk + i]      = d1 * @as(f32, @floatFromInt(q_lo)) - m1;
                out[b * QK_K + chunk + 32 + i] = d2 * @as(f32, @floatFromInt(q_hi)) - m2;
            }
        }
    }
}

const ScaleMin = struct { sc: u8, mn: u8 };

fn scaleMin(j: usize, scales: []const u8) ScaleMin {
    if (j < 4) {
        return .{ .sc = scales[j] & 0x3F, .mn = scales[j + 4] & 0x3F };
    } else {
        return .{
            .sc = (scales[j + 4] & 0x0F) | (@as(u8, scales[j - 4] >> 6) << 4),
            .mn = (scales[j + 4] >> 4)   | (@as(u8, scales[j    ] >> 6) << 4),
        };
    }
}

fn f16Bytes(b: *const [2]u8) f32 {
    const raw = std.mem.readInt(u16, b, .little);
    return @floatCast(@as(f16, @bitCast(raw)));
}

// ── tests ─────────────────────────────────────────────────────────────────────

test "dequantF32: round-trips" {
    const vals = [_]f32{ 1.5, -2.5, 0.0, 3.14 };
    const bytes = std.mem.sliceAsBytes(&vals);
    var out: [4]f32 = undefined;
    dequantF32(bytes, &out);
    for (vals, out) |a, b| try std.testing.expectEqual(a, b);
}

test "dequantF16: converts half-precision" {
    // 0x3C00 = 1.0 in f16, 0xBC00 = -1.0 in f16
    const data = [_]u8{ 0x00, 0x3C, 0x00, 0xBC };
    var out: [2]f32 = undefined;
    dequantF16(&data, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), out[1], 1e-4);
}

test "dequantQ8_0: single block round-trip" {
    // Craft a Q8_0 block: scale = 0.5 (f16), values = [2, -2, 4, 0, ...]
    var blk: [Q8_0_BLOCK_BYTES]u8 = @splat(0);
    // f16(0.5) = 0x3800
    blk[0] = 0x00; blk[1] = 0x38;
    blk[2] = 2; blk[3] = @bitCast(@as(i8, -2)); blk[4] = 4;
    var out: [Q8_0_BLOCK_ELEMS]f32 = undefined;
    dequantQ8_0(&blk, &out);
    try std.testing.expectApproxEqAbs(@as(f32,  1.0), out[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), out[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32,  2.0), out[2], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32,  0.0), out[3], 1e-5);
}

test "dequantQ4K: zero block gives all zeros" {
    var blk: [Q4_K_BLOCK_BYTES]u8 = @splat(0);
    // d=0, dmin=0 → everything is 0
    var out: [QK_K]f32 = undefined;
    dequantQ4K(&blk, &out);
    for (out) |v| try std.testing.expectApproxEqAbs(@as(f32, 0.0), v, 1e-6);
}
