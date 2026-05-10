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

// ── Q5_0 ──────────────────────────────────────────────────────────────────────
//
// Block layout (22 bytes for 32 elements):
//   [0..1]  f16   scale `d`
//   [2..5]  u8[4] qh  — 1 high bit per element, packed as a little-endian u32
//   [6..21] u8[16] qs — 4 low bits per element, 2 per byte (nibble-packed)
//
// Dequant: val = d * (x - 16), where x = (nibble | high_bit<<4) ∈ 0..31

pub const Q5_0_BLOCK_ELEMS = 32;
pub const Q5_0_BLOCK_BYTES = 2 + 4 + Q5_0_BLOCK_ELEMS / 2; // 22

pub fn dequantQ5_0(data: []const u8, out: []f32) void {
    const n_blocks = out.len / Q5_0_BLOCK_ELEMS;
    for (0..n_blocks) |b| {
        const blk = data[b * Q5_0_BLOCK_BYTES ..][0..Q5_0_BLOCK_BYTES];
        const d   = f16Bytes(blk[0..2]);
        const qh  = std.mem.readInt(u32, blk[2..6], .little);
        const qs  = blk[6..22];
        for (0..16) |j| {
            const hi0: u8   = @intCast(((qh >> @intCast(j))      & 1) << 4);
            const hi1: u8   = @intCast(((qh >> @intCast(j + 16)) & 1) << 4);
            const x0: i32   = @as(i32, (qs[j] & 0x0F) | hi0) - 16;
            const x1: i32   = @as(i32, (qs[j] >>    4) | hi1) - 16;
            out[b * Q5_0_BLOCK_ELEMS + j]      = d * @as(f32, @floatFromInt(x0));
            out[b * Q5_0_BLOCK_ELEMS + j + 16] = d * @as(f32, @floatFromInt(x1));
        }
    }
}

// ── Q6_K ──────────────────────────────────────────────────────────────────────
//
// Super-block layout (210 bytes for 256 elements):
//   [0..127]   u8[128]  ql  — lower 4 bits of each 6-bit value (2 per byte)
//   [128..191] u8[64]   qh  — upper 2 bits of each 6-bit value (4 per byte)
//   [192..207] i8[16]   sc  — signed 8-bit scales (one per 16-element sub-group)
//   [208..209] f16       d  — super-block scale
//
// Reconstruction for a pair from ql[l] / ql[l+32] and qh[l]:
//   q1 = (ql[l]    & 0xF) | ((qh[l] & 0x03) << 4)  — lo nibble, qh bits 0-1
//   q2 = (ql[l+32] & 0xF) | (((qh[l]>>2) & 0x03) << 4)
//   q3 = (ql[l]    >>  4) | (((qh[l]>>4) & 0x03) << 4)
//   q4 = (ql[l+32] >>  4) | (((qh[l]>>6) & 0x03) << 4)
//   signed_val = cast_i8(q) - 32   →  range −32..31
//   out = d × sc[sub_group] × signed_val

pub const Q6_K_BLOCK_BYTES = 128 + 64 + 16 + 2; // 210

pub fn dequantQ6K(data: []const u8, out: []f32) void {
    const n_blocks = out.len / QK_K;
    for (0..n_blocks) |b| {
        const blk = data[b * Q6_K_BLOCK_BYTES ..][0..Q6_K_BLOCK_BYTES];
        const ql  = blk[0..128];
        const qh  = blk[128..192];
        const sc  = blk[192..208]; // int8 scales
        const d   = f16Bytes(blk[208..210]);

        var ql_off: usize = 0;
        var qh_off: usize = 0;
        var sc_off: usize = 0;
        var chunk: usize = 0;
        while (chunk < QK_K) : ({
            chunk  += 128;
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }) {
            for (0..32) |l| {
                const is = l / 16; // 0 or 1 within this chunk's 8 scales

                const q1 = q6val(ql[ql_off + l],      qh[qh_off + l], 0, 0);
                const q2 = q6val(ql[ql_off + l + 32], qh[qh_off + l], 2, 0);
                const q3 = q6val(ql[ql_off + l],      qh[qh_off + l], 4, 4);
                const q4 = q6val(ql[ql_off + l + 32], qh[qh_off + l], 6, 4);

                const s0 = d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc[sc_off + is + 0]))));
                const s2 = d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc[sc_off + is + 2]))));
                const s4 = d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc[sc_off + is + 4]))));
                const s6 = d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc[sc_off + is + 6]))));

                const base = b * QK_K + chunk + l;
                out[base +  0] = s0 * q1;
                out[base + 32] = s2 * q2;
                out[base + 64] = s4 * q3;
                out[base + 96] = s6 * q4;
            }
        }
    }
}

/// Reconstruct one signed 6-bit value from its packed parts, subtract 32.
/// ql_byte: the byte holding the 4 low bits (select lo or hi nibble via ql_shift)
/// qh_byte: the byte holding the 2 high bits (select the 2-bit field via qh_shift)
fn q6val(ql_byte: u8, qh_byte: u8, qh_shift: u4, ql_shift: u4) f32 {
    const lo4: u8 = (ql_byte >> @intCast(ql_shift)) & 0x0F;
    const hi2: u8 = (qh_byte >> @intCast(qh_shift)) & 0x03;
    const raw: u8 = lo4 | (hi2 << 4); // 6-bit value, 0..63
    return @floatFromInt(@as(i32, raw) - 32);
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
