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

/// Fused Q8_0 dequant + dot product against `vec`.
///
/// Bypasses the f32 row_buf entirely — i8 values are sign-extended to f32
/// via vpmovsxbd+vcvtdq2ps and multiplied directly against the vec lane.
/// Each 32-element block: acc += scale × dot(@floatFromInt(qs[0..32]), vec_chunk)
pub fn dotQ8_0(data: []const u8, vec: []const f32) f32 {
    const n_blocks = vec.len / Q8_0_BLOCK_ELEMS;
    var total: f32 = 0.0;
    for (0..n_blocks) |b| {
        const blk = data[b * Q8_0_BLOCK_BYTES ..][0..Q8_0_BLOCK_BYTES];
        const d   = f16Bytes(blk[0..2]);
        const qs  = blk[2..][0..Q8_0_BLOCK_ELEMS];
        const base = b * Q8_0_BLOCK_ELEMS;
        var acc: @Vector(8, f32) = @splat(0.0);
        inline for (0..4) |j| {
            const raw8: @Vector(8, u8)  = qs[j * 8 ..][0..8].*;
            const qi:   @Vector(8, i8)  = @bitCast(raw8);
            const qf:   @Vector(8, f32) = @floatFromInt(qi);
            const vf:   @Vector(8, f32) = vec[base + j * 8 ..][0..8].*;
            acc += qf * vf;
        }
        total += @reduce(.Add, acc) * d;
    }
    return total;
}

/// Fused Q5_0 dequant + dot product.
///
/// Each 32-element block is processed in 4 groups of 8 via SIMD:
///   g=0: elements  0.. 7  —  lo nibbles of qs[0.. 7], qh bits  0.. 7
///   g=1: elements  8..15  —  lo nibbles of qs[8..15], qh bits  8..15
///   g=2: elements 16..23  —  hi nibbles of qs[0.. 7], qh bits 16..23
///   g=3: elements 24..31  —  hi nibbles of qs[8..15], qh bits 24..31
///
/// High-bit extraction: shift qh right by a comptime offset, isolate 8 bits
/// with {1,2,4,…,128} masks, then @min(bits, 1) via vpminud to normalise to
/// 0/1. vpmovzxbd + vcvtdq2ps convert the 5-bit values to f32 without
/// writing to an intermediate f32 row buffer.
pub fn dotQ5_0(data: []const u8, vec: []const f32) f32 {
    const n_blocks = vec.len / Q5_0_BLOCK_ELEMS;
    // {1, 2, 4, 8, 16, 32, 64, 128} — comptime mask for isolating 8 consecutive bits.
    const BIT_MASKS: @Vector(8, u32) = .{ 1, 2, 4, 8, 16, 32, 64, 128 };
    var total: f32 = 0.0;
    for (0..n_blocks) |b| {
        const blk  = data[b * Q5_0_BLOCK_BYTES ..][0..Q5_0_BLOCK_BYTES];
        const d    = f16Bytes(blk[0..2]);
        const qh   = std.mem.readInt(u32, blk[2..6], .little);
        const qs   = blk[6..22];
        const base = b * Q5_0_BLOCK_ELEMS;

        var acc: @Vector(8, f32) = @splat(0.0);
        inline for (0..4) |g| {
            const qs_off  = comptime (g & 1) * 8;  // qs[0..7] for g=0,2; qs[8..15] for g=1,3
            const use_hi  = comptime g >= 2;
            const qh_off: u5 = comptime if (g < 2) @as(u5, g * 8) else @as(u5, (g - 2) * 8 + 16);

            const bytes: @Vector(8, u8) = qs[qs_off..][0..8].*;
            const nib: @Vector(8, u8) = if (comptime use_hi)
                bytes >> @as(@Vector(8, u8), @splat(4))
            else
                bytes & @as(@Vector(8, u8), @splat(0x0F));

            // Scalar shift of qh by the comptime offset, then extract 8 bits via
            // bitmask + vpminud to normalise each to exactly 0 or 1.
            const qh_shifted: u32 = qh >> qh_off;
            const qh_v: @Vector(8, u32) = @splat(qh_shifted);
            const bits: @Vector(8, u32) = qh_v & BIT_MASKS;
            const hb_u32: @Vector(8, u32) = @min(bits, @as(@Vector(8, u32), @splat(@as(u32, 1))));
            const hb: @Vector(8, u8) = @intCast(hb_u32);

            // Combine: q5 = nibble | (hi_bit << 4)  →  0..31, then subtract 16.
            const q5_u8: @Vector(8, u8) = nib | (hb << @as(@Vector(8, u8), @splat(@as(u8, 4))));
            const q5_f:  @Vector(8, f32) = @floatFromInt(q5_u8);
            const q5:    @Vector(8, f32) = q5_f - @as(@Vector(8, f32), @splat(@as(f32, 16.0)));
            const vf:    @Vector(8, f32) = vec[base + g * 8 ..][0..8].*;
            acc += q5 * vf;
        }
        total += @reduce(.Add, acc) * d;
    }
    return total;
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

// ── Q5_K ──────────────────────────────────────────────────────────────────────
//
// Super-block layout (176 bytes for 256 elements):
//   [0..1]   f16  d       super-scale for sub-block scales
//   [2..3]   f16  dmin    super-scale for sub-block mins
//   [4..15]  u8[12] scales — same 6-bit (scale,min) packing as Q4_K
//   [16..47] u8[32] qh    — 1 high bit per element, packed 8/byte
//   [48..175] u8[128] ql  — 4 low bits per element, 2 per byte (same layout as Q4_K)
//
// High-bit packing: element at abs position n, with j = n/64, l = n%32, group = (n%64)/32.
//   bit mask = 1 << (2*j + group);  hi = (qh[l] & mask) != 0
//   Combined value: q5 = lo4 | (hi << 4)  → [0..31]; result = d*sc*q5 - dmin*mn
//
// This matches GGML dequantize_row_q5_K exactly (u1/u2 shifting pattern).

pub const Q5_K_BLOCK_BYTES = 2 + 2 + K_SCALE_SIZE + QK_K / 8 + QK_K / 2; // 176

pub fn dequantQ5K(data: []const u8, out: []f32) void {
    const n_blocks = out.len / QK_K;
    for (0..n_blocks) |b| {
        const blk  = data[b * Q5_K_BLOCK_BYTES ..][0..Q5_K_BLOCK_BYTES];
        const d    = f16Bytes(blk[0..2]);
        const dmin = f16Bytes(blk[2..4]);
        const sc_buf = blk[4..16];
        const qh   = blk[16..48];
        const ql   = blk[48..176];

        var is: usize = 0;
        var um1: u8 = 1;
        var um2: u8 = 2;
        var out_idx: usize = b * QK_K;
        var ql_off: usize = 0;

        for (0..4) |_| {
            const sm1 = scaleMin(is,     sc_buf);
            const sm2 = scaleMin(is + 1, sc_buf);
            is += 2;
            const d1 = d * @as(f32, @floatFromInt(sm1.sc));
            const m1 = dmin * @as(f32, @floatFromInt(sm1.mn));
            const d2 = d * @as(f32, @floatFromInt(sm2.sc));
            const m2 = dmin * @as(f32, @floatFromInt(sm2.mn));

            for (0..32) |l| {
                const hi: f32 = if ((qh[l] & um1) != 0) 16.0 else 0.0;
                out[out_idx] = d1 * (@as(f32, @floatFromInt(ql[ql_off + l] & 0xF)) + hi) - m1;
                out_idx += 1;
            }
            for (0..32) |l| {
                const hi: f32 = if ((qh[l] & um2) != 0) 16.0 else 0.0;
                out[out_idx] = d2 * (@as(f32, @floatFromInt(ql[ql_off + l] >> 4)) + hi) - m2;
                out_idx += 1;
            }
            ql_off += 32;
            um1 *%= 4; // wrapping multiply: 1→4→16→64→(0 but loop done)
            um2 *%= 4;
        }
    }
}

// ── Q3_K ──────────────────────────────────────────────────────────────────────
//
// Super-block layout (110 bytes for 256 elements):
//   [0..31]  u8[32] hmask — 1 high bit per element (bit j+4*(n/128) of hmask[l] for element n)
//   [32..95] u8[64] qs   — 2 low bits per element, 4 per byte
//   [96..107] u8[12] scales — 16 × 6-bit signed scale values packed via GGML bit-trick
//   [108..109] f16  d    — single super-scale (no dmin)
//
// Scale decode: read 12 bytes as 3 u32s, apply the GGML kmask bit manipulation, producing
// 16 u8 scale values each in [0..63]. Actual scale = d * (val - 32), range [-32..31].
//
// Element decode: for n in [0,128) and [128,256) halves:
//   qs_base = (n/128)*32;  shift = 2*(j%4);  m = 1 << (j + 4*(n/128))
//   lo2 = (qs[qs_base + l + group*16] >> shift) & 3
//   has_hi = (hmask[l + group*16] & m) != 0
//   val = dl * (lo2 - (has_hi ? 0 : 4))          — signed [-4..3]
//
// Matches GGML dequantize_row_q3_K exactly.

pub const Q3_K_BLOCK_BYTES = QK_K / 8 + QK_K / 4 + K_SCALE_SIZE + 2; // 110

pub fn dequantQ3K(data: []const u8, out: []f32) void {
    const n_blocks = out.len / QK_K;
    for (0..n_blocks) |b| {
        const blk   = data[b * Q3_K_BLOCK_BYTES ..][0..Q3_K_BLOCK_BYTES];
        const hmask = blk[0..32];
        const qs    = blk[32..96];
        const d_all = f16Bytes(blk[108..110]);

        // Unpack 12-byte scales into 16 u8 values via GGML's bit manipulation.
        const kmask1: u32 = 0x03030303;
        const kmask2: u32 = 0x0f0f0f0f;
        var aux: [4]u32 = undefined;
        aux[0] = std.mem.readInt(u32, blk[96..100],  .little);
        aux[1] = std.mem.readInt(u32, blk[100..104], .little);
        const tmp = std.mem.readInt(u32, blk[104..108], .little);
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2)         | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2)         | (((tmp >> 2) & kmask1) << 4);
        const sc: [16]u8 = @bitCast(aux);

        var is: usize = 0;
        var m: u8 = 1;
        var out_idx: usize = b * QK_K;

        for (0..2) |half| {
            const q_base: usize = half * 32;
            var shift: u3 = 0;
            for (0..4) |_| {
                const dl  = d_all * (@as(f32, @floatFromInt(sc[is])) - 32.0); is += 1;
                for (0..16) |l| {
                    const lo2 = @as(i32, (qs[q_base + l] >> shift) & 3);
                    const hi  = (hmask[l] & m) != 0;
                    out[out_idx] = dl * @as(f32, @floatFromInt(lo2 - if (hi) @as(i32, 0) else 4));
                    out_idx += 1;
                }
                const dl2 = d_all * (@as(f32, @floatFromInt(sc[is])) - 32.0); is += 1;
                for (0..16) |l| {
                    const lo2 = @as(i32, (qs[q_base + l + 16] >> shift) & 3);
                    const hi  = (hmask[l + 16] & m) != 0;
                    out[out_idx] = dl2 * @as(f32, @floatFromInt(lo2 - if (hi) @as(i32, 0) else 4));
                    out_idx += 1;
                }
                shift = @intCast((@as(u8, shift) +% 2) & 7);
                m *%= 2; // wrapping left-shift by 1: 1→2→4→8→(16 but loop ends)
            }
        }
    }
}

// ── Q5_1 ──────────────────────────────────────────────────────────────────────
//
// Block layout (24 bytes for 32 elements):
//   [0..1]  f16 d    — scale
//   [2..3]  f16 m    — offset (min)
//   [4..7]  u8[4] qh — 1 high bit per element, packed 8/byte
//   [8..23] u8[16] qs — 4 low bits per element, 2 per byte (nibble-packed)
//
// Dequant: q5 = lo4 | (hi << 4);  val = d * q5 + m   — unsigned range [0..31]

pub const Q5_1_BLOCK_ELEMS = 32;
pub const Q5_1_BLOCK_BYTES = 2 + 2 + 4 + 16; // 24

pub fn dequantQ5_1(data: []const u8, out: []f32) void {
    const n_blocks = out.len / Q5_1_BLOCK_ELEMS;
    for (0..n_blocks) |b| {
        const blk = data[b * Q5_1_BLOCK_BYTES ..][0..Q5_1_BLOCK_BYTES];
        const d   = f16Bytes(blk[0..2]);
        const m   = f16Bytes(blk[2..4]);
        const qh  = blk[4..8];
        const qs  = blk[8..24];
        for (0..16) |j| {
            const hi0: u8 = @intCast(((qh[j / 8] >> @intCast(j      % 8)) & 1) << 4);
            const hi1: u8 = @intCast(((qh[(j + 16) / 8] >> @intCast((j + 16) % 8)) & 1) << 4);
            const q0 = @as(f32, @floatFromInt((qs[j] & 0xF) | hi0));
            const q1 = @as(f32, @floatFromInt((qs[j] >> 4)  | hi1));
            out[b * Q5_1_BLOCK_ELEMS + j]      = d * q0 + m;
            out[b * Q5_1_BLOCK_ELEMS + j + 16] = d * q1 + m;
        }
    }
}

// ── IQ4_NL ────────────────────────────────────────────────────────────────────
//
// Block layout (18 bytes for 32 elements):
//   [0..1]  f16 d  — scale
//   [2..17] u8[16] — 4-bit nibble per element, 2 per byte
//
// Uses a fixed non-linear lookup table instead of linear scale.

pub const IQ4_NL_BLOCK_ELEMS = 32;
pub const IQ4_NL_BLOCK_BYTES = 2 + IQ4_NL_BLOCK_ELEMS / 2; // 18

const IQ4_NL_TABLE: [16]f32 = .{
    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0,
       1.0,   13.0,  25.0,  38.0,  53.0,  69.0,  89.0, 113.0,
};

pub fn dequantIQ4NL(data: []const u8, out: []f32) void {
    const n_blocks = out.len / IQ4_NL_BLOCK_ELEMS;
    for (0..n_blocks) |b| {
        const blk = data[b * IQ4_NL_BLOCK_BYTES ..][0..IQ4_NL_BLOCK_BYTES];
        const d   = f16Bytes(blk[0..2]);
        const qs  = blk[2..18];
        for (0..16) |j| {
            out[b * IQ4_NL_BLOCK_ELEMS + j]      = d * IQ4_NL_TABLE[qs[j] & 0xF];
            out[b * IQ4_NL_BLOCK_ELEMS + j + 16] = d * IQ4_NL_TABLE[qs[j] >> 4];
        }
    }
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

test "dotQ8_0: matches dequant+scalar-dot" {
    // Craft 2 Q8_0 blocks and check that dotQ8_0 == sum(dequant * vec)
    var data: [Q8_0_BLOCK_BYTES * 2]u8 = @splat(0);
    var vec:  [Q8_0_BLOCK_ELEMS * 2]f32 = undefined;
    // block 0: scale = 0.5, values = 1..32
    data[0] = 0x00; data[1] = 0x38; // f16(0.5)
    for (0..32) |i| data[2 + i] = @intCast(i + 1);
    // block 1: scale = 0.25, values = -16..15
    data[Q8_0_BLOCK_BYTES + 0] = 0x00; data[Q8_0_BLOCK_BYTES + 1] = 0x34; // f16(0.25)
    for (0..32) |i| data[Q8_0_BLOCK_BYTES + 2 + i] = @bitCast(@as(i8, @intCast(@as(i32, @intCast(i)) - 16)));
    for (&vec, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.1 - 3.0;

    // reference via dequant
    var decoded: [Q8_0_BLOCK_ELEMS * 2]f32 = undefined;
    dequantQ8_0(&data, &decoded);
    var expected: f32 = 0;
    for (decoded, vec) |d, v| expected += d * v;

    const got = dotQ8_0(&data, &vec);
    try std.testing.expectApproxEqAbs(expected, got, 1e-3);
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

test "dotQ5_0: matches dequant+scalar-dot" {
    var data: [Q5_0_BLOCK_BYTES * 2]u8 = @splat(0);
    var vec: [Q5_0_BLOCK_ELEMS * 2]f32 = undefined;
    // block 0: d=0.5, qh=0xAAAAAAAA (every other bit), nibbles vary
    data[0] = 0x00; data[1] = 0x38; // f16(0.5)
    data[2] = 0xAA; data[3] = 0xAA; data[4] = 0xAA; data[5] = 0xAA;
    for (0..16) |j| data[6 + j] = @intCast((j & 0xF) | (((j + 3) & 0xF) << 4));
    // block 1: d=0.125, qh=0, all nibbles = 5
    data[Q5_0_BLOCK_BYTES + 0] = 0x00; data[Q5_0_BLOCK_BYTES + 1] = 0x30; // f16(0.125)
    for (data[Q5_0_BLOCK_BYTES + 6 .. Q5_0_BLOCK_BYTES + 22]) |*b| b.* = 0x55;
    for (&vec, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.1 - 3.0;

    var decoded: [Q5_0_BLOCK_ELEMS * 2]f32 = undefined;
    dequantQ5_0(&data, &decoded);
    var expected: f32 = 0;
    for (decoded, vec) |d, v| expected += d * v;

    const got = dotQ5_0(&data, &vec);
    try std.testing.expectApproxEqAbs(expected, got, 1e-3);
}

test "dequantQ4K: zero block gives all zeros" {
    var blk: [Q4_K_BLOCK_BYTES]u8 = @splat(0);
    // d=0, dmin=0 → everything is 0
    var out: [QK_K]f32 = undefined;
    dequantQ4K(&blk, &out);
    for (out) |v| try std.testing.expectApproxEqAbs(@as(f32, 0.0), v, 1e-6);
}
