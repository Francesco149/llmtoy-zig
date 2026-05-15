#version 450
// Q3_K (weights) × Q8_1 (activations) integer-dot matvec.
//
// Q3_K is structurally similar to Q4_K but uses 3-bit quants split across a
// 2-bit `qs` array + a 1-bit `hmask`. The block also packs 16 × 6-bit scales
// in 12 bytes using the same scheme as Q4_K. Each 32-element sub-block needs
// TWO scales — one for each 16-element half — so the dot product splits
// naturally into two halves of 4 × dotPacked4x8EXT each.
//
// Block layout (110 bytes, 256 elements):
//   [0..31]    hmask[32]   — high bit per element (32 bytes × 8 bits = 256)
//   [32..95]   qs[64]      — low 2 bits per element (64 bytes × 4 quants/byte)
//   [96..107]  scales[12]  — 16 × 6-bit (4-low + 2-high) packed scales
//   [108..109] f16 d       — super-scale
//
// Quant value (signed): q3 = lo2 + (hi << 2) - 4    ∈ [-4, 3]
// Sub-block contribution:
//   contrib = d_b * d_a * (sc_lo * Σdot_lo + sc_hi * Σdot_hi)
// where d_a = block super-scale, d_b = Q8_1 block scale, sc_* = 6-bit signed
// (raw - 32), Σdot_* = Σ dotPacked4x8EXT(q3_word, act_word) over 4 i32s.

#extension GL_EXT_control_flow_attributes                 : require
#extension GL_EXT_shader_16bit_storage                    : require
#extension GL_EXT_shader_8bit_storage                     : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8   : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16  : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32  : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_EXT_integer_dot_product                     : require
#extension GL_KHR_shader_subgroup_basic                   : require
#extension GL_KHR_shader_subgroup_arithmetic              : require

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

struct block_q3_K {
    uint8_t   hmask[32];
    uint8_t   qs[64];
    uint8_t   scales[12];
    float16_t d;
};

struct block_q8_1_x4 {
    f16vec2  ds[4];
    int32_t  qs[32];
};

layout(std430, set = 0, binding = 0) readonly  buffer W { block_q3_K     weights[]; };
layout(std430, set = 0, binding = 1) readonly  buffer A { block_q8_1_x4  acts[];    };
layout(std430, set = 0, binding = 2) writeonly buffer D { float          out_vec[]; };

layout(push_constant) uniform PC { uint rows; uint cols; } pc;

// Decode one 6-bit scale (range [0, 63], signed value = result - 32) for
// scale index `is` ∈ [0..15] from the 12-byte packed scales array.
int get_scale_signed(uint blk_ib, uint is) {
    const uint lo4 = (uint(weights[blk_ib].scales[is & 7u]) >> (4u * (is >> 3u))) & 0xFu;
    const uint hi2 = (uint(weights[blk_ib].scales[8u + (is & 3u)]) >> (2u * (is >> 2u))) & 0x3u;
    return int(lo4 | (hi2 << 4u)) - 32;
}

// Pack 4 consecutive bytes from a uint8_t array into a uint32_t (LSB = qs[b]).
uint32_t pack4_qs(uint blk_ib, uint b) {
    return pack32(u8vec4(
        weights[blk_ib].qs[b    ], weights[blk_ib].qs[b + 1u],
        weights[blk_ib].qs[b + 2u], weights[blk_ib].qs[b + 3u]));
}
uint32_t pack4_hmask(uint blk_ib, uint b) {
    return pack32(u8vec4(
        weights[blk_ib].hmask[b    ], weights[blk_ib].hmask[b + 1u],
        weights[blk_ib].hmask[b + 2u], weights[blk_ib].hmask[b + 3u]));
}

void main() {
    const uint row = gl_WorkGroupID.x;
    if (row >= pc.rows) return;
    const uint tid = gl_LocalInvocationID.x;

    const uint n_sub        = pc.cols / 32u;       // 32-element sub-blocks per row
    const uint q3k_per_row  = pc.cols / 256u;
    const uint row_q3k_off  = row * q3k_per_row;

    float acc = 0.0;

    for (uint sub = tid; sub < n_sub; sub += 32u) {
        const uint q3k_idx   = sub >> 3u;                // sub / 8 (8 sub-blocks per Q3_K block)
        const uint s_in_q3k  = sub & 7u;
        const uint half_e    = s_in_q3k >> 2u;           // 0 (first half of block) or 1 (second)
        const uint iter      = s_in_q3k & 3u;            // 0..3 inside that half
        const uint shift     = iter << 1u;               // 0, 2, 4, 6
        const uint m_bit     = (half_e << 2u) | iter;    // 0..7

        const uint x4_idx = sub >> 2u;
        const uint x4_sub = sub & 3u;
        const uint blk_ib = row_q3k_off + q3k_idx;

        // Two 6-bit signed scales — one per 16-element half.
        const uint sc_idx_lo  = (half_e << 3u) | (iter << 1u);
        const int  sc_lo      = get_scale_signed(blk_ib, sc_idx_lo);
        const int  sc_hi      = get_scale_signed(blk_ib, sc_idx_lo | 1u);

        const float d_a = float(weights[blk_ib].d);
        const float d_b = float(acts[x4_idx].ds[x4_sub].x);

        // ── Half 0 (group=0, low 16 elements) ──
        // qs bytes at half*32 .. half*32+15;  hmask bytes at 0..15.
        const uint qs_off_lo = half_e * 32u;
        int32_t q_sum_lo = 0;
        [[unroll]] for (uint k = 0u; k < 4u; ++k) {
            const uint b = k * 4u;
            const uint32_t qs_word = pack4_qs(blk_ib, qs_off_lo + b);
            const uint32_t hm_word = pack4_hmask(blk_ib, b);
            const uint32_t lo2   = (qs_word >> shift) & 0x03030303u;
            const uint32_t hi_x4 = (hm_word >> m_bit) & 0x01010101u;
            // uns_word: each byte ∈ [0, 7] = lo2 + (hi << 2). No inter-byte
            // carry yet (max byte value 7).  Subtract 4 per byte via i8vec4
            // — straight u32 subtraction would propagate borrows when any
            // byte is < 4 (the common hi=0 case), corrupting neighbours.
            const uint32_t uns_word = lo2 + (hi_x4 << 2u);
            const int32_t  q3_word  = pack32(i8vec4(unpack8(uns_word)) - i8vec4(int8_t(4)));
            const int32_t  act_word = acts[x4_idx].qs[x4_sub * 8u + k];
            q_sum_lo += dotPacked4x8EXT(q3_word, act_word);
        }

        // ── Half 1 (group=1, high 16 elements) ──
        // qs bytes at half*32+16 .. half*32+31;  hmask bytes at 16..31.
        const uint qs_off_hi = half_e * 32u + 16u;
        int32_t q_sum_hi = 0;
        [[unroll]] for (uint k = 0u; k < 4u; ++k) {
            const uint b = k * 4u;
            const uint32_t qs_word = pack4_qs(blk_ib, qs_off_hi + b);
            const uint32_t hm_word = pack4_hmask(blk_ib, 16u + b);
            const uint32_t lo2   = (qs_word >> shift) & 0x03030303u;
            const uint32_t hi_x4 = (hm_word >> m_bit) & 0x01010101u;
            // uns_word: each byte ∈ [0, 7] = lo2 + (hi << 2). No inter-byte
            // carry yet (max byte value 7).  Subtract 4 per byte via i8vec4
            // — straight u32 subtraction would propagate borrows when any
            // byte is < 4 (the common hi=0 case), corrupting neighbours.
            const uint32_t uns_word = lo2 + (hi_x4 << 2u);
            const int32_t  q3_word  = pack32(i8vec4(unpack8(uns_word)) - i8vec4(int8_t(4)));
            const int32_t  act_word = acts[x4_idx].qs[x4_sub * 8u + 4u + k];
            q_sum_hi += dotPacked4x8EXT(q3_word, act_word);
        }

        acc += d_b * d_a * (float(sc_lo) * float(q_sum_lo) + float(sc_hi) * float(q_sum_hi));
    }

    acc = subgroupAdd(acc);
    if (tid == 0u) {
        out_vec[row] = acc;
    }
}
