#version 450
// Fused gate + GELU + up Q3_K × Q8_1 shader.
//
// Computes for each row:
//   out[row] = gelu(gate_mat[row] · acts) * (up_mat[row] · acts)
//
// where `acts` are Q8_1-quantized activations and both `gate_mat` and `up_mat`
// are Q3_K with identical dimensions.
//
// This shader replaces matvec_fused_gu_q3k.glsl in the integer-dot path: one
// f32 → Q8_1 quantize once per MoE call, then N (== n_experts_used) fused
// dispatches reading the same shared Q8_1 buffer.
//
// Algorithm mirrors matvec_q3_k_q8_1.glsl: 32 threads cooperate per output
// row, each thread strides through 32-element sub-blocks, two parallel
// dot products (one per matrix), final subgroupAdd + fuse.

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

layout(std430, set = 0, binding = 0) readonly  buffer G { block_q3_K     gate[];  };
layout(std430, set = 0, binding = 1) readonly  buffer U { block_q3_K     up[];    };
layout(std430, set = 0, binding = 2) readonly  buffer A { block_q8_1_x4  acts[];  };
layout(std430, set = 0, binding = 3) writeonly buffer D { float          out_vec[]; };

layout(push_constant) uniform PC { uint rows; uint cols; } pc;

int get_scale_signed_g(uint blk_ib, uint is) {
    const uint lo4 = (uint(gate[blk_ib].scales[is & 7u]) >> (4u * (is >> 3u))) & 0xFu;
    const uint hi2 = (uint(gate[blk_ib].scales[8u + (is & 3u)]) >> (2u * (is >> 2u))) & 0x3u;
    return int(lo4 | (hi2 << 4u)) - 32;
}
int get_scale_signed_u(uint blk_ib, uint is) {
    const uint lo4 = (uint(up[blk_ib].scales[is & 7u]) >> (4u * (is >> 3u))) & 0xFu;
    const uint hi2 = (uint(up[blk_ib].scales[8u + (is & 3u)]) >> (2u * (is >> 2u))) & 0x3u;
    return int(lo4 | (hi2 << 4u)) - 32;
}

uint32_t pack4_qs_g(uint blk_ib, uint b) {
    return pack32(u8vec4(
        gate[blk_ib].qs[b    ], gate[blk_ib].qs[b + 1u],
        gate[blk_ib].qs[b + 2u], gate[blk_ib].qs[b + 3u]));
}
uint32_t pack4_hmask_g(uint blk_ib, uint b) {
    return pack32(u8vec4(
        gate[blk_ib].hmask[b    ], gate[blk_ib].hmask[b + 1u],
        gate[blk_ib].hmask[b + 2u], gate[blk_ib].hmask[b + 3u]));
}
uint32_t pack4_qs_u(uint blk_ib, uint b) {
    return pack32(u8vec4(
        up[blk_ib].qs[b    ], up[blk_ib].qs[b + 1u],
        up[blk_ib].qs[b + 2u], up[blk_ib].qs[b + 3u]));
}
uint32_t pack4_hmask_u(uint blk_ib, uint b) {
    return pack32(u8vec4(
        up[blk_ib].hmask[b    ], up[blk_ib].hmask[b + 1u],
        up[blk_ib].hmask[b + 2u], up[blk_ib].hmask[b + 3u]));
}

// Q3_K → packed-i8 conversion: q3 = lo2 + (hi << 2) - 4 per byte.
// The "- 4" must be done as a per-byte signed subtract via i8vec4 to avoid
// borrow propagation in u32 arithmetic when bytes < 4.
int32_t q3_pack(uint32_t qs_word, uint32_t hm_word, uint shift, uint m_bit) {
    const uint32_t lo2     = (qs_word >> shift) & 0x03030303u;
    const uint32_t hi_x4   = (hm_word >> m_bit) & 0x01010101u;
    const uint32_t uns_word = lo2 + (hi_x4 << 2u);
    return pack32(i8vec4(unpack8(uns_word)) - i8vec4(int8_t(4)));
}

float gelu(float x) {
    const float c = 0.7978845608; // sqrt(2/π)
    return 0.5 * x * (1.0 + tanh(c * (x + 0.044715 * x * x * x)));
}

void main() {
    const uint row = gl_WorkGroupID.x;
    if (row >= pc.rows) return;
    const uint tid = gl_LocalInvocationID.x;

    const uint n_sub        = pc.cols / 32u;
    const uint q3k_per_row  = pc.cols / 256u;
    const uint row_q3k_off  = row * q3k_per_row;

    float gate_acc = 0.0;
    float up_acc   = 0.0;

    for (uint sub = tid; sub < n_sub; sub += 32u) {
        const uint q3k_idx   = sub >> 3u;
        const uint s_in_q3k  = sub & 7u;
        const uint half_e    = s_in_q3k >> 2u;
        const uint iter      = s_in_q3k & 3u;
        const uint shift     = iter << 1u;
        const uint m_bit     = (half_e << 2u) | iter;

        const uint x4_idx = sub >> 2u;
        const uint x4_sub = sub & 3u;
        const uint blk_ib = row_q3k_off + q3k_idx;

        const uint sc_idx_lo = (half_e << 3u) | (iter << 1u);
        const int  g_sc_lo   = get_scale_signed_g(blk_ib, sc_idx_lo);
        const int  g_sc_hi   = get_scale_signed_g(blk_ib, sc_idx_lo | 1u);
        const int  u_sc_lo   = get_scale_signed_u(blk_ib, sc_idx_lo);
        const int  u_sc_hi   = get_scale_signed_u(blk_ib, sc_idx_lo | 1u);

        const float g_d = float(gate[blk_ib].d);
        const float u_d = float(up[blk_ib].d);
        const float d_b = float(acts[x4_idx].ds[x4_sub].x);

        // ── Half 0 (lo, group=0) ──
        const uint qs_off_lo = half_e * 32u;
        int32_t g_qsum_lo = 0;
        int32_t u_qsum_lo = 0;
        [[unroll]] for (uint k = 0u; k < 4u; ++k) {
            const uint b = k * 4u;
            const int32_t act_word = acts[x4_idx].qs[x4_sub * 8u + k];
            g_qsum_lo += dotPacked4x8EXT(
                q3_pack(pack4_qs_g(blk_ib, qs_off_lo + b),
                        pack4_hmask_g(blk_ib, b), shift, m_bit), act_word);
            u_qsum_lo += dotPacked4x8EXT(
                q3_pack(pack4_qs_u(blk_ib, qs_off_lo + b),
                        pack4_hmask_u(blk_ib, b), shift, m_bit), act_word);
        }

        // ── Half 1 (hi, group=1) ──
        const uint qs_off_hi = half_e * 32u + 16u;
        int32_t g_qsum_hi = 0;
        int32_t u_qsum_hi = 0;
        [[unroll]] for (uint k = 0u; k < 4u; ++k) {
            const uint b = k * 4u;
            const int32_t act_word = acts[x4_idx].qs[x4_sub * 8u + 4u + k];
            g_qsum_hi += dotPacked4x8EXT(
                q3_pack(pack4_qs_g(blk_ib, qs_off_hi + b),
                        pack4_hmask_g(blk_ib, 16u + b), shift, m_bit), act_word);
            u_qsum_hi += dotPacked4x8EXT(
                q3_pack(pack4_qs_u(blk_ib, qs_off_hi + b),
                        pack4_hmask_u(blk_ib, 16u + b), shift, m_bit), act_word);
        }

        gate_acc += d_b * g_d * (float(g_sc_lo) * float(g_qsum_lo) + float(g_sc_hi) * float(g_qsum_hi));
        up_acc   += d_b * u_d * (float(u_sc_lo) * float(u_qsum_lo) + float(u_sc_hi) * float(u_qsum_hi));
    }

    gate_acc = subgroupAdd(gate_acc);
    up_acc   = subgroupAdd(up_acc);

    if (tid == 0u) {
        out_vec[row] = gelu(gate_acc) * up_acc;
    }
}
