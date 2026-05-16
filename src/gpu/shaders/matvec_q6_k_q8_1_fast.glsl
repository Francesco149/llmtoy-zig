#version 450
// Experimental Q6_K x Q8_1 matvec with packed Q6 decode.
//
// The production Q6_K shader decodes each 6-bit value through byte-indexed
// helper calls. This variant keeps the same one-row/subgroup shape but loads
// Q6 low/high bits as packed u32 words and unpacks four signed i8 values at a
// time before dotPacked4x8EXT. It is a low-risk first step toward the llama.cpp
// Q6_K MMVQ structure and targets the GPU-bound lm_head path.

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

struct block_q6_K_fast {
    uint8_t   ql[128];
    uint8_t   qh[64];
    int8_t    scales[16];
    float16_t d;
};

struct block_q8_1_x4 {
    f16vec2  ds[4];
    int32_t  qs[32];
};

layout(std430, set = 0, binding = 0) readonly  buffer W { block_q6_K_fast weights[]; };
layout(std430, set = 0, binding = 1) readonly  buffer A { block_q8_1_x4   acts[];    };
layout(std430, set = 0, binding = 2) writeonly buffer D { float           out_vec[]; };

layout(push_constant) uniform PC { uint rows; uint cols; } pc;

int32_t q6_pack4(uint32_t ql_word, uint32_t qh_word, uint ql_shift, uint qh_shift) {
    const uint32_t lo = (ql_word >> ql_shift) & 0x0F0F0F0Fu;
    const uint32_t hi = ((qh_word >> qh_shift) & 0x03030303u) << 4u;
    const uint32_t q = lo | hi;
    return pack32(i8vec4(
        int8_t(int( q        & 0xFFu) - 32),
        int8_t(int((q >>  8u) & 0xFFu) - 32),
        int8_t(int((q >> 16u) & 0xFFu) - 32),
        int8_t(int((q >> 24u) & 0xFFu) - 32)));
}

uint32_t load4_ql(uint blk, uint idx) {
    return uint32_t(weights[blk].ql[idx + 0u]) |
        (uint32_t(weights[blk].ql[idx + 1u]) << 8u) |
        (uint32_t(weights[blk].ql[idx + 2u]) << 16u) |
        (uint32_t(weights[blk].ql[idx + 3u]) << 24u);
}

uint32_t load4_qh(uint blk, uint idx) {
    return uint32_t(weights[blk].qh[idx + 0u]) |
        (uint32_t(weights[blk].qh[idx + 1u]) << 8u) |
        (uint32_t(weights[blk].qh[idx + 2u]) << 16u) |
        (uint32_t(weights[blk].qh[idx + 3u]) << 24u);
}

void main() {
    const uint row = gl_WorkGroupID.x;
    if (row >= pc.rows) return;
    const uint tid = gl_LocalInvocationID.x;

    const uint n_sub       = pc.cols / 32u;
    const uint q6k_per_row = pc.cols / 256u;
    const uint row_q6k_off = row * q6k_per_row;

    float acc = 0.0;

    for (uint sub = tid; sub < n_sub; sub += 32u) {
        const uint q6k_idx = sub >> 3u;
        const uint q6k_sub = sub & 7u;
        const uint half256 = q6k_sub >> 2u;
        const uint within  = q6k_sub & 3u;
        const uint blk_ib  = row_q6k_off + q6k_idx;

        const uint ql_base = half256 * 64u + ((within & 1u) * 32u);
        const uint qh_base = half256 * 32u;
        const uint ql_shift = (within >= 2u) ? 4u : 0u;
        const uint qh_shift = within * 2u;
        const uint sc_base = half256 * 8u + within * 2u;

        const uint x4_idx = sub >> 2u;
        const uint x4_sub = sub & 3u;
        const uint act_off = x4_sub * 8u;

        int32_t q_sum0 = 0;
        int32_t q_sum1 = 0;
        [[unroll]] for (uint j = 0u; j < 8u; ++j) {
            const int32_t qv = q6_pack4(
                load4_ql(blk_ib, ql_base + j * 4u),
                load4_qh(blk_ib, qh_base + j * 4u),
                ql_shift,
                qh_shift);
            const int32_t act_word = acts[x4_idx].qs[act_off + j];
            if (j < 4u) {
                q_sum0 += dotPacked4x8EXT(qv, act_word);
            } else {
                q_sum1 += dotPacked4x8EXT(qv, act_word);
            }
        }

        const float d_w = float(weights[blk_ib].d);
        const float d_b = float(acts[x4_idx].ds[x4_sub].x);
        const float sc0 = float(weights[blk_ib].scales[sc_base + 0u]);
        const float sc1 = float(weights[blk_ib].scales[sc_base + 1u]);
        acc += d_w * d_b * (sc0 * float(q_sum0) + sc1 * float(q_sum1));
    }

    acc = subgroupAdd(acc);
    if (tid == 0u) {
        out_vec[row] = acc;
    }
}
