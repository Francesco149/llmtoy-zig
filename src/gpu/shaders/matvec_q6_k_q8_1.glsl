#version 450
// Q6_K (weights) x Q8_1 (activations) integer-dot matvec.
//
// Q6_K stores signed 6-bit quants in 256-element super-blocks. Each 32-element
// sub-block has two i8 scales, one for each 16-element half:
//
//   val = d * scale[sub_half] * (q6 - 32)
//
// This shader mirrors the current Q*_K x Q8_1 row-per-workgroup path: one
// subgroup computes one output row, each lane strides over 32-element sub-blocks.
// It is a bridge to the endgame MMVQ kernels, but it removes the expensive CPU
// Q6_K fallback immediately, especially for Q6_K lm_head tensors.

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

struct block_q6_K {
    uint8_t   ql[128];
    uint8_t   qh[64];
    int8_t    scales[16];
    float16_t d;
};

struct block_q8_1_x4 {
    f16vec2  ds[4];
    int32_t  qs[32];
};

layout(std430, set = 0, binding = 0) readonly  buffer W { block_q6_K     weights[]; };
layout(std430, set = 0, binding = 1) readonly  buffer A { block_q8_1_x4  acts[];    };
layout(std430, set = 0, binding = 2) writeonly buffer D { float          out_vec[]; };

layout(push_constant) uniform PC { uint rows; uint cols; } pc;

int8_t q6_value(uint blk, uint ql_idx, uint qh_idx, uint ql_shift, uint qh_shift) {
    const uint lo = (uint(weights[blk].ql[ql_idx]) >> ql_shift) & 0x0Fu;
    const uint hi = ((uint(weights[blk].qh[qh_idx]) >> qh_shift) & 0x03u) << 4u;
    return int8_t(int(lo | hi) - 32);
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
            const uint b = j * 4u;
            const int32_t qv = pack32(i8vec4(
                q6_value(blk_ib, ql_base + b + 0u, qh_base + b + 0u, ql_shift, qh_shift),
                q6_value(blk_ib, ql_base + b + 1u, qh_base + b + 1u, ql_shift, qh_shift),
                q6_value(blk_ib, ql_base + b + 2u, qh_base + b + 2u, ql_shift, qh_shift),
                q6_value(blk_ib, ql_base + b + 3u, qh_base + b + 3u, ql_shift, qh_shift)));
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
