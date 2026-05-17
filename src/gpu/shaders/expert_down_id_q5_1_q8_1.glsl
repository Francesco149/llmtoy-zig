#version 450
// Batched expert-id Q5_1 down projection.
//
// One dispatch covers [active expert slot, output row]. `ids[slot]` selects the
// expert inside the flat [n_experts, rows, cols] weight tensor. `scales[slot]`
// is fused into the output, matching llama.cpp's MUL_MAT_ID_MUL shape.

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

struct block_q5_1 {
    float16_t d;
    float16_t m;
    uint16_t  qh[2];
    uint8_t   qs[16];
};

struct block_q8_1_x4 {
    f16vec2  ds[4];
    int32_t  qs[32];
};

layout(std430, set = 0, binding = 0) readonly  buffer W { block_q5_1    weights[]; };
layout(std430, set = 0, binding = 1) readonly  buffer A { block_q8_1_x4 acts[];    };
layout(std430, set = 0, binding = 2) readonly  buffer I { uint          ids[];     };
layout(std430, set = 0, binding = 3) readonly  buffer S { float         scales[];  };
layout(std430, set = 0, binding = 4) writeonly buffer D { float         out_vec[]; };

layout(push_constant) uniform PC { uint rows; uint cols; uint n_active; } pc;

void main() {
    const uint row = gl_WorkGroupID.x;
    const uint slot = gl_WorkGroupID.y;
    if (row >= pc.rows || slot >= pc.n_active) return;
    const uint tid = gl_LocalInvocationID.x;

    const uint n_sub = pc.cols / 32u;
    const uint x4_per_vec = (pc.cols + 127u) / 128u;
    const uint expert = ids[slot];
    const uint row_blk_off = (expert * pc.rows + row) * n_sub;

    float acc = 0.0;

    for (uint sub = tid; sub < n_sub; sub += 32u) {
        const uint blk_ib = row_blk_off + sub;
        const uint x4_idx = slot * x4_per_vec + (sub >> 2u);
        const uint x4_sub = sub & 3u;

        const float d_a = float(weights[blk_ib].d);
        const float m_a = float(weights[blk_ib].m);
        const float d_b = float(acts[x4_idx].ds[x4_sub].x);
        const float s_b = float(acts[x4_idx].ds[x4_sub].y);

        const uint32_t qh = uint32_t(weights[blk_ib].qh[0])
                         | (uint32_t(weights[blk_ib].qh[1]) << 16u);

        int32_t q_sum = 0;
        [[unroll]] for (uint iqs = 0u; iqs < 4u; ++iqs) {
            const uint b = iqs * 4u;
            const uint32_t vui = pack32(u8vec4(
                weights[blk_ib].qs[b], weights[blk_ib].qs[b + 1u],
                weights[blk_ib].qs[b + 2u], weights[blk_ib].qs[b + 3u]));

            const uint32_t qh_sh = qh >> (4u * iqs);
            const uint32_t qh_lo = (qh_sh & 0xFu) * 0x02040810u & 0x10101010u;
            const uint32_t qh_hi = ((qh_sh >> 16u) & 0xFu) * 0x02040810u & 0x10101010u;

            const int32_t v0 = int32_t( vui        & 0x0F0F0F0Fu | qh_lo);
            const int32_t v1 = int32_t((vui >> 4u) & 0x0F0F0F0Fu | qh_hi);

            const int32_t act_lo = acts[x4_idx].qs[x4_sub * 8u + iqs];
            const int32_t act_hi = acts[x4_idx].qs[x4_sub * 8u + 4u + iqs];
            q_sum += dotPacked4x8EXT(v0, act_lo);
            q_sum += dotPacked4x8EXT(v1, act_hi);
        }

        acc += d_a * d_b * float(q_sum) + m_a * s_b;
    }

    acc = subgroupAdd(acc);
    if (tid == 0u) {
        out_vec[slot * pc.rows + row] = acc * scales[slot];
    }
}
