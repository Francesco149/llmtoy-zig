#version 450
// Batched expert-id IQ4_NL down projection with 16 lanes per row.

#extension GL_EXT_control_flow_attributes                 : require
#extension GL_EXT_shader_16bit_storage                    : require
#extension GL_EXT_shader_8bit_storage                     : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8   : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32  : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_KHR_shader_subgroup_basic                   : require
#extension GL_KHR_shader_subgroup_arithmetic              : require

layout(local_size_x = 16, local_size_y = 1, local_size_z = 1) in;

struct block_iq4_nl {
    float16_t d;
    uint8_t   qs[16];
};

struct block_q8_1_x4 {
    f16vec2  ds[4];
    int32_t  qs[32];
};

layout(std430, set = 0, binding = 0) readonly  buffer W { block_iq4_nl weights[]; };
layout(std430, set = 0, binding = 1) readonly  buffer A { block_q8_1_x4 acts[];    };
layout(std430, set = 0, binding = 2) readonly  buffer I { uint          ids[];     };
layout(std430, set = 0, binding = 3) readonly  buffer S { float         scales[];  };
layout(std430, set = 0, binding = 4) writeonly buffer D { float         out_vec[]; };

layout(push_constant) uniform PC { uint rows; uint cols; uint n_active; } pc;

const float IQ4_NL_TABLE[16] = float[16](
    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0,
       1.0,   13.0,  25.0,  38.0,  53.0,  69.0,  89.0, 113.0
);

int act_byte(int32_t word, uint lane) {
    const uint shift = (3u - lane) * 8u;
    return int((word << shift) >> 24);
}

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

    for (uint sub = tid; sub < n_sub; sub += 16u) {
        const uint blk_ib = row_blk_off + sub;
        const uint x4_idx = slot * x4_per_vec + (sub >> 2u);
        const uint x4_sub = sub & 3u;
        const uint act_off = x4_sub * 8u;

        float block_sum = 0.0;
        [[unroll]] for (uint j = 0u; j < 16u; ++j) {
            const uint q = uint(weights[blk_ib].qs[j]);
            const uint word_idx = j >> 2u;
            const uint lane = j & 3u;
            const int a0 = act_byte(acts[x4_idx].qs[act_off + word_idx], lane);
            const int a1 = act_byte(acts[x4_idx].qs[act_off + 4u + word_idx], lane);
            block_sum += IQ4_NL_TABLE[q & 0x0Fu] * float(a0);
            block_sum += IQ4_NL_TABLE[q >> 4u]   * float(a1);
        }

        acc += float(weights[blk_ib].d) * float(acts[x4_idx].ds[x4_sub].x) * block_sum;
    }

    acc = subgroupAdd(acc);
    if (tid == 0u) {
        out_vec[slot * pc.rows + row] = acc * scales[slot];
    }
}
