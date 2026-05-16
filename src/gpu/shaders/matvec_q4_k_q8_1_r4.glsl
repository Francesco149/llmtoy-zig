#version 450
// Experimental Q4_K × Q8_1 matvec: four output rows per workgroup.
//
// This keeps the scalar math identical to matvec_q4_k_q8_1.glsl, but maps
// local_size_y lanes to independent rows. It is a first MMVQ-shape probe:
// fewer workgroups, more resident rows per workgroup, same descriptor layout.

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
#extension GL_KHR_shader_subgroup_clustered               : require

layout(local_size_x = 32, local_size_y = 4, local_size_z = 1) in;

struct block_q4_K {
    f16vec2  dm;
    uint32_t scales[3];
    uint32_t qs[32];
};

struct block_q8_1_x4 {
    f16vec2  ds[4];
    int32_t  qs[32];
};

layout(std430, set = 0, binding = 0) readonly  buffer W { block_q4_K     weights[]; };
layout(std430, set = 0, binding = 1) readonly  buffer A { block_q8_1_x4  acts[];    };
layout(std430, set = 0, binding = 2) writeonly buffer D { float          out_vec[]; };

layout(push_constant) uniform PC { uint rows; uint cols; } pc;

void get_sc_mb(uint32_t s0, uint32_t s1, uint32_t s2,
               uint is, out uint sc, out uint mb)
{
    const uvec3 scales        = uvec3(s0, s1, s2);
    const uint  scalesoffs    = (is & 3) * 8;
    const uint  scidx0        = (is < 4) ? 0u : 2u;
    const uint  scidxshift1   = (is < 4) ? scalesoffs : scalesoffs + 2u;
    const uint  mbidx0        = (is < 4) ? 1u : 2u;
    const uint  mbidxshift0   = (is < 4) ? scalesoffs : scalesoffs + 4u;
    const uint  mbidxshift1   = (is < 4) ? scalesoffs : scalesoffs + 2u;
    sc = ((scales[scidx0] >> scalesoffs)    & 0xFu) | ((scales[0] >> scidxshift1) & 0x30u);
    mb = ((scales[mbidx0] >> mbidxshift0)   & 0xFu) | ((scales[1] >> mbidxshift1) & 0x30u);
}

void main() {
    const uint row = gl_WorkGroupID.x * 4u + gl_LocalInvocationID.y;
    if (row >= pc.rows) return;
    const uint tid = gl_LocalInvocationID.x;

    const uint n_sub        = pc.cols / 32u;
    const uint q4k_per_row  = pc.cols / 256u;
    const uint row_q4k_off  = row * q4k_per_row;

    float acc = 0.0;

    for (uint sub = tid; sub < n_sub; sub += 32u) {
        const uint q4k_idx = sub >> 3u;
        const uint q4k_sub = sub & 7u;
        const uint chunk   = q4k_sub >> 1u;
        const uint hi      = q4k_sub & 1u;

        const uint x4_idx  = sub >> 2u;
        const uint x4_sub  = sub & 3u;

        const uint blk_ib = row_q4k_off + q4k_idx;
        const f16vec2 dm  = weights[blk_ib].dm;
        const float d_w    = float(dm.x);
        const float dmin_w = float(dm.y);

        uint sc, mb;
        get_sc_mb(weights[blk_ib].scales[0],
                  weights[blk_ib].scales[1],
                  weights[blk_ib].scales[2],
                  q4k_sub, sc, mb);
        const float d_w_sub = d_w    * float(sc);
        const float m_w_sub = dmin_w * float(mb);

        const f16vec2 ds = acts[x4_idx].ds[x4_sub];
        const float d_b = float(ds.x);
        const float s_b = float(ds.y);

        int32_t q_sum = 0;
        const uint qs_off = chunk * 8u;
        const uint act_off = x4_sub * 8u;
        [[unroll]] for (uint j = 0u; j < 8u; ++j) {
            const uint32_t raw_w = weights[blk_ib].qs[qs_off + j];
            const int32_t  nibbles = int32_t((hi == 0u)
                ? ( raw_w        & 0x0F0F0F0Fu)
                : ((raw_w >> 4u) & 0x0F0F0F0Fu));
            const int32_t act_word = acts[x4_idx].qs[act_off + j];
            q_sum += dotPacked4x8EXT(nibbles, act_word);
        }

        acc += d_b * d_w_sub * float(q_sum) - m_w_sub * s_b;
    }

    acc = subgroupClusteredAdd(acc, 32u);

    if (tid == 0u) {
        out_vec[row] = acc;
    }
}
