#version 450
// Q5_K (weights) x Q8_1 (activations) integer-dot matvec.
//
// This is a narrow fallback-removal kernel for the two Gemma4 Q5_K attention-V
// tensors. Q5_K uses the same 6-bit scale/min packing as Q4_K, plus one high
// bit per element. Contribution per 32-element sub-block:
//
//   d_b * d_w * scale * dot(q5, act_qs) - dmin_w * min * s_b

#extension GL_EXT_control_flow_attributes                 : require
#extension GL_EXT_shader_16bit_storage                    : require
#extension GL_EXT_shader_8bit_storage                     : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8   : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32  : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_EXT_integer_dot_product                     : require
#extension GL_KHR_shader_subgroup_basic                   : require
#extension GL_KHR_shader_subgroup_arithmetic              : require

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

struct block_q5_K {
    f16vec2  dm;
    uint8_t  scales[12];
    uint8_t  qh[32];
    uint8_t  qs[128];
};

struct block_q8_1_x4 {
    f16vec2  ds[4];
    int32_t  qs[32];
};

layout(std430, set = 0, binding = 0) readonly  buffer W { block_q5_K     weights[]; };
layout(std430, set = 0, binding = 1) readonly  buffer A { block_q8_1_x4  acts[];    };
layout(std430, set = 0, binding = 2) writeonly buffer D { float          out_vec[]; };

layout(push_constant) uniform PC { uint rows; uint cols; } pc;

void get_sc_mb(uint blk, uint is, out uint sc, out uint mb) {
    const uint s0 = uint(weights[blk].scales[0]) |
        (uint(weights[blk].scales[1]) << 8u) |
        (uint(weights[blk].scales[2]) << 16u) |
        (uint(weights[blk].scales[3]) << 24u);
    const uint s1 = uint(weights[blk].scales[4]) |
        (uint(weights[blk].scales[5]) << 8u) |
        (uint(weights[blk].scales[6]) << 16u) |
        (uint(weights[blk].scales[7]) << 24u);
    const uint s2 = uint(weights[blk].scales[8]) |
        (uint(weights[blk].scales[9]) << 8u) |
        (uint(weights[blk].scales[10]) << 16u) |
        (uint(weights[blk].scales[11]) << 24u);

    const uvec3 scales = uvec3(s0, s1, s2);
    const uint scalesoffs = (is & 3u) * 8u;
    const uint scidx0 = (is < 4u) ? 0u : 2u;
    const uint scidxshift1 = (is < 4u) ? scalesoffs : scalesoffs + 2u;
    const uint mbidx0 = (is < 4u) ? 1u : 2u;
    const uint mbidxshift0 = (is < 4u) ? scalesoffs : scalesoffs + 4u;
    const uint mbidxshift1 = (is < 4u) ? scalesoffs : scalesoffs + 2u;
    sc = ((scales[scidx0] >> scalesoffs) & 0xFu) | ((scales[0] >> scidxshift1) & 0x30u);
    mb = ((scales[mbidx0] >> mbidxshift0) & 0xFu) | ((scales[1] >> mbidxshift1) & 0x30u);
}

uint8_t q5_value(uint blk, uint ql_idx, uint qh_idx, uint ql_shift, uint qh_mask) {
    const uint lo = (uint(weights[blk].qs[ql_idx]) >> ql_shift) & 0x0Fu;
    const uint hi = ((uint(weights[blk].qh[qh_idx]) & qh_mask) != 0u) ? 16u : 0u;
    return uint8_t(lo | hi);
}

void main() {
    const uint row = gl_WorkGroupID.x;
    if (row >= pc.rows) return;
    const uint tid = gl_LocalInvocationID.x;

    const uint n_sub       = pc.cols / 32u;
    const uint q5k_per_row = pc.cols / 256u;
    const uint row_q5k_off = row * q5k_per_row;

    float acc = 0.0;

    for (uint sub = tid; sub < n_sub; sub += 32u) {
        const uint q5k_idx = sub >> 3u;
        const uint q5k_sub = sub & 7u;
        const uint blk_ib = row_q5k_off + q5k_idx;

        const uint ql_base = (q5k_sub >> 1u) * 32u;
        const uint ql_shift = (q5k_sub & 1u) * 4u;
        const uint qh_mask = 1u << q5k_sub;

        uint sc, mb;
        get_sc_mb(blk_ib, q5k_sub, sc, mb);

        const uint x4_idx = sub >> 2u;
        const uint x4_sub = sub & 3u;
        const uint act_off = x4_sub * 8u;

        int32_t q_sum = 0;
        [[unroll]] for (uint j = 0u; j < 8u; ++j) {
            const uint b = j * 4u;
            const int32_t qv = int32_t(pack32(u8vec4(
                q5_value(blk_ib, ql_base + b + 0u, b + 0u, ql_shift, qh_mask),
                q5_value(blk_ib, ql_base + b + 1u, b + 1u, ql_shift, qh_mask),
                q5_value(blk_ib, ql_base + b + 2u, b + 2u, ql_shift, qh_mask),
                q5_value(blk_ib, ql_base + b + 3u, b + 3u, ql_shift, qh_mask))));
            q_sum += dotPacked4x8EXT(qv, acts[x4_idx].qs[act_off + j]);
        }

        const f16vec2 dm = weights[blk_ib].dm;
        const f16vec2 ds = acts[x4_idx].ds[x4_sub];
        acc += float(ds.x) * float(dm.x) * float(sc) * float(q_sum)
             - float(dm.y) * float(mb) * float(ds.y);
    }

    acc = subgroupAdd(acc);
    if (tid == 0u) {
        out_vec[row] = acc;
    }
}
