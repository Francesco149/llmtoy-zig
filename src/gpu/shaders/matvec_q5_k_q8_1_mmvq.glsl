#version 450
// Experimental Q5_K x Q8_1 MMVQ-style matvec.

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

layout(constant_id = 0) const uint BLOCK_SIZE = 32u;
layout(constant_id = 1) const uint NUM_ROWS   = 1u;
layout(constant_id = 2) const uint NUM_COLS   = 1u;

layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

const uint K_PER_ITER = 16u;

struct block_q5_K {
    f16vec2  dm;
    uint32_t scales[3];
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

shared float tmpsh[NUM_ROWS][BLOCK_SIZE];

int32_t cache_b_qs[4];
vec2 cache_b_ds;

uint8_t q5_value(uint ib_k, uint ql_idx, uint qh_idx, uint ql_shift, uint qh_mask) {
    const uint lo = (uint(weights[ib_k].qs[ql_idx]) >> ql_shift) & 0x0Fu;
    const uint hi = ((uint(weights[ib_k].qh[qh_idx]) & qh_mask) != 0u) ? 16u : 0u;
    return uint8_t(lo | hi);
}

i32vec4 repack4(uint ib_a, uint iqs) {
    const uint ib_k = ib_a / 8u;
    const uint sub = ib_a & 7u;
    const uint elem_base = iqs * 4u;
    const uint ql_base = (sub >> 1u) * 32u;
    const uint ql_shift = (sub & 1u) * 4u;
    const uint qh_mask = 1u << sub;

    return i32vec4(
        int32_t(pack32(u8vec4(
            q5_value(ib_k, ql_base + elem_base + 0u, elem_base + 0u, ql_shift, qh_mask),
            q5_value(ib_k, ql_base + elem_base + 1u, elem_base + 1u, ql_shift, qh_mask),
            q5_value(ib_k, ql_base + elem_base + 2u, elem_base + 2u, ql_shift, qh_mask),
            q5_value(ib_k, ql_base + elem_base + 3u, elem_base + 3u, ql_shift, qh_mask)))),
        int32_t(pack32(u8vec4(
            q5_value(ib_k, ql_base + elem_base + 4u, elem_base + 4u, ql_shift, qh_mask),
            q5_value(ib_k, ql_base + elem_base + 5u, elem_base + 5u, ql_shift, qh_mask),
            q5_value(ib_k, ql_base + elem_base + 6u, elem_base + 6u, ql_shift, qh_mask),
            q5_value(ib_k, ql_base + elem_base + 7u, elem_base + 7u, ql_shift, qh_mask)))),
        int32_t(pack32(u8vec4(
            q5_value(ib_k, ql_base + elem_base + 8u, elem_base + 8u, ql_shift, qh_mask),
            q5_value(ib_k, ql_base + elem_base + 9u, elem_base + 9u, ql_shift, qh_mask),
            q5_value(ib_k, ql_base + elem_base + 10u, elem_base + 10u, ql_shift, qh_mask),
            q5_value(ib_k, ql_base + elem_base + 11u, elem_base + 11u, ql_shift, qh_mask)))),
        int32_t(pack32(u8vec4(
            q5_value(ib_k, ql_base + elem_base + 12u, elem_base + 12u, ql_shift, qh_mask),
            q5_value(ib_k, ql_base + elem_base + 13u, elem_base + 13u, ql_shift, qh_mask),
            q5_value(ib_k, ql_base + elem_base + 14u, elem_base + 14u, ql_shift, qh_mask),
            q5_value(ib_k, ql_base + elem_base + 15u, elem_base + 15u, ql_shift, qh_mask)))));
}

vec2 get_dm_scale(uint ib_a, uint iqs) {
    const uint ib_k = ib_a / 8u;
    const uint iqs_k = (ib_a & 7u) * 8u + iqs;
    const uint is = iqs_k / 8u;

    const uvec3 scales = uvec3(weights[ib_k].scales[0], weights[ib_k].scales[1], weights[ib_k].scales[2]);
    const uint scalesoffs = (is & 3u) * 8u;

    const uint scidx0 = (is < 4u) ? 0u : 2u;
    const uint scidxshift1 = (is < 4u) ? scalesoffs : scalesoffs + 2u;
    const uint mbidx0 = (is < 4u) ? 1u : 2u;
    const uint mbidxshift0 = (is < 4u) ? scalesoffs : scalesoffs + 4u;
    const uint mbidxshift1 = (is < 4u) ? scalesoffs : scalesoffs + 2u;

    const uint sc = ((scales[scidx0] >> scalesoffs) & 0x0Fu) | ((scales[0] >> scidxshift1) & 0x30u);
    const uint mb = ((scales[mbidx0] >> mbidxshift0) & 0x0Fu) | ((scales[1] >> mbidxshift1) & 0x30u);
    return vec2(weights[ib_k].dm) * vec2(float(sc), float(mb));
}

float mmvq_dot_product(uint ib_a, uint iqs) {
    const i32vec4 qs_a = repack4(ib_a, iqs * 4u);
    const vec2 dm_scale = get_dm_scale(ib_a, iqs * 4u);

    int32_t q_sum = 0;
    q_sum += dotPacked4x8EXT(qs_a.x, cache_b_qs[0]);
    q_sum += dotPacked4x8EXT(qs_a.y, cache_b_qs[1]);
    q_sum += dotPacked4x8EXT(qs_a.z, cache_b_qs[2]);
    q_sum += dotPacked4x8EXT(qs_a.w, cache_b_qs[3]);

    return cache_b_ds.x * dm_scale.x * float(q_sum) - dm_scale.y * (cache_b_ds.y * 0.5);
}

void preload_q8(uint col) {
    const uint q8_block = col >> 5u;
    const uint b_qs_idx = (col & 31u) >> 4u;
    const uint x4_idx = q8_block >> 2u;
    const uint x4_sub = q8_block & 3u;
    const uint act_off = x4_sub * 8u + b_qs_idx * 4u;
    cache_b_ds = vec2(acts[x4_idx].ds[x4_sub]);
    cache_b_qs[0] = acts[x4_idx].qs[act_off + 0u];
    cache_b_qs[1] = acts[x4_idx].qs[act_off + 1u];
    cache_b_qs[2] = acts[x4_idx].qs[act_off + 2u];
    cache_b_qs[3] = acts[x4_idx].qs[act_off + 3u];
}

void main() {
    const uint tid = gl_LocalInvocationID.x;
    const uint first_row = NUM_ROWS * gl_WorkGroupID.x;
    if (first_row >= pc.rows) return;

    float temp[NUM_ROWS];
    [[unroll]] for (uint n = 0u; n < NUM_ROWS; ++n) {
        temp[n] = 0.0;
    }

    const uint q5k_per_row = pc.cols / 256u;
    const uint num_iters = (pc.cols + K_PER_ITER * BLOCK_SIZE - 1u) / (K_PER_ITER * BLOCK_SIZE);
    for (uint i = 0u; i < num_iters; ++i) {
        const uint col = i * BLOCK_SIZE * K_PER_ITER + tid * K_PER_ITER;
        if (col >= pc.cols) continue;
        preload_q8(col);
        const uint iqs = tid & 1u;
        [[unroll]] for (uint n = 0u; n < NUM_ROWS; ++n) {
            const uint row = first_row + n;
            if (row < pc.rows) {
                const uint ib_a = row * q5k_per_row * 8u + (col >> 5u);
                temp[n] += mmvq_dot_product(ib_a, iqs);
            }
        }
    }

    [[unroll]] for (uint n = 0u; n < NUM_ROWS; ++n) {
        temp[n] = subgroupAdd(temp[n]);
        if (gl_SubgroupInvocationID == 0u) {
            tmpsh[n][gl_SubgroupID] = temp[n];
        }
    }
    barrier();

    if (tid == 0u) {
        [[unroll]] for (uint n = 0u; n < NUM_ROWS; ++n) {
            temp[n] = 0.0;
            for (uint s = 0u; s < gl_NumSubgroups; ++s) {
                temp[n] += tmpsh[n][s];
            }
        }
    }

    if (tid == 0u) {
        [[unroll]] for (uint n = 0u; n < NUM_ROWS; ++n) {
            const uint row = first_row + n;
            if (row < pc.rows) {
                out_vec[row] = temp[n];
            }
        }
    }
}
