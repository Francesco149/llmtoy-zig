#version 450
// Experimental Q6_K x Q8_1 MMVQ-style matvec.
//
// This is the first generated-style port target. It keeps llmtoy's existing
// Q6_K and Q8_1 buffer storage, but aliases Q6_K quants through llama.cpp's
// packed16 view so each thread can use 2-byte ql/qh loads:
// - local_size_x is specialization constant BLOCK_SIZE
// - NUM_ROWS rows are accumulated by each invocation
// - each thread processes K_PER_ITER=16 columns per loop iteration
// - reduction uses shared memory across BLOCK_SIZE lanes

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

struct block_q6_K {
    uint8_t   ql[128];
    uint8_t   qh[64];
    int8_t    scales[16];
    float16_t d;
};

struct block_q6_K_packed16 {
    uint16_t  ql[64];
    uint16_t  qh[32];
    int16_t   scales[8];
    float16_t d;
};

struct block_q8_1_x4 {
    f16vec2  ds[4];
    int32_t  qs[32];
};

layout(std430, set = 0, binding = 0) readonly  buffer W { block_q6_K     weights[]; };
layout(std430, set = 0, binding = 0) readonly  buffer WP16 { block_q6_K_packed16 weights_p16[]; };
layout(std430, set = 0, binding = 1) readonly  buffer A { block_q8_1_x4  acts[];    };
layout(std430, set = 0, binding = 2) writeonly buffer D { float          out_vec[]; };

layout(push_constant) uniform PC { uint rows; uint cols; } pc;

shared float tmpsh[NUM_ROWS][BLOCK_SIZE];

int32_t cache_b_qs[4];
vec2 cache_b_ds;

i8vec2 q6_vals2(uint blk, uint ql_idx16, uint qh_idx16, uint ql_shift, uint qh_shift) {
    const uint32_t ql = (uint32_t(weights_p16[blk].ql[ql_idx16]) >> ql_shift) & 0x0F0Fu;
    const uint32_t qh = ((uint32_t(weights_p16[blk].qh[qh_idx16]) >> qh_shift) & 0x0303u) << 4u;
    const uint32_t q = ql | qh;
    return i8vec2(
        int8_t(int(q & 0xFFu) - 32),
        int8_t(int((q >> 8u) & 0xFFu) - 32));
}

i32vec4 repack4(uint ib_a, uint iqs) {
    const uint ib_k = ib_a / 8u;
    const uint iqs_k = (ib_a & 7u) * 8u + iqs;

    const uint ql_idx = (iqs_k / 32u) * 16u + (iqs_k & 15u);
    const uint ql_shift = (((iqs_k & 31u) / 16u) * 4u);

    const uint qh_idx = (iqs_k / 32u) * 8u + iqs;
    const uint qh_shift = (((iqs_k & 31u) / 8u) * 2u);

    const i8vec2 vals00 = q6_vals2(ib_k, ql_idx * 2u + 0u, qh_idx * 2u + 0u, ql_shift, qh_shift);
    const i8vec2 vals01 = q6_vals2(ib_k, ql_idx * 2u + 1u, qh_idx * 2u + 1u, ql_shift, qh_shift);
    const i8vec2 vals10 = q6_vals2(ib_k, ql_idx * 2u + 2u, qh_idx * 2u + 2u, ql_shift, qh_shift);
    const i8vec2 vals11 = q6_vals2(ib_k, ql_idx * 2u + 3u, qh_idx * 2u + 3u, ql_shift, qh_shift);
    const i8vec2 vals20 = q6_vals2(ib_k, ql_idx * 2u + 4u, qh_idx * 2u + 4u, ql_shift, qh_shift);
    const i8vec2 vals21 = q6_vals2(ib_k, ql_idx * 2u + 5u, qh_idx * 2u + 5u, ql_shift, qh_shift);
    const i8vec2 vals30 = q6_vals2(ib_k, ql_idx * 2u + 6u, qh_idx * 2u + 6u, ql_shift, qh_shift);
    const i8vec2 vals31 = q6_vals2(ib_k, ql_idx * 2u + 7u, qh_idx * 2u + 7u, ql_shift, qh_shift);

    return i32vec4(
        pack32(i8vec4(vals00.x, vals00.y, vals01.x, vals01.y)),
        pack32(i8vec4(vals10.x, vals10.y, vals11.x, vals11.y)),
        pack32(i8vec4(vals20.x, vals20.y, vals21.x, vals21.y)),
        pack32(i8vec4(vals30.x, vals30.y, vals31.x, vals31.y)));
}

float get_d_scale(uint ib_a, uint iqs) {
    const uint ib_k = ib_a / 8u;
    const uint iqs_k = (ib_a & 7u) * 8u + iqs;
    return float(weights[ib_k].d) * float(weights[ib_k].scales[iqs_k / 4u]);
}

float mmvq_dot_product(uint ib_a, uint iqs) {
    const i32vec4 qs_a = repack4(ib_a, iqs * 4u);
    const float d_scale = get_d_scale(ib_a, iqs * 4u);

    int32_t q_sum = 0;
    q_sum += dotPacked4x8EXT(qs_a.x, cache_b_qs[0]);
    q_sum += dotPacked4x8EXT(qs_a.y, cache_b_qs[1]);
    q_sum += dotPacked4x8EXT(qs_a.z, cache_b_qs[2]);
    q_sum += dotPacked4x8EXT(qs_a.w, cache_b_qs[3]);

    return cache_b_ds.x * d_scale * float(q_sum);
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

    const uint q6k_per_row = pc.cols / 256u;
    const uint num_iters = (pc.cols + K_PER_ITER * BLOCK_SIZE - 1u) / (K_PER_ITER * BLOCK_SIZE);
    for (uint i = 0u; i < num_iters; ++i) {
        const uint col = i * BLOCK_SIZE * K_PER_ITER + tid * K_PER_ITER;
        if (col >= pc.cols) continue;
        preload_q8(col);
        const uint iqs = tid & 1u;
        [[unroll]] for (uint n = 0u; n < NUM_ROWS; ++n) {
            const uint row = first_row + n;
            if (row < pc.rows) {
                const uint ib_a = row * q6k_per_row * 8u + (col >> 5u);
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
