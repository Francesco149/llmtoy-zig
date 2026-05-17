#version 450
// Experimental Q3_K x Q8_1 MMVQ-style matvec.
//
// Mirrors llama.cpp's DATA_A_Q3_K helper in mul_mat_vecq_funcs.glsl:
// packed16 hmask/qs loads, raw byte scales/d, Q8_1 block cache, and
// subgroup-plus-shared-memory reduction.

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

struct block_q3_K {
    uint8_t   hmask[32];
    uint8_t   qs[64];
    uint8_t   scales[12];
    float16_t d;
};

struct block_q3_K_packed16 {
    uint16_t  hmask[16];
    uint16_t  qs[32];
    uint16_t  scales[6];
    float16_t d;
};

struct block_q8_1_x4 {
    f16vec2  ds[4];
    int32_t  qs[32];
};

layout(std430, set = 0, binding = 0) readonly  buffer W { block_q3_K     weights[]; };
layout(std430, set = 0, binding = 0) readonly  buffer WP16 { block_q3_K_packed16 weights_p16[]; };
layout(std430, set = 0, binding = 1) readonly  buffer A { block_q8_1_x4  acts[];    };
layout(std430, set = 0, binding = 2) writeonly buffer D { float          out_vec[]; };

layout(push_constant) uniform PC { uint rows; uint cols; } pc;

shared float tmpsh[NUM_ROWS][BLOCK_SIZE];

int32_t cache_b_qs[4];
vec2 cache_b_ds;

i8vec2 q3_vals2(uint blk, uint qs_idx16, uint hm_idx16, uint qs_shift, uint hm_shift) {
    const uint32_t lo = (uint32_t(weights_p16[blk].qs[qs_idx16]) >> qs_shift) & 0x0303u;
    const uint32_t hi = ((uint32_t(weights_p16[blk].hmask[hm_idx16]) >> hm_shift) & 0x0101u) << 2u;
    const uint32_t q = lo | hi;
    return i8vec2(
        int8_t(int(q & 0xFFu) - 4),
        int8_t(int((q >> 8u) & 0xFFu) - 4));
}

i32vec4 repack4(uint ib_a, uint iqs) {
    const uint ib_k = ib_a / 8u;
    const uint iqs_k = (ib_a & 7u) * 8u + iqs;

    const uint qs_idx = (iqs_k / 32u) * 8u + (iqs_k & 7u);
    const uint qs_shift = ((iqs_k & 31u) / 8u) * 2u;
    const uint hm_shift = iqs_k / 8u;

    const i8vec2 vals00 = q3_vals2(ib_k, qs_idx * 2u + 0u, iqs * 2u + 0u, qs_shift, hm_shift);
    const i8vec2 vals01 = q3_vals2(ib_k, qs_idx * 2u + 1u, iqs * 2u + 1u, qs_shift, hm_shift);
    const i8vec2 vals10 = q3_vals2(ib_k, qs_idx * 2u + 2u, iqs * 2u + 2u, qs_shift, hm_shift);
    const i8vec2 vals11 = q3_vals2(ib_k, qs_idx * 2u + 3u, iqs * 2u + 3u, qs_shift, hm_shift);
    const i8vec2 vals20 = q3_vals2(ib_k, qs_idx * 2u + 4u, iqs * 2u + 4u, qs_shift, hm_shift);
    const i8vec2 vals21 = q3_vals2(ib_k, qs_idx * 2u + 5u, iqs * 2u + 5u, qs_shift, hm_shift);
    const i8vec2 vals30 = q3_vals2(ib_k, qs_idx * 2u + 6u, iqs * 2u + 6u, qs_shift, hm_shift);
    const i8vec2 vals31 = q3_vals2(ib_k, qs_idx * 2u + 7u, iqs * 2u + 7u, qs_shift, hm_shift);

    return i32vec4(
        pack32(i8vec4(vals00.x, vals00.y, vals01.x, vals01.y)),
        pack32(i8vec4(vals10.x, vals10.y, vals11.x, vals11.y)),
        pack32(i8vec4(vals20.x, vals20.y, vals21.x, vals21.y)),
        pack32(i8vec4(vals30.x, vals30.y, vals31.x, vals31.y)));
}

float get_d_scale(uint ib_a, uint iqs) {
    const uint ib_k = ib_a / 8u;
    const uint iqs_k = (ib_a & 7u) * 8u + iqs;
    const uint is = iqs_k / 4u;
    const uint lo4 = (uint(weights[ib_k].scales[is & 7u]) >> (4u * (is >> 3u))) & 0x0Fu;
    const uint hi2 = (uint(weights[ib_k].scales[8u + (is & 3u)]) >> (2u * (is >> 2u))) & 0x03u;
    return float(weights[ib_k].d) * float(int(lo4 | (hi2 << 4u)) - 32);
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

    const uint q3k_per_row = pc.cols / 256u;
    const uint num_iters = (pc.cols + K_PER_ITER * BLOCK_SIZE - 1u) / (K_PER_ITER * BLOCK_SIZE);
    for (uint i = 0u; i < num_iters; ++i) {
        const uint col = i * BLOCK_SIZE * K_PER_ITER + tid * K_PER_ITER;
        if (col >= pc.cols) continue;
        preload_q8(col);
        const uint iqs = tid & 1u;
        [[unroll]] for (uint n = 0u; n < NUM_ROWS; ++n) {
            const uint row = first_row + n;
            if (row < pc.rows) {
                const uint ib_a = row * q3k_per_row * 8u + (col >> 5u);
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
