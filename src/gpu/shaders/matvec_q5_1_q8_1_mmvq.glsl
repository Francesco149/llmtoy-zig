#version 450
// Experimental Q5_1 x Q8_1 MMVQ-style matvec.
//
// Mirrors llama.cpp's DATA_A_Q5_1 legacy-quant path in mul_mat_vecq.comp:
// K_PER_ITER=8, packed16 quant reads, packed32 scale/min reads, Q8_1 block
// cache, and subgroup-plus-shared-memory reduction. Bench-only until it wins.

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

const uint K_PER_ITER = 8u;

struct block_q5_1 {
    float16_t d;
    float16_t m;
    uint16_t  qh[2];
    uint8_t   qs[16];
};

struct block_q5_1_packed16 {
    float16_t d;
    float16_t m;
    uint32_t  qh;
    uint16_t  qs[8];
};

struct block_q5_1_packed32 {
    f16vec2   dm;
    uint32_t  qh;
    uint32_t  qs[4];
};

struct block_q8_1_x4 {
    f16vec2  ds[4];
    int32_t  qs[32];
};

layout(std430, set = 0, binding = 0) readonly  buffer W { block_q5_1 weights[]; };
layout(std430, set = 0, binding = 0) readonly  buffer WP16 { block_q5_1_packed16 weights_p16[]; };
layout(std430, set = 0, binding = 0) readonly  buffer WP32 { block_q5_1_packed32 weights_p32[]; };
layout(std430, set = 0, binding = 1) readonly  buffer A { block_q8_1_x4 acts[]; };
layout(std430, set = 0, binding = 2) writeonly buffer D { float out_vec[]; };

layout(push_constant) uniform PC { uint rows; uint cols; } pc;

shared float tmpsh[NUM_ROWS][BLOCK_SIZE];

int32_t cache_b_qs[2];
vec2 cache_b_ds;

i32vec2 repack(uint ib, uint iqs) {
    const u16vec2 quants = u16vec2(
        weights_p16[ib].qs[iqs * 2u],
        weights_p16[ib].qs[iqs * 2u + 1u]);
    const uint32_t vui = pack32(quants);
    const uint32_t qh_sh = weights_p32[ib].qh >> (4u * iqs);
    const int32_t v0 = int32_t(vui & 0x0F0F0F0Fu)
                     | int32_t((qh_sh & 0xFu) * 0x02040810u & 0x10101010u);
    const int32_t v1 = int32_t((vui >> 4u) & 0x0F0F0F0Fu)
                     | int32_t(((qh_sh >> 16u) & 0xFu) * 0x02040810u & 0x10101010u);
    return i32vec2(v0, v1);
}

float mmvq_dot_product(uint ib_a, uint iqs) {
    const i32vec2 qs_a = repack(ib_a, iqs);
    const vec2 dm = vec2(weights_p32[ib_a].dm);
    int32_t q_sum = 0;
    q_sum += dotPacked4x8EXT(qs_a.x, cache_b_qs[0]);
    q_sum += dotPacked4x8EXT(qs_a.y, cache_b_qs[1]);
    return float(q_sum) * dm.x * cache_b_ds.x + dm.y * cache_b_ds.y * 0.25;
}

void preload_q8(uint col, uint tid) {
    const uint q8_block = col >> 5u;
    const uint b_qs_idx = tid & 3u;
    const uint x4_idx = q8_block >> 2u;
    const uint x4_sub = q8_block & 3u;
    cache_b_ds = vec2(acts[x4_idx].ds[x4_sub]);
    cache_b_qs[0] = acts[x4_idx].qs[x4_sub * 8u + b_qs_idx];
    cache_b_qs[1] = acts[x4_idx].qs[x4_sub * 8u + b_qs_idx + 4u];
}

void main() {
    const uint tid = gl_LocalInvocationID.x;
    const uint first_row = NUM_ROWS * gl_WorkGroupID.x;
    if (first_row >= pc.rows) return;

    float temp[NUM_ROWS];
    [[unroll]] for (uint n = 0u; n < NUM_ROWS; ++n) {
        temp[n] = 0.0;
    }

    const uint blocks_per_row = pc.cols >> 5u;
    const uint num_iters = (pc.cols + K_PER_ITER * BLOCK_SIZE - 1u) / (K_PER_ITER * BLOCK_SIZE);
    for (uint i = 0u; i < num_iters; ++i) {
        const uint col = i * BLOCK_SIZE * K_PER_ITER + tid * K_PER_ITER;
        if (col >= pc.cols) continue;
        preload_q8(col, tid);
        const uint iqs = tid & 3u;
        [[unroll]] for (uint n = 0u; n < NUM_ROWS; ++n) {
            const uint row = first_row + n;
            if (row < pc.rows) {
                const uint ib_a = row * blocks_per_row + (col >> 5u);
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
        [[unroll]] for (uint n = 0u; n < NUM_ROWS; ++n) {
            const uint row = first_row + n;
            if (row < pc.rows) out_vec[row] = temp[n];
        }
    }
}
