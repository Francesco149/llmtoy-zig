#version 450
// Experimental Q6_K x Q8_1 MMVQ-style matvec.
//
// This is the first generated-style port target. It keeps llmtoy's existing
// Q6_K and Q8_1 buffer layouts, but changes the work shape toward llama.cpp:
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

struct block_q8_1_x4 {
    f16vec2  ds[4];
    int32_t  qs[32];
};

layout(std430, set = 0, binding = 0) readonly  buffer W { block_q6_K     weights[]; };
layout(std430, set = 0, binding = 1) readonly  buffer A { block_q8_1_x4  acts[];    };
layout(std430, set = 0, binding = 2) writeonly buffer D { float          out_vec[]; };

layout(push_constant) uniform PC { uint rows; uint cols; } pc;

shared float tmpsh[NUM_ROWS][BLOCK_SIZE];

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

int32_t q6_pack4(uint blk, uint elem) {
    const uint sub = elem >> 5u;             // 32-element sub-block
    const uint word = (elem & 31u) >> 2u;    // 4 values per word
    const uint half256 = sub >> 2u;
    const uint within = sub & 3u;
    const uint ql_base = half256 * 64u + ((within & 1u) * 32u);
    const uint qh_base = half256 * 32u;
    const uint ql_shift = (within >= 2u) ? 4u : 0u;
    const uint qh_shift = within * 2u;

    const uint32_t lo = (load4_ql(blk, ql_base + word * 4u) >> ql_shift) & 0x0F0F0F0Fu;
    const uint32_t hi = ((load4_qh(blk, qh_base + word * 4u) >> qh_shift) & 0x03030303u) << 4u;
    const uint32_t q = lo | hi;
    return pack32(i8vec4(
        int8_t(int( q        & 0xFFu) - 32),
        int8_t(int((q >>  8u) & 0xFFu) - 32),
        int8_t(int((q >> 16u) & 0xFFu) - 32),
        int8_t(int((q >> 24u) & 0xFFu) - 32)));
}

float q6_scale(uint blk, uint elem) {
    const uint sub = elem >> 5u;
    const uint half256 = sub >> 2u;
    const uint within = sub & 3u;
    const uint first_or_second_16 = (elem & 31u) >> 4u;
    return float(weights[blk].scales[half256 * 8u + within * 2u + first_or_second_16]);
}

float dot16_q6_q8(uint row, uint col) {
    if (col >= pc.cols) return 0.0;

    const uint q6k_per_row = pc.cols / 256u;
    const uint blk = row * q6k_per_row + (col >> 8u);
    const uint elem = col & 255u;

    const uint q8_block = col >> 5u;
    const uint q8_word = (col & 31u) >> 2u;
    const uint x4_idx = q8_block >> 2u;
    const uint x4_sub = q8_block & 3u;
    const uint act_off = x4_sub * 8u + q8_word;
    const float d_b = float(acts[x4_idx].ds[x4_sub].x);

    const float d_w = float(weights[blk].d);
    const float sc0 = q6_scale(blk, elem);
    const float sc1 = q6_scale(blk, elem + 8u);

    int32_t sum0 = 0;
    int32_t sum1 = 0;
    [[unroll]] for (uint k = 0u; k < 2u; ++k) {
        const uint e = elem + k * 4u;
        sum0 += dotPacked4x8EXT(q6_pack4(blk, e), acts[x4_idx].qs[act_off + k]);
    }
    [[unroll]] for (uint k = 0u; k < 2u; ++k) {
        const uint e = elem + 8u + k * 4u;
        sum1 += dotPacked4x8EXT(q6_pack4(blk, e), acts[x4_idx].qs[act_off + 2u + k]);
    }

    return d_w * d_b * (sc0 * float(sum0) + sc1 * float(sum1));
}

void main() {
    const uint tid = gl_LocalInvocationID.x;
    const uint first_row = NUM_ROWS * gl_WorkGroupID.x;
    if (first_row >= pc.rows) return;

    float temp[NUM_ROWS];
    [[unroll]] for (uint n = 0u; n < NUM_ROWS; ++n) {
        temp[n] = 0.0;
    }

    const uint stride = BLOCK_SIZE * K_PER_ITER;
    for (uint base = tid * K_PER_ITER; base < pc.cols; base += stride) {
        [[unroll]] for (uint n = 0u; n < NUM_ROWS; ++n) {
            const uint row = first_row + n;
            if (row < pc.rows) {
                temp[n] += dot16_q6_q8(row, base);
            }
        }
    }

    [[unroll]] for (uint n = 0u; n < NUM_ROWS; ++n) {
        tmpsh[n][tid] = temp[n];
    }
    barrier();

    for (uint s = BLOCK_SIZE >> 1u; s > 0u; s >>= 1u) {
        if (tid < s) {
            [[unroll]] for (uint n = 0u; n < NUM_ROWS; ++n) {
                tmpsh[n][tid] += tmpsh[n][tid + s];
            }
        }
        barrier();
    }

    if (tid == 0u) {
        [[unroll]] for (uint n = 0u; n < NUM_ROWS; ++n) {
            const uint row = first_row + n;
            if (row < pc.rows) {
                out_vec[row] = tmpsh[n][0];
            }
        }
    }
}
