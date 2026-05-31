#version 450
// Four-row workgroup variant of expert_gate_up_id_q3_k_q8_1.glsl.
//
// Each 128-thread workgroup contains four 32-lane row subgroups. This preserves
// the original one-subgroup-per-output-row math while quartering the number of
// workgroups launched for the selected-expert gate/up dispatch.

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

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

struct block_q3_K {
    uint8_t   hmask[32];
    uint8_t   qs[64];
    uint8_t   scales[12];
    float16_t d;
};

struct block_q8_1_x4 {
    f16vec2 ds[4];
    int32_t qs[32];
};

layout(std430, set = 0, binding = 0) readonly  buffer W { block_q3_K    weights[]; };
layout(std430, set = 0, binding = 1) readonly  buffer A { block_q8_1_x4 acts[];    };
layout(std430, set = 0, binding = 2) readonly  buffer I { uint          ids[];     };
layout(std430, set = 0, binding = 3) writeonly buffer D { float         out_vec[]; };

layout(push_constant) uniform PC { uint rows; uint cols; uint n_active; } pc;

int get_scale_signed(uint blk_ib, uint is) {
    const uint lo4 = (uint(weights[blk_ib].scales[is & 7u]) >> (4u * (is >> 3u))) & 0xFu;
    const uint hi2 = (uint(weights[blk_ib].scales[8u + (is & 3u)]) >> (2u * (is >> 2u))) & 0x3u;
    return int(lo4 | (hi2 << 4u)) - 32;
}

uint32_t pack4_qs(uint blk_ib, uint b) {
    return pack32(u8vec4(
        weights[blk_ib].qs[b], weights[blk_ib].qs[b + 1u],
        weights[blk_ib].qs[b + 2u], weights[blk_ib].qs[b + 3u]));
}

uint32_t pack4_hmask(uint blk_ib, uint b) {
    return pack32(u8vec4(
        weights[blk_ib].hmask[b], weights[blk_ib].hmask[b + 1u],
        weights[blk_ib].hmask[b + 2u], weights[blk_ib].hmask[b + 3u]));
}

int32_t q3_pack(uint32_t qs_word, uint32_t hm_word, uint shift, uint m_bit) {
    const uint32_t lo2 = (qs_word >> shift) & 0x03030303u;
    const uint32_t hi_x4 = (hm_word >> m_bit) & 0x01010101u;
    const uint32_t uns_word = lo2 + (hi_x4 << 2u);
    return pack32(i8vec4(unpack8(uns_word)) - i8vec4(int8_t(4)));
}

float gelu(float x) {
    const float c = 0.7978845608;
    return 0.5 * x * (1.0 + tanh(c * (x + 0.044715 * x * x * x)));
}

void q3_dot_pair(uint gate_block_base, uint up_block_base, uint lane, uint n_sub, out float gate_acc, out float up_acc) {
    gate_acc = 0.0;
    up_acc = 0.0;

    for (uint sub = lane; sub < n_sub; sub += 32u) {
        const uint q3k_idx = sub >> 3u;
        const uint s_in_q3k = sub & 7u;
        const uint half_e = s_in_q3k >> 2u;
        const uint iter = s_in_q3k & 3u;
        const uint shift = iter << 1u;
        const uint m_bit = (half_e << 2u) | iter;

        const uint x4_idx = sub >> 2u;
        const uint x4_sub = sub & 3u;
        const uint gate_blk = gate_block_base + q3k_idx;
        const uint up_blk = up_block_base + q3k_idx;

        const uint sc_idx_lo = (half_e << 3u) | (iter << 1u);
        const int gate_sc_lo = get_scale_signed(gate_blk, sc_idx_lo);
        const int gate_sc_hi = get_scale_signed(gate_blk, sc_idx_lo | 1u);
        const int up_sc_lo = get_scale_signed(up_blk, sc_idx_lo);
        const int up_sc_hi = get_scale_signed(up_blk, sc_idx_lo | 1u);
        const float gate_d_a = float(weights[gate_blk].d);
        const float up_d_a = float(weights[up_blk].d);
        const float d_b = float(acts[x4_idx].ds[x4_sub].x);

        const uint qs_off_lo = half_e * 32u;
        int32_t gate_qsum_lo = 0;
        int32_t up_qsum_lo = 0;
        [[unroll]] for (uint k = 0u; k < 4u; ++k) {
            const uint b = k * 4u;
            const int32_t act_word = acts[x4_idx].qs[x4_sub * 8u + k];
            gate_qsum_lo += dotPacked4x8EXT(
                q3_pack(pack4_qs(gate_blk, qs_off_lo + b),
                        pack4_hmask(gate_blk, b), shift, m_bit), act_word);
            up_qsum_lo += dotPacked4x8EXT(
                q3_pack(pack4_qs(up_blk, qs_off_lo + b),
                        pack4_hmask(up_blk, b), shift, m_bit), act_word);
        }

        const uint qs_off_hi = half_e * 32u + 16u;
        int32_t gate_qsum_hi = 0;
        int32_t up_qsum_hi = 0;
        [[unroll]] for (uint k = 0u; k < 4u; ++k) {
            const uint b = k * 4u;
            const int32_t act_word = acts[x4_idx].qs[x4_sub * 8u + 4u + k];
            gate_qsum_hi += dotPacked4x8EXT(
                q3_pack(pack4_qs(gate_blk, qs_off_hi + b),
                        pack4_hmask(gate_blk, 16u + b), shift, m_bit), act_word);
            up_qsum_hi += dotPacked4x8EXT(
                q3_pack(pack4_qs(up_blk, qs_off_hi + b),
                        pack4_hmask(up_blk, 16u + b), shift, m_bit), act_word);
        }

        gate_acc += d_b * gate_d_a * (float(gate_sc_lo) * float(gate_qsum_lo) + float(gate_sc_hi) * float(gate_qsum_hi));
        up_acc += d_b * up_d_a * (float(up_sc_lo) * float(up_qsum_lo) + float(up_sc_hi) * float(up_qsum_hi));
    }

    gate_acc = subgroupClusteredAdd(gate_acc, 32u);
    up_acc = subgroupClusteredAdd(up_acc, 32u);
}

void main() {
    const uint lane = gl_LocalInvocationID.x & 31u;
    const uint row_in_group = gl_LocalInvocationID.x >> 5u;
    const uint row = gl_WorkGroupID.x * 4u + row_in_group;
    const uint slot = gl_WorkGroupID.y;
    if (row >= pc.rows || slot >= pc.n_active) return;

    const uint blocks_per_row = pc.cols / 256u;
    const uint blocks_per_matrix = pc.rows * blocks_per_row;
    const uint expert = ids[slot];
    const uint expert_base = expert * 2u * blocks_per_matrix;
    const uint gate_base = expert_base + row * blocks_per_row;
    const uint up_base = expert_base + blocks_per_matrix + row * blocks_per_row;
    const uint n_sub = pc.cols / 32u;

    float gate_acc;
    float up_acc;
    q3_dot_pair(gate_base, up_base, lane, n_sub, gate_acc, up_acc);

    if (lane == 0u) {
        out_vec[slot * pc.rows + row] = gelu(gate_acc) * up_acc;
    }
}
