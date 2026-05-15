#version 450
// Q4_K (weights) × Q8_1 (activations) integer-dot matvec.
//
// 32 threads cooperate per output row. Each thread strides through the row's
// 32-element sub-blocks; the final subgroup reduction combines partial sums.
// dotPacked4x8EXT computes 4 i8 × i8 multiply-accumulates in one VOPD on RDNA3
// (integerDotProduct4x8BitPackedSignedAccelerated = true).
//
// Per 32-element sub-block math (matches llama.cpp mul_mat_vec_q4_k):
//   (d_w, dmin_w) = weights[blk].dm                — super-scales for this Q4_K block
//   (sc,  mb)     = unpack_6bit(scales, q4k_sub)   — per-sub-block (scale, min) bytes
//   d_w_sub = d_w  * sc                            — effective weight scale
//   m_w_sub = dmin * mb                            — effective weight bias
//   (d_b, s_b) = acts[..].ds                       — activation block: scale, d_b * sum(qs)
//   q_sum   = Σ dotPacked4x8EXT(nibbles, act_qs)   — 32 i8 weights × 32 i8 acts
//   contrib = d_b * d_w_sub * q_sum - m_w_sub * s_b
//
// Indexing notes:
//   The Q4_K nibble layout splits 256 elements into 4 chunks of 64; each chunk
//   uses 32 bytes of `qs` (= 8 packed u32). Sub-block s ∈ [0..7] maps to
//   chunk = s/2 and `hi = s%2` (low or high nibble of each byte).
//
// Restrictions: cols % 256 == 0 (one Q4_K block per 256 cols).

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

struct block_q4_K {
    f16vec2  dm;
    uint32_t scales[3];     // 12 bytes of packed (sc, mb) for 8 sub-blocks
    uint32_t qs[32];        // 128 bytes of 4-bit nibbles
};

struct block_q8_1_x4 {
    f16vec2  ds[4];
    int32_t  qs[32];
};

layout(std430, set = 0, binding = 0) readonly  buffer W { block_q4_K     weights[]; };
layout(std430, set = 0, binding = 1) readonly  buffer A { block_q8_1_x4  acts[];    };
layout(std430, set = 0, binding = 2) writeonly buffer D { float          out_vec[]; };

layout(push_constant) uniform PC { uint rows; uint cols; } pc;

// 6-bit (sc, mb) decode for sub-block is ∈ [0..7].
// Mirrors get_scale_min_k4 in ggml-quants.c and llama.cpp's get_dm_scale.
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
    const uint row = gl_WorkGroupID.x;
    if (row >= pc.rows) return;
    const uint tid = gl_LocalInvocationID.x;

    const uint n_sub        = pc.cols / 32u;     // sub-blocks of 32 elements
    const uint q4k_per_row  = pc.cols / 256u;
    const uint row_q4k_off  = row * q4k_per_row;

    float acc = 0.0;

    for (uint sub = tid; sub < n_sub; sub += 32u) {
        const uint q4k_idx = sub >> 3u;          // sub / 8
        const uint q4k_sub = sub & 7u;
        const uint chunk   = q4k_sub >> 1u;      // 0..3
        const uint hi      = q4k_sub & 1u;       // 0 = lo nibble, 1 = hi nibble

        const uint x4_idx  = sub >> 2u;          // sub / 4
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

    acc = subgroupAdd(acc);

    if (tid == 0u) {
        out_vec[row] = acc;
    }
}
