#version 450
// Q4_K × fp32 matrix-vector multiply.
// Block layout (144 bytes, QK_K = 256 elements):
//   [0..1]    f16  d     – super-scale for sub-block scales
//   [2..3]    f16  dmin  – super-scale for sub-block mins
//   [4..15]   u8[12] scales – 8 × (scale,min) pairs packed as 6-bit values
//   [16..143] u8[128] qs – 4-bit nibbles, 2 per byte
//
// 256 elements are split into 4 chunks of 64, each chunk into 2 sub-blocks of 32.
// For element e:
//   c = e / 64, p = e % 64, s = p / 32, i = p % 32
//   sub_idx = c*2 + s
//   qs_byte = qs[c*32 + i]
//   q4 = lo nibble (s==0) or hi nibble (s==1)
//   val = d * scale[sub_idx] * q4 - dmin * min[sub_idx]
//
// Scale/min packing (get_scale_min_k4 from ggml-quants.c):
//   j < 4:  sc = scales[j] & 0x3F,      mn = scales[j+4] & 0x3F
//   j >= 4: sc = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
//           mn = (scales[j+4] >> 4)   | ((scales[j  ] >> 6) << 4)

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly buffer MatBuf  { uint mat[];     };
layout(std430, set = 0, binding = 1) readonly buffer VecBuf  { float vec_in[]; };
layout(std430, set = 0, binding = 2) writeonly buffer OutBuf { float vec_out[]; };

layout(push_constant) uniform PC { uint rows; uint cols; } pc;

// cols must be a multiple of 256 (QK_K)
#define QK_K     256u
#define BLK_SIZE 144u

// ── byte helpers ──────────────────────────────────────────────────────────────

uint byte_at(uint i) {
    return (mat[i >> 2u] >> ((i & 3u) << 3u)) & 0xffu;
}

uint u16_at(uint i) {
    return byte_at(i) | (byte_at(i + 1u) << 8u);
}

float f16_to_f32(uint h) {
    uint s = h >> 15u;
    uint e = (h >> 10u) & 0x1fu;
    uint m = h & 0x3ffu;
    if (e == 0u)  return uintBitsToFloat(s << 31u);
    if (e == 31u) return uintBitsToFloat((s << 31u) | 0x7f800000u | (m << 13u));
    return uintBitsToFloat((s << 31u) | ((e + 112u) << 23u) | (m << 13u));
}

// ── scale/min unpacking ───────────────────────────────────────────────────────
//
// Extracts effective scale (d * sc) and effective min (dmin * mn) for sub-block j.

void scale_min_k4(uint scales_off, uint j, float d_all, float dmin_all,
                  out float d_eff, out float m_eff) {
    uint sc, mn;
    if (j < 4u) {
        sc = byte_at(scales_off + j) & 0x3fu;
        mn = byte_at(scales_off + j + 4u) & 0x3fu;
    } else {
        uint extra = byte_at(scales_off + j + 4u);
        sc = (extra & 0x0fu) | ((byte_at(scales_off + j - 4u) >> 6u) << 4u);
        mn = (extra >> 4u)   | ((byte_at(scales_off + j     ) >> 6u) << 4u);
    }
    d_eff = d_all * float(sc);
    m_eff = dmin_all * float(mn);
}

// ── main ──────────────────────────────────────────────────────────────────────

void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= pc.rows) return;

    uint n_blocks = pc.cols / QK_K;
    float sum = 0.0;

    for (uint b = 0u; b < n_blocks; b++) {
        uint blk   = (row * n_blocks + b) * BLK_SIZE;
        float d_all    = f16_to_f32(u16_at(blk));
        float dmin_all = f16_to_f32(u16_at(blk + 2u));

        // Process 4 chunks of 64 elements; each chunk = 2 sub-blocks of 32.
        for (uint c = 0u; c < 4u; c++) {
            float d0, m0, d1, m1;
            scale_min_k4(blk + 4u, c * 2u,      d_all, dmin_all, d0, m0);
            scale_min_k4(blk + 4u, c * 2u + 1u, d_all, dmin_all, d1, m1);

            uint qs_base = blk + 16u + c * 32u;
            uint vec_base = b * QK_K + c * 64u;

            for (uint i = 0u; i < 32u; i++) {
                uint qb  = byte_at(qs_base + i);
                float q_lo = float(qb & 0xfu);
                float q_hi = float(qb >> 4u);
                sum = fma(d0 * q_lo - m0, vec_in[vec_base + i],       sum);
                sum = fma(d1 * q_hi - m1, vec_in[vec_base + 32u + i], sum);
            }
        }
    }

    vec_out[row] = sum;
}
