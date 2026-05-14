#version 450
// Q5_0 × fp32 matrix-vector multiply.
// Block layout (22 bytes, QK5_0 = 32 elements):
//   [0..1]  f16 d    – scale; x = d * (q5 - 16)
//   [2..5]  u32 qh   – 5th bit per element: bit e is the 5th bit of element e
//   [6..21] u8[16] qs – interleaved: lo nibble = element j, hi nibble = element j+16
//
// BLK_SIZE=22 is not divisible by 4, so qh may be unaligned → use u32_at().
//
// Decode (j = 0..15):
//   lo: q5 = (qs[j] & 0xF) | ((qh>>j      ) & 1) << 4  → x = d*(q5-16)
//   hi: q5 = (qs[j] >> 4)  | ((qh>>(j+16u)) & 1) << 4  → x = d*(q5-16)

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly buffer MatBuf  { uint mat[];      };
layout(std430, set = 0, binding = 1) readonly buffer VecBuf  { float vec_in[];  };
layout(std430, set = 0, binding = 2) writeonly buffer OutBuf { float vec_out[]; };

layout(push_constant) uniform PC { uint rows; uint cols; } pc;

// cols must be a multiple of 32 (QK5_0)
#define QK5_0    32u
#define BLK_SIZE 22u

uint byte_at(uint i) {
    return (mat[i >> 2u] >> ((i & 3u) << 3u)) & 0xffu;
}

uint u16_at(uint i) {
    return byte_at(i) | (byte_at(i + 1u) << 8u);
}

// Unaligned little-endian uint32 read.
uint u32_at(uint i) {
    uint lo_shift = (i & 3u) << 3u;
    if (lo_shift == 0u) return mat[i >> 2u];
    return (mat[i >> 2u] >> lo_shift) | (mat[(i >> 2u) + 1u] << (32u - lo_shift));
}

float f16_to_f32(uint h) {
    uint s = h >> 15u;
    uint e = (h >> 10u) & 0x1fu;
    uint m = h & 0x3ffu;
    if (e == 0u)  return uintBitsToFloat(s << 31u);
    if (e == 31u) return uintBitsToFloat((s << 31u) | 0x7f800000u | (m << 13u));
    return uintBitsToFloat((s << 31u) | ((e + 112u) << 23u) | (m << 13u));
}

void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= pc.rows) return;

    uint n_blocks = pc.cols / QK5_0;
    float sum = 0.0;

    for (uint b = 0u; b < n_blocks; b++) {
        uint blk = (row * n_blocks + b) * BLK_SIZE;
        float d = f16_to_f32(u16_at(blk));
        uint qh = u32_at(blk + 2u);

        uint vec_base = b * QK5_0;
        for (uint j = 0u; j < 16u; j++) {
            uint qb   = byte_at(blk + 6u + j);
            uint q5_lo = (qb & 0xFu)  | (((qh >> j)        & 1u) << 4u);
            uint q5_hi = (qb >> 4u)   | (((qh >> (j + 16u)) & 1u) << 4u);
            sum = fma(d * (float(q5_lo) - 16.0), vec_in[vec_base + j],        sum);
            sum = fma(d * (float(q5_hi) - 16.0), vec_in[vec_base + j + 16u],  sum);
        }
    }

    vec_out[row] = sum;
}
