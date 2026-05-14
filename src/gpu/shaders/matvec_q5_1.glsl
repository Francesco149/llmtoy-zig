#version 450
// Q5_1 × fp32 matrix-vector multiply.
// Block layout (24 bytes, QK5_1 = 32 elements):
//   [0..1]  f16  d  – scale
//   [2..3]  f16  m  – addend
//   [4..7]  u32  qh – 5th bit per element: bit i → bit 4 of element i
//   [8..23] u8[16] qs – 4-bit nibbles, 2 per byte; element i: nibble i%2 of byte i/2
//
// Decode: q5 = (qs[i/2] >> (4*(i%2)) & 0xF) | (((qh>>i)&1) << 4)
//          x  = d * float(q5) + m

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly buffer MatBuf  { uint mat[];      };
layout(std430, set = 0, binding = 1) readonly buffer VecBuf  { float vec_in[];  };
layout(std430, set = 0, binding = 2) writeonly buffer OutBuf { float vec_out[]; };

layout(push_constant) uniform PC { uint rows; uint cols; } pc;

// cols must be a multiple of 32 (QK5_1)
#define QK5_1    32u
#define BLK_SIZE 24u

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

void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= pc.rows) return;

    uint n_blocks = pc.cols / QK5_1;
    float sum = 0.0;

    for (uint b = 0u; b < n_blocks; b++) {
        uint blk = (row * n_blocks + b) * BLK_SIZE;
        float d = f16_to_f32(u16_at(blk));
        float m = f16_to_f32(u16_at(blk + 2u));
        // BLK_SIZE=24 is divisible by 4 so block starts are 4-byte aligned.
        // qh is at intra-block offset 4 (also aligned) — read as a single uint.
        uint qh = mat[(blk + 4u) >> 2u];

        uint vec_base = b * QK5_1;
        for (uint i = 0u; i < QK5_1; i++) {
            uint q4 = (byte_at(blk + 8u + (i >> 1u)) >> ((i & 1u) << 2u)) & 0xFu;
            uint q5 = q4 | (((qh >> i) & 1u) << 4u);
            sum = fma(d * float(q5) + m, vec_in[vec_base + i], sum);
        }
    }

    vec_out[row] = sum;
}
