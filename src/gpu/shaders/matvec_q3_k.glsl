#version 450
// Q3_K × fp32 matrix-vector multiply.
// Block layout (110 bytes, QK_K = 256 elements):
//   [0..31]    hmask  – 1 high bit per element
//   [32..95]   qs     – 2 low bits per element, 4 packed per byte
//   [96..107]  scales – 16 × 6-bit scale values (12-byte GGML packing)
//   [108..109] d      – f16 super-block scale
//
// Element ordering matches ggml dequantize_row_q3_K / dequantQ3K exactly:
//   two halves × four iters × two groups × 16 lanes.
// For element e (0..255):
//   half  = e >> 7,  iter = (e >> 5) & 3,  group = (e >> 4) & 1,  lane = e & 15
//   qs_off  = (half << 5) | (group << 4) | lane  (index into qs section)
//   hm_off  = (group << 4) | lane               (index into hmask section)
//   shift   = iter << 1                          (which 2-bit pair in qs byte)
//   m_bit   = (half << 2) | iter                (which bit in hmask byte)
//   sc_idx  = (half << 3) | (iter << 1) | group (which of 16 scale values)
//   lo2  = (qs[qs_off] >> shift) & 3
//   hi   = (hmask[hm_off] >> m_bit) & 1
//   q3   = lo2 - (hi ? 0 : 4)          – signed range [-4, 3]
//   val  = d * (scale[sc_idx] - 32) * q3

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly buffer MatBuf  { uint mat[];     };
layout(std430, set = 0, binding = 1) readonly buffer VecBuf  { float vec_in[]; };
layout(std430, set = 0, binding = 2) writeonly buffer OutBuf { float vec_out[]; };

layout(push_constant) uniform PC { uint rows; uint cols; } pc;

// cols must be a multiple of 256 (QK_K)
#define QK_K     256u
#define BLK_SIZE 110u

// ── byte helpers ──────────────────────────────────────────────────────────────

uint byte_at(uint i) {
    return (mat[i >> 2u] >> ((i & 3u) << 3u)) & 0xffu;
}

uint u16_at(uint i) {
    return byte_at(i) | (byte_at(i + 1u) << 8u);
}

// Unaligned little-endian uint32 read (crosses a word boundary when i%4 != 0).
uint u32_at(uint i) {
    uint lo_shift = (i & 3u) << 3u;
    if (lo_shift == 0u) return mat[i >> 2u];
    uint w0 = mat[i >> 2u];
    uint w1 = mat[(i >> 2u) + 1u];
    return (w0 >> lo_shift) | (w1 << (32u - lo_shift));
}

float f16_to_f32(uint h) {
    uint s = h >> 15u;
    uint e = (h >> 10u) & 0x1fu;
    uint m = h & 0x3ffu;
    if (e == 0u)  return uintBitsToFloat(s << 31u);
    if (e == 31u) return uintBitsToFloat((s << 31u) | 0x7f800000u | (m << 13u));
    return uintBitsToFloat((s << 31u) | ((e + 112u) << 23u) | (m << 13u));
}

// ── scale unpacking ───────────────────────────────────────────────────────────
//
// The 12 scale bytes encode 16 × 6-bit unsigned values via the GGML packing:
//   sc01 = u32(scales[0..3]),  sc23 = u32(scales[4..7]),  tmp = u32(scales[8..11])
//   aux[0] = (sc01 & kmask2) | ((tmp        & kmask1) << 4)  → sc[0..3]
//   aux[1] = (sc23 & kmask2) | (((tmp >> 2) & kmask1) << 4)  → sc[4..7]
//   aux[2] = ((sc01 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)  → sc[8..11]
//   aux[3] = ((sc23 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)  → sc[12..15]
// Each byte of aux holds one 6-bit scale in bits [5:0].
// Signed scale = float(aux_byte) - 32.0   (range [-32, 31]).

void unpack_scales(uint scales_off, out uint aux[4]) {
    const uint kmask1 = 0x03030303u;
    const uint kmask2 = 0x0f0f0f0fu;
    uint sc01 = u32_at(scales_off);
    uint sc23 = u32_at(scales_off + 4u);
    uint tmp  = u32_at(scales_off + 8u);
    aux[0] = (sc01 & kmask2)         | ((tmp         & kmask1) << 4u);
    aux[1] = (sc23 & kmask2)         | (((tmp >> 2u) & kmask1) << 4u);
    aux[2] = ((sc01 >> 4u) & kmask2) | (((tmp >> 4u) & kmask1) << 4u);
    aux[3] = ((sc23 >> 4u) & kmask2) | (((tmp >> 6u) & kmask1) << 4u);
}

float get_scale(uint aux[4], float d_all, uint sc_idx) {
    uint raw = (aux[sc_idx >> 2u] >> ((sc_idx & 3u) << 3u)) & 0x3fu;
    return d_all * (float(int(raw)) - 32.0);
}

// ── main ──────────────────────────────────────────────────────────────────────

void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= pc.rows) return;

    uint n_blocks = pc.cols / QK_K;
    float sum = 0.0;

    for (uint b = 0u; b < n_blocks; b++) {
        uint blk = (row * n_blocks + b) * BLK_SIZE;
        float d_all = f16_to_f32(u16_at(blk + 108u));

        uint aux[4];
        unpack_scales(blk + 96u, aux);

        for (uint e = 0u; e < QK_K; e++) {
            uint half_e  = e >> 7u;
            uint iter_e  = (e >> 5u) & 3u;
            uint group_e = (e >> 4u) & 1u;
            uint lane_e  = e & 15u;

            uint shift  = iter_e << 1u;
            uint m_bit  = (half_e << 2u) | iter_e;
            uint qs_off = (half_e << 5u) | (group_e << 4u) | lane_e;
            uint hm_off = (group_e << 4u) | lane_e;
            uint sc_idx = (half_e << 3u) | (iter_e << 1u) | group_e;

            uint lo2    = (byte_at(blk + 32u + qs_off) >> shift) & 3u;
            uint hi_bit = (byte_at(blk + hm_off) >> m_bit) & 1u;
            int  q3     = int(lo2) - (hi_bit == 0u ? 4 : 0);

            float scale = get_scale(aux, d_all, sc_idx);
            sum = fma(scale * float(q3), vec_in[b * QK_K + e], sum);
        }
    }

    vec_out[row] = sum;
}
