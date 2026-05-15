#version 450
// Fused gate-gelu-up Q3_K × fp32 shader.
// Computes output[row] = gelu(gate_mat[row] · vec) * (up_mat[row] · vec)
// in a single pass — eliminates the CPU roundtrip between gate+up and down.
//
// Q3_K block layout (110 bytes, QK_K = 256 elements):
//   [0..31]    hmask  – 1 high bit per element
//   [32..95]   qs     – 2 low bits per element, 4 packed per byte
//   [96..107]  scales – 16 × 6-bit scale values (12-byte GGML packing)
//   [108..109] d      – f16 super-block scale
//
// Bindings:  0=gate_mat  1=up_mat  2=vec_in  3=vec_out
// Both matrices have identical dimensions and quantization.

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly buffer GateBuf { uint gate_mat[]; };
layout(std430, set = 0, binding = 1) readonly buffer UpBuf   { uint up_mat[];   };
layout(std430, set = 0, binding = 2) readonly buffer VecBuf  { float vec_in[];  };
layout(std430, set = 0, binding = 3) writeonly buffer OutBuf { float vec_out[];  };

layout(push_constant) uniform PC { uint rows; uint cols; } pc;

#define QK_K     256u
#define BLK_SIZE 110u

// ── byte helpers (gate_mat) ───────────────────────────────────────────────────

uint gbyte(uint i) { return (gate_mat[i >> 2u] >> ((i & 3u) << 3u)) & 0xffu; }
uint gu16(uint i)  { return gbyte(i) | (gbyte(i + 1u) << 8u); }
uint gu32(uint i) {
    uint sh = (i & 3u) << 3u;
    if (sh == 0u) return gate_mat[i >> 2u];
    return (gate_mat[i >> 2u] >> sh) | (gate_mat[(i >> 2u) + 1u] << (32u - sh));
}

// ── byte helpers (up_mat) ────────────────────────────────────────────────────

uint ubyte(uint i) { return (up_mat[i >> 2u] >> ((i & 3u) << 3u)) & 0xffu; }
uint uu16(uint i)  { return ubyte(i) | (ubyte(i + 1u) << 8u); }
uint uu32(uint i) {
    uint sh = (i & 3u) << 3u;
    if (sh == 0u) return up_mat[i >> 2u];
    return (up_mat[i >> 2u] >> sh) | (up_mat[(i >> 2u) + 1u] << (32u - sh));
}

float f16_to_f32(uint h) {
    uint s = h >> 15u;
    uint e = (h >> 10u) & 0x1fu;
    uint m = h & 0x3ffu;
    if (e == 0u)  return uintBitsToFloat(s << 31u);
    if (e == 31u) return uintBitsToFloat((s << 31u) | 0x7f800000u | (m << 13u));
    return uintBitsToFloat((s << 31u) | ((e + 112u) << 23u) | (m << 13u));
}

// ── scale unpack ─────────────────────────────────────────────────────────────

void unpack_g_scales(uint off, out uint aux[4]) {
    const uint k1 = 0x03030303u, k2 = 0x0f0f0f0fu;
    uint sc01 = gu32(off), sc23 = gu32(off+4u), tmp = gu32(off+8u);
    aux[0] = (sc01 & k2)         | ((tmp         & k1) << 4u);
    aux[1] = (sc23 & k2)         | (((tmp >> 2u) & k1) << 4u);
    aux[2] = ((sc01 >> 4u) & k2) | (((tmp >> 4u) & k1) << 4u);
    aux[3] = ((sc23 >> 4u) & k2) | (((tmp >> 6u) & k1) << 4u);
}

void unpack_u_scales(uint off, out uint aux[4]) {
    const uint k1 = 0x03030303u, k2 = 0x0f0f0f0fu;
    uint sc01 = uu32(off), sc23 = uu32(off+4u), tmp = uu32(off+8u);
    aux[0] = (sc01 & k2)         | ((tmp         & k1) << 4u);
    aux[1] = (sc23 & k2)         | (((tmp >> 2u) & k1) << 4u);
    aux[2] = ((sc01 >> 4u) & k2) | (((tmp >> 4u) & k1) << 4u);
    aux[3] = ((sc23 >> 4u) & k2) | (((tmp >> 6u) & k1) << 4u);
}

float get_scale(uint aux[4], float d, uint sc_idx) {
    uint raw = (aux[sc_idx >> 2u] >> ((sc_idx & 3u) << 3u)) & 0x3fu;
    return d * (float(int(raw)) - 32.0);
}

float gelu(float x) {
    const float c = 0.7978845608; // sqrt(2/π)
    return 0.5 * x * (1.0 + tanh(c * (x + 0.044715 * x * x * x)));
}

// ── main ──────────────────────────────────────────────────────────────────────

void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= pc.rows) return;

    uint n_blocks = pc.cols / QK_K;
    float gate_sum = 0.0;
    float up_sum   = 0.0;

    for (uint b = 0u; b < n_blocks; b++) {
        uint gblk = (row * n_blocks + b) * BLK_SIZE;
        uint ublk = gblk; // same row, same block index, separate buffer

        float gd = f16_to_f32(gu16(gblk + 108u));
        float ud = f16_to_f32(uu16(ublk + 108u));

        uint gaux[4]; unpack_g_scales(gblk + 96u, gaux);
        uint uaux[4]; unpack_u_scales(ublk + 96u, uaux);

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

            float v = vec_in[b * QK_K + e];

            uint glo2  = (gbyte(gblk + 32u + qs_off) >> shift) & 3u;
            uint ghi   = (gbyte(gblk + hm_off) >> m_bit) & 1u;
            int  gq3   = int(glo2) - (ghi == 0u ? 4 : 0);
            gate_sum   = fma(get_scale(gaux, gd, sc_idx) * float(gq3), v, gate_sum);

            uint ulo2  = (ubyte(ublk + 32u + qs_off) >> shift) & 3u;
            uint uhi   = (ubyte(ublk + hm_off) >> m_bit) & 1u;
            int  uq3   = int(ulo2) - (uhi == 0u ? 4 : 0);
            up_sum     = fma(get_scale(uaux, ud, sc_idx) * float(uq3), v, up_sum);
        }
    }

    vec_out[row] = gelu(gate_sum) * up_sum;
}
