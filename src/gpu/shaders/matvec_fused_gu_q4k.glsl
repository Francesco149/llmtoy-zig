#version 450
// Fused gate-gelu-up Q4_K × fp32 shader.
// Computes output[row] = gelu(gate_mat[row] · vec) * (up_mat[row] · vec)
// in a single pass — eliminates the CPU gelu and the gate/up round-trip.
//
// Q4_K block layout (144 bytes, QK_K = 256 elements):
//   [0..1]    f16  d       – super-scale for sub-block scales
//   [2..3]    f16  dmin    – super-scale for sub-block mins
//   [4..15]   u8[12] scales – 8 × (sc,mn) pairs packed as 6-bit values
//   [16..143] u8[128] qs   – 4-bit nibbles, 2 per byte
//
// Bindings: 0=gate_mat  1=up_mat  2=vec_in  3=vec_out

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly buffer GateBuf { uint gate_mat[]; };
layout(std430, set = 0, binding = 1) readonly buffer UpBuf   { uint up_mat[];   };
layout(std430, set = 0, binding = 2) readonly buffer VecBuf  { float vec_in[];  };
layout(std430, set = 0, binding = 3) writeonly buffer OutBuf { float vec_out[];  };

layout(push_constant) uniform PC { uint rows; uint cols; } pc;

#define QK_K     256u
#define BLK_SIZE 144u

// ── byte helpers ──────────────────────────────────────────────────────────────

uint gbyte(uint i) { return (gate_mat[i >> 2u] >> ((i & 3u) << 3u)) & 0xffu; }
uint ubyte(uint i) { return (up_mat  [i >> 2u] >> ((i & 3u) << 3u)) & 0xffu; }

uint gu16(uint i) { return gbyte(i) | (gbyte(i + 1u) << 8u); }
uint uu16(uint i) { return ubyte(i) | (ubyte(i + 1u) << 8u); }

float f16_to_f32(uint h) {
    uint s = h >> 15u;
    uint e = (h >> 10u) & 0x1fu;
    uint m = h & 0x3ffu;
    if (e == 0u)  return uintBitsToFloat(s << 31u);
    if (e == 31u) return uintBitsToFloat((s << 31u) | 0x7f800000u | (m << 13u));
    return uintBitsToFloat((s << 31u) | ((e + 112u) << 23u) | (m << 13u));
}

// ── scale/min unpacking ───────────────────────────────────────────────────────

void g_scale_min(uint scales_off, uint j, float d_all, float dmin_all,
                 out float d_eff, out float m_eff) {
    uint sc, mn;
    if (j < 4u) {
        sc = gbyte(scales_off + j) & 0x3fu;
        mn = gbyte(scales_off + j + 4u) & 0x3fu;
    } else {
        uint extra = gbyte(scales_off + j + 4u);
        sc = (extra & 0x0fu) | ((gbyte(scales_off + j - 4u) >> 6u) << 4u);
        mn = (extra >> 4u)   | ((gbyte(scales_off + j     ) >> 6u) << 4u);
    }
    d_eff = d_all * float(sc);
    m_eff = dmin_all * float(mn);
}

void u_scale_min(uint scales_off, uint j, float d_all, float dmin_all,
                 out float d_eff, out float m_eff) {
    uint sc, mn;
    if (j < 4u) {
        sc = ubyte(scales_off + j) & 0x3fu;
        mn = ubyte(scales_off + j + 4u) & 0x3fu;
    } else {
        uint extra = ubyte(scales_off + j + 4u);
        sc = (extra & 0x0fu) | ((ubyte(scales_off + j - 4u) >> 6u) << 4u);
        mn = (extra >> 4u)   | ((ubyte(scales_off + j     ) >> 6u) << 4u);
    }
    d_eff = d_all * float(sc);
    m_eff = dmin_all * float(mn);
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
        uint ublk = gblk;

        float gd    = f16_to_f32(gu16(gblk));
        float gdmin = f16_to_f32(gu16(gblk + 2u));
        float ud    = f16_to_f32(uu16(ublk));
        float udmin = f16_to_f32(uu16(ublk + 2u));

        for (uint c = 0u; c < 4u; c++) {
            float gd0, gm0, gd1, gm1;
            float ud0, um0, ud1, um1;
            g_scale_min(gblk + 4u, c * 2u,      gd, gdmin, gd0, gm0);
            g_scale_min(gblk + 4u, c * 2u + 1u, gd, gdmin, gd1, gm1);
            u_scale_min(ublk + 4u, c * 2u,      ud, udmin, ud0, um0);
            u_scale_min(ublk + 4u, c * 2u + 1u, ud, udmin, ud1, um1);

            uint gqs_base = gblk + 16u + c * 32u;
            uint uqs_base = ublk + 16u + c * 32u;
            uint vec_base = b * QK_K + c * 64u;

            for (uint i = 0u; i < 32u; i++) {
                float v_lo = vec_in[vec_base + i];
                float v_hi = vec_in[vec_base + 32u + i];

                uint gqb = gbyte(gqs_base + i);
                gate_sum = fma(gd0 * float(gqb & 0xfu) - gm0, v_lo, gate_sum);
                gate_sum = fma(gd1 * float(gqb >> 4u)  - gm1, v_hi, gate_sum);

                uint uqb = ubyte(uqs_base + i);
                up_sum = fma(ud0 * float(uqb & 0xfu) - um0, v_lo, up_sum);
                up_sum = fma(ud1 * float(uqb >> 4u)  - um1, v_hi, up_sum);
            }
        }
    }

    vec_out[row] = gelu(gate_sum) * up_sum;
}
