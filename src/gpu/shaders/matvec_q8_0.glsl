#version 450
// Q8_0 × fp32 matrix-vector multiply.
// Each block covers 32 elements: 2 bytes f16 scale + 32 bytes i8 quants = 34 bytes.
// Dequant on the fly: out[row] = Σ_i( scale * q[i] * vec[i] )

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// Quantized matrix as raw bytes packed into uint32 words.
layout(std430, set = 0, binding = 0) readonly buffer MatBuf  { uint    mat[];    };
layout(std430, set = 0, binding = 1) readonly buffer VecBuf  { float   vec_in[]; };
layout(std430, set = 0, binding = 2) writeonly buffer OutBuf { float   vec_out[]; };

layout(push_constant) uniform PC { uint rows; uint cols; } pc;

// ── byte helpers ──────────────────────────────────────────────────────────────

uint byte_at(uint byte_idx) {
    return (mat[byte_idx >> 2u] >> ((byte_idx & 3u) << 3u)) & 0xffu;
}

uint u16_at(uint byte_idx) {
    return byte_at(byte_idx) | (byte_at(byte_idx + 1u) << 8u);
}

// ── fp16 → fp32 ───────────────────────────────────────────────────────────────

float f16_to_f32(uint h) {
    uint s = h >> 15u;
    uint e = (h >> 10u) & 0x1fu;
    uint m = h & 0x3ffu;
    if (e == 0u)  return uintBitsToFloat(s << 31u);              // ±0 (denormals → 0)
    if (e == 31u) return uintBitsToFloat((s << 31u) | 0x7f800000u | (m << 13u)); // inf/nan
    return uintBitsToFloat((s << 31u) | ((e + 112u) << 23u) | (m << 13u));
}

// ── main ──────────────────────────────────────────────────────────────────────

void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= pc.rows) return;

    uint n_blocks = pc.cols >> 5u; // cols / 32
    float sum = 0.0;

    for (uint b = 0u; b < n_blocks; b++) {
        uint blk   = (row * n_blocks + b) * 34u;
        float scale = f16_to_f32(u16_at(blk));
        for (uint i = 0u; i < 32u; i++) {
            // bitfieldExtract sign-extends the 8-bit quant to int32.
            float q = float(bitfieldExtract(int(byte_at(blk + 2u + i)), 0, 8));
            sum = fma(scale * q, vec_in[b * 32u + i], sum);
        }
    }

    vec_out[row] = sum;
}
