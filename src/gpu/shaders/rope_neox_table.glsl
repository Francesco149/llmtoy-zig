#version 450
// NeoX-style RoPE using a precomputed inverse-frequency table (Gemma4
// global-attention layers read `rope_freqs.weight`).
//
//   freq_i = freqs[i]  (i ∈ [0, head_dim/2))
//   angle  = pos * freq_i
//   [x_i, x_{i+half}] ← [x_i cos − x_{i+half} sin, x_i sin + x_{i+half} cos]
//
// One workgroup per head; 256 threads (≥ head_dim/2 for both head_dim_swa=256
// and head_dim_global=512). Threads past head_dim/2 exit early.

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0)          buffer Vec   { float vec_data[]; };
layout(std430, set = 0, binding = 1) readonly buffer Freqs { float freqs[]; };

layout(push_constant) uniform PC {
    uint pos;
    uint head_dim;
} pc;

void main() {
    const uint head = gl_WorkGroupID.x;
    const uint i    = gl_LocalInvocationID.x;
    const uint hd_half = pc.head_dim >> 1u;
    if (i >= hd_half) return;

    const uint head_off = head * pc.head_dim;
    const float angle = float(pc.pos) * freqs[i];
    const float cos_a = cos(angle);
    const float sin_a = sin(angle);

    const float x0 = vec_data[head_off + i];
    const float x1 = vec_data[head_off + i + hd_half];
    vec_data[head_off + i]           = x0 * cos_a - x1 * sin_a;
    vec_data[head_off + i + hd_half] = x0 * sin_a + x1 * cos_a;
}
