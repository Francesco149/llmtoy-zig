#version 450
// NeoX-style RoPE computing inverse frequencies from a single `theta` base
// (Gemma4 SWA layers use this variant with theta = rope_theta_swa).
//
//   freq_i = 1 / theta^(2i/head_dim)
//   angle  = pos * freq_i
//   [x_i, x_{i+half}] ← [x_i cos − x_{i+half} sin, x_i sin + x_{i+half} cos]

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) buffer Vec { float vec_data[]; };

layout(push_constant) uniform PC {
    uint  pos;
    uint  head_dim;
    float theta;
} pc;

void main() {
    const uint head = gl_WorkGroupID.x;
    const uint i    = gl_LocalInvocationID.x;
    const uint hd_half = pc.head_dim >> 1u;
    if (i >= hd_half) return;

    const uint head_off = head * pc.head_dim;
    const float exponent = float(2u * i) / float(pc.head_dim);
    const float freq  = 1.0 / pow(pc.theta, exponent);
    const float angle = float(pc.pos) * freq;
    const float cos_a = cos(angle);
    const float sin_a = sin(angle);

    const float x0 = vec_data[head_off + i];
    const float x1 = vec_data[head_off + i + hd_half];
    vec_data[head_off + i]           = x0 * cos_a - x1 * sin_a;
    vec_data[head_off + i + hd_half] = x0 * sin_a + x1 * cos_a;
}
