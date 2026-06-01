#version 450
// Fused per-head RMSNorm + NeoX RoPE for Gemma4 global attention.

#extension GL_KHR_shader_subgroup_basic      : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly  buffer X     { float x_in[];  };
layout(std430, set = 0, binding = 1) readonly  buffer W     { float w[];     };
layout(std430, set = 0, binding = 2) readonly  buffer Freqs { float freqs[]; };
layout(std430, set = 0, binding = 3) writeonly buffer Y     { float y_out[]; };

layout(push_constant) uniform PC {
    uint head_dim;
    float eps;
    uint pos;
} pc;

shared float normalized[512];
shared float sub_sums[8];

void main() {
    const uint head = gl_WorkGroupID.x;
    const uint base = head * pc.head_dim;
    const uint tid = gl_LocalInvocationID.x;

    float ss = 0.0;
    for (uint i = tid; i < pc.head_dim; i += gl_WorkGroupSize.x) {
        const float v = x_in[base + i];
        ss += v * v;
    }
    ss = subgroupAdd(ss);
    if (gl_SubgroupInvocationID == 0u) sub_sums[gl_SubgroupID] = ss;
    barrier();

    if (gl_SubgroupID == 0u) {
        const float s = (gl_SubgroupInvocationID < gl_NumSubgroups)
            ? sub_sums[gl_SubgroupInvocationID] : 0.0;
        const float total = subgroupAdd(s);
        if (gl_SubgroupInvocationID == 0u) sub_sums[0] = total;
    }
    barrier();

    const float rms_inv = inversesqrt(sub_sums[0] / float(pc.head_dim) + pc.eps);
    for (uint i = tid; i < pc.head_dim; i += gl_WorkGroupSize.x) {
        normalized[i] = x_in[base + i] * rms_inv * (1.0 + w[i]);
    }
    barrier();

    const uint hd_half = pc.head_dim >> 1u;
    if (tid >= hd_half) return;
    const float angle = float(pc.pos) * freqs[tid];
    const float cos_a = cos(angle);
    const float sin_a = sin(angle);
    const float x0 = normalized[tid];
    const float x1 = normalized[tid + hd_half];
    y_out[base + tid] = x0 * cos_a - x1 * sin_a;
    y_out[base + tid + hd_half] = x0 * sin_a + x1 * cos_a;
}
