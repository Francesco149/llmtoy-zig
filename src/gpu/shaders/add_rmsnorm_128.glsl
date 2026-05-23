#version 450
// 128-thread variant of add_rmsnorm.glsl for benchmarking smaller single-row norms.

#extension GL_KHR_shader_subgroup_basic      : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly  buffer A { float a[]; };
layout(std430, set = 0, binding = 1) readonly  buffer B { float b[]; };
layout(std430, set = 0, binding = 2) readonly  buffer W { float w[]; };
layout(std430, set = 0, binding = 3) writeonly buffer Y { float y[]; };

layout(push_constant) uniform PC {
    uint  n;
    float eps;
    uint  flags;
} pc;

shared float sub_sums[8];

void main() {
    const uint tid    = gl_LocalInvocationID.x;
    const uint stride = gl_WorkGroupSize.x;

    if ((pc.flags & 2u) != 0u) {
        if (tid == 0u) {
            float ss = 0.0;
            for (uint i = 0u; i < pc.n; i++) {
                const float v = a[i] + b[i];
                ss += v * v;
            }
            sub_sums[0] = ss;
        }
        barrier();
    } else {
        float ss = 0.0;
        for (uint i = tid; i < pc.n; i += stride) {
            const float v = a[i] + b[i];
            ss += v * v;
        }

        ss = subgroupAdd(ss);
        if (gl_SubgroupInvocationID == 0u) {
            sub_sums[gl_SubgroupID] = ss;
        }
        barrier();

        if (gl_SubgroupID == 0u) {
            const float s = (gl_SubgroupInvocationID < gl_NumSubgroups)
                ? sub_sums[gl_SubgroupInvocationID] : 0.0;
            const float total = subgroupAdd(s);
            if (gl_SubgroupInvocationID == 0u) sub_sums[0] = total;
        }
        barrier();
    }

    const float rms_inv = inversesqrt(sub_sums[0] / float(pc.n) + pc.eps);
    const float bias = ((pc.flags & 1u) != 0u) ? 1.0 : 0.0;
    for (uint i = tid; i < pc.n; i += stride) {
        y[i] = (a[i] + b[i]) * rms_inv * (bias + w[i]);
    }
}
