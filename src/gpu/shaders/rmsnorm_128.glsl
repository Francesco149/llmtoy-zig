#version 450
// 128-thread variant of rmsnorm.glsl for benchmarking smaller single-row norms.

#extension GL_KHR_shader_subgroup_basic      : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly  buffer X { float x_in[];  };
layout(std430, set = 0, binding = 1) readonly  buffer W { float w[];     };
layout(std430, set = 0, binding = 2) writeonly buffer Y { float y_out[]; };

layout(push_constant) uniform PC {
    uint  n;
    float eps;
    uint  weight_offset;
} pc;

shared float sub_sums[8];

void main() {
    const uint tid    = gl_LocalInvocationID.x;
    const uint stride = gl_WorkGroupSize.x;

    if ((pc.weight_offset & 2u) != 0u) {
        if (tid == 0u) {
            float ss = 0.0;
            for (uint i = 0u; i < pc.n; i++) {
                const float v = x_in[i];
                ss += v * v;
            }
            sub_sums[0] = ss;
        }
        barrier();
    } else {
        float ss = 0.0;
        for (uint i = tid; i < pc.n; i += stride) {
            const float v = x_in[i];
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

    const float mean_sq = sub_sums[0] / float(pc.n);
    const float rms_inv = inversesqrt(mean_sq + pc.eps);
    const float bias = ((pc.weight_offset & 1u) != 0u) ? 1.0 : 0.0;
    for (uint i = tid; i < pc.n; i += stride) {
        y_out[i] = x_in[i] * rms_inv * (bias + w[i]);
    }
}
