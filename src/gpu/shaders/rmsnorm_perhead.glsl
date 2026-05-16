#version 450
// Per-head RMS normalization for Gemma4 attention.
//
// X and Y are [n_heads * head_dim] f32 buffers laid out as concatenated head
// rows.  Each workgroup handles one head: gl_WorkGroupID.x = head index.
// 256 threads cooperate via subgroupAdd + shared memory to reduce Σ x² across
// the head row.
//
// W is a [head_dim] f32 vector shared across all heads (Gemma4's q_norm /
// k_norm tensors).  When use_weight==0 the W binding is ignored (callers may
// bind any valid buffer) and the output is x*rms_inv — the rmsnormRaw path
// used for V projections.
//
// `head_dim` MUST be a multiple of the workgroup size (256).  Gemma4 uses
// head_dim ∈ {256 (SWA), 512 (global)}, both compatible.
//
// `weight_offset` carries Gemma's (1 + w) weight convention (1 → bias=1;
// 0 → bias=0).  Ignored when use_weight==0.

#extension GL_KHR_shader_subgroup_basic      : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly  buffer X { float x_in[];  };
layout(std430, set = 0, binding = 1) readonly  buffer W { float w[];     };
layout(std430, set = 0, binding = 2) writeonly buffer Y { float y_out[]; };

layout(push_constant) uniform PC {
    uint  head_dim;
    float eps;
    uint  weight_offset; // 0 → w; 1 → (1 + w); ignored when use_weight==0
    uint  use_weight;    // 0 → y = x*rms_inv (rmsnormRaw); 1 → multiply by w
} pc;

shared float sub_sums[8];

void main() {
    const uint head   = gl_WorkGroupID.x;
    const uint base   = head * pc.head_dim;
    const uint tid    = gl_LocalInvocationID.x;
    const uint stride = gl_WorkGroupSize.x;

    float ss = 0.0;
    for (uint i = tid; i < pc.head_dim; i += stride) {
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

    const float mean_sq = sub_sums[0] / float(pc.head_dim);
    const float rms_inv = inversesqrt(mean_sq + pc.eps);

    const float bias = (pc.weight_offset != 0u) ? 1.0 : 0.0;
    if (pc.use_weight != 0u) {
        for (uint i = tid; i < pc.head_dim; i += stride) {
            y_out[base + i] = x_in[base + i] * rms_inv * (bias + w[i]);
        }
    } else {
        for (uint i = tid; i < pc.head_dim; i += stride) {
            y_out[base + i] = x_in[base + i] * rms_inv;
        }
    }
}
