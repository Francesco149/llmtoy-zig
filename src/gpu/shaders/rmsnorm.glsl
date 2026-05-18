#version 450
// RMS normalization (single-row): y[i] = x[i] * rms_inv * w[i]
// where rms_inv = 1 / sqrt(Σ x[i]² / n + eps).
//
// One workgroup per call. 256 threads cooperate via subgroupAdd + shared
// memory to reduce Σ x² across the row (RDNA3 wave32 → 8 subgroups per
// workgroup, then a final 8-way subgroup reduction).
//
// Low bit of `weight_offset` controls Gemma's weight convention:
//   0 → y = x * rms_inv * w        (this codebase's existing convention)
//   1 → y = x * rms_inv * (1 + w)  (some Gemma variants pre-shift weights)
// Bit 1 requests a CPU-order scalar sum for numerically sensitive tail norms.
// The flag is parameterized so the same shader serves both layouts.
//
// `pc.n` MUST be a multiple of the workgroup size (256). All Gemma4 norm
// dimensions (d_model=2816, head_dim_global=512, head_dim_swa=256) satisfy
// this except for non-256-divisible ones — those need a tail loop, not in
// this shader's scope.

#extension GL_KHR_shader_subgroup_basic      : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly  buffer X { float x_in[];  };
layout(std430, set = 0, binding = 1) readonly  buffer W { float w[];     };
layout(std430, set = 0, binding = 2) writeonly buffer Y { float y_out[]; };

layout(push_constant) uniform PC {
    uint  n;             // vector length
    float eps;           // small epsilon added to mean_sq before sqrt
    uint  weight_offset; // 0 = use w directly; 1 = use (1 + w)
} pc;

// Up to 8 subgroups per workgroup at subgroup=32 (RDNA3 wave32).
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
        // Phase 1 — per-thread partial sum of x².
        float ss = 0.0;
        for (uint i = tid; i < pc.n; i += stride) {
            const float v = x_in[i];
            ss += v * v;
        }

        // Phase 2 — subgroup reduction (each subgroup of 32 reduces to 1 value).
        ss = subgroupAdd(ss);
        if (gl_SubgroupInvocationID == 0u) {
            sub_sums[gl_SubgroupID] = ss;
        }
        barrier();

        // Phase 3 — final reduction across subgroups by subgroup 0.
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

    // Phase 4 — write normalized output.
    const float bias = ((pc.weight_offset & 1u) != 0u) ? 1.0 : 0.0;
    for (uint i = tid; i < pc.n; i += stride) {
        y_out[i] = x_in[i] * rms_inv * (bias + w[i]);
    }
}
