#version 450
// Fused QK softmax and AV for decode attention windows up to 1024 positions.
//
// One workgroup handles one Q head. Scores live in shared memory instead of a
// global scores buffer, so this replaces the qk_softmax + av dispatch pair for
// short decode/SWA windows while keeping the old two-pass path for long global
// contexts.

#extension GL_KHR_shader_subgroup_basic      : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly  buffer Q   { float q[];       };
layout(std430, set = 0, binding = 1) readonly  buffer K   { float k_cache[]; };
layout(std430, set = 0, binding = 2) readonly  buffer V   { float v_cache[]; };
layout(std430, set = 0, binding = 3) writeonly buffer Out { float out_[];    };

layout(push_constant) uniform PC {
    uint  seq;
    uint  win_len;
    uint  head_dim;
    uint  n_kv_heads;
    uint  n_q_per_kv;
    uint  cap;
    float scale;
} pc;

shared float sh_scores[1024];
shared float sub_max[8];
shared float sub_sum[8];

void main() {
    const uint h        = gl_WorkGroupID.x;
    const uint tid      = gl_LocalInvocationID.x;
    const uint nthreads = gl_WorkGroupSize.x;
    const uint kv_h     = h / pc.n_q_per_kv;

    const uint q_base      = h * pc.head_dim;
    const uint kv_row      = pc.n_kv_heads * pc.head_dim;
    const uint kv_head_off = kv_h * pc.head_dim;
    const uint start_pos   = pc.seq - pc.win_len;
    const uint out_base    = h * pc.head_dim;

    float local_max = -3.4e38;
    for (uint i = tid; i < pc.win_len; i += nthreads) {
        const uint slot = (start_pos + i) % pc.cap;
        const uint k_off = slot * kv_row + kv_head_off;
        float dot = 0.0;
        for (uint d = 0; d < pc.head_dim; ++d) {
            dot += q[q_base + d] * k_cache[k_off + d];
        }
        const float s = dot * pc.scale;
        sh_scores[i] = s;
        local_max = max(local_max, s);
    }

    local_max = subgroupMax(local_max);
    if (gl_SubgroupInvocationID == 0u) sub_max[gl_SubgroupID] = local_max;
    barrier();
    if (gl_SubgroupID == 0u) {
        float v = (gl_SubgroupInvocationID < gl_NumSubgroups)
            ? sub_max[gl_SubgroupInvocationID] : -3.4e38;
        v = subgroupMax(v);
        if (gl_SubgroupInvocationID == 0u) sub_max[0] = v;
    }
    barrier();
    const float max_val = sub_max[0];

    float local_sum = 0.0;
    for (uint i = tid; i < pc.win_len; i += nthreads) {
        const float v = exp(sh_scores[i] - max_val);
        sh_scores[i] = v;
        local_sum += v;
    }

    local_sum = subgroupAdd(local_sum);
    if (gl_SubgroupInvocationID == 0u) sub_sum[gl_SubgroupID] = local_sum;
    barrier();
    if (gl_SubgroupID == 0u) {
        float v = (gl_SubgroupInvocationID < gl_NumSubgroups)
            ? sub_sum[gl_SubgroupInvocationID] : 0.0;
        v = subgroupAdd(v);
        if (gl_SubgroupInvocationID == 0u) sub_sum[0] = v;
    }
    barrier();
    const float inv_sum = 1.0 / sub_sum[0];

    for (uint i = tid; i < pc.win_len; i += nthreads) {
        sh_scores[i] *= inv_sum;
    }
    barrier();

    for (uint dd = tid; dd < pc.head_dim; dd += nthreads) {
        float acc = 0.0;
        for (uint i = 0; i < pc.win_len; ++i) {
            const uint slot = (start_pos + i) % pc.cap;
            const uint v_off = slot * kv_row + kv_head_off + dd;
            acc += sh_scores[i] * v_cache[v_off];
        }
        out_[out_base + dd] = acc;
    }
}
