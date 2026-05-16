#version 450
// Fused Q·K^T + softmax for one current token.
//
// One workgroup per Q head: gl_WorkGroupID.x = h. 256 threads parallelise
// across the `win_len` attended positions; each thread sequentially computes
// the full head_dim dot product for the position(s) it owns.
//
// Output layout: scores[h * win_len + i] = softmax(scale · q[h]·k[kv_h][slot(i)])_i
// where slot(i) = ((seq - win_len) + i) % cap is the circular KV-cache slot
// of the i-th attended position.
//
// For sliding-window attention the caller sets win_len = min(seq, sliding_window);
// for global attention win_len = seq. The mask is implicit in win_len.
//
// Bindings:
//   0: Q       — [n_heads · head_dim] f32, current token's per-head queries
//   1: K_cache — [cap · n_kv_heads · head_dim] f32, slot-major KV cache
//   2: scores  — [n_heads · win_len] f32 (output)

#extension GL_KHR_shader_subgroup_basic      : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly  buffer Q       { float q[];        };
layout(std430, set = 0, binding = 1) readonly  buffer K       { float k_cache[];  };
layout(std430, set = 0, binding = 2)           buffer Scores  { float scores[];   };

layout(push_constant) uniform PC {
    uint  seq;          // pos + 1 (number of tokens seen so far)
    uint  win_len;      // number of attended positions ≤ seq
    uint  head_dim;
    uint  n_kv_heads;
    uint  n_q_per_kv;   // n_heads / n_kv_heads (GQA group size)
    uint  cap;          // KV cache capacity for this layer
    float scale;
} pc;

// One slot per subgroup — RDNA3 wave32 → up to 8 subgroups per workgroup.
shared float sub_max[8];
shared float sub_sum[8];

void main() {
    const uint h        = gl_WorkGroupID.x;
    const uint tid      = gl_LocalInvocationID.x;
    const uint nthreads = gl_WorkGroupSize.x;
    const uint kv_h     = h / pc.n_q_per_kv;

    const uint scores_base = h * pc.win_len;
    const uint q_base      = h * pc.head_dim;
    const uint kv_row      = pc.n_kv_heads * pc.head_dim;
    const uint k_head_off  = kv_h * pc.head_dim;
    const uint start_pos   = pc.seq - pc.win_len;

    // ── Phase 1: scores[h, i] = scale · q[h]·k[slot(i), kv_h], track max
    float local_max = -3.4e38;
    for (uint i = tid; i < pc.win_len; i += nthreads) {
        const uint slot   = (start_pos + i) % pc.cap;
        const uint k_off  = slot * kv_row + k_head_off;
        float dot = 0.0;
        for (uint d = 0; d < pc.head_dim; ++d) {
            dot += q[q_base + d] * k_cache[k_off + d];
        }
        const float s = dot * pc.scale;
        scores[scores_base + i] = s;
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

    // ── Phase 2: exp(scores - max), in-place; track sum
    float local_sum = 0.0;
    for (uint i = tid; i < pc.win_len; i += nthreads) {
        const float v = exp(scores[scores_base + i] - max_val);
        scores[scores_base + i] = v;
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

    // ── Phase 3: normalise
    for (uint i = tid; i < pc.win_len; i += nthreads) {
        scores[scores_base + i] *= inv_sum;
    }
}
