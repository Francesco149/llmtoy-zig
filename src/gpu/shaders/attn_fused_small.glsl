#version 450
// Fused QK softmax and AV for decode attention windows up to 1024 positions.
//
// One workgroup handles one Q head. Scores live in shared memory instead of a
// global scores buffer, so this replaces the qk_softmax + av dispatch pair for
// short decode/SWA windows while keeping the old two-pass path for long global
// contexts.
//
// QK parallelization: each subgroup cooperates on one K position at a time.
// Lanes within the subgroup split head_dim and reduce via subgroupAdd. For
// small win_len (e.g. decode at 33 tokens) this avoids the previous
// thread-per-position pattern where most lanes sat idle. Q is preloaded into
// shared memory once so the inner QK loop only reads global K.

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

// head_dim ≤ 512 for Gemma4 (256 SWA, 512 global), win_len ≤ 1024.
shared float sh_q[512];
shared float sh_scores[1024];
shared float sub_max[8];
shared float sub_sum[8];

void main() {
    const uint h        = gl_WorkGroupID.x;
    const uint tid      = gl_LocalInvocationID.x;
    const uint nthreads = gl_WorkGroupSize.x;
    const uint kv_h     = h / pc.n_q_per_kv;

    const uint sg_size  = gl_SubgroupSize;
    const uint sg_id    = gl_SubgroupID;
    const uint sg_inv   = gl_SubgroupInvocationID;
    const uint n_sg     = gl_NumSubgroups;

    const uint q_base      = h * pc.head_dim;
    const uint kv_row      = pc.n_kv_heads * pc.head_dim;
    const uint kv_head_off = kv_h * pc.head_dim;
    const uint start_pos   = pc.seq - pc.win_len;
    const uint out_base    = h * pc.head_dim;

    // Phase 0: preload Q for this head into shared memory once.
    for (uint d = tid; d < pc.head_dim; d += nthreads) {
        sh_q[d] = q[q_base + d];
    }
    barrier();

    // Phase 1: QK dot. Each subgroup handles a strided set of K positions; the
    // subgroup's lanes cooperate on head_dim with a final subgroupAdd reduce.
    for (uint pos = sg_id; pos < pc.win_len; pos += n_sg) {
        const uint slot  = (start_pos + pos) % pc.cap;
        const uint k_off = slot * kv_row + kv_head_off;
        float partial = 0.0;
        for (uint d = sg_inv; d < pc.head_dim; d += sg_size) {
            partial += sh_q[d] * k_cache[k_off + d];
        }
        const float dot = subgroupAdd(partial);
        if (sg_inv == 0u) {
            sh_scores[pos] = dot * pc.scale;
        }
    }
    barrier();

    // Phase 2: max reduction across all scores.
    float local_max = -3.4e38;
    for (uint i = tid; i < pc.win_len; i += nthreads) {
        local_max = max(local_max, sh_scores[i]);
    }
    local_max = subgroupMax(local_max);
    if (sg_inv == 0u) sub_max[sg_id] = local_max;
    barrier();
    if (sg_id == 0u) {
        float v = (sg_inv < n_sg) ? sub_max[sg_inv] : -3.4e38;
        v = subgroupMax(v);
        if (sg_inv == 0u) sub_max[0] = v;
    }
    barrier();
    const float max_val = sub_max[0];

    // Phase 3: exp + sum reduction.
    float local_sum = 0.0;
    for (uint i = tid; i < pc.win_len; i += nthreads) {
        const float v = exp(sh_scores[i] - max_val);
        sh_scores[i] = v;
        local_sum += v;
    }
    local_sum = subgroupAdd(local_sum);
    if (sg_inv == 0u) sub_sum[sg_id] = local_sum;
    barrier();
    if (sg_id == 0u) {
        float v = (sg_inv < n_sg) ? sub_sum[sg_inv] : 0.0;
        v = subgroupAdd(v);
        if (sg_inv == 0u) sub_sum[0] = v;
    }
    barrier();
    const float inv_sum = 1.0 / sub_sum[0];

    for (uint i = tid; i < pc.win_len; i += nthreads) {
        sh_scores[i] *= inv_sum;
    }
    barrier();

    // Phase 4: attn · V. One thread per head_dim element. Loop over positions
    // and accumulate the softmax-weighted V. Stays the same as before — for
    // head_dim ≥ nthreads (= 256) all threads have real work.
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
