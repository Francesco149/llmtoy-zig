#version 450
// Attention · V: out[h, d] = Σ_i scores[h, i] · v_cache[slot(i), kv_h, d]
//
// One workgroup per Q head: gl_WorkGroupID.x = h. 256 threads parallelise
// across the head_dim output dimension; each thread accumulates one (or two,
// for head_dim=512) output element over all `win_len` attended positions.
//
// Memory pattern per inner-loop iteration i: all 256 threads broadcast-read
// scores[h, i] (one value), then read v_cache[slot(i), kv_h, 0..head_dim-1]
// (256 contiguous floats — perfectly coalesced).
//
// Bindings:
//   0: scores  — [n_heads · win_len] f32, the softmaxed attention weights
//   1: V_cache — [cap · n_kv_heads · head_dim] f32, slot-major KV cache
//   2: out     — [n_heads · head_dim] f32 (output: per-head attention result)

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly  buffer Scores { float scores[];   };
layout(std430, set = 0, binding = 1) readonly  buffer V      { float v_cache[];  };
layout(std430, set = 0, binding = 2) writeonly buffer Out    { float out_[];     };

layout(push_constant) uniform PC {
    uint  seq;
    uint  win_len;
    uint  head_dim;
    uint  n_kv_heads;
    uint  n_q_per_kv;
    uint  cap;
} pc;

void main() {
    const uint h        = gl_WorkGroupID.x;
    const uint tid      = gl_LocalInvocationID.x;
    const uint nthreads = gl_WorkGroupSize.x;
    const uint kv_h     = h / pc.n_q_per_kv;

    const uint scores_base = h * pc.win_len;
    const uint kv_row      = pc.n_kv_heads * pc.head_dim;
    const uint v_head_off  = kv_h * pc.head_dim;
    const uint start_pos   = pc.seq - pc.win_len;
    const uint out_base    = h * pc.head_dim;

    for (uint dd = tid; dd < pc.head_dim; dd += nthreads) {
        float acc = 0.0;
        for (uint i = 0; i < pc.win_len; ++i) {
            const uint slot   = (start_pos + i) % pc.cap;
            const uint v_off  = slot * kv_row + v_head_off + dd;
            acc += scores[scores_base + i] * v_cache[v_off];
        }
        out_[out_base + dd] = acc;
    }
}
