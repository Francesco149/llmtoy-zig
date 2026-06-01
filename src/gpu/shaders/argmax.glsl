#version 450
// Return the lowest index with the highest value in x.
// Pass 0 splits the row across workgroups and writes one partial winner per
// group. Pass 1 reduces those partial winners into the final output index.

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly buffer X { float x[]; };
struct Winner {
    float val;
    uint idx;
};
layout(std430, set = 0, binding = 1) buffer Scratch { Winner partials[]; };
layout(std430, set = 0, binding = 2) writeonly buffer Out { uint out_idx; };

layout(push_constant) uniform PC {
    uint n;
    uint n_groups;
    uint pass;
} pc;

shared float best_vals[256];
shared uint best_idxs[256];

bool better(float lhs_val, uint lhs_idx, float rhs_val, uint rhs_idx) {
    return lhs_val > rhs_val || (lhs_val == rhs_val && lhs_idx < rhs_idx);
}

void main() {
    const uint tid = gl_LocalInvocationID.x;
    const uint group = gl_WorkGroupID.x;
    float best_val = -1.0 / 0.0;
    uint best_idx = 0u;

    if (pc.pass == 0u) {
        for (uint i = group * gl_WorkGroupSize.x + tid; i < pc.n; i += pc.n_groups * gl_WorkGroupSize.x) {
            if (better(x[i], i, best_val, best_idx)) {
                best_val = x[i];
                best_idx = i;
            }
        }
    } else if (tid < pc.n) {
        const Winner winner = partials[tid];
        best_val = winner.val;
        best_idx = winner.idx;
    }

    best_vals[tid] = best_val;
    best_idxs[tid] = best_idx;
    barrier();

    for (uint stride = gl_WorkGroupSize.x / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride && better(best_vals[tid + stride], best_idxs[tid + stride], best_vals[tid], best_idxs[tid])) {
            best_vals[tid] = best_vals[tid + stride];
            best_idxs[tid] = best_idxs[tid + stride];
        }
        barrier();
    }

    if (tid != 0u) return;
    if (pc.pass == 0u) {
        partials[group].val = best_vals[0];
        partials[group].idx = best_idxs[0];
    } else {
        out_idx = best_idxs[0];
    }
}
