#version 450
// Return the lowest index with the highest value in x.
// One 256-thread workgroup scans the whole row, then reduces local winners.

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly buffer X { float x[]; };
layout(std430, set = 0, binding = 1) writeonly buffer Out { uint out_idx; };

layout(push_constant) uniform PC {
    uint n;
} pc;

shared float best_vals[256];
shared uint best_idxs[256];

bool better(float lhs_val, uint lhs_idx, float rhs_val, uint rhs_idx) {
    return lhs_val > rhs_val || (lhs_val == rhs_val && lhs_idx < rhs_idx);
}

void main() {
    const uint tid = gl_LocalInvocationID.x;
    float best_val = -1.0 / 0.0;
    uint best_idx = 0u;

    for (uint i = tid; i < pc.n; i += gl_WorkGroupSize.x) {
        if (better(x[i], i, best_val, best_idx)) {
            best_val = x[i];
            best_idx = i;
        }
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

    if (tid == 0u) out_idx = best_idxs[0];
}
