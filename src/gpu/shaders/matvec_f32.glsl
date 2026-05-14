#version 450

// fp32 matrix-vector multiply: out[row] = sum_c mat[row*cols + c] * vec[c]
// One invocation per output row. Each workgroup handles WORKGROUP_SIZE rows.
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer MatBuf { float mat[]; };
layout(set = 0, binding = 1) readonly buffer VecBuf { float vec_in[]; };
layout(set = 0, binding = 2) writeonly buffer OutBuf { float vec_out[]; };

layout(push_constant) uniform PC {
    uint rows;
    uint cols;
} pc;

void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= pc.rows) return;

    float acc = 0.0;
    uint base = row * pc.cols;
    for (uint c = 0; c < pc.cols; c++) {
        acc = fma(mat[base + c], vec_in[c], acc);
    }
    vec_out[row] = acc;
}
