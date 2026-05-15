#version 450
// Elementwise scale: x[i] *= s
// Used for embedding scale (× sqrt(d_model)) and layer_output_scale.
// One thread per element; 256 threads per workgroup.

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) buffer X { float x[]; };

layout(push_constant) uniform PC {
    uint  n;
    float s;
} pc;

void main() {
    const uint i = gl_GlobalInvocationID.x;
    if (i >= pc.n) return;
    x[i] *= pc.s;
}
