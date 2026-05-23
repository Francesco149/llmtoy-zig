#version 450
// Elementwise fused residual add and scale: a[i] = (a[i] + b[i]) * s
// One thread per element; 256 threads per workgroup.

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) buffer A          { float a[]; };
layout(std430, set = 0, binding = 1) readonly buffer B { float b[]; };

layout(push_constant) uniform PC {
    uint  n;
    float s;
} pc;

void main() {
    const uint i = gl_GlobalInvocationID.x;
    if (i >= pc.n) return;
    a[i] = (a[i] + b[i]) * pc.s;
}
