#version 450
// Gemma final logit softcap: x[i] = tanh(x[i] / cap) * cap.
// One thread per element; 256 threads per workgroup.

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) buffer X { float x[]; };

layout(push_constant) uniform PC {
    uint  n;
    float cap;
} pc;

void main() {
    const uint i = gl_GlobalInvocationID.x;
    if (i >= pc.n) return;
    x[i] = tanh(x[i] / pc.cap) * pc.cap;
}
