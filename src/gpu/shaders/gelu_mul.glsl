#version 450
// Elementwise GELU + multiply: a[i] = gelu(a[i]) * b[i]
// GELU is the approximate tanh variant matching ops/math.zig:
//   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
// Used between gate+up matvecs and the w_down matvec in the dense FFN.
// One thread per element; 256 threads per workgroup.

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) buffer A          { float a[]; };
layout(std430, set = 0, binding = 1) readonly buffer B { float b[]; };

layout(push_constant) uniform PC { uint n; } pc;

const float SQRT_2_OVER_PI = 0.7978845608028654;
const float GELU_COEFF     = 0.044715;

void main() {
    const uint i = gl_GlobalInvocationID.x;
    if (i >= pc.n) return;
    const float x = a[i];
    const float t = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
    const float g = 0.5 * x * (1.0 + tanh(t));
    a[i] = g * b[i];
}
