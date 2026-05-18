#version 450
// Weighted accumulation of n expert outputs into a single output vector.
// output[i] = sum over k of (scales[k] * inputs[k * d_model + i])
//
// Bindings: 0=inputs (n × d_model f32), 1=scales (n f32), 2=output (d_model f32, read-write)
// Push constants: d_model, n (number of active experts, ≤16)

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly buffer InputsBuf { float inputs[]; };
layout(std430, set = 0, binding = 1) readonly buffer ScalesBuf { float scales[]; };
layout(std430, set = 0, binding = 2) buffer OutputBuf { float output_[]; };

layout(push_constant) uniform PC { uint d_model; uint n; } pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.d_model) return;

    float acc = 0.0;
    for (uint k = 0u; k < pc.n; k++) {
        acc += scales[k] * inputs[k * pc.d_model + i];
    }
    output_[i] = acc;
}
