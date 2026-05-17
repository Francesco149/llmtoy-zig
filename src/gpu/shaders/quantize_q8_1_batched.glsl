#version 450
// Quantize a flat [slot, ncols] f32 buffer to slot-major Q8_1_x4 blocks.

#extension GL_EXT_control_flow_attributes                 : require
#extension GL_EXT_shader_16bit_storage                    : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8   : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16  : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32  : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_KHR_shader_subgroup_basic                   : require
#extension GL_KHR_shader_subgroup_clustered               : require
#extension GL_KHR_shader_subgroup_arithmetic              : require

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly buffer A {
    vec4 data_a[];
};

struct block_q8_1_x4 {
    f16vec2 ds[4];
    int32_t qs[32];
};

layout(std430, set = 0, binding = 1) writeonly buffer B {
    block_q8_1_x4 data_b[];
};

layout(push_constant) uniform PC { uint ncols; uint n_active; } pc;

void main() {
    const uint tid = gl_LocalInvocationID.x;
    const uint block = gl_WorkGroupID.x;
    const uint slot = gl_WorkGroupID.y;
    if (slot >= pc.n_active) return;

    const uint block_in_wg = tid / 8u;
    const uint iqs = tid % 8u;
    const uint blocks_per_slot = (pc.ncols + 127u) / 128u;
    const uint v4_per_slot = (pc.ncols + 3u) / 4u;
    const uint v4_idx = slot * v4_per_slot + block * 32u + tid;
    const uint block_out = slot * blocks_per_slot + block;

    vec4 vals = (block * 32u + tid < v4_per_slot) ? data_a[v4_idx] : vec4(0.0);
    const vec4 abs_vals = abs(vals);
    const float lane_max = max(max(abs_vals.x, abs_vals.y), max(abs_vals.z, abs_vals.w));

    const float amax = subgroupClusteredMax(lane_max, 8);
    const float d = amax / 127.0;
    const float d_inv = (d != 0.0) ? (1.0 / d) : 0.0;

    vec4 vq = round(vals * d_inv);
    vq = clamp(vq, vec4(-127.0), vec4(127.0));

    data_b[block_out].qs[block_in_wg * 8u + iqs] =
        pack32(i8vec4(int8_t(vq.x), int8_t(vq.y), int8_t(vq.z), int8_t(vq.w)));

    const float lane_sum = vq.x + vq.y + vq.z + vq.w;
    const float sum_qs = subgroupClusteredAdd(lane_sum, 8);

    if (iqs == 0u) {
        data_b[block_out].ds[block_in_wg] = f16vec2(d, sum_qs * d);
    }
}
