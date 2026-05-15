#version 450
// Quantize an f32 activation vector to Q8_1 in the x4-packed memory layout that
// the integer-dot matvec shaders consume.
//
// Output layout per 128 input elements:
//   struct block_q8_1_x4 {        // 144 bytes
//       f16vec2 ds[4];            //  16 bytes — d, d*sum_qs per sub-block
//       int32_t qs[32];           // 128 bytes — i8 quants, 4 packed per i32
//   };
//
// One workgroup handles one block_q8_1_x4 = 128 input elements = 32 vec4s.
// Threads cooperate via subgroupClusteredMax/Add(cluster=8); each cluster of 8
// threads handles one Q8_1 sub-block (32 elements / 4 elements per vec4).
//
// Partial trailing blocks: if ncols isn't a multiple of 128, the host pads the
// output buffer up to ceil(ncols/128) groups; vec4 reads past ncols clamp to
// zero, producing a zero block.  Vec4 alignment is required (ncols % 4 == 0),
// which all model activations satisfy.

#extension GL_EXT_control_flow_attributes               : require
#extension GL_EXT_shader_16bit_storage                  : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16: require
#extension GL_EXT_shader_explicit_arithmetic_types_int32: require
#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_KHR_shader_subgroup_basic                 : require
#extension GL_KHR_shader_subgroup_clustered             : require
#extension GL_KHR_shader_subgroup_arithmetic            : require

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

layout(push_constant) uniform PC { uint ncols; } pc;

void main() {
    const uint tid          = gl_LocalInvocationID.x;   // 0..31
    const uint wgid         = gl_WorkGroupID.x;         // x4-group index
    const uint block_in_wg  = tid / 8;                  // which sub-block (0..3)
    const uint iqs          = tid % 8;                  // which i32 in sub-block (0..7)

    const uint v4_idx = wgid * 32 + tid;
    const uint ne_v4  = (pc.ncols + 3u) / 4u;

    vec4 vals = (v4_idx < ne_v4) ? data_a[v4_idx] : vec4(0.0);
    const vec4 abs_vals = abs(vals);
    const float lane_max = max(max(abs_vals.x, abs_vals.y),
                               max(abs_vals.z, abs_vals.w));

    // Per-sub-block amax across 8 cooperating lanes.
    const float amax  = subgroupClusteredMax(lane_max, 8);
    const float d     = amax / 127.0;
    const float d_inv = (d != 0.0) ? (1.0 / d) : 0.0;

    vec4 vq = round(vals * d_inv);
    vq      = clamp(vq, vec4(-127.0), vec4(127.0));

    data_b[wgid].qs[block_in_wg * 8u + iqs] =
        pack32(i8vec4(int8_t(vq.x), int8_t(vq.y), int8_t(vq.z), int8_t(vq.w)));

    // d * sum(qs) → ds.y; consumed by Q4_1 / Q5_1 / Q4_K / Q5_K dot products.
    const float lane_sum = vq.x + vq.y + vq.z + vq.w;
    const float sum_qs   = subgroupClusteredAdd(lane_sum, 8);

    if (iqs == 0u) {
        data_b[wgid].ds[block_in_wg] = f16vec2(d, sum_qs * d);
    }
}
