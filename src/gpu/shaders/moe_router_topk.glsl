#version 450
// Gemma4 decode router:
//   moe_in    = rmsnorm(x, norm_w)
//   router_in = x * rms_inv * router_scale / sqrt(d_model)
//   logits    = router_w * router_in
//   ids       = stable top-k(logits)
//   scales    = softmax(logits)[ids] * down_scale[ids]
//
// One workgroup handles one token. Gemma4 has 128 experts and selects 8.

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly  buffer X          { float x[]; };
layout(std430, set = 0, binding = 1) readonly  buffer NormW      { float norm_w[]; };
layout(std430, set = 0, binding = 2) readonly  buffer RouterScale { float router_scale[]; };
layout(std430, set = 0, binding = 3) readonly  buffer RouterW     { float router_w[]; };
layout(std430, set = 0, binding = 4) readonly  buffer DownScale   { float down_scale[]; };
layout(std430, set = 0, binding = 5) writeonly buffer MoeIn       { float moe_in[]; };
layout(std430, set = 0, binding = 6) buffer Ids                  { uint ids[]; };
layout(std430, set = 0, binding = 7) writeonly buffer Scales      { float scales[]; };
layout(std430, set = 0, binding = 8) writeonly buffer AccumScales { float accum_scales[]; };

layout(push_constant) uniform PC {
    uint  d_model;
    uint  n_experts;
    uint  n_active;
    float eps;
} pc;

shared float partials[128];
shared float router_in[4096];
shared float logits[128];

void main() {
    const uint tid = gl_LocalInvocationID.x;

    float ss = 0.0;
    for (uint i = tid; i < pc.d_model; i += gl_WorkGroupSize.x) {
        ss += x[i] * x[i];
    }
    partials[tid] = ss;
    barrier();
    for (uint stride = gl_WorkGroupSize.x / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) partials[tid] += partials[tid + stride];
        barrier();
    }

    const float rms_inv = inversesqrt(partials[0] / float(pc.d_model) + pc.eps);
    const float router_factor = rms_inv * inversesqrt(float(pc.d_model));
    for (uint i = tid; i < pc.d_model; i += gl_WorkGroupSize.x) {
        moe_in[i] = x[i] * rms_inv * norm_w[i];
        router_in[i] = x[i] * router_factor * router_scale[i];
    }
    barrier();

    if (tid < pc.n_experts) {
        float score = 0.0;
        const uint base = tid * pc.d_model;
        for (uint i = 0u; i < pc.d_model; i++) {
            score += router_w[base + i] * router_in[i];
        }
        logits[tid] = score;
    }
    barrier();

    if (tid != 0u) return;

    float max_logit = -1.0 / 0.0;
    float exp_sum = 0.0;
    for (uint e = 0u; e < pc.n_experts; e++) {
        const float score = logits[e];
        if (score > max_logit) {
            exp_sum = exp_sum * exp(max_logit - score) + 1.0;
            max_logit = score;
        } else {
            exp_sum += exp(score - max_logit);
        }
    }

    for (uint k = 0u; k < pc.n_active; k++) {
        uint best = 0u;
        float best_val = -1.0 / 0.0;
        for (uint e = 0u; e < pc.n_experts; e++) {
            bool already_selected = false;
            for (uint prev = 0u; prev < k; prev++) {
                if (ids[prev] == e) {
                    already_selected = true;
                    break;
                }
            }
            if (!already_selected && logits[e] > best_val) {
                best = e;
                best_val = logits[e];
            }
        }
        ids[k] = best;
        scales[k] = down_scale[best] * exp(best_val - max_logit) / exp_sum;
        accum_scales[k] = 1.0;
    }
}
