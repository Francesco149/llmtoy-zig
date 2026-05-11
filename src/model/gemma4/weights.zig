/// Gemma4 weight storage.
///
/// Each layer has up to 7 RMSNorm weight vectors plus quantized attention and
/// FFN matrices.  Global attention layers omit wv (use K as V instead).
/// All matrices are zero-copy slices into the GGUF mmap.

const std      = @import("std");
const GgmlType = @import("../../gguf/types.zig").GgmlType;

pub const RawMatrix = struct {
    data:  []const u8,
    type_: GgmlType,
    rows:  usize,
    cols:  usize,
};

pub const Gemma4LayerWeights = struct {
    // RMSNorm weights (always F32 in this model)
    attn_norm:          []f32,  // [d_model]
    post_attention_norm:[]f32,  // [d_model]
    q_norm:             []f32,  // [head_dim] for this layer
    k_norm:             []f32,  // [head_dim]
    ffn_norm:           []f32,  // [d_model]   — dense FFN input norm
    pre_ffw_norm_2:     []f32,  // [d_model]   — MoE expert input norm
    post_ffw_norm_1:    []f32,  // [d_model]   — post dense-FFN norm
    post_ffw_norm_2:    []f32,  // [d_model]   — post MoE norm
    post_ffw_norm:      []f32,  // [d_model]   — final combined FFN norm

    layer_output_scale: f32,    // scalar applied after second residual

    // Attention projections
    wq: RawMatrix,              // [nq,  d_model]
    wk: RawMatrix,              // [nkv, d_model]
    wv: ?RawMatrix,             // [nkv, d_model]; null for global layers → V = K
    wo: RawMatrix,              // [d_model, nq]

    // Dense SwiGLU-GELU FFN
    w_gate: RawMatrix,          // [d_ffn, d_model]
    w_up:   RawMatrix,          // [d_ffn, d_model]
    w_down: RawMatrix,          // [d_model, d_ffn]

    // Sparse MoE FFN
    router_w:        RawMatrix, // [n_experts, d_model]  F32 router weights
    router_scale:    []f32,     // [d_model]  element-wise router input scale
    gate_up_exps:    RawMatrix, // 3D: [d_model, 2*d_expert, n_experts]
    down_exps:       RawMatrix, // 3D: [d_expert, d_model, n_experts]
    down_exps_scale: []f32,     // [n_experts]  per-expert output scale (F32)
};

pub const Gemma4Weights = struct {
    token_emb:   RawMatrix,          // [vocab_size, d_model]
    layers:      []Gemma4LayerWeights,
    out_norm:    []f32,              // [d_model]
    lm_head:     RawMatrix,          // [vocab_size, d_model]
    rope_freqs:  []f32,              // [head_dim_global/2] explicit global RoPE freqs

    // Arena owns all dequantized norm slices
    arena: std.heap.ArenaAllocator,

    pub fn deinit(self: *Gemma4Weights) void {
        self.arena.deinit();
    }
};
