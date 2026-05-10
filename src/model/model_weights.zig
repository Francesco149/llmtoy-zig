/// ModelWeights — weight storage for real GGUF-loaded models.
///
/// Norm weights (d_model elements) are always dequantized to f32 at load time
/// — they're tiny and accessed element-wise by rmsnorm.
///
/// Large matrices (wq, wk, wv, wo, w_gate, w_up, w_down, lm_head, token_emb)
/// are held as raw byte slices pointing directly into the mmap'd GGUF file
/// (zero-copy) together with their GgmlType. The forward pass calls
/// math.quantMatvec to dequantize one row at a time during the dot product.
///
/// The GgufReader (and its mmap) MUST outlive this struct.

const std      = @import("std");
const Config   = @import("config.zig").Config;
const GgmlType = @import("../gguf/types.zig").GgmlType;

pub const RawMatrix = struct {
    data:  []const u8,
    type_: GgmlType,
    rows:  usize, // output dimension  (first argument to quantMatvec)
    cols:  usize, // input  dimension
};

pub const LayerWeights = struct {
    attn_norm: []f32, // [d_model]
    ffn_norm:  []f32, // [d_model]
    wq:        RawMatrix, // [nq,  d_model]
    wk:        RawMatrix, // [nkv, d_model]
    wv:        RawMatrix, // [nkv, d_model]
    wo:        RawMatrix, // [d_model, nq ]
    w_gate:    RawMatrix, // [d_ffn, d_model]
    w_up:      RawMatrix, // [d_ffn, d_model]
    w_down:    RawMatrix, // [d_model, d_ffn]
};

pub const ModelWeights = struct {
    token_emb: RawMatrix, // [vocab_size, d_model]
    layers:    []LayerWeights,
    out_norm:  []f32,     // [d_model]
    lm_head:   RawMatrix, // [vocab_size, d_model]

    // Arena that owns the norm weight f32 slices.
    arena: std.heap.ArenaAllocator,

    pub fn deinit(self: *ModelWeights) void {
        self.arena.deinit();
    }
};
