const std = @import("std");
const Config = @import("config.zig").Config;

/// Plain f32 weight arrays for a minimal LLaMA-style transformer.
/// Layout matches what we'll dequantize from GGUF in Phase 4.
///
/// All slices are owned by this struct — call deinit to free.
/// Use initZero for a clean slate, then overwrite individual weights for tests.
pub const Weights = struct {
    token_emb: []f32,   // [vocab_size × d_model]

    // Per-layer (n_layers slices each)
    attn_norm: [][]f32, // [n_layers][d_model]
    wq: [][]f32,        // [n_layers][d_model × head_dim]
    wk: [][]f32,        // [n_layers][d_model × head_dim]
    wv: [][]f32,        // [n_layers][d_model × head_dim]
    wo: [][]f32,        // [n_layers][d_model × head_dim]  (head_dim → d_model)
    ffn_norm: [][]f32,  // [n_layers][d_model]
    w_gate: [][]f32,    // [n_layers][d_ffn × d_model]
    w_up: [][]f32,      // [n_layers][d_ffn × d_model]
    w_down: [][]f32,    // [n_layers][d_model × d_ffn]

    out_norm: []f32,    // [d_model]
    lm_head: []f32,     // [vocab_size × d_model]

    allocator: std.mem.Allocator,

    /// Allocate all weight arrays and zero-initialise them.
    pub fn initZero(cfg: Config, allocator: std.mem.Allocator) !Weights {
        const d = cfg.d_model;
        const hd = cfg.headDim();
        const v = cfg.vocab_size;
        const ff = cfg.d_ffn;
        const l = cfg.n_layers;

        const token_emb = try zeroSlice(allocator, v * d);
        errdefer allocator.free(token_emb);

        const attn_norm = try layerSlices(allocator, l, d);
        errdefer freeLayerSlices(allocator, attn_norm);

        const wq = try layerSlices(allocator, l, d * hd);
        errdefer freeLayerSlices(allocator, wq);
        const wk = try layerSlices(allocator, l, d * hd);
        errdefer freeLayerSlices(allocator, wk);
        const wv = try layerSlices(allocator, l, d * hd);
        errdefer freeLayerSlices(allocator, wv);
        const wo = try layerSlices(allocator, l, d * hd);
        errdefer freeLayerSlices(allocator, wo);

        const ffn_norm = try layerSlices(allocator, l, d);
        errdefer freeLayerSlices(allocator, ffn_norm);
        const w_gate = try layerSlices(allocator, l, ff * d);
        errdefer freeLayerSlices(allocator, w_gate);
        const w_up = try layerSlices(allocator, l, ff * d);
        errdefer freeLayerSlices(allocator, w_up);
        const w_down = try layerSlices(allocator, l, d * ff);
        errdefer freeLayerSlices(allocator, w_down);

        const out_norm = try zeroSlice(allocator, d);
        errdefer allocator.free(out_norm);
        const lm_head = try zeroSlice(allocator, v * d);
        errdefer allocator.free(lm_head);

        return .{
            .token_emb = token_emb,
            .attn_norm = attn_norm,
            .wq = wq, .wk = wk, .wv = wv, .wo = wo,
            .ffn_norm = ffn_norm,
            .w_gate = w_gate, .w_up = w_up, .w_down = w_down,
            .out_norm = out_norm,
            .lm_head = lm_head,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Weights) void {
        const a = self.allocator;
        a.free(self.token_emb);
        freeLayerSlices(a, self.attn_norm);
        freeLayerSlices(a, self.wq);
        freeLayerSlices(a, self.wk);
        freeLayerSlices(a, self.wv);
        freeLayerSlices(a, self.wo);
        freeLayerSlices(a, self.ffn_norm);
        freeLayerSlices(a, self.w_gate);
        freeLayerSlices(a, self.w_up);
        freeLayerSlices(a, self.w_down);
        a.free(self.out_norm);
        a.free(self.lm_head);
    }

    /// Set all RMSNorm weight vectors to 1 (identity normalisation).
    /// Useful in tests to remove the weight's effect from normalisation.
    pub fn setNormWeightsOne(self: *Weights) void {
        for (self.attn_norm) |w| @memset(w, 1.0);
        for (self.ffn_norm) |w| @memset(w, 1.0);
        @memset(self.out_norm, 1.0);
    }
};

fn zeroSlice(allocator: std.mem.Allocator, n: usize) ![]f32 {
    const s = try allocator.alloc(f32, n);
    @memset(s, 0.0);
    return s;
}

fn layerSlices(allocator: std.mem.Allocator, n_layers: usize, elem_len: usize) ![][]f32 {
    const outer = try allocator.alloc([]f32, n_layers);
    for (outer, 0..) |*s, i| {
        s.* = try zeroSlice(allocator, elem_len);
        errdefer for (outer[0..i]) |prev| allocator.free(prev);
    }
    return outer;
}

fn freeLayerSlices(allocator: std.mem.Allocator, slices: [][]f32) void {
    for (slices) |s| allocator.free(s);
    allocator.free(slices);
}
