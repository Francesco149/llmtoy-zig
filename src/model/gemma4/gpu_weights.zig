/// GPU-resident weight sessions for Gemma4.
///
/// Uploads per-layer attention and dense-FFN weight matrices to VRAM once at
/// model-load time. MoE expert tensors stay on CPU: they are too large to
/// pre-upload in full (≈20 GB across 62 layers) and are accessed sparsely
/// (8 of 128 experts per token), so staging overhead would exceed the benefit.
///
/// Matrices whose quant type has no GPU shader (Q5_K, IQ4_NL, …) get a null
/// session and fall back to the CPU path transparently.

const std       = @import("std");
const GgmlType  = @import("../../gguf/types.zig").GgmlType;
const GpuCtx    = @import("../../gpu/context.zig").GpuContext;
const mv_mod    = @import("../../gpu/matvec.zig");
const MatvecPipeline = mv_mod.MatvecPipeline;
const MatvecSession  = mv_mod.MatvecSession;
const wt_       = @import("weights.zig");
const Gemma4Weights  = wt_.Gemma4Weights;
const RawMatrix      = wt_.RawMatrix;
const Gemma4Config   = @import("config.zig").Gemma4Config;

// Sessions for one layer's attention + dense FFN projections.
pub const GpuLayerWeights = struct {
    wq:     ?MatvecSession,
    wk:     ?MatvecSession,
    wv:     ?MatvecSession,
    wo:     ?MatvecSession,
    w_gate: ?MatvecSession,
    w_up:   ?MatvecSession,
    w_down: ?MatvecSession,

    pub fn deinitAll(self: *GpuLayerWeights) void {
        if (self.wq)     |*s| s.deinit();
        if (self.wk)     |*s| s.deinit();
        if (self.wv)     |*s| s.deinit();
        if (self.wo)     |*s| s.deinit();
        if (self.w_gate) |*s| s.deinit();
        if (self.w_up)   |*s| s.deinit();
        if (self.w_down) |*s| s.deinit();
    }
};

pub const GpuWeights = struct {
    ctx:     GpuCtx,
    pl_f32:  MatvecPipeline,
    pl_q8_0: MatvecPipeline,
    pl_q3_k: MatvecPipeline,
    pl_q4_k: MatvecPipeline,
    layers:  []GpuLayerWeights,
    lm_head: ?MatvecSession,
    allocator: std.mem.Allocator,

    pub fn init(g4w: *const Gemma4Weights, g4cfg: Gemma4Config, allocator: std.mem.Allocator) !GpuWeights {
        var ctx = try GpuCtx.init();
        errdefer ctx.deinit();

        var pl_f32  = try MatvecPipeline.initF32(&ctx);
        errdefer pl_f32.deinit();
        var pl_q8_0 = try MatvecPipeline.initQ8_0(&ctx);
        errdefer pl_q8_0.deinit();
        var pl_q3_k = try MatvecPipeline.initQ3K(&ctx);
        errdefer pl_q3_k.deinit();
        var pl_q4_k = try MatvecPipeline.initQ4K(&ctx);
        errdefer pl_q4_k.deinit();

        const layers = try allocator.alloc(GpuLayerWeights, g4cfg.n_layers);
        errdefer allocator.free(layers);
        for (layers) |*l| l.* = .{
            .wq = null, .wk = null, .wv = null, .wo = null,
            .w_gate = null, .w_up = null, .w_down = null,
        };

        // n_layers_done tracks how many layers have fully-uploaded sessions so
        // that the outer errdefer only deinits layers we've completed.
        var n_layers_done: usize = 0;
        errdefer for (0..n_layers_done) |l| layers[l].deinitAll();

        for (0..g4cfg.n_layers) |l| {
            const lw = &g4w.layers[l];
            // Per-iteration errdefer cleans up any partial uploads within this layer.
            errdefer layers[l].deinitAll();

            layers[l].wq     = try upload(&ctx, lw.wq);
            layers[l].wk     = try upload(&ctx, lw.wk);
            if (lw.wv) |wv| layers[l].wv = try upload(&ctx, wv);
            layers[l].wo     = try upload(&ctx, lw.wo);
            layers[l].w_gate = try upload(&ctx, lw.w_gate);
            layers[l].w_up   = try upload(&ctx, lw.w_up);
            layers[l].w_down = try upload(&ctx, lw.w_down);

            n_layers_done = l + 1;
        }

        const lm_head = try upload(&ctx, g4w.lm_head);

        return .{
            .ctx     = ctx,
            .pl_f32  = pl_f32,
            .pl_q8_0 = pl_q8_0,
            .pl_q3_k = pl_q3_k,
            .pl_q4_k = pl_q4_k,
            .layers  = layers,
            .lm_head = lm_head,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *GpuWeights) void {
        if (self.lm_head) |*s| s.deinit();
        for (self.layers) |*l| l.deinitAll();
        self.allocator.free(self.layers);
        self.pl_q4_k.deinit();
        self.pl_q3_k.deinit();
        self.pl_q8_0.deinit();
        self.pl_f32.deinit();
        self.ctx.deinit();
    }

    // Return the pipeline for a supported quant type. Only call when a
    // session exists for that type (session creation implies support).
    pub fn pipelineFor(self: *const GpuWeights, t: GgmlType) *const MatvecPipeline {
        return switch (t) {
            .f32  => &self.pl_f32,
            .q8_0 => &self.pl_q8_0,
            .q3_k => &self.pl_q3_k,
            .q4_k => &self.pl_q4_k,
            else  => unreachable,
        };
    }
};

fn upload(ctx: *const GpuCtx, mat: RawMatrix) !?MatvecSession {
    return MatvecSession.initFromRaw(ctx, mat.data, mat.type_,
        @intCast(mat.rows), @intCast(mat.cols));
}
