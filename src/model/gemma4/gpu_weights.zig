/// GPU-resident weight sessions for Gemma4.
///
/// Phase 1: attention + dense FFN — one layer's ≤7 matrices per vkQueueSubmit.
/// Phase 2: MoE experts — all 128 experts per layer uploaded in batches of 16
///          (~47 MiB staging peak per batch). Expert sessions stored in flat
///          arrays [n_layers × n_experts]; null entries fall back to CPU.
///
/// lm_head is uploaded separately and falls back to CPU on alloc failure.
///
/// Matrices with no GPU shader (Q5_K, IQ4_NL, …) get null sessions and fall
/// back to the CPU path transparently.

const std        = @import("std");
const math       = @import("../../ops/math.zig");
const vk_mod     = @import("../../gpu/vk.zig");
const vk         = vk_mod.vk;
const GgmlType   = @import("../../gguf/types.zig").GgmlType;
const GpuCtx     = @import("../../gpu/context.zig").GpuContext;
const GpuBuffer  = @import("../../gpu/buffer.zig").GpuBuffer;
const mv_mod     = @import("../../gpu/matvec.zig");
const MatvecPipeline      = mv_mod.MatvecPipeline;
const MatvecSession       = mv_mod.MatvecSession;
const FusedGateUpPipeline = mv_mod.FusedGateUpPipeline;
const AccumPipeline       = mv_mod.AccumPipeline;
const QuantizeQ8_1Pipeline = mv_mod.QuantizeQ8_1Pipeline;
const wt_                = @import("weights.zig");
const Gemma4Weights      = wt_.Gemma4Weights;
const Gemma4LayerWeights = wt_.Gemma4LayerWeights;
const RawMatrix          = wt_.RawMatrix;
const Gemma4Config   = @import("config.zig").Gemma4Config;

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
    ctx:        GpuCtx,
    pl_f32:     MatvecPipeline,
    pl_q8_0:    MatvecPipeline,
    pl_q3_k:    MatvecPipeline,
    pl_q4_k:    MatvecPipeline,
    pl_q4_k_q8_1:    MatvecPipeline,
    pl_quantize_q8_1: QuantizeQ8_1Pipeline,
    pl_q5_1:    MatvecPipeline,
    pl_q5_0:    MatvecPipeline,
    pl_fused_gu: FusedGateUpPipeline,
    pl_accum:    AccumPipeline,
    layers:     []GpuLayerWeights,
    lm_head:    ?MatvecSession,
    // Shared host-coherent I/O buffers sized to the largest matrix across all
    // sessions. Eliminates 420+ individual per-session HOST_COHERENT allocations.
    shared_vec:  ?GpuBuffer,
    shared_out:  ?GpuBuffer,
    // Device-local Q8_1-quantized activation, sized to q8_1OutBytes(max_cols).
    // Quantize-shader writes; mul_mat_vec_q*_q8_1 shaders read.  Lives in VRAM
    // so neither path crosses PCIe per token.
    shared_acts_q8_1: ?GpuBuffer,
    // Flat [n_layers × n_experts] expert session arrays; null = CPU fallback.
    expert_gate: ?[]?MatvecSession,
    expert_up:   ?[]?MatvecSession,
    expert_down: ?[]?MatvecSession,
    n_experts:      usize,
    n_experts_used: usize,
    // I/O buffers for batched expert dispatch:
    // expert_in_buf:      HOST_COHERENT, persistently mapped — CPU writes moe_in.
    // expert_mid_bufs:    device-local VRAM — fused shader writes, down shader reads.
    // expert_all_out_buf: device-local VRAM — flat [n_experts_used × d_model] f32;
    //                     down shader writes sub-ranges, accum shader reads the whole.
    // expert_scales_buf:  HOST_COHERENT, persistently mapped — CPU writes router scales.
    // moe_gpu_buf:        HOST_COHERENT, persistently mapped — accum shader writes result,
    //                     CPU reads it after submit.
    expert_in_buf:      ?GpuBuffer,
    expert_mid_bufs:    ?[]GpuBuffer,
    expert_all_out_buf: ?GpuBuffer,
    expert_scales_buf:  ?GpuBuffer,
    moe_gpu_buf:        ?GpuBuffer,
    expert_in_slice:    ?[]f32,
    expert_scales_slice: ?[]f32,
    moe_gpu_slice:      ?[]f32,
    // Per-projection output buffers for batched dispatch.
    // QKV: all three read the same input → one upload, three parallel dispatches.
    // gate+up: both read the same FFN-norm input → same pattern.
    q_out_buf:    ?GpuBuffer,
    k_out_buf:    ?GpuBuffer,
    v_out_buf:    ?GpuBuffer,
    gate_out_buf: ?GpuBuffer,
    up_out_buf:   ?GpuBuffer,
    allocator:   std.mem.Allocator,

    pub fn init(g4w: *const Gemma4Weights, g4cfg: Gemma4Config, allocator: std.mem.Allocator) !GpuWeights {
        const avail_mb = availableMemoryMB();
        std.debug.print("  available system RAM: {} MiB\n", .{avail_mb});
        if (avail_mb < 500) return error.InsufficientMemory;

        std.debug.print("  init: creating Vulkan context (VRAM={} MiB GTT={} MiB)\n",
            .{ vramUsedMB(), gttUsedMB() });
        var ctx = try GpuCtx.init();
        errdefer ctx.deinit();
        std.debug.print("  init: VkDevice ready (VRAM={} MiB GTT={} MiB sys={} MiB)\n",
            .{ vramUsedMB(), gttUsedMB(), availableMemoryMB() });

        var pl_f32  = try MatvecPipeline.initF32(&ctx);
        errdefer pl_f32.deinit();
        std.debug.print("  init: pl_f32  ok (VRAM={} MiB GTT={} MiB sys={} MiB)\n",
            .{ vramUsedMB(), gttUsedMB(), availableMemoryMB() });
        var pl_q8_0 = try MatvecPipeline.initQ8_0(&ctx);
        errdefer pl_q8_0.deinit();
        std.debug.print("  init: pl_q8_0 ok (VRAM={} MiB GTT={} MiB sys={} MiB)\n",
            .{ vramUsedMB(), gttUsedMB(), availableMemoryMB() });
        var pl_q3_k = try MatvecPipeline.initQ3K(&ctx);
        errdefer pl_q3_k.deinit();
        std.debug.print("  init: pl_q3_k ok (VRAM={} MiB GTT={} MiB sys={} MiB)\n",
            .{ vramUsedMB(), gttUsedMB(), availableMemoryMB() });
        var pl_q4_k = try MatvecPipeline.initQ4K(&ctx);
        errdefer pl_q4_k.deinit();
        std.debug.print("  init: pl_q4_k ok (VRAM={} MiB GTT={} MiB sys={} MiB)\n",
            .{ vramUsedMB(), gttUsedMB(), availableMemoryMB() });
        var pl_q4_k_q8_1 = try MatvecPipeline.initQ4KQ8_1(&ctx);
        errdefer pl_q4_k_q8_1.deinit();
        std.debug.print("  init: pl_q4_k_q8_1 ok\n", .{});
        var pl_quantize_q8_1 = try QuantizeQ8_1Pipeline.init(&ctx);
        errdefer pl_quantize_q8_1.deinit();
        std.debug.print("  init: pl_quantize_q8_1 ok\n", .{});
        var pl_q5_1 = try MatvecPipeline.initQ5_1(&ctx);
        errdefer pl_q5_1.deinit();
        std.debug.print("  init: pl_q5_1 ok (VRAM={} MiB GTT={} MiB sys={} MiB)\n",
            .{ vramUsedMB(), gttUsedMB(), availableMemoryMB() });
        var pl_q5_0 = try MatvecPipeline.initQ5_0(&ctx);
        errdefer pl_q5_0.deinit();
        std.debug.print("  init: pl_q5_0 ok (VRAM={} MiB GTT={} MiB sys={} MiB)\n",
            .{ vramUsedMB(), gttUsedMB(), availableMemoryMB() });
        var pl_fused_gu = try FusedGateUpPipeline.init(&ctx);
        errdefer pl_fused_gu.deinit();
        std.debug.print("  init: pl_fused_gu ok\n", .{});
        var pl_accum = try AccumPipeline.init(&ctx);
        errdefer pl_accum.deinit();
        std.debug.print("  init: pl_accum ok\n", .{});

        const layers = try allocator.alloc(GpuLayerWeights, g4cfg.n_layers);
        errdefer allocator.free(layers);
        for (layers) |*l| l.* = .{
            .wq = null, .wk = null, .wv = null, .wo = null,
            .w_gate = null, .w_up = null, .w_down = null,
        };

        var gw = GpuWeights{
            .ctx = ctx, .pl_f32 = pl_f32, .pl_q8_0 = pl_q8_0,
            .pl_q3_k = pl_q3_k, .pl_q4_k = pl_q4_k,
            .pl_q4_k_q8_1 = pl_q4_k_q8_1,
            .pl_quantize_q8_1 = pl_quantize_q8_1,
            .pl_q5_1 = pl_q5_1, .pl_q5_0 = pl_q5_0,
            .pl_fused_gu = pl_fused_gu, .pl_accum = pl_accum,
            .layers = layers, .lm_head = null,
            .shared_vec = null, .shared_out = null,
            .shared_acts_q8_1 = null,
            .expert_gate = null, .expert_up = null, .expert_down = null,
            .n_experts = g4cfg.n_experts, .n_experts_used = g4cfg.n_experts_used,
            .expert_in_buf = null, .expert_mid_bufs = null,
            .expert_all_out_buf = null, .expert_scales_buf = null, .moe_gpu_buf = null,
            .expert_in_slice = null, .expert_scales_slice = null, .moe_gpu_slice = null,
            .q_out_buf = null, .k_out_buf = null, .v_out_buf = null,
            .gate_out_buf = null, .up_out_buf = null,
            .allocator = allocator,
        };
        errdefer gw.deinit();

        // One vkQueueSubmit per layer: peak staging ~22 MiB instead of ~1 GiB.
        // Preemptive abort: if sys RAM drops below 8 GiB, return error so the
        // caller falls back to CPU instead of letting the OOM killer fire.
        for (0..g4cfg.n_layers) |l| {
            const mem_before = availableMemoryMB();
            if (mem_before < 8 * 1024) {
                std.debug.print("  GPU upload aborted at layer {}: only {} MiB available\n",
                    .{ l, mem_before });
                return error.InsufficientMemory;
            }
            try uploadLayerBatch(&ctx, &gw.layers[l], &g4w.layers[l]);
            const vram_used = vramUsedMB();
            const gtt_used  = gttUsedMB();
            const mem_avail = availableMemoryMB();
            std.debug.print("  layer {:2}: VRAM={} MiB GTT={} MiB sys_avail={} MiB\n",
                .{ l, vram_used, gtt_used, mem_avail });
        }

        // lm_head staging can be 400+ MiB; fall back to CPU on alloc failure.
        gw.lm_head = uploadSingleBatch(&ctx, g4w.lm_head) catch |e| blk: {
            std.debug.print("  lm_head GPU upload failed ({s}), using CPU\n", .{@errorName(e)});
            break :blk null;
        };

        // Create ONE shared vec_buf + out_buf sized to the largest matrix.
        // Replaces 420+ per-session HOST_COHERENT allocations with 2.
        var max_rows: u32 = 1;
        var max_cols: u32 = 1;
        for (gw.layers) |l| {
            for ([_]?MatvecSession{ l.wq, l.wk, l.wv, l.wo, l.w_gate, l.w_up, l.w_down }) |ms| {
                if (ms) |s| {
                    if (s.rows > max_rows) max_rows = s.rows;
                    if (s.cols > max_cols) max_cols = s.cols;
                }
            }
        }
        if (gw.lm_head) |s| {
            if (s.rows > max_rows) max_rows = s.rows;
            if (s.cols > max_cols) max_cols = s.cols;
        }
        gw.shared_vec = try GpuBuffer.initHostCoherent(&gw.ctx,
            max_cols * @sizeOf(f32), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        gw.shared_out = try GpuBuffer.initHostCoherent(&gw.ctx,
            max_rows * @sizeOf(f32), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        // Q8_1 activation buffer lives in VRAM; quantize-shader writes, matvec reads.
        const q8_1_bytes = mv_mod.q8_1OutBytes(max_cols);
        gw.shared_acts_q8_1 = try GpuBuffer.initDeviceLocal(&gw.ctx,
            q8_1_bytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        std.debug.print("  shared I/O bufs: vec={} KiB out={} KiB q8_1_acts={} KiB\n",
            .{ max_cols * @sizeOf(f32) / 1024, max_rows * @sizeOf(f32) / 1024,
               q8_1_bytes / 1024 });

        // Per-projection output buffers for batched QKV and gate+up dispatch.
        // Sized per-projection so all three (or two) can be in-flight simultaneously.
        var max_q_rows: u32 = 1; var max_k_rows: u32 = 1; var max_v_rows: u32 = 1;
        var max_gate_rows: u32 = 1; var max_up_rows: u32 = 1;
        for (gw.layers) |l| {
            if (l.wq)     |s| if (s.rows > max_q_rows)    { max_q_rows    = s.rows; };
            if (l.wk)     |s| if (s.rows > max_k_rows)    { max_k_rows    = s.rows; };
            if (l.wv)     |s| if (s.rows > max_v_rows)    { max_v_rows    = s.rows; };
            if (l.w_gate) |s| if (s.rows > max_gate_rows) { max_gate_rows = s.rows; };
            if (l.w_up)   |s| if (s.rows > max_up_rows)   { max_up_rows   = s.rows; };
        }
        gw.q_out_buf    = try GpuBuffer.initHostCoherent(&gw.ctx, max_q_rows    * @sizeOf(f32), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        gw.k_out_buf    = try GpuBuffer.initHostCoherent(&gw.ctx, max_k_rows    * @sizeOf(f32), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        gw.v_out_buf    = try GpuBuffer.initHostCoherent(&gw.ctx, max_v_rows    * @sizeOf(f32), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        gw.gate_out_buf = try GpuBuffer.initHostCoherent(&gw.ctx, max_gate_rows * @sizeOf(f32), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        gw.up_out_buf   = try GpuBuffer.initHostCoherent(&gw.ctx, max_up_rows   * @sizeOf(f32), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        std.debug.print("  batch output bufs: q={} KiB k={} KiB v={} KiB gate={} KiB up={} KiB\n", .{
            max_q_rows    * @sizeOf(f32) / 1024, max_k_rows    * @sizeOf(f32) / 1024,
            max_v_rows    * @sizeOf(f32) / 1024, max_gate_rows * @sizeOf(f32) / 1024,
            max_up_rows   * @sizeOf(f32) / 1024,
        });

        // Upload all MoE experts to VRAM (128 experts × 30 layers ≈ 9.6 GiB).
        // Batched by 16 experts per command buffer (~47 MiB staging peak per batch).
        const n_total = g4cfg.n_layers * g4cfg.n_experts;
        gw.expert_gate = try allocator.alloc(?MatvecSession, n_total);
        @memset(gw.expert_gate.?, null);
        gw.expert_up   = try allocator.alloc(?MatvecSession, n_total);
        @memset(gw.expert_up.?, null);
        gw.expert_down = try allocator.alloc(?MatvecSession, n_total);
        @memset(gw.expert_down.?, null);
        std.debug.print("  uploading {} experts × {} layers to GPU...\n",
            .{g4cfg.n_experts, g4cfg.n_layers});
        for (0..g4cfg.n_layers) |l| {
            const off = l * g4cfg.n_experts;
            uploadExpertsBatch(
                &gw.ctx,
                gw.expert_gate.?[off..][0..g4cfg.n_experts],
                gw.expert_up.?[off..][0..g4cfg.n_experts],
                gw.expert_down.?[off..][0..g4cfg.n_experts],
                &g4w.layers[l], g4cfg.d_model, g4cfg.d_expert,
            ) catch |e| std.debug.print("  layer {}: expert upload failed ({s})\n",
                .{l, @errorName(e)});
        }
        std.debug.print("  experts done: VRAM={} MiB GTT={} MiB\n",
            .{vramUsedMB(), gttUsedMB()});

        // Expert I/O buffers for fused batched dispatch:
        // - expert_in_buf:      HOST_COHERENT, persistently mapped (CPU writes moe_in)
        // - expert_mid_bufs:    device-local VRAM (fused writes, down reads — GPU only)
        // - expert_all_out_buf: device-local VRAM, flat n×d_model; down writes sub-ranges,
        //                       accum reads the whole
        // - expert_scales_buf:  HOST_COHERENT, persistently mapped (CPU writes router scales)
        // - moe_gpu_buf:        HOST_COHERENT, persistently mapped (accum writes, CPU reads back)
        const nu = g4cfg.n_experts_used;
        gw.expert_in_buf    = try GpuBuffer.initHostCoherent(&gw.ctx,
            g4cfg.d_model * @sizeOf(f32), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        gw.expert_in_slice  = try gw.expert_in_buf.?.mapSlice(f32, g4cfg.d_model);

        gw.expert_mid_bufs  = try allocator.alloc(GpuBuffer, nu);
        for (0..nu) |k| {
            gw.expert_mid_bufs.?[k] = try GpuBuffer.initDeviceLocal(&gw.ctx,
                g4cfg.d_expert * @sizeOf(f32), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        }

        gw.expert_all_out_buf = try GpuBuffer.initDeviceLocal(&gw.ctx,
            nu * g4cfg.d_model * @sizeOf(f32), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        gw.expert_scales_buf  = try GpuBuffer.initHostCoherent(&gw.ctx,
            nu * @sizeOf(f32), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        gw.expert_scales_slice = try gw.expert_scales_buf.?.mapSlice(f32, nu);
        gw.moe_gpu_buf        = try GpuBuffer.initHostCoherent(&gw.ctx,
            g4cfg.d_model * @sizeOf(f32), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        gw.moe_gpu_slice      = try gw.moe_gpu_buf.?.mapSlice(f32, g4cfg.d_model);

        std.debug.print("  expert I/O bufs: in={} KiB, {} × mid={} KiB (VRAM), all_out={} KiB (VRAM), moe_out={} KiB\n", .{
            g4cfg.d_model * @sizeOf(f32) / 1024,
            nu, g4cfg.d_expert * @sizeOf(f32) / 1024,
            nu * g4cfg.d_model * @sizeOf(f32) / 1024,
            g4cfg.d_model * @sizeOf(f32) / 1024,
        });

        return gw;
    }

    pub fn deinit(self: *GpuWeights) void {
        if (self.moe_gpu_buf)       |*b| { b.unmap(); b.deinit(); }
        if (self.expert_scales_buf) |*b| { b.unmap(); b.deinit(); }
        if (self.expert_all_out_buf)|*b| b.deinit();
        if (self.expert_mid_bufs)   |bs| {
            for (bs) |*b| b.deinit();
            self.allocator.free(bs);
        }
        if (self.expert_in_buf) |*b| { b.unmap(); b.deinit(); }
        if (self.expert_down) |ed| { for (ed) |*ms| if (ms.*) |*s| s.deinit(); self.allocator.free(ed); }
        if (self.expert_up)   |eu| { for (eu) |*ms| if (ms.*) |*s| s.deinit(); self.allocator.free(eu); }
        if (self.expert_gate) |eg| { for (eg) |*ms| if (ms.*) |*s| s.deinit(); self.allocator.free(eg); }
        if (self.up_out_buf)   |*b| b.deinit();
        if (self.gate_out_buf) |*b| b.deinit();
        if (self.v_out_buf)    |*b| b.deinit();
        if (self.k_out_buf)    |*b| b.deinit();
        if (self.q_out_buf)    |*b| b.deinit();
        if (self.shared_acts_q8_1) |*b| b.deinit();
        if (self.shared_out) |*b| b.deinit();
        if (self.shared_vec) |*b| b.deinit();
        if (self.lm_head) |*s| s.deinit();
        for (self.layers) |*l| l.deinitAll();
        self.allocator.free(self.layers);
        self.pl_accum.deinit();
        self.pl_fused_gu.deinit();
        self.pl_q5_0.deinit();
        self.pl_q5_1.deinit();
        self.pl_quantize_q8_1.deinit();
        self.pl_q4_k_q8_1.deinit();
        self.pl_q4_k.deinit();
        self.pl_q3_k.deinit();
        self.pl_q8_0.deinit();
        self.pl_f32.deinit();
        self.ctx.deinit();
    }

    // Dispatch wq, wk, (optionally wv) in a single command buffer.
    // All three read the same xb → upload once, 2–3 parallel dispatches, 1 submit.
    // Pipelines are passed by the caller so gpu_weights.zig doesn't need raw types.
    // Returns error.NotOnGpu if any needed session is missing.
    pub fn runLayerQKV(
        self: *const GpuWeights,
        layer: usize,
        wq_pl: *const MatvecPipeline,
        wk_pl: *const MatvecPipeline,
        wv_pl: ?*const MatvecPipeline,
        xb: []const f32,
        q_out: []f32,
        k_out: []f32,
        v_out: []f32,
    ) !void {
        const lw    = &self.layers[layer];
        const wq    = lw.wq orelse return error.NotOnGpu;
        const wk    = lw.wk orelse return error.NotOnGpu;
        const q_buf = &(self.q_out_buf orelse return error.NotOnGpu);
        const k_buf = &(self.k_out_buf orelse return error.NotOnGpu);

        try self.shared_vec.?.upload(std.mem.sliceAsBytes(xb));

        const cmd = try self.ctx.beginBatch();
        var q_dset: vk.VkDescriptorSet = null;
        var k_dset: vk.VkDescriptorSet = null;
        var v_dset: ?vk.VkDescriptorSet = null;

        q_dset = try wq.recordMv(cmd, wq_pl, &self.shared_vec.?, q_buf);
        k_dset = try wk.recordMv(cmd, wk_pl, &self.shared_vec.?, k_buf);
        if (wv_pl) |vpl| {
            const wv    = lw.wv orelse return error.NotOnGpu;
            const v_buf = &(self.v_out_buf orelse return error.NotOnGpu);
            v_dset = try wv.recordMv(cmd, vpl, &self.shared_vec.?, v_buf);
        }

        try self.ctx.submitBatch(cmd);

        _ = vk.vkFreeDescriptorSets(self.ctx.device, wq_pl.desc_pool, 1, &q_dset);
        _ = vk.vkFreeDescriptorSets(self.ctx.device, wk_pl.desc_pool, 1, &k_dset);
        if (v_dset) |*ds| _ = vk.vkFreeDescriptorSets(self.ctx.device, wv_pl.?.desc_pool, 1, ds);

        try q_buf.download(std.mem.sliceAsBytes(q_out));
        try k_buf.download(std.mem.sliceAsBytes(k_out));
        if (v_dset != null) try self.v_out_buf.?.download(std.mem.sliceAsBytes(v_out));
    }

    // Q8_1-activation matvec: one f32 → Q8_1 quantize pass, then one matvec.
    // The two dispatches share a command buffer with a compute-compute barrier
    // between them so the matvec sees the freshly-written Q8_1 acts.
    // Currently only Q4_K weights are wired up (pl_q4_k_q8_1).
    pub fn runQ8_1Mv(
        self: *const GpuWeights,
        sess: *const MatvecSession,
        xb: []const f32,
        out: []f32,
    ) !void {
        std.debug.assert(sess.cols % 256 == 0);
        const vec_buf  = &self.shared_vec.?;
        const out_buf  = &self.shared_out.?;
        const acts_buf = &self.shared_acts_q8_1.?;

        try vec_buf.upload(std.mem.sliceAsBytes(xb));

        const cmd = try self.ctx.beginBatch();
        const q_dset = try self.pl_quantize_q8_1.record(
            cmd, vec_buf, acts_buf, sess.cols);
        GpuCtx.recordShaderBarrier(cmd);
        const mv_dset = try self.pl_q4_k_q8_1.record(
            cmd, &sess.mat_buf, acts_buf, out_buf, sess.rows, sess.cols);
        try self.ctx.submitBatch(cmd);

        _ = vk.vkFreeDescriptorSets(self.ctx.device,
            self.pl_quantize_q8_1.desc_pool, 1, &q_dset);
        _ = vk.vkFreeDescriptorSets(self.ctx.device,
            self.pl_q4_k_q8_1.desc_pool, 1, &mv_dset);

        try out_buf.download(std.mem.sliceAsBytes(out));
    }

    // QKV variant of runQ8_1Mv: one upload, one quantize, then 2–3 matvecs all
    // reading the same Q8_1 acts buffer. Saves 2 PCIe uploads + 2 quantize
    // dispatches vs three independent runQ8_1Mv calls.
    //
    // Caller must have already verified that wq, wk (and wv if not shared) are
    // all Q4_K and have GPU sessions. `v_out` is null for shared-V layers
    // (caller will @memcpy k into v).
    pub fn runLayerQKVQ8_1(
        self: *const GpuWeights,
        layer: usize,
        xb: []const f32,
        q_out: []f32,
        k_out: []f32,
        v_out: ?[]f32,
    ) !void {
        const lw   = &self.layers[layer];
        const wq   = lw.wq orelse return error.NotOnGpu;
        const wk   = lw.wk orelse return error.NotOnGpu;
        std.debug.assert(wq.cols % 256 == 0);
        std.debug.assert(wq.cols == wk.cols);

        const vec_buf  = &self.shared_vec.?;
        const acts_buf = &self.shared_acts_q8_1.?;
        const q_buf    = &(self.q_out_buf orelse return error.NotOnGpu);
        const k_buf    = &(self.k_out_buf orelse return error.NotOnGpu);

        try vec_buf.upload(std.mem.sliceAsBytes(xb));

        const cmd = try self.ctx.beginBatch();
        const quant_dset = try self.pl_quantize_q8_1.record(cmd, vec_buf, acts_buf, wq.cols);
        GpuCtx.recordShaderBarrier(cmd);

        const q_mv_dset = try self.pl_q4_k_q8_1.record(
            cmd, &wq.mat_buf, acts_buf, q_buf, wq.rows, wq.cols);
        const k_mv_dset = try self.pl_q4_k_q8_1.record(
            cmd, &wk.mat_buf, acts_buf, k_buf, wk.rows, wk.cols);

        var v_mv_dset: ?vk.VkDescriptorSet = null;
        if (v_out != null) {
            const wv    = lw.wv orelse return error.NotOnGpu;
            const v_buf = &(self.v_out_buf orelse return error.NotOnGpu);
            std.debug.assert(wv.cols == wq.cols);
            v_mv_dset = try self.pl_q4_k_q8_1.record(
                cmd, &wv.mat_buf, acts_buf, v_buf, wv.rows, wv.cols);
        }

        try self.ctx.submitBatch(cmd);

        const dev = self.ctx.device;
        _ = vk.vkFreeDescriptorSets(dev, self.pl_quantize_q8_1.desc_pool, 1, &quant_dset);
        _ = vk.vkFreeDescriptorSets(dev, self.pl_q4_k_q8_1.desc_pool, 1, &q_mv_dset);
        _ = vk.vkFreeDescriptorSets(dev, self.pl_q4_k_q8_1.desc_pool, 1, &k_mv_dset);
        if (v_mv_dset) |*ds| _ = vk.vkFreeDescriptorSets(dev, self.pl_q4_k_q8_1.desc_pool, 1, ds);

        try q_buf.download(std.mem.sliceAsBytes(q_out));
        try k_buf.download(std.mem.sliceAsBytes(k_out));
        if (v_out) |v| try self.v_out_buf.?.download(std.mem.sliceAsBytes(v));
    }

    // Dispatch w_gate and w_up in one command buffer (both read the same FFN-norm xb).
    pub fn runLayerGateUp(
        self: *const GpuWeights,
        layer: usize,
        gate_pl: *const MatvecPipeline,
        up_pl:   *const MatvecPipeline,
        xb: []const f32,
        gate_out: []f32,
        up_out:   []f32,
    ) !void {
        const lw      = &self.layers[layer];
        const w_gate  = lw.w_gate orelse return error.NotOnGpu;
        const w_up    = lw.w_up   orelse return error.NotOnGpu;
        const gate_buf = &(self.gate_out_buf orelse return error.NotOnGpu);
        const up_buf   = &(self.up_out_buf   orelse return error.NotOnGpu);

        try self.shared_vec.?.upload(std.mem.sliceAsBytes(xb));

        const cmd = try self.ctx.beginBatch();
        var gate_dset: vk.VkDescriptorSet = null;
        var up_dset:   vk.VkDescriptorSet = null;

        gate_dset = try w_gate.recordMv(cmd, gate_pl, &self.shared_vec.?, gate_buf);
        up_dset   = try w_up.recordMv(cmd, up_pl,   &self.shared_vec.?, up_buf);

        try self.ctx.submitBatch(cmd);

        _ = vk.vkFreeDescriptorSets(self.ctx.device, gate_pl.desc_pool, 1, &gate_dset);
        _ = vk.vkFreeDescriptorSets(self.ctx.device, up_pl.desc_pool,   1, &up_dset);

        try gate_buf.download(std.mem.sliceAsBytes(gate_out));
        try up_buf.download(std.mem.sliceAsBytes(up_out));
    }

    pub fn expertGate(self: *const GpuWeights, l: usize, e: usize) ?MatvecSession {
        const eg = self.expert_gate orelse return null;
        return eg[l * self.n_experts + e];
    }
    pub fn expertUp(self: *const GpuWeights, l: usize, e: usize) ?MatvecSession {
        const eu = self.expert_up orelse return null;
        return eu[l * self.n_experts + e];
    }
    pub fn expertDown(self: *const GpuWeights, l: usize, e: usize) ?MatvecSession {
        const ed = self.expert_down orelse return null;
        return ed[l * self.n_experts + e];
    }

    pub fn pipelineFor(self: *const GpuWeights, t: GgmlType) *const MatvecPipeline {
        return switch (t) {
            .f32  => &self.pl_f32,
            .q8_0 => &self.pl_q8_0,
            .q3_k => &self.pl_q3_k,
            .q4_k => &self.pl_q4_k,
            .q5_1 => &self.pl_q5_1,
            .q5_0 => &self.pl_q5_0,
            else  => unreachable,
        };
    }

    // Run all active experts for one layer in ONE GPU submission:
    //   Phase 1: n fused gate-gelu-up dispatches, write device-local mid_bufs
    //   Phase 2: n down dispatches, write sub-ranges of device-local expert_all_out_buf
    //   Phase 3: 1 accum dispatch, reads all_out + scales, writes HOST_COHERENT moe_gpu_buf
    // CPU writes moe_in and scales, then reads moe_gpu_buf after submit.
    // Returns error.ExpertNotOnGpu if any session is missing; caller falls back to CPU.
    pub fn runExpertBatch(
        self: *const GpuWeights,
        layer: usize,
        top_idx: []const usize,
        gate_up_type: GgmlType,
        down_type: GgmlType,
        down_exps_scale: []const f32,
        moe_in: []const f32,
        router_out: []const f32,
        moe_buf: []f32,
    ) !void {
        _ = gate_up_type; // fused pipeline is Q3_K-specific; gate_up_type must be q3_k
        const n = top_idx.len;
        const eg_sessions = self.expert_gate orelse return error.ExpertNotOnGpu;
        const eu_sessions = self.expert_up   orelse return error.ExpertNotOnGpu;
        const ed_sessions = self.expert_down orelse return error.ExpertNotOnGpu;
        const mid_bufs    = self.expert_mid_bufs     orelse return error.ExpertNotOnGpu;
        const all_out_buf = &(self.expert_all_out_buf orelse return error.ExpertNotOnGpu);
        const in_buf      = &(self.expert_in_buf      orelse return error.ExpertNotOnGpu);
        const scales_buf  = &(self.expert_scales_buf  orelse return error.ExpertNotOnGpu);
        const moe_out_buf = &(self.moe_gpu_buf        orelse return error.ExpertNotOnGpu);
        const in_slice     = self.expert_in_slice     orelse return error.ExpertNotOnGpu;
        const scales_slice = self.expert_scales_slice orelse return error.ExpertNotOnGpu;
        const moe_slice    = self.moe_gpu_slice       orelse return error.ExpertNotOnGpu;

        for (top_idx) |eidx| {
            if (eg_sessions[layer * self.n_experts + eidx] == null or
                eu_sessions[layer * self.n_experts + eidx] == null or
                ed_sessions[layer * self.n_experts + eidx] == null)
                return error.ExpertNotOnGpu;
        }

        const pl_dn = self.pipelineFor(down_type);

        // Write inputs and per-expert scales to HOST_COHERENT buffers.
        @memcpy(in_slice, moe_in);
        for (0..n) |k| {
            const eidx = top_idx[k];
            scales_slice[k] = down_exps_scale[eidx] * router_out[eidx];
        }
        // Zero the accumulation output — accum shader does +=.
        @memset(moe_slice, 0);

        const cmd = try self.ctx.beginBatch();
        var fused_dsets: [16]vk.VkDescriptorSet = undefined;
        var down_dsets:  [16]vk.VkDescriptorSet = undefined;

        // Phase 1: fused gate-gelu-up for each expert
        for (0..n) |k| {
            const sg = eg_sessions[layer * self.n_experts + top_idx[k]].?;
            const su = eu_sessions[layer * self.n_experts + top_idx[k]].?;
            fused_dsets[k] = try self.pl_fused_gu.record(
                cmd, &sg.mat_buf, &su.mat_buf, in_buf, &mid_bufs[k], sg.rows, sg.cols);
        }

        // Barrier: fused writes mid_bufs, down reads mid_bufs
        GpuCtx.recordShaderBarrier(cmd);

        // Phase 2: down matmul; each expert writes into its sub-range of expert_all_out_buf.
        const d_model: u64 = blk: {
            const sd0 = ed_sessions[layer * self.n_experts + top_idx[0]].?;
            break :blk sd0.rows;
        };
        for (0..n) |k| {
            const sd = ed_sessions[layer * self.n_experts + top_idx[k]].?;
            const out_off = k * d_model * @sizeOf(f32);
            const out_sz  = d_model * @sizeOf(f32);
            down_dsets[k] = try pl_dn.recordToRange(
                cmd, &sd.mat_buf, &mid_bufs[k],
                all_out_buf.handle, out_off, out_sz, sd.rows, sd.cols);
        }

        // Barrier: down writes all_out, accum reads all_out
        GpuCtx.recordShaderBarrier(cmd);

        // Phase 3: weighted accumulation on GPU
        const accum_dset = try self.pl_accum.record(
            cmd, all_out_buf, scales_buf, moe_out_buf, @intCast(d_model), @intCast(n));

        try self.ctx.submitBatch(cmd);

        for (fused_dsets[0..n]) |*ds|
            _ = vk.vkFreeDescriptorSets(self.ctx.device, self.pl_fused_gu.desc_pool, 1, ds);
        for (down_dsets[0..n]) |*ds|
            _ = vk.vkFreeDescriptorSets(self.ctx.device, pl_dn.desc_pool, 1, ds);
        var accum_ds = accum_dset;
        _ = vk.vkFreeDescriptorSets(self.ctx.device, self.pl_accum.desc_pool, 1, &accum_ds);

        // Add GPU-accumulated expert result into moe_buf.
        for (moe_buf, moe_slice) |*m, v| m.* += v;
    }
};

// --- module-level helpers ---

fn isGpuSupported(t: GgmlType) bool {
    return switch (t) {
        .f32, .q8_0, .q3_k, .q4_k, .q5_1, .q5_0 => true,
        else => false,
    };
}

// Schedule one matrix upload into an open command buffer.
// Commits the staging buffer to sbufs[nsb.*] and increments nsb.
// Returns null for unsupported quant types (→ CPU fallback).
fn schedUpload(
    ctx: *const GpuCtx,
    mat: RawMatrix,
    cmd: vk.VkCommandBuffer,
    sbufs: []GpuBuffer,
    nsb: *usize,
) !?MatvecSession {
    if (!isGpuSupported(mat.type_)) return null;

    var tmp: ?GpuBuffer = null;
    errdefer if (tmp) |*s| s.deinit();

    tmp = try GpuBuffer.initStaging(ctx, mat.data.len);
    try tmp.?.upload(mat.data);

    const sess = try MatvecSession.allocEmpty(
        ctx, mat.data.len,
        @intCast(mat.rows), @intCast(mat.cols));

    GpuCtx.recordCopy(cmd, tmp.?.handle, sess.mat_buf.handle, mat.data.len);

    sbufs[nsb.*] = tmp.?;
    tmp = null; // committed → cancel errdefer
    nsb.* += 1;

    return sess;
}

// Upload one layer's ≤7 matrices in a single command buffer submission.
fn uploadLayerBatch(ctx: *const GpuCtx, glayer: *GpuLayerWeights, lw: *const Gemma4LayerWeights) !void {
    var stagings: [7]GpuBuffer = undefined;
    var n_stagings: usize = 0;
    errdefer for (0..n_stagings) |i| stagings[i].deinit();

    const cmd = try ctx.beginBatchCopy();
    glayer.wq     = try schedUpload(ctx, lw.wq,     cmd, &stagings, &n_stagings);
    glayer.wk     = try schedUpload(ctx, lw.wk,     cmd, &stagings, &n_stagings);
    if (lw.wv) |wv|
        glayer.wv = try schedUpload(ctx, wv,          cmd, &stagings, &n_stagings);
    glayer.wo     = try schedUpload(ctx, lw.wo,     cmd, &stagings, &n_stagings);
    glayer.w_gate = try schedUpload(ctx, lw.w_gate, cmd, &stagings, &n_stagings);
    glayer.w_up   = try schedUpload(ctx, lw.w_up,   cmd, &stagings, &n_stagings);
    glayer.w_down = try schedUpload(ctx, lw.w_down, cmd, &stagings, &n_stagings);

    try ctx.submitBatchCopy(cmd);
    for (0..n_stagings) |i| stagings[i].deinit();
}

// Upload all experts for one layer in batches of EXPERT_BATCH per command buffer.
// Each batch creates at most EXPERT_BATCH×3 staging buffers (~47 MiB at batch=16).
const EXPERT_BATCH: usize = 16;

fn uploadExpertsBatch(
    ctx:      *const GpuCtx,
    gate_out: []?MatvecSession,
    up_out:   []?MatvecSession,
    down_out: []?MatvecSession,
    lw:       *const Gemma4LayerWeights,
    d_model:  usize,
    d_expert: usize,
) !void {
    const gu_row  = math.rowBytes(lw.gate_up_exps.type_, d_model);
    const gu_each = d_expert * gu_row;  // bytes for one expert's gate or up
    const dn_row  = math.rowBytes(lw.down_exps.type_, d_expert);
    const dn_each = d_model * dn_row;   // bytes for one expert's down

    var e: usize = 0;
    while (e < gate_out.len) : (e += EXPERT_BATCH) {
        const end = @min(e + EXPERT_BATCH, gate_out.len);
        var stagings: [EXPERT_BATCH * 3]GpuBuffer = undefined;
        var n_stagings: usize = 0;
        errdefer for (0..n_stagings) |i| stagings[i].deinit();

        const cmd = try ctx.beginBatchCopy();
        for (e..end) |eidx| {
            gate_out[eidx] = try schedUpload(ctx,
                .{ .data = lw.gate_up_exps.data[eidx * 2 * gu_each..][0..gu_each],
                   .type_ = lw.gate_up_exps.type_, .rows = d_expert, .cols = d_model },
                cmd, &stagings, &n_stagings);
            up_out[eidx] = try schedUpload(ctx,
                .{ .data = lw.gate_up_exps.data[eidx * 2 * gu_each + gu_each..][0..gu_each],
                   .type_ = lw.gate_up_exps.type_, .rows = d_expert, .cols = d_model },
                cmd, &stagings, &n_stagings);
            down_out[eidx] = try schedUpload(ctx,
                .{ .data = lw.down_exps.data[eidx * dn_each..][0..dn_each],
                   .type_ = lw.down_exps.type_, .rows = d_model, .cols = d_expert },
                cmd, &stagings, &n_stagings);
        }
        try ctx.submitBatchCopy(cmd);
        for (0..n_stagings) |i| stagings[i].deinit();
    }
}

// Upload a single matrix in its own command buffer submission.
fn uploadSingleBatch(ctx: *const GpuCtx, mat: RawMatrix) !?MatvecSession {
    var stagings: [1]GpuBuffer = undefined;
    var n_stagings: usize = 0;
    errdefer for (0..n_stagings) |i| stagings[i].deinit();

    const cmd = try ctx.beginBatchCopy();
    const sess = try schedUpload(ctx, mat, cmd, &stagings, &n_stagings);
    try ctx.submitBatchCopy(cmd);
    for (0..n_stagings) |i| stagings[i].deinit();
    return sess;
}

fn readSysU64(path: []const u8) u64 {
    const fd = std.posix.openat(std.posix.AT.FDCWD, path, .{}, 0) catch return 0;
    defer _ = std.os.linux.close(fd);
    var buf: [32]u8 = undefined;
    const n = std.posix.read(fd, &buf) catch return 0;
    const s = std.mem.trim(u8, buf[0..n], " \t\n");
    return std.fmt.parseInt(u64, s, 10) catch 0;
}

fn vramUsedMB() u64 {
    return readSysU64("/sys/class/drm/card1/device/mem_info_vram_used") / (1024 * 1024);
}

fn gttUsedMB() u64 {
    return readSysU64("/sys/class/drm/card1/device/mem_info_gtt_used") / (1024 * 1024);
}

// Read MemAvailable from /proc/meminfo; returns maxInt on any failure.
fn availableMemoryMB() u64 {
    const fd = std.posix.openat(std.posix.AT.FDCWD, "/proc/meminfo", .{}, 0) catch return std.math.maxInt(u64);
    defer _ = std.os.linux.close(fd);
    var buf: [4096]u8 = undefined;
    const n = std.posix.read(fd, &buf) catch return std.math.maxInt(u64);
    const text = buf[0..n];
    const prefix = "MemAvailable:";
    const pos = std.mem.indexOf(u8, text, prefix) orelse return std.math.maxInt(u64);
    const after = std.mem.trim(u8, text[pos + prefix.len..][0..@min(32, text.len - pos - prefix.len)], " \t\n");
    const end = std.mem.indexOfAny(u8, after, " \t\n") orelse after.len;
    const kb = std.fmt.parseInt(u64, after[0..end], 10) catch return std.math.maxInt(u64);
    return kb / 1024;
}
