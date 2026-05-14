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
const MatvecPipeline     = mv_mod.MatvecPipeline;
const MatvecSession      = mv_mod.MatvecSession;
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
    pl_q5_1:    MatvecPipeline,
    pl_q5_0:    MatvecPipeline,
    layers:     []GpuLayerWeights,
    lm_head:    ?MatvecSession,
    // Shared host-coherent I/O buffers sized to the largest matrix across all
    // sessions. Eliminates 420+ individual per-session HOST_COHERENT allocations.
    shared_vec:  ?GpuBuffer,
    shared_out:  ?GpuBuffer,
    // Flat [n_layers × n_experts] expert session arrays; null = CPU fallback.
    expert_gate: ?[]?MatvecSession,
    expert_up:   ?[]?MatvecSession,
    expert_down: ?[]?MatvecSession,
    n_experts:   usize,
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
        var pl_q5_1 = try MatvecPipeline.initQ5_1(&ctx);
        errdefer pl_q5_1.deinit();
        std.debug.print("  init: pl_q5_1 ok (VRAM={} MiB GTT={} MiB sys={} MiB)\n",
            .{ vramUsedMB(), gttUsedMB(), availableMemoryMB() });
        var pl_q5_0 = try MatvecPipeline.initQ5_0(&ctx);
        errdefer pl_q5_0.deinit();
        std.debug.print("  init: pl_q5_0 ok (VRAM={} MiB GTT={} MiB sys={} MiB)\n",
            .{ vramUsedMB(), gttUsedMB(), availableMemoryMB() });

        const layers = try allocator.alloc(GpuLayerWeights, g4cfg.n_layers);
        errdefer allocator.free(layers);
        for (layers) |*l| l.* = .{
            .wq = null, .wk = null, .wv = null, .wo = null,
            .w_gate = null, .w_up = null, .w_down = null,
        };

        var gw = GpuWeights{
            .ctx = ctx, .pl_f32 = pl_f32, .pl_q8_0 = pl_q8_0,
            .pl_q3_k = pl_q3_k, .pl_q4_k = pl_q4_k,
            .pl_q5_1 = pl_q5_1, .pl_q5_0 = pl_q5_0,
            .layers = layers, .lm_head = null,
            .shared_vec = null, .shared_out = null,
            .expert_gate = null, .expert_up = null, .expert_down = null,
            .n_experts = g4cfg.n_experts,
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
        std.debug.print("  shared I/O bufs: vec={} KiB out={} KiB\n",
            .{ max_cols * @sizeOf(f32) / 1024, max_rows * @sizeOf(f32) / 1024 });

        // MoE experts intentionally NOT uploaded to VRAM.
        // Measured: per-expert GPU dispatch (720 vkQueueSubmit/token at 8 experts×3
        // matmuls×30 layers) costs more than it saves — 1.52 tok/s vs 2.44 tok/s.
        // expert_gate/up/down stay null → expertGate/Up/Down return null → CPU fallback.
        // TODO: batched dispatch (all 8 active experts in one command buffer per layer).

        return gw;
    }

    pub fn deinit(self: *GpuWeights) void {
        if (self.expert_down) |ed| { for (ed) |*ms| if (ms.*) |*s| s.deinit(); self.allocator.free(ed); }
        if (self.expert_up)   |eu| { for (eu) |*ms| if (ms.*) |*s| s.deinit(); self.allocator.free(eu); }
        if (self.expert_gate) |eg| { for (eg) |*ms| if (ms.*) |*s| s.deinit(); self.allocator.free(eg); }
        if (self.shared_out) |*b| b.deinit();
        if (self.shared_vec) |*b| b.deinit();
        if (self.lm_head) |*s| s.deinit();
        for (self.layers) |*l| l.deinitAll();
        self.allocator.free(self.layers);
        self.pl_q5_0.deinit();
        self.pl_q5_1.deinit();
        self.pl_q4_k.deinit();
        self.pl_q3_k.deinit();
        self.pl_q8_0.deinit();
        self.pl_f32.deinit();
        self.ctx.deinit();
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
