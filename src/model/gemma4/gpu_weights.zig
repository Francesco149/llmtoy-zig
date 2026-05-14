/// GPU-resident weight sessions for Gemma4.
///
/// Uploads per-layer attention and dense-FFN weight matrices to VRAM one
/// layer at a time: each layer's ≤7 matrices go into a single command buffer
/// → single vkQueueSubmit → free staging.  Peak HOST_COHERENT staging is
/// ~22 MiB (one layer) rather than ~1 GiB (all layers at once).
///
/// lm_head is uploaded separately and falls back to CPU if the allocation
/// fails (vocab_size=262144 can make it 400+ MiB of staging).
///
/// MoE expert tensors stay on CPU: too large to pre-upload (~20 GB across
/// all layers) and accessed sparsely (8 of 128 experts per token).
///
/// Matrices whose quant type has no GPU shader (Q5_K, IQ4_NL, …) get a null
/// session and fall back to the CPU path transparently.

const std        = @import("std");
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
    layers:     []GpuLayerWeights,
    lm_head:    ?MatvecSession,
    // Shared host-coherent I/O buffers sized to the largest matrix across all
    // sessions. Eliminates 420+ individual per-session HOST_COHERENT allocations.
    shared_vec: ?GpuBuffer,
    shared_out: ?GpuBuffer,
    allocator:  std.mem.Allocator,

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

        const layers = try allocator.alloc(GpuLayerWeights, g4cfg.n_layers);
        errdefer allocator.free(layers);
        for (layers) |*l| l.* = .{
            .wq = null, .wk = null, .wv = null, .wo = null,
            .w_gate = null, .w_up = null, .w_down = null,
        };

        var gw = GpuWeights{
            .ctx = ctx, .pl_f32 = pl_f32, .pl_q8_0 = pl_q8_0,
            .pl_q3_k = pl_q3_k, .pl_q4_k = pl_q4_k, .pl_q5_1 = pl_q5_1,
            .layers = layers, .lm_head = null,
            .shared_vec = null, .shared_out = null,
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

        return gw;
    }

    pub fn deinit(self: *GpuWeights) void {
        if (self.shared_out) |*b| b.deinit();
        if (self.shared_vec) |*b| b.deinit();
        if (self.lm_head) |*s| s.deinit();
        for (self.layers) |*l| l.deinitAll();
        self.allocator.free(self.layers);
        self.pl_q5_1.deinit();
        self.pl_q4_k.deinit();
        self.pl_q3_k.deinit();
        self.pl_q8_0.deinit();
        self.pl_f32.deinit();
        self.ctx.deinit();
    }

    pub fn pipelineFor(self: *const GpuWeights, t: GgmlType) *const MatvecPipeline {
        return switch (t) {
            .f32  => &self.pl_f32,
            .q8_0 => &self.pl_q8_0,
            .q3_k => &self.pl_q3_k,
            .q4_k => &self.pl_q4_k,
            .q5_1 => &self.pl_q5_1,
            else  => unreachable,
        };
    }
};

// --- module-level helpers ---

fn isGpuSupported(t: GgmlType) bool {
    return switch (t) {
        .f32, .q8_0, .q3_k, .q4_k, .q5_1 => true,
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
