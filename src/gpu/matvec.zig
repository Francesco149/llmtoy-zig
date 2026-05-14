// GPU matrix-vector multiply for fp32 and quantized weight formats.
// Wraps pipeline creation, descriptor management, and dispatch into a simple API.
const std = @import("std");
const vk = @import("vk.zig").vk;
const GpuContext = @import("context.zig").GpuContext;
const GpuBuffer = @import("buffer.zig").GpuBuffer;
const shaders = @import("gpu_shaders");

const WORKGROUP_SIZE: u32 = 64;

// Push-constant layout must match the GLSL shaders.
const PushConst = extern struct { rows: u32, cols: u32 };

// ── MatvecPipeline ────────────────────────────────────────────────────────────

// Compiled compute pipeline for one matvec shader variant.
// Create one per quant format; reuse across all calls with that format.
pub const MatvecPipeline = struct {
    pipeline: vk.VkPipeline,
    layout: vk.VkPipelineLayout,
    dset_layout: vk.VkDescriptorSetLayout,
    desc_pool: vk.VkDescriptorPool,
    device: vk.VkDevice,

    pub fn initF32(ctx: *const GpuContext) !MatvecPipeline {
        // SPIR-V size must be a multiple of 4 words — checked at comptime.
        comptime std.debug.assert(shaders.matvec_f32.len % 4 == 0);
        return initFromSpv(ctx, &shaders.matvec_f32);
    }

    pub fn initQ8_0(ctx: *const GpuContext) !MatvecPipeline {
        comptime std.debug.assert(shaders.matvec_q8_0.len % 4 == 0);
        return initFromSpv(ctx, &shaders.matvec_q8_0);
    }

    pub fn initQ3K(ctx: *const GpuContext) !MatvecPipeline {
        comptime std.debug.assert(shaders.matvec_q3_k.len % 4 == 0);
        return initFromSpv(ctx, &shaders.matvec_q3_k);
    }

    fn initFromSpv(ctx: *const GpuContext, spv: anytype) !MatvecPipeline {
        // align(4) on the const in shaders.zig should guarantee this, but
        // assert the actual runtime address in case @embedFile doesn't honour it.
        std.debug.assert(@intFromPtr(spv) % 4 == 0);
        // Redundant with the comptime len check at each call site, but belt-and-suspenders.
        std.debug.assert(spv.len % 4 == 0);

        const dev = ctx.device;

        // Descriptor set layout: bindings 0/1/2 are storage buffers (mat, vec, out)
        const bindings = [3]vk.VkDescriptorSetLayoutBinding{
            mkStorageBuf(0), mkStorageBuf(1), mkStorageBuf(2),
        };
        const dsl_ci = vk.VkDescriptorSetLayoutCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .bindingCount = bindings.len,
            .pBindings = &bindings,
        };
        var dset_layout: vk.VkDescriptorSetLayout = null;
        if (vk.vkCreateDescriptorSetLayout(dev, &dsl_ci, null, &dset_layout) != vk.VK_SUCCESS)
            return error.VkDescSetLayoutFailed;
        errdefer vk.vkDestroyDescriptorSetLayout(dev, dset_layout, null);

        const pc_range = vk.VkPushConstantRange{
            .stageFlags = vk.VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = @sizeOf(PushConst),
        };
        const layout_ci = vk.VkPipelineLayoutCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .setLayoutCount = 1,
            .pSetLayouts = &dset_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pc_range,
        };
        var layout: vk.VkPipelineLayout = null;
        if (vk.vkCreatePipelineLayout(dev, &layout_ci, null, &layout) != vk.VK_SUCCESS)
            return error.VkPipelineLayoutFailed;
        errdefer vk.vkDestroyPipelineLayout(dev, layout, null);

        const shader_ci = vk.VkShaderModuleCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .codeSize = spv.len,
            .pCode = @ptrCast(spv),
        };
        var shader_mod: vk.VkShaderModule = null;
        if (vk.vkCreateShaderModule(dev, &shader_ci, null, &shader_mod) != vk.VK_SUCCESS)
            return error.VkShaderModuleFailed;
        defer vk.vkDestroyShaderModule(dev, shader_mod, null);

        const stage = vk.VkPipelineShaderStageCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .stage = vk.VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader_mod,
            .pName = "main",
            .pSpecializationInfo = null,
        };
        const pipeline_ci = vk.VkComputePipelineCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .stage = stage,
            .layout = layout,
            .basePipelineHandle = null,
            .basePipelineIndex = -1,
        };
        var pipeline: vk.VkPipeline = null;
        if (vk.vkCreateComputePipelines(dev, null, 1, &pipeline_ci, null, &pipeline) != vk.VK_SUCCESS)
            return error.VkComputePipelineFailed;
        errdefer vk.vkDestroyPipeline(dev, pipeline, null);

        const pool_size = vk.VkDescriptorPoolSize{
            .type = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 3 * 8,
        };
        const pool_ci = vk.VkDescriptorPoolCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = null,
            .flags = vk.VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            .maxSets = 8,
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size,
        };
        var desc_pool: vk.VkDescriptorPool = null;
        if (vk.vkCreateDescriptorPool(dev, &pool_ci, null, &desc_pool) != vk.VK_SUCCESS)
            return error.VkDescriptorPoolFailed;

        return .{
            .pipeline = pipeline,
            .layout = layout,
            .dset_layout = dset_layout,
            .desc_pool = desc_pool,
            .device = dev,
        };
    }

    pub fn deinit(self: *MatvecPipeline) void {
        vk.vkDestroyDescriptorPool(self.device, self.desc_pool, null);
        vk.vkDestroyPipeline(self.device, self.pipeline, null);
        vk.vkDestroyPipelineLayout(self.device, self.layout, null);
        vk.vkDestroyDescriptorSetLayout(self.device, self.dset_layout, null);
    }

    // Submit one matvec dispatch and block until complete.
    // mat_buf, vec_buf, out_buf must already be bound to the right data.
    pub fn run(
        self: *const MatvecPipeline,
        ctx: *const GpuContext,
        mat_buf: *const GpuBuffer,
        vec_buf: *const GpuBuffer,
        out_buf: *const GpuBuffer,
        rows: u32,
        cols: u32,
    ) !void {
        const dev = ctx.device;

        const alloc_ci = vk.VkDescriptorSetAllocateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = null,
            .descriptorPool = self.desc_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &self.dset_layout,
        };
        var dset: vk.VkDescriptorSet = null;
        if (vk.vkAllocateDescriptorSets(dev, &alloc_ci, &dset) != vk.VK_SUCCESS)
            return error.VkDescriptorSetAllocFailed;
        defer _ = vk.vkFreeDescriptorSets(dev, self.desc_pool, 1, &dset);

        const buf_infos = [3]vk.VkDescriptorBufferInfo{
            .{ .buffer = mat_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = vec_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = out_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        };
        const writes = [3]vk.VkWriteDescriptorSet{
            mkWrite(dset, 0, &buf_infos[0]),
            mkWrite(dset, 1, &buf_infos[1]),
            mkWrite(dset, 2, &buf_infos[2]),
        };
        vk.vkUpdateDescriptorSets(dev, writes.len, &writes, 0, null);

        const cmd_alloc = vk.VkCommandBufferAllocateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = null,
            .commandPool = ctx.cmd_pool,
            .level = vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        var cmd: vk.VkCommandBuffer = undefined;
        if (vk.vkAllocateCommandBuffers(dev, &cmd_alloc, &cmd) != vk.VK_SUCCESS)
            return error.VkCommandBufferAllocFailed;
        defer vk.vkFreeCommandBuffers(dev, ctx.cmd_pool, 1, &cmd);

        const begin_ci = vk.VkCommandBufferBeginInfo{
            .sType = vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = null,
            .flags = vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = null,
        };
        _ = vk.vkBeginCommandBuffer(cmd, &begin_ci);

        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.layout, 0, 1, &dset, 0, null);

        const pc = PushConst{ .rows = rows, .cols = cols };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT,
            0, @sizeOf(PushConst), &pc);

        const groups = (rows + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        vk.vkCmdDispatch(cmd, groups, 1, 1);

        _ = vk.vkEndCommandBuffer(cmd);

        const submit = vk.VkSubmitInfo{
            .sType = vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = null,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = null,
            .pWaitDstStageMask = null,
            .commandBufferCount = 1,
            .pCommandBuffers = &cmd,
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = null,
        };
        if (vk.vkQueueSubmit(ctx.queue, 1, &submit, null) != vk.VK_SUCCESS)
            return error.VkQueueSubmitFailed;
        _ = vk.vkQueueWaitIdle(ctx.queue);
    }
};

// ── MatvecSession ─────────────────────────────────────────────────────────────

// Pre-loaded weight matrix that stays resident in VRAM across token steps.
// Upload the matrix once at model-load time, then call run() for every token.
// The quant format is baked into the MatvecPipeline passed to run().
pub const MatvecSession = struct {
    mat_buf: GpuBuffer, // device-local VRAM: holds raw matrix bytes (any format)
    vec_buf: GpuBuffer, // host-coherent: fp32 activation input, updated each token
    out_buf: GpuBuffer, // host-coherent: fp32 output, read each token
    rows: u32,
    cols: u32,

    // Upload an fp32 row-major matrix to VRAM.
    pub fn init(ctx: *const GpuContext, mat: []const f32, rows: u32, cols: u32) !MatvecSession {
        return initBytes(ctx, std.mem.sliceAsBytes(mat), rows, cols);
    }

    // Upload a Q8_0 quantized matrix (raw GGUF bytes) to VRAM.
    // mat_bytes must be exactly rows * (cols/32) * 34 bytes.
    pub fn initQ8_0(ctx: *const GpuContext, mat_bytes: []const u8, rows: u32, cols: u32) !MatvecSession {
        std.debug.assert(cols % 32 == 0);
        std.debug.assert(mat_bytes.len == rows * (cols / 32) * 34);
        return initBytes(ctx, mat_bytes, rows, cols);
    }

    // Upload a Q3_K quantized matrix (raw GGUF bytes) to VRAM.
    // mat_bytes must be exactly rows * (cols/256) * 110 bytes.
    pub fn initQ3K(ctx: *const GpuContext, mat_bytes: []const u8, rows: u32, cols: u32) !MatvecSession {
        std.debug.assert(cols % 256 == 0);
        std.debug.assert(mat_bytes.len == rows * (cols / 256) * 110);
        return initBytes(ctx, mat_bytes, rows, cols);
    }

    fn initBytes(ctx: *const GpuContext, mat_bytes: []const u8, rows: u32, cols: u32) !MatvecSession {
        var staging = try GpuBuffer.initStaging(ctx, mat_bytes.len);
        defer staging.deinit();
        try staging.upload(mat_bytes);

        var mat_buf = try GpuBuffer.initDeviceLocal(ctx, mat_bytes.len,
            vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        errdefer mat_buf.deinit();
        try ctx.copyBuffer(staging.handle, mat_buf.handle, mat_bytes.len);

        var vec_buf = try GpuBuffer.initHostCoherent(ctx, cols * @sizeOf(f32),
            vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        errdefer vec_buf.deinit();

        var out_buf = try GpuBuffer.initHostCoherent(ctx, rows * @sizeOf(f32),
            vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        errdefer out_buf.deinit();

        return .{
            .mat_buf = mat_buf,
            .vec_buf = vec_buf,
            .out_buf = out_buf,
            .rows = rows,
            .cols = cols,
        };
    }

    pub fn deinit(self: *MatvecSession) void {
        self.out_buf.deinit();
        self.vec_buf.deinit();
        self.mat_buf.deinit();
    }

    pub fn run(
        self: *const MatvecSession,
        ctx: *const GpuContext,
        pipeline: *const MatvecPipeline,
        vec: []const f32,
        out: []f32,
    ) !void {
        try self.vec_buf.upload(std.mem.sliceAsBytes(vec));
        try pipeline.run(ctx, &self.mat_buf, &self.vec_buf, &self.out_buf, self.rows, self.cols);
        try self.out_buf.download(std.mem.sliceAsBytes(out));
    }
};

// ── convenience functions ─────────────────────────────────────────────────────

// One-shot fp32 matvec: upload matrix to VRAM, run, download result.
// For repeated calls on the same matrix use MatvecSession instead.
pub fn matvecF32(
    ctx: *const GpuContext,
    pipeline: *const MatvecPipeline,
    mat: []const f32,
    vec: []const f32,
    out: []f32,
    rows: u32,
    cols: u32,
) !void {
    var session = try MatvecSession.init(ctx, mat, rows, cols);
    defer session.deinit();
    try session.run(ctx, pipeline, vec, out);
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn mkStorageBuf(binding: u32) vk.VkDescriptorSetLayoutBinding {
    return .{
        .binding = binding,
        .descriptorType = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = vk.VK_SHADER_STAGE_COMPUTE_BIT,
        .pImmutableSamplers = null,
    };
}

fn mkWrite(dset: vk.VkDescriptorSet, binding: u32, info: *const vk.VkDescriptorBufferInfo) vk.VkWriteDescriptorSet {
    return .{
        .sType = vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = null,
        .dstSet = dset,
        .dstBinding = binding,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pImageInfo = null,
        .pBufferInfo = info,
        .pTexelBufferView = null,
    };
}

// ── tests ─────────────────────────────────────────────────────────────────────

test "gpu matvec f32 correctness" {
    const ctx = GpuContext.init() catch |e| {
        std.debug.print("gpu init failed: {}\n", .{e});
        return;
    };
    var gpu = ctx;
    defer gpu.deinit();

    var pipeline = try MatvecPipeline.initF32(&gpu);
    defer pipeline.deinit();

    // 4×4 identity matrix times [1,2,3,4] = [1,2,3,4]
    const rows: u32 = 4;
    const cols: u32 = 4;
    const mat = [16]f32{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };
    const vec = [4]f32{ 1, 2, 3, 4 };
    var out = [4]f32{ 0, 0, 0, 0 };

    try matvecF32(&gpu, &pipeline, &mat, &vec, &out, rows, cols);

    try std.testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
    try std.testing.expectApproxEqAbs(out[1], 2.0, 1e-5);
    try std.testing.expectApproxEqAbs(out[2], 3.0, 1e-5);
    try std.testing.expectApproxEqAbs(out[3], 4.0, 1e-5);
}

test "gpu matvec Q8_0 correctness" {
    const ctx = GpuContext.init() catch |e| {
        std.debug.print("gpu init failed: {}\n", .{e});
        return;
    };
    var gpu = ctx;
    defer gpu.deinit();

    var pipeline = try MatvecPipeline.initQ8_0(&gpu);
    defer pipeline.deinit();

    // 32 rows × 32 cols — one Q8_0 block per row (n_blocks = 1).
    // Row i: scale = 1.0 (f16 = 0x3C00, LE bytes [0x00, 0x3C]),
    //        quants all zero except qs[i] = 1.
    // Input vec = [1, 2, ..., 32].
    // Expected output[i] = 1.0 * 1 * vec[i] = i + 1.
    const rows: u32 = 32;
    const cols: u32 = 32;
    const block_bytes = 34; // 2 (f16) + 32 (i8)

    var mat_bytes: [rows * block_bytes]u8 = [_]u8{0} ** (rows * block_bytes);
    for (0..rows) |row| {
        const b = row * block_bytes;
        mat_bytes[b + 0] = 0x00; // f16 1.0 low byte
        mat_bytes[b + 1] = 0x3C; // f16 1.0 high byte
        mat_bytes[b + 2 + row] = 1; // qs[row] = 1, rest = 0
    }

    var vec: [cols]f32 = undefined;
    for (0..cols) |i| vec[i] = @floatFromInt(i + 1);
    var out: [rows]f32 = [_]f32{0} ** rows;

    var session = try MatvecSession.initQ8_0(&gpu, &mat_bytes, rows, cols);
    defer session.deinit();
    try session.run(&gpu, &pipeline, &vec, &out);

    for (0..rows) |i| {
        try std.testing.expectApproxEqAbs(out[i], @as(f32, @floatFromInt(i + 1)), 1e-4);
    }
}

test "gpu matvec Q3_K positive q3" {
    // Verifies correct scale unpacking and element ordering for a positive quant value.
    //
    // Block (1 row × 256 cols, 110 bytes):
    //   hmask[0..31] = 0xFF  →  hi_bit = 1 for every element
    //   qs[0]        = 0x01  →  lo2 = 1 for element 0 (shift=0); lo2=0 for e=32,64,96
    //   qs[1..63]    = 0x00  →  lo2 = 0 for all other elements
    //   scales[0..3] = [0x01, 0x00, 0x00, 0x00]
    //   scales[4..7] = [0x00, 0x00, 0x00, 0x00]
    //   scales[8..11]= [0xAA, 0xAA, 0xAA, 0xAA]
    //   → sc[0] = 33, sc[1..15] = 32
    //   → scale for group 0 (elements 0..15) = d_all*(33-32) = 1.0
    //   → scale for all other groups = d_all*(32-32) = 0.0
    //   d = f16(1.0) = [0x00, 0x3C]
    //
    // Element 0: hi=1, lo2=1 → q3=1, scale=1.0, vec[0]=1.0  → contributes 1.0
    // All other elements: q3=0 or scale=0                     → contribute 0.0
    // Expected output: 1.0
    const ctx = GpuContext.init() catch |e| {
        std.debug.print("gpu init failed: {}\n", .{e});
        return;
    };
    var gpu = ctx;
    defer gpu.deinit();

    var pipeline = try MatvecPipeline.initQ3K(&gpu);
    defer pipeline.deinit();

    const blk_size = 110;
    var mat_bytes: [blk_size]u8 = [_]u8{0} ** blk_size;
    @memset(mat_bytes[0..32], 0xFF);    // hmask: all hi bits set
    mat_bytes[32] = 0x01;              // qs[0]: lo2=1 at shift=0 for element 0
    // scales: sc[0]=33, sc[1..15]=32
    mat_bytes[96] = 0x01; mat_bytes[97] = 0x00; mat_bytes[98] = 0x00; mat_bytes[99] = 0x00;
    mat_bytes[100] = 0x00; mat_bytes[101] = 0x00; mat_bytes[102] = 0x00; mat_bytes[103] = 0x00;
    mat_bytes[104] = 0xAA; mat_bytes[105] = 0xAA; mat_bytes[106] = 0xAA; mat_bytes[107] = 0xAA;
    mat_bytes[108] = 0x00; mat_bytes[109] = 0x3C; // d = f16(1.0)

    var vec: [256]f32 = [_]f32{1.0} ** 256;
    var out: [1]f32 = .{0.0};

    var session = try MatvecSession.initQ3K(&gpu, &mat_bytes, 1, 256);
    defer session.deinit();
    try session.run(&gpu, &pipeline, &vec, &out);

    try std.testing.expectApproxEqAbs(out[0], 1.0, 1e-4);
}

test "gpu matvec Q3_K negative q3" {
    // All hi_bits = 0, all lo2 = 0  →  q3 = 0 - 4 = -4 for every element.
    // All 16 scales = 33  →  each sub-block scale = d_all*(33-32) = 1.0.
    // vec = all 1.0,  256 elements.
    // Expected: 1.0 * (-4) * 256 = -1024.0.
    //
    // Scale encoding for all sc[i]=33:
    //   sc01 = sc23 = 0x11111111, tmp = 0xAAAAAAAA
    //   → scales[0..3]  = [0x11,0x11,0x11,0x11]
    //   → scales[4..7]  = [0x11,0x11,0x11,0x11]
    //   → scales[8..11] = [0xAA,0xAA,0xAA,0xAA]
    const ctx = GpuContext.init() catch |e| {
        std.debug.print("gpu init failed: {}\n", .{e});
        return;
    };
    var gpu = ctx;
    defer gpu.deinit();

    var pipeline = try MatvecPipeline.initQ3K(&gpu);
    defer pipeline.deinit();

    const blk_size = 110;
    var mat_bytes: [blk_size]u8 = [_]u8{0} ** blk_size;
    // hmask = 0, qs = 0  →  hi_bit=0, lo2=0  →  q3 = -4 for all elements
    // scales: all sc[i] = 33
    mat_bytes[96] = 0x11; mat_bytes[97] = 0x11; mat_bytes[98] = 0x11; mat_bytes[99] = 0x11;
    mat_bytes[100] = 0x11; mat_bytes[101] = 0x11; mat_bytes[102] = 0x11; mat_bytes[103] = 0x11;
    mat_bytes[104] = 0xAA; mat_bytes[105] = 0xAA; mat_bytes[106] = 0xAA; mat_bytes[107] = 0xAA;
    mat_bytes[108] = 0x00; mat_bytes[109] = 0x3C; // d = f16(1.0)

    var vec: [256]f32 = [_]f32{1.0} ** 256;
    var out: [1]f32 = .{0.0};

    var session = try MatvecSession.initQ3K(&gpu, &mat_bytes, 1, 256);
    defer session.deinit();
    try session.run(&gpu, &pipeline, &vec, &out);

    try std.testing.expectApproxEqAbs(out[0], -1024.0, 0.1);
}
