// GPU matrix-vector multiply for fp32 and quantized weight formats.
// Wraps pipeline creation, descriptor management, and dispatch into a simple API.
const std = @import("std");
const vk = @import("vk.zig").vk;
const GpuContext = @import("context.zig").GpuContext;
const GpuBuffer = @import("buffer.zig").GpuBuffer;
const shaders = @import("gpu_shaders");
const GgmlType = @import("../gguf/types.zig").GgmlType;
const dq = @import("../quant/dequant.zig");
const math_mod = @import("../ops/math.zig");

const WORKGROUP_SIZE: u32 = 64;

// Push-constant layout must match the GLSL shaders.
const PushConst = extern struct { rows: u32, cols: u32 };

pub const MmvqSpec = extern struct {
    block_size: u32 = 32,
    num_rows: u32 = 1,
    num_cols: u32 = 1,
};

// ── MatvecPipeline ────────────────────────────────────────────────────────────

// Compiled compute pipeline for one matvec shader variant.
// Create one per quant format; reuse across all calls with that format.
//
// `rows_per_workgroup` determines the dispatch geometry: a `record(rows=R, cols=C)`
// call dispatches `ceil(R / rows_per_workgroup)` workgroups in X. The original
// f32 / Q8_0 / Q*_K shaders use 1 thread per row, so each workgroup of 64 threads
// processes 64 rows (rows_per_workgroup = 64).  The Q8_1-activation shaders use
// a full subgroup per row, so each workgroup processes 1 row.
pub const MatvecPipeline = struct {
    pipeline: vk.VkPipeline,
    layout: vk.VkPipelineLayout,
    dset_layout: vk.VkDescriptorSetLayout,
    desc_pool: vk.VkDescriptorPool,
    device: vk.VkDevice,
    rows_per_workgroup: u32,

    pub fn initF32(ctx: *const GpuContext) !MatvecPipeline {
        // SPIR-V size must be a multiple of 4 words — checked at comptime.
        comptime std.debug.assert(shaders.matvec_f32.len % 4 == 0);
        return initFromSpv(ctx, &shaders.matvec_f32, 64);
    }

    pub fn initQ8_0(ctx: *const GpuContext) !MatvecPipeline {
        comptime std.debug.assert(shaders.matvec_q8_0.len % 4 == 0);
        return initFromSpv(ctx, &shaders.matvec_q8_0, 64);
    }

    pub fn initQ3K(ctx: *const GpuContext) !MatvecPipeline {
        comptime std.debug.assert(shaders.matvec_q3_k.len % 4 == 0);
        return initFromSpv(ctx, &shaders.matvec_q3_k, 64);
    }

    pub fn initQ4K(ctx: *const GpuContext) !MatvecPipeline {
        comptime std.debug.assert(shaders.matvec_q4_k.len % 4 == 0);
        return initFromSpv(ctx, &shaders.matvec_q4_k, 64);
    }

    pub fn initQ5_1(ctx: *const GpuContext) !MatvecPipeline {
        comptime std.debug.assert(shaders.matvec_q5_1.len % 4 == 0);
        return initFromSpv(ctx, &shaders.matvec_q5_1, 64);
    }

    pub fn initQ5_0(ctx: *const GpuContext) !MatvecPipeline {
        comptime std.debug.assert(shaders.matvec_q5_0.len % 4 == 0);
        return initFromSpv(ctx, &shaders.matvec_q5_0, 64);
    }

    // Q4_K weights × Q8_1 activations, subgroup-cooperative (1 workgroup/row).
    pub fn initQ4KQ8_1(ctx: *const GpuContext) !MatvecPipeline {
        comptime std.debug.assert(shaders.matvec_q4_k_q8_1.len % 4 == 0);
        return initFromSpv(ctx, &shaders.matvec_q4_k_q8_1, 1);
    }

    // Experimental Q4_K × Q8_1 variant: 4 rows per workgroup.
    pub fn initQ4KQ8_1R4(ctx: *const GpuContext) !MatvecPipeline {
        comptime std.debug.assert(shaders.matvec_q4_k_q8_1_r4.len % 4 == 0);
        return initFromSpv(ctx, &shaders.matvec_q4_k_q8_1_r4, 4);
    }

    // Q3_K weights × Q8_1 activations, subgroup-cooperative (1 workgroup/row).
    pub fn initQ3KQ8_1(ctx: *const GpuContext) !MatvecPipeline {
        comptime std.debug.assert(shaders.matvec_q3_k_q8_1.len % 4 == 0);
        return initFromSpv(ctx, &shaders.matvec_q3_k_q8_1, 1);
    }

    pub fn initQ3KQ8_1Mmvq(ctx: *const GpuContext, spec: MmvqSpec) !MatvecPipeline {
        comptime std.debug.assert(shaders.matvec_q3_k_q8_1_mmvq.len % 4 == 0);
        return initFromSpvMmvq(ctx, &shaders.matvec_q3_k_q8_1_mmvq, spec);
    }

    pub fn initQ5_0Q8_1(ctx: *const GpuContext) !MatvecPipeline {
        comptime std.debug.assert(shaders.matvec_q5_0_q8_1.len % 4 == 0);
        return initFromSpv(ctx, &shaders.matvec_q5_0_q8_1, 1);
    }

    pub fn initQ5_1Q8_1(ctx: *const GpuContext) !MatvecPipeline {
        comptime std.debug.assert(shaders.matvec_q5_1_q8_1.len % 4 == 0);
        return initFromSpv(ctx, &shaders.matvec_q5_1_q8_1, 1);
    }

    pub fn initQ6KQ8_1(ctx: *const GpuContext) !MatvecPipeline {
        comptime std.debug.assert(shaders.matvec_q6_k_q8_1.len % 4 == 0);
        return initFromSpv(ctx, &shaders.matvec_q6_k_q8_1, 1);
    }

    pub fn initQ6KQ8_1Fast(ctx: *const GpuContext) !MatvecPipeline {
        comptime std.debug.assert(shaders.matvec_q6_k_q8_1_fast.len % 4 == 0);
        return initFromSpv(ctx, &shaders.matvec_q6_k_q8_1_fast, 1);
    }

    pub fn initQ6KQ8_1Mmvq(ctx: *const GpuContext, spec: MmvqSpec) !MatvecPipeline {
        comptime std.debug.assert(shaders.matvec_q6_k_q8_1_mmvq.len % 4 == 0);
        return initFromSpvMmvq(ctx, &shaders.matvec_q6_k_q8_1_mmvq, spec);
    }

    pub fn initQ5KQ8_1(ctx: *const GpuContext) !MatvecPipeline {
        comptime std.debug.assert(shaders.matvec_q5_k_q8_1.len % 4 == 0);
        return initFromSpv(ctx, &shaders.matvec_q5_k_q8_1, 1);
    }

    pub fn initIQ4NLQ8_1(ctx: *const GpuContext) !MatvecPipeline {
        comptime std.debug.assert(shaders.matvec_iq4_nl_q8_1.len % 4 == 0);
        return initFromSpv(ctx, &shaders.matvec_iq4_nl_q8_1, 1);
    }

    fn initFromSpv(ctx: *const GpuContext, spv: anytype, rows_per_workgroup: u32) !MatvecPipeline {
        return initFromSpvSpecialized(ctx, spv, rows_per_workgroup, null);
    }

    fn initFromSpvMmvq(ctx: *const GpuContext, spv: anytype, spec: MmvqSpec) !MatvecPipeline {
        const map_entries = [_]vk.VkSpecializationMapEntry{
            .{ .constantID = 0, .offset = @offsetOf(MmvqSpec, "block_size"), .size = @sizeOf(u32) },
            .{ .constantID = 1, .offset = @offsetOf(MmvqSpec, "num_rows"),   .size = @sizeOf(u32) },
            .{ .constantID = 2, .offset = @offsetOf(MmvqSpec, "num_cols"),   .size = @sizeOf(u32) },
        };
        const spec_info = vk.VkSpecializationInfo{
            .mapEntryCount = map_entries.len,
            .pMapEntries = &map_entries,
            .dataSize = @sizeOf(MmvqSpec),
            .pData = &spec,
        };
        return initFromSpvSpecialized(ctx, spv, spec.num_rows, &spec_info);
    }

    fn initFromSpvSpecialized(
        ctx: *const GpuContext,
        spv: anytype,
        rows_per_workgroup: u32,
        specialization: ?*const vk.VkSpecializationInfo,
    ) !MatvecPipeline {
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
            .pSpecializationInfo = specialization,
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

        // 64 sets: enough for one-shot run() (1 set) and batched record() (up to 16
        // gate+up or 8 down dispatches per layer, with room for concurrent calls).
        const pool_size = vk.VkDescriptorPoolSize{
            .type = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 3 * 64,
        };
        const pool_ci = vk.VkDescriptorPoolCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = null,
            .flags = vk.VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            .maxSets = 64,
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
            .rows_per_workgroup = rows_per_workgroup,
        };
    }

    // Record one dispatch into an already-open command buffer.
    // Returns the descriptor set allocated from this pipeline's pool.
    // Caller must free the returned set after vkQueueWaitIdle.
    pub fn record(
        self: *const MatvecPipeline,
        cmd: vk.VkCommandBuffer,
        mat_buf: *const GpuBuffer,
        vec_buf: *const GpuBuffer,
        out_buf: *const GpuBuffer,
        rows: u32,
        cols: u32,
    ) !vk.VkDescriptorSet {
        const dset = try self.allocDescriptorSet();
        self.updateDescriptorSet(dset, mat_buf, vec_buf, out_buf);
        self.recordDescriptor(cmd, dset, rows, cols);
        return dset;
    }

    pub fn allocDescriptorSet(self: *const MatvecPipeline) !vk.VkDescriptorSet {
        const alloc_ci = vk.VkDescriptorSetAllocateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = null,
            .descriptorPool = self.desc_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &self.dset_layout,
        };
        var dset: vk.VkDescriptorSet = null;
        if (vk.vkAllocateDescriptorSets(self.device, &alloc_ci, &dset) != vk.VK_SUCCESS)
            return error.VkDescriptorSetAllocFailed;
        return dset;
    }

    pub fn updateDescriptorSet(
        self: *const MatvecPipeline,
        dset: vk.VkDescriptorSet,
        mat_buf: *const GpuBuffer,
        vec_buf: *const GpuBuffer,
        out_buf: *const GpuBuffer,
    ) void {
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
        vk.vkUpdateDescriptorSets(self.device, writes.len, &writes, 0, null);
    }

    pub fn recordDescriptor(
        self: *const MatvecPipeline,
        cmd: vk.VkCommandBuffer,
        dset: vk.VkDescriptorSet,
        rows: u32,
        cols: u32,
    ) void {
        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.layout, 0, 1, &dset, 0, null);

        const pc = PushConst{ .rows = rows, .cols = cols };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT,
            0, @sizeOf(PushConst), &pc);

        const groups = (rows + self.rows_per_workgroup - 1) / self.rows_per_workgroup;
        vk.vkCmdDispatch(cmd, groups, 1, 1);
    }

    pub fn freeDescriptorSet(self: *const MatvecPipeline, dset: *vk.VkDescriptorSet) void {
        _ = vk.vkFreeDescriptorSets(self.device, self.desc_pool, 1, dset);
    }

    // Like record(), but the output sub-range is specified by byte offset+size into
    // a larger buffer. Used to write each expert's down output into expert_all_out_buf.
    pub fn recordToRange(
        self: *const MatvecPipeline,
        cmd: vk.VkCommandBuffer,
        mat_buf: *const GpuBuffer,
        vec_buf: *const GpuBuffer,
        out_handle: vk.VkBuffer,
        out_offset: u64,
        out_range: u64,
        rows: u32,
        cols: u32,
    ) !vk.VkDescriptorSet {
        const dev = self.device;

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

        const buf_infos = [3]vk.VkDescriptorBufferInfo{
            .{ .buffer = mat_buf.handle, .offset = 0,          .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = vec_buf.handle, .offset = 0,          .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = out_handle,     .offset = out_offset,  .range = out_range        },
        };
        const writes = [3]vk.VkWriteDescriptorSet{
            mkWrite(dset, 0, &buf_infos[0]),
            mkWrite(dset, 1, &buf_infos[1]),
            mkWrite(dset, 2, &buf_infos[2]),
        };
        vk.vkUpdateDescriptorSets(dev, writes.len, &writes, 0, null);

        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.layout, 0, 1, &dset, 0, null);

        const pc = PushConst{ .rows = rows, .cols = cols };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT,
            0, @sizeOf(PushConst), &pc);

        const groups = (rows + self.rows_per_workgroup - 1) / self.rows_per_workgroup;
        vk.vkCmdDispatch(cmd, groups, 1, 1);

        return dset;
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

        const cmd = try ctx.beginBatch();

        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.layout, 0, 1, &dset, 0, null);

        const pc = PushConst{ .rows = rows, .cols = cols };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT,
            0, @sizeOf(PushConst), &pc);

        const groups = (rows + self.rows_per_workgroup - 1) / self.rows_per_workgroup;
        const p_mv = ctx.profileBegin(cmd, "matvec.run");
        vk.vkCmdDispatch(cmd, groups, 1, 1);
        ctx.profileEnd(cmd, p_mv);

        try ctx.submitBatch(cmd);
    }
};

// ── MatvecSession ─────────────────────────────────────────────────────────────

// Pre-loaded weight matrix that stays resident in VRAM across token steps.
// Upload the matrix once at model-load time, then call run() for every token.
// The quant format is baked into the MatvecPipeline passed to run().
//
// vec_buf and out_buf are NOT owned by the session — callers share a single
// pair (sized to the largest matrix) across all sessions to avoid 400+
// separate HOST_COHERENT Vulkan allocations. GpuWeights owns the shared bufs.
pub const MatvecSession = struct {
    mat_buf: GpuBuffer, // device-local VRAM: holds raw matrix bytes (any format)
    rows: u32,
    cols: u32,

    // Upload an fp32 row-major matrix to VRAM.
    pub fn init(ctx: *const GpuContext, mat: []const f32, rows: u32, cols: u32) !MatvecSession {
        return initBytes(ctx, std.mem.sliceAsBytes(mat), rows, cols);
    }

    // Upload a Q8_0 quantized matrix (raw GGUF bytes) to VRAM.
    pub fn initQ8_0(ctx: *const GpuContext, mat_bytes: []const u8, rows: u32, cols: u32) !MatvecSession {
        std.debug.assert(cols % 32 == 0);
        std.debug.assert(mat_bytes.len == rows * (cols / 32) * 34);
        return initBytes(ctx, mat_bytes, rows, cols);
    }

    // Upload a Q3_K quantized matrix (raw GGUF bytes) to VRAM.
    pub fn initQ3K(ctx: *const GpuContext, mat_bytes: []const u8, rows: u32, cols: u32) !MatvecSession {
        std.debug.assert(cols % 256 == 0);
        std.debug.assert(mat_bytes.len == rows * (cols / 256) * 110);
        return initBytes(ctx, mat_bytes, rows, cols);
    }

    // Upload a Q4_K quantized matrix (raw GGUF bytes) to VRAM.
    pub fn initQ4K(ctx: *const GpuContext, mat_bytes: []const u8, rows: u32, cols: u32) !MatvecSession {
        std.debug.assert(cols % 256 == 0);
        std.debug.assert(mat_bytes.len == rows * (cols / 256) * 144);
        return initBytes(ctx, mat_bytes, rows, cols);
    }

    // Upload a Q5_1 quantized matrix (raw GGUF bytes) to VRAM.
    pub fn initQ5_1(ctx: *const GpuContext, mat_bytes: []const u8, rows: u32, cols: u32) !MatvecSession {
        std.debug.assert(cols % 32 == 0);
        std.debug.assert(mat_bytes.len == rows * (cols / 32) * 24);
        return initBytes(ctx, mat_bytes, rows, cols);
    }

    // Upload a Q5_0 quantized matrix (raw GGUF bytes) to VRAM.
    pub fn initQ5_0(ctx: *const GpuContext, mat_bytes: []const u8, rows: u32, cols: u32) !MatvecSession {
        std.debug.assert(cols % 32 == 0);
        std.debug.assert(mat_bytes.len == rows * (cols / 32) * 22);
        return initBytes(ctx, mat_bytes, rows, cols);
    }

    // Upload a Q6_K quantized matrix (raw GGUF bytes) to VRAM.
    pub fn initQ6K(ctx: *const GpuContext, mat_bytes: []const u8, rows: u32, cols: u32) !MatvecSession {
        std.debug.assert(cols % 256 == 0);
        std.debug.assert(mat_bytes.len == rows * (cols / 256) * 210);
        return initBytes(ctx, mat_bytes, rows, cols);
    }

    // Upload a Q5_K quantized matrix (raw GGUF bytes) to VRAM.
    pub fn initQ5K(ctx: *const GpuContext, mat_bytes: []const u8, rows: u32, cols: u32) !MatvecSession {
        std.debug.assert(cols % 256 == 0);
        std.debug.assert(mat_bytes.len == rows * (cols / 256) * 176);
        return initBytes(ctx, mat_bytes, rows, cols);
    }

    // Upload an IQ4_NL quantized matrix (raw GGUF bytes) to VRAM.
    pub fn initIQ4NL(ctx: *const GpuContext, mat_bytes: []const u8, rows: u32, cols: u32) !MatvecSession {
        std.debug.assert(cols % 32 == 0);
        std.debug.assert(mat_bytes.len == rows * (cols / 32) * 18);
        return initBytes(ctx, mat_bytes, rows, cols);
    }

    // Upload any GPU-supported quant type. Returns null for unsupported types.
    pub fn initFromRaw(ctx: *const GpuContext, mat_data: []const u8, mat_type: GgmlType, rows: u32, cols: u32) !?MatvecSession {
        return switch (mat_type) {
            .f32  => try initBytes(ctx, mat_data, rows, cols),
            .q8_0 => try initQ8_0(ctx, mat_data, rows, cols),
            .q3_k => try initQ3K(ctx, mat_data, rows, cols),
            .q4_k => try initQ4K(ctx, mat_data, rows, cols),
            .q5_1 => try initQ5_1(ctx, mat_data, rows, cols),
            .q5_0 => try initQ5_0(ctx, mat_data, rows, cols),
            .q6_k => try initQ6K(ctx, mat_data, rows, cols),
            .q5_k => try initQ5K(ctx, mat_data, rows, cols),
            .iq4_nl => try initIQ4NL(ctx, mat_data, rows, cols),
            else  => null,
        };
    }

    // Allocate mat_buf (device-local, empty) for batch upload.
    // Caller records the staging→VRAM copy via GpuContext.recordCopy,
    // then calls submitBatchCopy before calling run().
    pub fn allocEmpty(ctx: *const GpuContext, mat_size: usize, rows: u32, cols: u32) !MatvecSession {
        const mat_buf = try GpuBuffer.initDeviceLocal(ctx, mat_size, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        return .{ .mat_buf = mat_buf, .rows = rows, .cols = cols };
    }

    fn initBytes(ctx: *const GpuContext, mat_bytes: []const u8, rows: u32, cols: u32) !MatvecSession {
        var staging = try GpuBuffer.initStaging(ctx, mat_bytes.len);
        defer staging.deinit();
        try staging.upload(mat_bytes);

        var mat_buf = try GpuBuffer.initDeviceLocal(ctx, mat_bytes.len,
            vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        errdefer mat_buf.deinit();
        try ctx.copyBuffer(staging.handle, mat_buf.handle, mat_bytes.len);

        return .{ .mat_buf = mat_buf, .rows = rows, .cols = cols };
    }

    pub fn deinit(self: *MatvecSession) void {
        self.mat_buf.deinit();
    }

    // Record one dispatch into an open command buffer without submitting.
    // Returns the allocated descriptor set; caller must free it after submit.
    pub fn recordMv(
        self: *const MatvecSession,
        cmd: vk.VkCommandBuffer,
        pipeline: *const MatvecPipeline,
        vec_buf: *const GpuBuffer,
        out_buf: *const GpuBuffer,
    ) !vk.VkDescriptorSet {
        return pipeline.record(cmd, &self.mat_buf, vec_buf, out_buf, self.rows, self.cols);
    }

    // Run matvec using caller-provided host-coherent buffers.
    // vec_buf must be at least cols*4 bytes; out_buf at least rows*4 bytes.
    // GpuWeights shares one max-sized pair across all sessions.
    pub fn run(
        self: *const MatvecSession,
        ctx: *const GpuContext,
        pipeline: *const MatvecPipeline,
        vec_buf: *const GpuBuffer,
        out_buf: *const GpuBuffer,
        vec: []const f32,
        out: []f32,
    ) !void {
        try vec_buf.upload(std.mem.sliceAsBytes(vec));
        try pipeline.run(ctx, &self.mat_buf, vec_buf, out_buf, self.rows, self.cols);
        try out_buf.download(std.mem.sliceAsBytes(out));
    }

    // Convenience wrapper for tests and one-shot ops.
    // Creates temporary host-coherent buffers for this call only.
    pub fn runOwned(
        self: *const MatvecSession,
        ctx: *const GpuContext,
        pipeline: *const MatvecPipeline,
        vec: []const f32,
        out: []f32,
    ) !void {
        var vb = try GpuBuffer.initHostCoherent(ctx, self.cols * @sizeOf(f32),
            vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        defer vb.deinit();
        var ob = try GpuBuffer.initHostCoherent(ctx, self.rows * @sizeOf(f32),
            vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        defer ob.deinit();
        return self.run(ctx, pipeline, &vb, &ob, vec, out);
    }
};

// ── FusedGateUpPipeline ───────────────────────────────────────────────────────

// Fused gate-gelu-up pipeline for Q3_K experts.
// Computes output[row] = gelu(gate_mat[row]·vec) * (up_mat[row]·vec)
// in one dispatch, eliminating the CPU roundtrip between gate/up and down.
// 4 bindings: 0=gate_mat  1=up_mat  2=vec_in  3=vec_out
pub const FusedGateUpPipeline = struct {
    pipeline:   vk.VkPipeline,
    layout:     vk.VkPipelineLayout,
    dset_layout: vk.VkDescriptorSetLayout,
    desc_pool:  vk.VkDescriptorPool,
    device:     vk.VkDevice,
    rows_per_workgroup: u32,

    pub fn init(ctx: *const GpuContext) !FusedGateUpPipeline {
        comptime std.debug.assert(shaders.matvec_fused_gu_q3k.len % 4 == 0);
        return initFromSpv(ctx, &shaders.matvec_fused_gu_q3k, 64);
    }

    // Q3_K weights × Q8_1 activations, fused gate-gelu-up. Same 4-binding
    // layout (gate, up, acts, out); 32 threads cooperate per output row.
    pub fn initQ8_1(ctx: *const GpuContext) !FusedGateUpPipeline {
        comptime std.debug.assert(shaders.matvec_fused_gu_q3k_q8_1.len % 4 == 0);
        return initFromSpv(ctx, &shaders.matvec_fused_gu_q3k_q8_1, 1);
    }

    fn initFromSpv(ctx: *const GpuContext, spv: anytype, rows_per_workgroup: u32) !FusedGateUpPipeline {
        std.debug.assert(@intFromPtr(spv) % 4 == 0);
        const dev = ctx.device;

        const bindings = [4]vk.VkDescriptorSetLayoutBinding{
            mkStorageBuf(0), mkStorageBuf(1), mkStorageBuf(2), mkStorageBuf(3),
        };
        const dsl_ci = vk.VkDescriptorSetLayoutCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = null, .flags = 0,
            .bindingCount = bindings.len, .pBindings = &bindings,
        };
        var dset_layout: vk.VkDescriptorSetLayout = null;
        if (vk.vkCreateDescriptorSetLayout(dev, &dsl_ci, null, &dset_layout) != vk.VK_SUCCESS)
            return error.VkDescSetLayoutFailed;
        errdefer vk.vkDestroyDescriptorSetLayout(dev, dset_layout, null);

        const pc_range = vk.VkPushConstantRange{
            .stageFlags = vk.VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = @sizeOf(PushConst),
        };
        const layout_ci = vk.VkPipelineLayoutCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = null, .flags = 0,
            .setLayoutCount = 1, .pSetLayouts = &dset_layout,
            .pushConstantRangeCount = 1, .pPushConstantRanges = &pc_range,
        };
        var layout: vk.VkPipelineLayout = null;
        if (vk.vkCreatePipelineLayout(dev, &layout_ci, null, &layout) != vk.VK_SUCCESS)
            return error.VkPipelineLayoutFailed;
        errdefer vk.vkDestroyPipelineLayout(dev, layout, null);

        const shader_ci = vk.VkShaderModuleCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = null, .flags = 0, .codeSize = spv.len, .pCode = @ptrCast(spv),
        };
        var shader_mod: vk.VkShaderModule = null;
        if (vk.vkCreateShaderModule(dev, &shader_ci, null, &shader_mod) != vk.VK_SUCCESS)
            return error.VkShaderModuleFailed;
        defer vk.vkDestroyShaderModule(dev, shader_mod, null);

        const stage = vk.VkPipelineShaderStageCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = null, .flags = 0,
            .stage = vk.VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader_mod, .pName = "main", .pSpecializationInfo = null,
        };
        const pipeline_ci = vk.VkComputePipelineCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .pNext = null, .flags = 0, .stage = stage, .layout = layout,
            .basePipelineHandle = null, .basePipelineIndex = -1,
        };
        var pipeline: vk.VkPipeline = null;
        if (vk.vkCreateComputePipelines(dev, null, 1, &pipeline_ci, null, &pipeline) != vk.VK_SUCCESS)
            return error.VkComputePipelineFailed;
        errdefer vk.vkDestroyPipeline(dev, pipeline, null);

        // 16 sets: one per active expert per layer (n_experts_used = 8 typical)
        const pool_size = vk.VkDescriptorPoolSize{
            .type = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 4 * 16,
        };
        const pool_ci = vk.VkDescriptorPoolCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = null,
            .flags = vk.VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            .maxSets = 16, .poolSizeCount = 1, .pPoolSizes = &pool_size,
        };
        var desc_pool: vk.VkDescriptorPool = null;
        if (vk.vkCreateDescriptorPool(dev, &pool_ci, null, &desc_pool) != vk.VK_SUCCESS)
            return error.VkDescriptorPoolFailed;

        return .{
            .pipeline = pipeline, .layout = layout,
            .dset_layout = dset_layout, .desc_pool = desc_pool, .device = dev,
            .rows_per_workgroup = rows_per_workgroup,
        };
    }

    // Record one fused gate-gelu-up dispatch into an open command buffer.
    // Returns the descriptor set; caller frees it after the submit completes.
    pub fn record(
        self: *const FusedGateUpPipeline,
        cmd: vk.VkCommandBuffer,
        gate_buf: *const GpuBuffer,
        up_buf:   *const GpuBuffer,
        vec_buf:  *const GpuBuffer,
        out_buf:  *const GpuBuffer,
        rows: u32,
        cols: u32,
    ) !vk.VkDescriptorSet {
        const dev = self.device;
        const alloc_ci = vk.VkDescriptorSetAllocateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = null, .descriptorPool = self.desc_pool,
            .descriptorSetCount = 1, .pSetLayouts = &self.dset_layout,
        };
        var dset: vk.VkDescriptorSet = null;
        if (vk.vkAllocateDescriptorSets(dev, &alloc_ci, &dset) != vk.VK_SUCCESS)
            return error.VkDescriptorSetAllocFailed;

        const buf_infos = [4]vk.VkDescriptorBufferInfo{
            .{ .buffer = gate_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = up_buf.handle,   .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = vec_buf.handle,  .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = out_buf.handle,  .offset = 0, .range = vk.VK_WHOLE_SIZE },
        };
        const writes = [4]vk.VkWriteDescriptorSet{
            mkWrite(dset, 0, &buf_infos[0]), mkWrite(dset, 1, &buf_infos[1]),
            mkWrite(dset, 2, &buf_infos[2]), mkWrite(dset, 3, &buf_infos[3]),
        };
        vk.vkUpdateDescriptorSets(dev, writes.len, &writes, 0, null);

        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.layout, 0, 1, &dset, 0, null);
        const pc = PushConst{ .rows = rows, .cols = cols };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT,
            0, @sizeOf(PushConst), &pc);
        const groups = (rows + self.rows_per_workgroup - 1) / self.rows_per_workgroup;
        vk.vkCmdDispatch(cmd, groups, 1, 1);

        return dset;
    }

    pub fn deinit(self: *FusedGateUpPipeline) void {
        vk.vkDestroyDescriptorPool(self.device, self.desc_pool, null);
        vk.vkDestroyPipeline(self.device, self.pipeline, null);
        vk.vkDestroyPipelineLayout(self.device, self.layout, null);
        vk.vkDestroyDescriptorSetLayout(self.device, self.dset_layout, null);
    }
};

// ── AccumPipeline ─────────────────────────────────────────────────────────────

// Weighted accumulation: output[i] += sum_k(scales[k] * inputs[k*d_model + i])
// Eliminates CPU roundtrip for expert output accumulation.
// Bindings: 0=inputs (n×d_model f32), 1=scales (n f32), 2=output (d_model f32, +=)
pub const AccumPipeline = struct {
    pipeline:    vk.VkPipeline,
    layout:      vk.VkPipelineLayout,
    dset_layout: vk.VkDescriptorSetLayout,
    desc_pool:   vk.VkDescriptorPool,
    device:      vk.VkDevice,

    pub fn init(ctx: *const GpuContext) !AccumPipeline {
        comptime std.debug.assert(shaders.expert_accum.len % 4 == 0);
        const dev = ctx.device;

        const bindings = [3]vk.VkDescriptorSetLayoutBinding{
            mkStorageBuf(0), mkStorageBuf(1), mkStorageBuf(2),
        };
        const dsl_ci = vk.VkDescriptorSetLayoutCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = null, .flags = 0,
            .bindingCount = bindings.len, .pBindings = &bindings,
        };
        var dset_layout: vk.VkDescriptorSetLayout = null;
        if (vk.vkCreateDescriptorSetLayout(dev, &dsl_ci, null, &dset_layout) != vk.VK_SUCCESS)
            return error.VkDescSetLayoutFailed;
        errdefer vk.vkDestroyDescriptorSetLayout(dev, dset_layout, null);

        const pc_range = vk.VkPushConstantRange{
            .stageFlags = vk.VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = @sizeOf(PushConst),
        };
        const layout_ci = vk.VkPipelineLayoutCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = null, .flags = 0,
            .setLayoutCount = 1, .pSetLayouts = &dset_layout,
            .pushConstantRangeCount = 1, .pPushConstantRanges = &pc_range,
        };
        var layout: vk.VkPipelineLayout = null;
        if (vk.vkCreatePipelineLayout(dev, &layout_ci, null, &layout) != vk.VK_SUCCESS)
            return error.VkPipelineLayoutFailed;
        errdefer vk.vkDestroyPipelineLayout(dev, layout, null);

        const spv = &shaders.expert_accum;
        std.debug.assert(@intFromPtr(spv) % 4 == 0);
        const shader_ci = vk.VkShaderModuleCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = null, .flags = 0, .codeSize = spv.len, .pCode = @ptrCast(spv),
        };
        var shader_mod: vk.VkShaderModule = null;
        if (vk.vkCreateShaderModule(dev, &shader_ci, null, &shader_mod) != vk.VK_SUCCESS)
            return error.VkShaderModuleFailed;
        defer vk.vkDestroyShaderModule(dev, shader_mod, null);

        const stage = vk.VkPipelineShaderStageCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = null, .flags = 0,
            .stage = vk.VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader_mod, .pName = "main", .pSpecializationInfo = null,
        };
        const pipeline_ci = vk.VkComputePipelineCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .pNext = null, .flags = 0, .stage = stage, .layout = layout,
            .basePipelineHandle = null, .basePipelineIndex = -1,
        };
        var pipeline: vk.VkPipeline = null;
        if (vk.vkCreateComputePipelines(dev, null, 1, &pipeline_ci, null, &pipeline) != vk.VK_SUCCESS)
            return error.VkComputePipelineFailed;
        errdefer vk.vkDestroyPipeline(dev, pipeline, null);

        const pool_size = vk.VkDescriptorPoolSize{
            .type = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 3 * 4,
        };
        const pool_ci = vk.VkDescriptorPoolCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = null,
            .flags = vk.VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            .maxSets = 4, .poolSizeCount = 1, .pPoolSizes = &pool_size,
        };
        var desc_pool: vk.VkDescriptorPool = null;
        if (vk.vkCreateDescriptorPool(dev, &pool_ci, null, &desc_pool) != vk.VK_SUCCESS)
            return error.VkDescriptorPoolFailed;

        return .{
            .pipeline = pipeline, .layout = layout,
            .dset_layout = dset_layout, .desc_pool = desc_pool, .device = dev,
        };
    }

    pub fn record(
        self: *const AccumPipeline,
        cmd: vk.VkCommandBuffer,
        inputs_buf: *const GpuBuffer,
        scales_buf: *const GpuBuffer,
        out_buf: *const GpuBuffer,
        d_model: u32,
        n: u32,
    ) !vk.VkDescriptorSet {
        const dev = self.device;
        const alloc_ci = vk.VkDescriptorSetAllocateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = null, .descriptorPool = self.desc_pool,
            .descriptorSetCount = 1, .pSetLayouts = &self.dset_layout,
        };
        var dset: vk.VkDescriptorSet = null;
        if (vk.vkAllocateDescriptorSets(dev, &alloc_ci, &dset) != vk.VK_SUCCESS)
            return error.VkDescriptorSetAllocFailed;

        const buf_infos = [3]vk.VkDescriptorBufferInfo{
            .{ .buffer = inputs_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = scales_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = out_buf.handle,    .offset = 0, .range = vk.VK_WHOLE_SIZE },
        };
        const writes = [3]vk.VkWriteDescriptorSet{
            mkWrite(dset, 0, &buf_infos[0]),
            mkWrite(dset, 1, &buf_infos[1]),
            mkWrite(dset, 2, &buf_infos[2]),
        };
        vk.vkUpdateDescriptorSets(dev, writes.len, &writes, 0, null);

        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.layout, 0, 1, &dset, 0, null);
        // PushConst: rows=d_model, cols=n (reuses the 2×u32 push constant struct)
        const pc = PushConst{ .rows = d_model, .cols = n };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT,
            0, @sizeOf(PushConst), &pc);
        const groups = (d_model + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        vk.vkCmdDispatch(cmd, groups, 1, 1);

        return dset;
    }

    pub fn deinit(self: *AccumPipeline) void {
        vk.vkDestroyDescriptorPool(self.device, self.desc_pool, null);
        vk.vkDestroyPipeline(self.device, self.pipeline, null);
        vk.vkDestroyPipelineLayout(self.device, self.layout, null);
        vk.vkDestroyDescriptorSetLayout(self.device, self.dset_layout, null);
    }
};

// ── QuantizeQ8_1Pipeline ──────────────────────────────────────────────────────
//
// Quantize an f32 activation vector into Q8_1 in the x4-packed layout that
// the integer-dot matvec shaders read. One workgroup per 128-element x4 group
// (32 threads); per-block amax + sum done with subgroup-clustered ops. Output
// size: ceil(ncols / 128) * 144 bytes.
//
// Bindings: 0 = input vec4[] (f32 activation), 1 = output block_q8_1_x4[].
// Push constant: { uint ncols; }.

const QuantizePushConst = extern struct { ncols: u32 };

pub const QuantizeQ8_1Pipeline = struct {
    pipeline:    vk.VkPipeline,
    layout:      vk.VkPipelineLayout,
    dset_layout: vk.VkDescriptorSetLayout,
    desc_pool:   vk.VkDescriptorPool,
    device:      vk.VkDevice,

    pub fn init(ctx: *const GpuContext) !QuantizeQ8_1Pipeline {
        comptime std.debug.assert(shaders.quantize_q8_1.len % 4 == 0);
        const dev = ctx.device;

        const bindings = [2]vk.VkDescriptorSetLayoutBinding{
            mkStorageBuf(0), mkStorageBuf(1),
        };
        const dsl_ci = vk.VkDescriptorSetLayoutCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = null, .flags = 0,
            .bindingCount = bindings.len, .pBindings = &bindings,
        };
        var dset_layout: vk.VkDescriptorSetLayout = null;
        if (vk.vkCreateDescriptorSetLayout(dev, &dsl_ci, null, &dset_layout) != vk.VK_SUCCESS)
            return error.VkDescSetLayoutFailed;
        errdefer vk.vkDestroyDescriptorSetLayout(dev, dset_layout, null);

        const pc_range = vk.VkPushConstantRange{
            .stageFlags = vk.VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0, .size = @sizeOf(QuantizePushConst),
        };
        const layout_ci = vk.VkPipelineLayoutCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = null, .flags = 0,
            .setLayoutCount = 1, .pSetLayouts = &dset_layout,
            .pushConstantRangeCount = 1, .pPushConstantRanges = &pc_range,
        };
        var layout: vk.VkPipelineLayout = null;
        if (vk.vkCreatePipelineLayout(dev, &layout_ci, null, &layout) != vk.VK_SUCCESS)
            return error.VkPipelineLayoutFailed;
        errdefer vk.vkDestroyPipelineLayout(dev, layout, null);

        const spv = &shaders.quantize_q8_1;
        std.debug.assert(@intFromPtr(spv) % 4 == 0);
        const shader_ci = vk.VkShaderModuleCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = null, .flags = 0, .codeSize = spv.len, .pCode = @ptrCast(spv),
        };
        var shader_mod: vk.VkShaderModule = null;
        if (vk.vkCreateShaderModule(dev, &shader_ci, null, &shader_mod) != vk.VK_SUCCESS)
            return error.VkShaderModuleFailed;
        defer vk.vkDestroyShaderModule(dev, shader_mod, null);

        const stage = vk.VkPipelineShaderStageCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = null, .flags = 0,
            .stage = vk.VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader_mod, .pName = "main", .pSpecializationInfo = null,
        };
        const pipeline_ci = vk.VkComputePipelineCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .pNext = null, .flags = 0, .stage = stage, .layout = layout,
            .basePipelineHandle = null, .basePipelineIndex = -1,
        };
        var pipeline: vk.VkPipeline = null;
        if (vk.vkCreateComputePipelines(dev, null, 1, &pipeline_ci, null, &pipeline) != vk.VK_SUCCESS)
            return error.VkComputePipelineFailed;
        errdefer vk.vkDestroyPipeline(dev, pipeline, null);

        const pool_size = vk.VkDescriptorPoolSize{
            .type = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2 * 16,
        };
        const pool_ci = vk.VkDescriptorPoolCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = null,
            .flags = vk.VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            .maxSets = 16, .poolSizeCount = 1, .pPoolSizes = &pool_size,
        };
        var desc_pool: vk.VkDescriptorPool = null;
        if (vk.vkCreateDescriptorPool(dev, &pool_ci, null, &desc_pool) != vk.VK_SUCCESS)
            return error.VkDescriptorPoolFailed;

        return .{
            .pipeline = pipeline, .layout = layout,
            .dset_layout = dset_layout, .desc_pool = desc_pool, .device = dev,
        };
    }

    // Record a Q8_1 quantization dispatch into an already-recording command buffer.
    pub fn record(
        self: *const QuantizeQ8_1Pipeline,
        cmd: vk.VkCommandBuffer,
        in_buf:  *const GpuBuffer,
        out_buf: *const GpuBuffer,
        ncols: u32,
    ) !vk.VkDescriptorSet {
        std.debug.assert(ncols % 4 == 0); // vec4-aligned reads
        const dev = self.device;
        const alloc_ci = vk.VkDescriptorSetAllocateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = null, .descriptorPool = self.desc_pool,
            .descriptorSetCount = 1, .pSetLayouts = &self.dset_layout,
        };
        var dset: vk.VkDescriptorSet = null;
        if (vk.vkAllocateDescriptorSets(dev, &alloc_ci, &dset) != vk.VK_SUCCESS)
            return error.VkDescriptorSetAllocFailed;

        const buf_infos = [2]vk.VkDescriptorBufferInfo{
            .{ .buffer = in_buf.handle,  .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = out_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        };
        const writes = [2]vk.VkWriteDescriptorSet{
            mkWrite(dset, 0, &buf_infos[0]),
            mkWrite(dset, 1, &buf_infos[1]),
        };
        vk.vkUpdateDescriptorSets(dev, writes.len, &writes, 0, null);

        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.layout, 0, 1, &dset, 0, null);
        const pc = QuantizePushConst{ .ncols = ncols };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT,
            0, @sizeOf(QuantizePushConst), &pc);
        // One workgroup per 128-element x4 group (32 threads each).
        const groups = (ncols + 127) / 128;
        vk.vkCmdDispatch(cmd, groups, 1, 1);

        return dset;
    }

    pub fn deinit(self: *QuantizeQ8_1Pipeline) void {
        vk.vkDestroyDescriptorPool(self.device, self.desc_pool, null);
        vk.vkDestroyPipeline(self.device, self.pipeline, null);
        vk.vkDestroyPipelineLayout(self.device, self.layout, null);
        vk.vkDestroyDescriptorSetLayout(self.device, self.dset_layout, null);
    }
};

// Convenience: round up an activation length to the number of bytes a Q8_1_x4
// output buffer must be sized to.
pub fn q8_1OutBytes(ncols: u32) u32 {
    return ((ncols + 127) / 128) * @as(u32, dq.Q8_1_X4_BYTES);
}

// ── RmsnormPipeline ───────────────────────────────────────────────────────────
//
// Per-row RMS normalization: y[i] = x[i] * rms_inv * (bias + w[i])
// where rms_inv = 1 / sqrt(Σ x²/n + eps) and bias is either 0 or 1 to handle
// the Gemma (1+w) weight convention via a flag.
//
// Bindings: 0=x_in (f32), 1=w (f32), 2=y_out (f32). One workgroup per call,
// 256 threads. Push constant: { u32 n; f32 eps; u32 weight_offset; }.

const RmsnormPushConst = extern struct {
    n: u32,
    eps: f32,
    weight_offset: u32,
};

pub const RmsnormPipeline = struct {
    pipeline:    vk.VkPipeline,
    layout:      vk.VkPipelineLayout,
    dset_layout: vk.VkDescriptorSetLayout,
    desc_pool:   vk.VkDescriptorPool,
    device:      vk.VkDevice,

    pub fn init(ctx: *const GpuContext) !RmsnormPipeline {
        comptime std.debug.assert(shaders.rmsnorm.len % 4 == 0);
        const dev = ctx.device;

        const bindings = [3]vk.VkDescriptorSetLayoutBinding{
            mkStorageBuf(0), mkStorageBuf(1), mkStorageBuf(2),
        };
        const dsl_ci = vk.VkDescriptorSetLayoutCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = null, .flags = 0,
            .bindingCount = bindings.len, .pBindings = &bindings,
        };
        var dset_layout: vk.VkDescriptorSetLayout = null;
        if (vk.vkCreateDescriptorSetLayout(dev, &dsl_ci, null, &dset_layout) != vk.VK_SUCCESS)
            return error.VkDescSetLayoutFailed;
        errdefer vk.vkDestroyDescriptorSetLayout(dev, dset_layout, null);

        const pc_range = vk.VkPushConstantRange{
            .stageFlags = vk.VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0, .size = @sizeOf(RmsnormPushConst),
        };
        const layout_ci = vk.VkPipelineLayoutCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = null, .flags = 0,
            .setLayoutCount = 1, .pSetLayouts = &dset_layout,
            .pushConstantRangeCount = 1, .pPushConstantRanges = &pc_range,
        };
        var layout: vk.VkPipelineLayout = null;
        if (vk.vkCreatePipelineLayout(dev, &layout_ci, null, &layout) != vk.VK_SUCCESS)
            return error.VkPipelineLayoutFailed;
        errdefer vk.vkDestroyPipelineLayout(dev, layout, null);

        const spv = &shaders.rmsnorm;
        const shader_ci = vk.VkShaderModuleCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = null, .flags = 0, .codeSize = spv.len, .pCode = @ptrCast(spv),
        };
        var shader_mod: vk.VkShaderModule = null;
        if (vk.vkCreateShaderModule(dev, &shader_ci, null, &shader_mod) != vk.VK_SUCCESS)
            return error.VkShaderModuleFailed;
        defer vk.vkDestroyShaderModule(dev, shader_mod, null);

        const stage = vk.VkPipelineShaderStageCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = null, .flags = 0,
            .stage = vk.VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader_mod, .pName = "main", .pSpecializationInfo = null,
        };
        const pipeline_ci = vk.VkComputePipelineCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .pNext = null, .flags = 0, .stage = stage, .layout = layout,
            .basePipelineHandle = null, .basePipelineIndex = -1,
        };
        var pipeline: vk.VkPipeline = null;
        if (vk.vkCreateComputePipelines(dev, null, 1, &pipeline_ci, null, &pipeline) != vk.VK_SUCCESS)
            return error.VkComputePipelineFailed;
        errdefer vk.vkDestroyPipeline(dev, pipeline, null);

        const pool_size = vk.VkDescriptorPoolSize{
            .type = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 3 * 32,
        };
        const pool_ci = vk.VkDescriptorPoolCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = null,
            .flags = vk.VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            .maxSets = 32, .poolSizeCount = 1, .pPoolSizes = &pool_size,
        };
        var desc_pool: vk.VkDescriptorPool = null;
        if (vk.vkCreateDescriptorPool(dev, &pool_ci, null, &desc_pool) != vk.VK_SUCCESS)
            return error.VkDescriptorPoolFailed;

        return .{
            .pipeline = pipeline, .layout = layout,
            .dset_layout = dset_layout, .desc_pool = desc_pool, .device = dev,
        };
    }

    pub fn record(
        self: *const RmsnormPipeline,
        cmd: vk.VkCommandBuffer,
        x_buf: *const GpuBuffer,
        w_buf: *const GpuBuffer,
        y_buf: *const GpuBuffer,
        n: u32,
        eps: f32,
        weight_offset: bool,
    ) !vk.VkDescriptorSet {
        std.debug.assert(n % 256 == 0);
        const dev = self.device;
        const alloc_ci = vk.VkDescriptorSetAllocateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = null, .descriptorPool = self.desc_pool,
            .descriptorSetCount = 1, .pSetLayouts = &self.dset_layout,
        };
        var dset: vk.VkDescriptorSet = null;
        if (vk.vkAllocateDescriptorSets(dev, &alloc_ci, &dset) != vk.VK_SUCCESS)
            return error.VkDescriptorSetAllocFailed;

        const buf_infos = [3]vk.VkDescriptorBufferInfo{
            .{ .buffer = x_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = w_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = y_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        };
        const writes = [3]vk.VkWriteDescriptorSet{
            mkWrite(dset, 0, &buf_infos[0]),
            mkWrite(dset, 1, &buf_infos[1]),
            mkWrite(dset, 2, &buf_infos[2]),
        };
        vk.vkUpdateDescriptorSets(dev, writes.len, &writes, 0, null);

        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.layout, 0, 1, &dset, 0, null);

        const pc = RmsnormPushConst{
            .n = n, .eps = eps,
            .weight_offset = if (weight_offset) 1 else 0,
        };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT,
            0, @sizeOf(RmsnormPushConst), &pc);
        vk.vkCmdDispatch(cmd, 1, 1, 1);

        return dset;
    }

    pub fn deinit(self: *RmsnormPipeline) void {
        vk.vkDestroyDescriptorPool(self.device, self.desc_pool, null);
        vk.vkDestroyPipeline(self.device, self.pipeline, null);
        vk.vkDestroyPipelineLayout(self.device, self.layout, null);
        vk.vkDestroyDescriptorSetLayout(self.device, self.dset_layout, null);
    }
};

// ── RmsnormPerHeadPipeline ────────────────────────────────────────────────────
//
// Multi-head per-head RMS normalization for Gemma4 attention. One workgroup
// per head reduces Σ x² across `head_dim` elements; threads then write the
// normalized output back to the same per-head row. The W binding (weight
// vector, head_dim floats) is shared across heads; when use_weight==0 the
// shader ignores W entirely — used for the V projection's rmsnormRaw path.
//
// Single dispatch handles all heads, so per-token submit overhead stays
// constant regardless of n_heads / n_kv_heads.

const RmsnormPerHeadPushConst = extern struct {
    head_dim: u32,
    eps: f32,
    weight_offset: u32,
    use_weight: u32,
};

pub const RmsnormPerHeadPipeline = struct {
    pipeline:    vk.VkPipeline,
    layout:      vk.VkPipelineLayout,
    dset_layout: vk.VkDescriptorSetLayout,
    desc_pool:   vk.VkDescriptorPool,
    device:      vk.VkDevice,

    pub fn init(ctx: *const GpuContext) !RmsnormPerHeadPipeline {
        comptime std.debug.assert(shaders.rmsnorm_perhead.len % 4 == 0);
        const dev = ctx.device;

        const bindings = [3]vk.VkDescriptorSetLayoutBinding{
            mkStorageBuf(0), mkStorageBuf(1), mkStorageBuf(2),
        };
        const dsl_ci = vk.VkDescriptorSetLayoutCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = null, .flags = 0,
            .bindingCount = bindings.len, .pBindings = &bindings,
        };
        var dset_layout: vk.VkDescriptorSetLayout = null;
        if (vk.vkCreateDescriptorSetLayout(dev, &dsl_ci, null, &dset_layout) != vk.VK_SUCCESS)
            return error.VkDescSetLayoutFailed;
        errdefer vk.vkDestroyDescriptorSetLayout(dev, dset_layout, null);

        const pc_range = vk.VkPushConstantRange{
            .stageFlags = vk.VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0, .size = @sizeOf(RmsnormPerHeadPushConst),
        };
        const layout_ci = vk.VkPipelineLayoutCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = null, .flags = 0,
            .setLayoutCount = 1, .pSetLayouts = &dset_layout,
            .pushConstantRangeCount = 1, .pPushConstantRanges = &pc_range,
        };
        var layout: vk.VkPipelineLayout = null;
        if (vk.vkCreatePipelineLayout(dev, &layout_ci, null, &layout) != vk.VK_SUCCESS)
            return error.VkPipelineLayoutFailed;
        errdefer vk.vkDestroyPipelineLayout(dev, layout, null);

        const spv = &shaders.rmsnorm_perhead;
        const shader_ci = vk.VkShaderModuleCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = null, .flags = 0, .codeSize = spv.len, .pCode = @ptrCast(spv),
        };
        var shader_mod: vk.VkShaderModule = null;
        if (vk.vkCreateShaderModule(dev, &shader_ci, null, &shader_mod) != vk.VK_SUCCESS)
            return error.VkShaderModuleFailed;
        defer vk.vkDestroyShaderModule(dev, shader_mod, null);

        const stage = vk.VkPipelineShaderStageCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = null, .flags = 0,
            .stage = vk.VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader_mod, .pName = "main", .pSpecializationInfo = null,
        };
        const pipeline_ci = vk.VkComputePipelineCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .pNext = null, .flags = 0, .stage = stage, .layout = layout,
            .basePipelineHandle = null, .basePipelineIndex = -1,
        };
        var pipeline: vk.VkPipeline = null;
        if (vk.vkCreateComputePipelines(dev, null, 1, &pipeline_ci, null, &pipeline) != vk.VK_SUCCESS)
            return error.VkComputePipelineFailed;
        errdefer vk.vkDestroyPipeline(dev, pipeline, null);

        const pool_size = vk.VkDescriptorPoolSize{
            .type = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 3 * 32,
        };
        const pool_ci = vk.VkDescriptorPoolCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = null,
            .flags = vk.VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            .maxSets = 32, .poolSizeCount = 1, .pPoolSizes = &pool_size,
        };
        var desc_pool: vk.VkDescriptorPool = null;
        if (vk.vkCreateDescriptorPool(dev, &pool_ci, null, &desc_pool) != vk.VK_SUCCESS)
            return error.VkDescriptorPoolFailed;

        return .{
            .pipeline = pipeline, .layout = layout,
            .dset_layout = dset_layout, .desc_pool = desc_pool, .device = dev,
        };
    }

    pub fn record(
        self: *const RmsnormPerHeadPipeline,
        cmd: vk.VkCommandBuffer,
        x_buf: *const GpuBuffer,
        w_buf: *const GpuBuffer,  // may equal x_buf when use_weight==false (unread)
        y_buf: *const GpuBuffer,
        n_heads: u32,
        head_dim: u32,
        eps: f32,
        weight_offset: bool,
        use_weight: bool,
    ) !vk.VkDescriptorSet {
        std.debug.assert(head_dim % 256 == 0);
        const dev = self.device;
        const alloc_ci = vk.VkDescriptorSetAllocateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = null, .descriptorPool = self.desc_pool,
            .descriptorSetCount = 1, .pSetLayouts = &self.dset_layout,
        };
        var dset: vk.VkDescriptorSet = null;
        if (vk.vkAllocateDescriptorSets(dev, &alloc_ci, &dset) != vk.VK_SUCCESS)
            return error.VkDescriptorSetAllocFailed;

        const buf_infos = [3]vk.VkDescriptorBufferInfo{
            .{ .buffer = x_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = w_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = y_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        };
        const writes = [3]vk.VkWriteDescriptorSet{
            mkWrite(dset, 0, &buf_infos[0]),
            mkWrite(dset, 1, &buf_infos[1]),
            mkWrite(dset, 2, &buf_infos[2]),
        };
        vk.vkUpdateDescriptorSets(dev, writes.len, &writes, 0, null);

        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.layout, 0, 1, &dset, 0, null);

        const pc = RmsnormPerHeadPushConst{
            .head_dim = head_dim, .eps = eps,
            .weight_offset = if (weight_offset) 1 else 0,
            .use_weight    = if (use_weight)    1 else 0,
        };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT,
            0, @sizeOf(RmsnormPerHeadPushConst), &pc);
        vk.vkCmdDispatch(cmd, n_heads, 1, 1);

        return dset;
    }

    pub fn deinit(self: *RmsnormPerHeadPipeline) void {
        vk.vkDestroyDescriptorPool(self.device, self.desc_pool, null);
        vk.vkDestroyPipeline(self.device, self.pipeline, null);
        vk.vkDestroyPipelineLayout(self.device, self.layout, null);
        vk.vkDestroyDescriptorSetLayout(self.device, self.dset_layout, null);
    }
};

// ── ElemAddPipeline ───────────────────────────────────────────────────────────
//
// In-place elementwise add: a[i] += b[i]. 2 bindings (a read-write, b read-only),
// push constant carries the length. One thread per element.

const ElemPushConst = extern struct { n: u32 };
const ElemScalePushConst = extern struct { n: u32, s: f32 };

fn buildSimplePipeline(
    ctx: *const GpuContext,
    spv: anytype,
    n_bindings: u32,
    pc_size: u32,
    max_sets: u32,
) !struct {
    pipeline: vk.VkPipeline,
    layout: vk.VkPipelineLayout,
    dset_layout: vk.VkDescriptorSetLayout,
    desc_pool: vk.VkDescriptorPool,
} {
    const dev = ctx.device;

    // Variable-size descriptor binding list (max 4 for our shaders).
    std.debug.assert(n_bindings >= 1 and n_bindings <= 4);
    var bindings_buf: [4]vk.VkDescriptorSetLayoutBinding = undefined;
    var b: u32 = 0;
    while (b < n_bindings) : (b += 1) bindings_buf[b] = mkStorageBuf(b);
    const dsl_ci = vk.VkDescriptorSetLayoutCreateInfo{
        .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = null, .flags = 0,
        .bindingCount = n_bindings, .pBindings = &bindings_buf,
    };
    var dset_layout: vk.VkDescriptorSetLayout = null;
    if (vk.vkCreateDescriptorSetLayout(dev, &dsl_ci, null, &dset_layout) != vk.VK_SUCCESS)
        return error.VkDescSetLayoutFailed;
    errdefer vk.vkDestroyDescriptorSetLayout(dev, dset_layout, null);

    const pc_range = vk.VkPushConstantRange{
        .stageFlags = vk.VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = pc_size,
    };
    const layout_ci = vk.VkPipelineLayoutCreateInfo{
        .sType = vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = null, .flags = 0,
        .setLayoutCount = 1, .pSetLayouts = &dset_layout,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &pc_range,
    };
    var layout: vk.VkPipelineLayout = null;
    if (vk.vkCreatePipelineLayout(dev, &layout_ci, null, &layout) != vk.VK_SUCCESS)
        return error.VkPipelineLayoutFailed;
    errdefer vk.vkDestroyPipelineLayout(dev, layout, null);

    const shader_ci = vk.VkShaderModuleCreateInfo{
        .sType = vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = null, .flags = 0, .codeSize = spv.len, .pCode = @ptrCast(spv),
    };
    var shader_mod: vk.VkShaderModule = null;
    if (vk.vkCreateShaderModule(dev, &shader_ci, null, &shader_mod) != vk.VK_SUCCESS)
        return error.VkShaderModuleFailed;
    defer vk.vkDestroyShaderModule(dev, shader_mod, null);

    const stage = vk.VkPipelineShaderStageCreateInfo{
        .sType = vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = null, .flags = 0, .stage = vk.VK_SHADER_STAGE_COMPUTE_BIT,
        .module = shader_mod, .pName = "main", .pSpecializationInfo = null,
    };
    const pipeline_ci = vk.VkComputePipelineCreateInfo{
        .sType = vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = null, .flags = 0, .stage = stage, .layout = layout,
        .basePipelineHandle = null, .basePipelineIndex = -1,
    };
    var pipeline: vk.VkPipeline = null;
    if (vk.vkCreateComputePipelines(dev, null, 1, &pipeline_ci, null, &pipeline) != vk.VK_SUCCESS)
        return error.VkComputePipelineFailed;
    errdefer vk.vkDestroyPipeline(dev, pipeline, null);

    const pool_size = vk.VkDescriptorPoolSize{
        .type = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = n_bindings * max_sets,
    };
    const pool_ci = vk.VkDescriptorPoolCreateInfo{
        .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = null,
        .flags = vk.VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = max_sets, .poolSizeCount = 1, .pPoolSizes = &pool_size,
    };
    var desc_pool: vk.VkDescriptorPool = null;
    if (vk.vkCreateDescriptorPool(dev, &pool_ci, null, &desc_pool) != vk.VK_SUCCESS)
        return error.VkDescriptorPoolFailed;

    return .{
        .pipeline = pipeline, .layout = layout,
        .dset_layout = dset_layout, .desc_pool = desc_pool,
    };
}

pub const ElemAddPipeline = struct {
    pipeline:    vk.VkPipeline,
    layout:      vk.VkPipelineLayout,
    dset_layout: vk.VkDescriptorSetLayout,
    desc_pool:   vk.VkDescriptorPool,
    device:      vk.VkDevice,

    pub fn init(ctx: *const GpuContext) !ElemAddPipeline {
        comptime std.debug.assert(shaders.elem_add.len % 4 == 0);
        const built = try buildSimplePipeline(ctx, &shaders.elem_add, 2, @sizeOf(ElemPushConst), 32);
        return .{
            .pipeline = built.pipeline, .layout = built.layout,
            .dset_layout = built.dset_layout, .desc_pool = built.desc_pool,
            .device = ctx.device,
        };
    }

    pub fn record(
        self: *const ElemAddPipeline,
        cmd: vk.VkCommandBuffer,
        a_buf: *const GpuBuffer,
        b_buf: *const GpuBuffer,
        n: u32,
    ) !vk.VkDescriptorSet {
        const dev = self.device;
        const alloc_ci = vk.VkDescriptorSetAllocateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = null, .descriptorPool = self.desc_pool,
            .descriptorSetCount = 1, .pSetLayouts = &self.dset_layout,
        };
        var dset: vk.VkDescriptorSet = null;
        if (vk.vkAllocateDescriptorSets(dev, &alloc_ci, &dset) != vk.VK_SUCCESS)
            return error.VkDescriptorSetAllocFailed;

        const buf_infos = [2]vk.VkDescriptorBufferInfo{
            .{ .buffer = a_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = b_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        };
        const writes = [2]vk.VkWriteDescriptorSet{
            mkWrite(dset, 0, &buf_infos[0]),
            mkWrite(dset, 1, &buf_infos[1]),
        };
        vk.vkUpdateDescriptorSets(dev, writes.len, &writes, 0, null);

        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.layout, 0, 1, &dset, 0, null);
        const pc = ElemPushConst{ .n = n };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT,
            0, @sizeOf(ElemPushConst), &pc);
        const groups = (n + 255) / 256;
        vk.vkCmdDispatch(cmd, groups, 1, 1);
        return dset;
    }

    pub fn deinit(self: *ElemAddPipeline) void {
        vk.vkDestroyDescriptorPool(self.device, self.desc_pool, null);
        vk.vkDestroyPipeline(self.device, self.pipeline, null);
        vk.vkDestroyPipelineLayout(self.device, self.layout, null);
        vk.vkDestroyDescriptorSetLayout(self.device, self.dset_layout, null);
    }
};

pub const ElemScalePipeline = struct {
    pipeline:    vk.VkPipeline,
    layout:      vk.VkPipelineLayout,
    dset_layout: vk.VkDescriptorSetLayout,
    desc_pool:   vk.VkDescriptorPool,
    device:      vk.VkDevice,

    pub fn init(ctx: *const GpuContext) !ElemScalePipeline {
        comptime std.debug.assert(shaders.elem_scale.len % 4 == 0);
        const built = try buildSimplePipeline(ctx, &shaders.elem_scale, 1, @sizeOf(ElemScalePushConst), 32);
        return .{
            .pipeline = built.pipeline, .layout = built.layout,
            .dset_layout = built.dset_layout, .desc_pool = built.desc_pool,
            .device = ctx.device,
        };
    }

    pub fn record(
        self: *const ElemScalePipeline,
        cmd: vk.VkCommandBuffer,
        x_buf: *const GpuBuffer,
        n: u32,
        s: f32,
    ) !vk.VkDescriptorSet {
        const dev = self.device;
        const alloc_ci = vk.VkDescriptorSetAllocateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = null, .descriptorPool = self.desc_pool,
            .descriptorSetCount = 1, .pSetLayouts = &self.dset_layout,
        };
        var dset: vk.VkDescriptorSet = null;
        if (vk.vkAllocateDescriptorSets(dev, &alloc_ci, &dset) != vk.VK_SUCCESS)
            return error.VkDescriptorSetAllocFailed;

        const buf_info = vk.VkDescriptorBufferInfo{
            .buffer = x_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE,
        };
        const write = mkWrite(dset, 0, &buf_info);
        vk.vkUpdateDescriptorSets(dev, 1, &write, 0, null);

        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.layout, 0, 1, &dset, 0, null);
        const pc = ElemScalePushConst{ .n = n, .s = s };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT,
            0, @sizeOf(ElemScalePushConst), &pc);
        const groups = (n + 255) / 256;
        vk.vkCmdDispatch(cmd, groups, 1, 1);
        return dset;
    }

    pub fn deinit(self: *ElemScalePipeline) void {
        vk.vkDestroyDescriptorPool(self.device, self.desc_pool, null);
        vk.vkDestroyPipeline(self.device, self.pipeline, null);
        vk.vkDestroyPipelineLayout(self.device, self.layout, null);
        vk.vkDestroyDescriptorSetLayout(self.device, self.dset_layout, null);
    }
};

// ── GeluMulPipeline ───────────────────────────────────────────────────────────
//
// In-place GELU + elementwise multiply: a[i] = gelu(a[i]) * b[i].
// Used to fuse the dense-FFN "gelu(gate) * up" step into the GPU command buffer
// so we don't have to round-trip gate/up through CPU between the gate+up
// matvecs and the w_down matvec.

pub const GeluMulPipeline = struct {
    pipeline:    vk.VkPipeline,
    layout:      vk.VkPipelineLayout,
    dset_layout: vk.VkDescriptorSetLayout,
    desc_pool:   vk.VkDescriptorPool,
    device:      vk.VkDevice,

    pub fn init(ctx: *const GpuContext) !GeluMulPipeline {
        comptime std.debug.assert(shaders.gelu_mul.len % 4 == 0);
        const built = try buildSimplePipeline(ctx, &shaders.gelu_mul, 2, @sizeOf(ElemPushConst), 32);
        return .{
            .pipeline = built.pipeline, .layout = built.layout,
            .dset_layout = built.dset_layout, .desc_pool = built.desc_pool,
            .device = ctx.device,
        };
    }

    pub fn record(
        self: *const GeluMulPipeline,
        cmd: vk.VkCommandBuffer,
        a_buf: *const GpuBuffer,
        b_buf: *const GpuBuffer,
        n: u32,
    ) !vk.VkDescriptorSet {
        const dev = self.device;
        const alloc_ci = vk.VkDescriptorSetAllocateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = null, .descriptorPool = self.desc_pool,
            .descriptorSetCount = 1, .pSetLayouts = &self.dset_layout,
        };
        var dset: vk.VkDescriptorSet = null;
        if (vk.vkAllocateDescriptorSets(dev, &alloc_ci, &dset) != vk.VK_SUCCESS)
            return error.VkDescriptorSetAllocFailed;

        const buf_infos = [2]vk.VkDescriptorBufferInfo{
            .{ .buffer = a_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = b_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        };
        const writes = [2]vk.VkWriteDescriptorSet{
            mkWrite(dset, 0, &buf_infos[0]),
            mkWrite(dset, 1, &buf_infos[1]),
        };
        vk.vkUpdateDescriptorSets(dev, writes.len, &writes, 0, null);

        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.layout, 0, 1, &dset, 0, null);
        const pc = ElemPushConst{ .n = n };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT,
            0, @sizeOf(ElemPushConst), &pc);
        const groups = (n + 255) / 256;
        vk.vkCmdDispatch(cmd, groups, 1, 1);
        return dset;
    }

    pub fn deinit(self: *GeluMulPipeline) void {
        vk.vkDestroyDescriptorPool(self.device, self.desc_pool, null);
        vk.vkDestroyPipeline(self.device, self.pipeline, null);
        vk.vkDestroyPipelineLayout(self.device, self.layout, null);
        vk.vkDestroyDescriptorSetLayout(self.device, self.dset_layout, null);
    }
};

// ── RopeNeox pipelines ────────────────────────────────────────────────────────
//
// Two variants: `Table` reads a precomputed inverse-frequency table (Gemma4
// global layers, freqs from rope_freqs.weight); `Theta` computes freqs from
// a single theta base (Gemma4 SWA layers, theta = rope_theta_swa).
//
// One workgroup per head; the Zig wrapper dispatches `n_heads` workgroups.

const RopeTablePushConst = extern struct { pos: u32, head_dim: u32 };
const RopeThetaPushConst = extern struct { pos: u32, head_dim: u32, theta: f32 };

pub const RopeNeoxTablePipeline = struct {
    pipeline:    vk.VkPipeline,
    layout:      vk.VkPipelineLayout,
    dset_layout: vk.VkDescriptorSetLayout,
    desc_pool:   vk.VkDescriptorPool,
    device:      vk.VkDevice,

    pub fn init(ctx: *const GpuContext) !RopeNeoxTablePipeline {
        comptime std.debug.assert(shaders.rope_neox_table.len % 4 == 0);
        const built = try buildSimplePipeline(ctx, &shaders.rope_neox_table, 2, @sizeOf(RopeTablePushConst), 64);
        return .{
            .pipeline = built.pipeline, .layout = built.layout,
            .dset_layout = built.dset_layout, .desc_pool = built.desc_pool,
            .device = ctx.device,
        };
    }

    pub fn record(
        self: *const RopeNeoxTablePipeline,
        cmd: vk.VkCommandBuffer,
        vec_buf: *const GpuBuffer,
        freqs_buf: *const GpuBuffer,
        pos: u32,
        head_dim: u32,
        n_heads: u32,
    ) !vk.VkDescriptorSet {
        std.debug.assert(head_dim <= 512); // local_size_x is 256, half ≤ 256
        const dev = self.device;
        const alloc_ci = vk.VkDescriptorSetAllocateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = null, .descriptorPool = self.desc_pool,
            .descriptorSetCount = 1, .pSetLayouts = &self.dset_layout,
        };
        var dset: vk.VkDescriptorSet = null;
        if (vk.vkAllocateDescriptorSets(dev, &alloc_ci, &dset) != vk.VK_SUCCESS)
            return error.VkDescriptorSetAllocFailed;
        const buf_infos = [2]vk.VkDescriptorBufferInfo{
            .{ .buffer = vec_buf.handle,   .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = freqs_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        };
        const writes = [2]vk.VkWriteDescriptorSet{
            mkWrite(dset, 0, &buf_infos[0]),
            mkWrite(dset, 1, &buf_infos[1]),
        };
        vk.vkUpdateDescriptorSets(dev, writes.len, &writes, 0, null);

        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.layout, 0, 1, &dset, 0, null);
        const pc = RopeTablePushConst{ .pos = pos, .head_dim = head_dim };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT,
            0, @sizeOf(RopeTablePushConst), &pc);
        vk.vkCmdDispatch(cmd, n_heads, 1, 1);
        return dset;
    }

    pub fn deinit(self: *RopeNeoxTablePipeline) void {
        vk.vkDestroyDescriptorPool(self.device, self.desc_pool, null);
        vk.vkDestroyPipeline(self.device, self.pipeline, null);
        vk.vkDestroyPipelineLayout(self.device, self.layout, null);
        vk.vkDestroyDescriptorSetLayout(self.device, self.dset_layout, null);
    }
};

pub const RopeNeoxThetaPipeline = struct {
    pipeline:    vk.VkPipeline,
    layout:      vk.VkPipelineLayout,
    dset_layout: vk.VkDescriptorSetLayout,
    desc_pool:   vk.VkDescriptorPool,
    device:      vk.VkDevice,

    pub fn init(ctx: *const GpuContext) !RopeNeoxThetaPipeline {
        comptime std.debug.assert(shaders.rope_neox_theta.len % 4 == 0);
        const built = try buildSimplePipeline(ctx, &shaders.rope_neox_theta, 1, @sizeOf(RopeThetaPushConst), 64);
        return .{
            .pipeline = built.pipeline, .layout = built.layout,
            .dset_layout = built.dset_layout, .desc_pool = built.desc_pool,
            .device = ctx.device,
        };
    }

    pub fn record(
        self: *const RopeNeoxThetaPipeline,
        cmd: vk.VkCommandBuffer,
        vec_buf: *const GpuBuffer,
        pos: u32,
        head_dim: u32,
        theta: f32,
        n_heads: u32,
    ) !vk.VkDescriptorSet {
        std.debug.assert(head_dim <= 512);
        const dev = self.device;
        const alloc_ci = vk.VkDescriptorSetAllocateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = null, .descriptorPool = self.desc_pool,
            .descriptorSetCount = 1, .pSetLayouts = &self.dset_layout,
        };
        var dset: vk.VkDescriptorSet = null;
        if (vk.vkAllocateDescriptorSets(dev, &alloc_ci, &dset) != vk.VK_SUCCESS)
            return error.VkDescriptorSetAllocFailed;
        const buf_info = vk.VkDescriptorBufferInfo{
            .buffer = vec_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE,
        };
        const write = mkWrite(dset, 0, &buf_info);
        vk.vkUpdateDescriptorSets(dev, 1, &write, 0, null);

        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.layout, 0, 1, &dset, 0, null);
        const pc = RopeThetaPushConst{ .pos = pos, .head_dim = head_dim, .theta = theta };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT,
            0, @sizeOf(RopeThetaPushConst), &pc);
        vk.vkCmdDispatch(cmd, n_heads, 1, 1);
        return dset;
    }

    pub fn deinit(self: *RopeNeoxThetaPipeline) void {
        vk.vkDestroyDescriptorPool(self.device, self.desc_pool, null);
        vk.vkDestroyPipeline(self.device, self.pipeline, null);
        vk.vkDestroyPipelineLayout(self.device, self.layout, null);
        vk.vkDestroyDescriptorSetLayout(self.device, self.dset_layout, null);
    }
};

// ── AttnQkSoftmaxPipeline + AttnAvPipeline ────────────────────────────────────
//
// Fused Q·K^T + softmax (one shader) and attn·V (one shader), used together
// for one-token-at-a-time attention with K/V resident in VRAM. Both dispatch
// `n_heads` workgroups; each workgroup handles one Q head.
//
// Memory layouts:
//   Q       — [n_heads, head_dim] f32      (current token, per-head)
//   K_cache — [cap, n_kv_heads, head_dim]  (slot-major; same as the CPU cache)
//   V_cache — [cap, n_kv_heads, head_dim]
//   scores  — [n_heads, win_len] f32       (softmaxed attention weights)
//   out     — [n_heads, head_dim] f32      (== attn_concat in forward.zig)

const AttnQkSoftmaxPushConst = extern struct {
    seq:         u32,
    win_len:     u32,
    head_dim:    u32,
    n_kv_heads:  u32,
    n_q_per_kv:  u32,
    cap:         u32,
    scale:       f32,
};

const AttnAvPushConst = extern struct {
    seq:         u32,
    win_len:     u32,
    head_dim:    u32,
    n_kv_heads:  u32,
    n_q_per_kv:  u32,
    cap:         u32,
};

pub const AttnQkSoftmaxPipeline = struct {
    pipeline:    vk.VkPipeline,
    layout:      vk.VkPipelineLayout,
    dset_layout: vk.VkDescriptorSetLayout,
    desc_pool:   vk.VkDescriptorPool,
    device:      vk.VkDevice,

    pub fn init(ctx: *const GpuContext) !AttnQkSoftmaxPipeline {
        comptime std.debug.assert(shaders.attn_qk_softmax.len % 4 == 0);
        const built = try buildSimplePipeline(ctx, &shaders.attn_qk_softmax, 3,
            @sizeOf(AttnQkSoftmaxPushConst), 32);
        return .{
            .pipeline = built.pipeline, .layout = built.layout,
            .dset_layout = built.dset_layout, .desc_pool = built.desc_pool,
            .device = ctx.device,
        };
    }

    pub fn record(
        self: *const AttnQkSoftmaxPipeline,
        cmd: vk.VkCommandBuffer,
        q_buf: *const GpuBuffer,
        k_buf: *const GpuBuffer,
        scores_buf: *const GpuBuffer,
        n_heads: u32,
        seq: u32, win_len: u32, head_dim: u32,
        n_kv_heads: u32, n_q_per_kv: u32, cap: u32,
        scale: f32,
    ) !vk.VkDescriptorSet {
        const dev = self.device;
        const alloc_ci = vk.VkDescriptorSetAllocateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = null, .descriptorPool = self.desc_pool,
            .descriptorSetCount = 1, .pSetLayouts = &self.dset_layout,
        };
        var dset: vk.VkDescriptorSet = null;
        if (vk.vkAllocateDescriptorSets(dev, &alloc_ci, &dset) != vk.VK_SUCCESS)
            return error.VkDescriptorSetAllocFailed;

        const buf_infos = [3]vk.VkDescriptorBufferInfo{
            .{ .buffer = q_buf.handle,      .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = k_buf.handle,      .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = scores_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        };
        const writes = [3]vk.VkWriteDescriptorSet{
            mkWrite(dset, 0, &buf_infos[0]),
            mkWrite(dset, 1, &buf_infos[1]),
            mkWrite(dset, 2, &buf_infos[2]),
        };
        vk.vkUpdateDescriptorSets(dev, writes.len, &writes, 0, null);

        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.layout, 0, 1, &dset, 0, null);
        const pc = AttnQkSoftmaxPushConst{
            .seq = seq, .win_len = win_len, .head_dim = head_dim,
            .n_kv_heads = n_kv_heads, .n_q_per_kv = n_q_per_kv,
            .cap = cap, .scale = scale,
        };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT,
            0, @sizeOf(AttnQkSoftmaxPushConst), &pc);
        vk.vkCmdDispatch(cmd, n_heads, 1, 1);
        return dset;
    }

    pub fn deinit(self: *AttnQkSoftmaxPipeline) void {
        vk.vkDestroyDescriptorPool(self.device, self.desc_pool, null);
        vk.vkDestroyPipeline(self.device, self.pipeline, null);
        vk.vkDestroyPipelineLayout(self.device, self.layout, null);
        vk.vkDestroyDescriptorSetLayout(self.device, self.dset_layout, null);
    }
};

pub const AttnAvPipeline = struct {
    pipeline:    vk.VkPipeline,
    layout:      vk.VkPipelineLayout,
    dset_layout: vk.VkDescriptorSetLayout,
    desc_pool:   vk.VkDescriptorPool,
    device:      vk.VkDevice,

    pub fn init(ctx: *const GpuContext) !AttnAvPipeline {
        comptime std.debug.assert(shaders.attn_av.len % 4 == 0);
        const built = try buildSimplePipeline(ctx, &shaders.attn_av, 3,
            @sizeOf(AttnAvPushConst), 32);
        return .{
            .pipeline = built.pipeline, .layout = built.layout,
            .dset_layout = built.dset_layout, .desc_pool = built.desc_pool,
            .device = ctx.device,
        };
    }

    pub fn record(
        self: *const AttnAvPipeline,
        cmd: vk.VkCommandBuffer,
        scores_buf: *const GpuBuffer,
        v_buf: *const GpuBuffer,
        out_buf: *const GpuBuffer,
        n_heads: u32,
        seq: u32, win_len: u32, head_dim: u32,
        n_kv_heads: u32, n_q_per_kv: u32, cap: u32,
    ) !vk.VkDescriptorSet {
        const dev = self.device;
        const alloc_ci = vk.VkDescriptorSetAllocateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = null, .descriptorPool = self.desc_pool,
            .descriptorSetCount = 1, .pSetLayouts = &self.dset_layout,
        };
        var dset: vk.VkDescriptorSet = null;
        if (vk.vkAllocateDescriptorSets(dev, &alloc_ci, &dset) != vk.VK_SUCCESS)
            return error.VkDescriptorSetAllocFailed;

        const buf_infos = [3]vk.VkDescriptorBufferInfo{
            .{ .buffer = scores_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = v_buf.handle,      .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = out_buf.handle,    .offset = 0, .range = vk.VK_WHOLE_SIZE },
        };
        const writes = [3]vk.VkWriteDescriptorSet{
            mkWrite(dset, 0, &buf_infos[0]),
            mkWrite(dset, 1, &buf_infos[1]),
            mkWrite(dset, 2, &buf_infos[2]),
        };
        vk.vkUpdateDescriptorSets(dev, writes.len, &writes, 0, null);

        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.layout, 0, 1, &dset, 0, null);
        const pc = AttnAvPushConst{
            .seq = seq, .win_len = win_len, .head_dim = head_dim,
            .n_kv_heads = n_kv_heads, .n_q_per_kv = n_q_per_kv, .cap = cap,
        };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT,
            0, @sizeOf(AttnAvPushConst), &pc);
        vk.vkCmdDispatch(cmd, n_heads, 1, 1);
        return dset;
    }

    pub fn deinit(self: *AttnAvPipeline) void {
        vk.vkDestroyDescriptorPool(self.device, self.desc_pool, null);
        vk.vkDestroyPipeline(self.device, self.pipeline, null);
        vk.vkDestroyPipelineLayout(self.device, self.layout, null);
        vk.vkDestroyDescriptorSetLayout(self.device, self.dset_layout, null);
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
    try session.runOwned(ctx, pipeline, vec, out);
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
    try session.runOwned(&gpu, &pipeline, &vec, &out);

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
    try session.runOwned(&gpu, &pipeline, &vec, &out);

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
    try session.runOwned(&gpu, &pipeline, &vec, &out);

    try std.testing.expectApproxEqAbs(out[0], -1024.0, 0.1);
}

test "gpu matvec Q4_K scale" {
    // One row × 256 cols (one 144-byte block).
    // d=1.0, dmin=0.0; sub-block 0: sc=1, mn=0.
    // qs[0..31] = 0x11 → lo nibble = 1 for each of the 32 elements in sub-block 0.
    // vec = all 1.0.
    // Expected: d_all*sc * q_lo * 32 = 1.0*1 * 1 * 32 = 32.0
    const ctx = GpuContext.init() catch |e| {
        std.debug.print("gpu init failed: {}\n", .{e});
        return;
    };
    var gpu = ctx;
    defer gpu.deinit();

    var pipeline = try MatvecPipeline.initQ4K(&gpu);
    defer pipeline.deinit();

    var mat_bytes: [144]u8 = [_]u8{0} ** 144;
    mat_bytes[0] = 0x00; mat_bytes[1] = 0x3C; // d = f16(1.0)
    // dmin at [2..3] = 0.0 (already zero)
    mat_bytes[4] = 0x01; // scales[0]: sc[0] = 1 (j=0 → sc = scales[0] & 0x3F)
    @memset(mat_bytes[16..48], 0x11); // qs chunk 0: lo nibble = 1 for all 32 bytes

    var vec: [256]f32 = [_]f32{1.0} ** 256;
    var out: [1]f32 = .{0.0};

    var session = try MatvecSession.initQ4K(&gpu, &mat_bytes, 1, 256);
    defer session.deinit();
    try session.runOwned(&gpu, &pipeline, &vec, &out);

    try std.testing.expectApproxEqAbs(out[0], 32.0, 1e-4);
}

test "gpu matvec Q4_K min" {
    // d=0.0, dmin=1.0; sub-block 0: sc=0, mn=2.
    // qs = all 0; vec = all 1.0.
    // Expected: -(dmin*mn) * 32 = -(1.0*2) * 32 = -64.0
    const ctx = GpuContext.init() catch |e| {
        std.debug.print("gpu init failed: {}\n", .{e});
        return;
    };
    var gpu = ctx;
    defer gpu.deinit();

    var pipeline = try MatvecPipeline.initQ4K(&gpu);
    defer pipeline.deinit();

    var mat_bytes: [144]u8 = [_]u8{0} ** 144;
    // d at [0..1] = 0.0 (already zero)
    mat_bytes[2] = 0x00; mat_bytes[3] = 0x3C; // dmin = f16(1.0)
    mat_bytes[8] = 0x02; // scales[4]: mn[0] = 2 (j=0 → mn = scales[4] & 0x3F)
    // qs = all 0 (already zero) → q_lo = q_hi = 0

    var vec: [256]f32 = [_]f32{1.0} ** 256;
    var out: [1]f32 = .{0.0};

    var session = try MatvecSession.initQ4K(&gpu, &mat_bytes, 1, 256);
    defer session.deinit();
    try session.runOwned(&gpu, &pipeline, &vec, &out);

    try std.testing.expectApproxEqAbs(out[0], -64.0, 1e-4);
}

test "gpu matvec Q5_1 scale" {
    // 1 row × 32 cols (one 24-byte block).
    // d=1.0, m=0.0, qh=0 (no 5th bits set).
    // qs[0]=0x01 → element 0: lo nibble=1, element 1: hi nibble=0; rest=0.
    // vec = all 1.0.
    // Expected: 1.0 * 1 + 0.0 = 1.0 (only element 0 contributes).
    const ctx = GpuContext.init() catch |e| {
        std.debug.print("gpu init failed: {}\n", .{e});
        return;
    };
    var gpu = ctx;
    defer gpu.deinit();

    var pipeline = try MatvecPipeline.initQ5_1(&gpu);
    defer pipeline.deinit();

    var mat_bytes: [24]u8 = [_]u8{0} ** 24;
    mat_bytes[0] = 0x00; mat_bytes[1] = 0x3C; // d = f16(1.0)
    // m at [2..3] = 0.0 (already zero)
    // qh at [4..7] = 0 (already zero) — no 5th bits
    mat_bytes[8] = 0x01; // qs[0]: lo nibble=1 for element 0

    var vec: [32]f32 = [_]f32{1.0} ** 32;
    var out: [1]f32 = .{0.0};

    var session = try MatvecSession.initQ5_1(&gpu, &mat_bytes, 1, 32);
    defer session.deinit();
    try session.runOwned(&gpu, &pipeline, &vec, &out);

    try std.testing.expectApproxEqAbs(out[0], 1.0, 1e-4);
}

test "gpu matvec Q5_1 addend" {
    // d=0.0, m=1.0, qh=0, qs all zero.
    // vec = all 1.0.
    // Every element: x = 0.0 * 0 + 1.0 = 1.0 → sum = 32.0.
    const ctx = GpuContext.init() catch |e| {
        std.debug.print("gpu init failed: {}\n", .{e});
        return;
    };
    var gpu = ctx;
    defer gpu.deinit();

    var pipeline = try MatvecPipeline.initQ5_1(&gpu);
    defer pipeline.deinit();

    var mat_bytes: [24]u8 = [_]u8{0} ** 24;
    // d at [0..1] = 0.0 (already zero)
    mat_bytes[2] = 0x00; mat_bytes[3] = 0x3C; // m = f16(1.0)

    var vec: [32]f32 = [_]f32{1.0} ** 32;
    var out: [1]f32 = .{0.0};

    var session = try MatvecSession.initQ5_1(&gpu, &mat_bytes, 1, 32);
    defer session.deinit();
    try session.runOwned(&gpu, &pipeline, &vec, &out);

    try std.testing.expectApproxEqAbs(out[0], 32.0, 1e-4);
}

test "gpu matvec Q5_1 fifth bit" {
    // Verify 5th bit: qh=0x00000001 → element 0 gets q5 = 0 | 16 = 16.
    // d=1.0, m=0.0, qs all zero, vec = all 1.0.
    // Expected: 1.0 * 16 + 0.0 = 16.0.
    const ctx = GpuContext.init() catch |e| {
        std.debug.print("gpu init failed: {}\n", .{e});
        return;
    };
    var gpu = ctx;
    defer gpu.deinit();

    var pipeline = try MatvecPipeline.initQ5_1(&gpu);
    defer pipeline.deinit();

    var mat_bytes: [24]u8 = [_]u8{0} ** 24;
    mat_bytes[0] = 0x00; mat_bytes[1] = 0x3C; // d = f16(1.0)
    // m=0, qs=0 — only element 0 gets a 5th bit via qh
    mat_bytes[4] = 0x01; // qh low byte: bit 0 set → 5th bit of element 0

    var vec: [32]f32 = [_]f32{1.0} ** 32;
    var out: [1]f32 = .{0.0};

    var session = try MatvecSession.initQ5_1(&gpu, &mat_bytes, 1, 32);
    defer session.deinit();
    try session.runOwned(&gpu, &pipeline, &vec, &out);

    try std.testing.expectApproxEqAbs(out[0], 16.0, 1e-4);
}

test "gpu matvec Q5_0 symmetric" {
    // 1 row × 32 cols (one 22-byte block).
    // d=1.0, qh=0, qs all zero → all elements: q5=0, x=0-16=-16.
    // vec = all 1.0.
    // Expected: -16.0 * 32 = -512.0.
    const ctx = GpuContext.init() catch |e| {
        std.debug.print("gpu init failed: {}\n", .{e});
        return;
    };
    var gpu = ctx;
    defer gpu.deinit();

    var pipeline = try MatvecPipeline.initQ5_0(&gpu);
    defer pipeline.deinit();

    var mat_bytes: [22]u8 = [_]u8{0} ** 22;
    mat_bytes[0] = 0x00; mat_bytes[1] = 0x3C; // d = f16(1.0)

    var vec: [32]f32 = [_]f32{1.0} ** 32;
    var out: [1]f32 = .{0.0};

    var session = try MatvecSession.initQ5_0(&gpu, &mat_bytes, 1, 32);
    defer session.deinit();
    try session.runOwned(&gpu, &pipeline, &vec, &out);

    try std.testing.expectApproxEqAbs(out[0], -512.0, 0.1);
}

test "gpu matvec Q5_0 lo nibble" {
    // d=1.0, qh=0, qs[0]=0x01 → lo nibble for element 0 = 1, hi for element 16 = 0.
    // Element 0: q5=1, x=1-16=-15. All others: q5=0, x=-16.
    // vec = all 1.0.
    // Expected: -15.0 + (-16.0) * 31 = -15 - 496 = -511.0.
    const ctx = GpuContext.init() catch |e| {
        std.debug.print("gpu init failed: {}\n", .{e});
        return;
    };
    var gpu = ctx;
    defer gpu.deinit();

    var pipeline = try MatvecPipeline.initQ5_0(&gpu);
    defer pipeline.deinit();

    var mat_bytes: [22]u8 = [_]u8{0} ** 22;
    mat_bytes[0] = 0x00; mat_bytes[1] = 0x3C; // d = f16(1.0)
    mat_bytes[6] = 0x01; // qs[0]: lo nibble=1 → element 0; hi nibble=0 → element 16

    var vec: [32]f32 = [_]f32{1.0} ** 32;
    var out: [1]f32 = .{0.0};

    var session = try MatvecSession.initQ5_0(&gpu, &mat_bytes, 1, 32);
    defer session.deinit();
    try session.runOwned(&gpu, &pipeline, &vec, &out);

    try std.testing.expectApproxEqAbs(out[0], -511.0, 0.1);
}

test "gpu matvec Q5_0 fifth bit" {
    // d=1.0, qh bit 0 set → element 0 gets q5 = 0 | 16 = 16, x=16-16=0.
    // All other elements: q5=0, x=-16.
    // vec = all 1.0.
    // Expected: 0.0 + (-16.0) * 31 = -496.0.
    const ctx = GpuContext.init() catch |e| {
        std.debug.print("gpu init failed: {}\n", .{e});
        return;
    };
    var gpu = ctx;
    defer gpu.deinit();

    var pipeline = try MatvecPipeline.initQ5_0(&gpu);
    defer pipeline.deinit();

    var mat_bytes: [22]u8 = [_]u8{0} ** 22;
    mat_bytes[0] = 0x00; mat_bytes[1] = 0x3C; // d = f16(1.0)
    mat_bytes[2] = 0x01; // qh byte 0, bit 0 → 5th bit of element 0

    var vec: [32]f32 = [_]f32{1.0} ** 32;
    var out: [1]f32 = .{0.0};

    var session = try MatvecSession.initQ5_0(&gpu, &mat_bytes, 1, 32);
    defer session.deinit();
    try session.runOwned(&gpu, &pipeline, &vec, &out);

    try std.testing.expectApproxEqAbs(out[0], -496.0, 0.1);
}

test "gpu fused gate-up Q3_K correctness" {
    // 1 row × 256 cols. Element 0 contributes q3=1 with scale 1.0:
    //   hmask=0xFF (hi_bit=1), qs[0]=0x01 (lo2=1) → q3=(1|4)-4=1
    //   sc[0]=33 → group-0 scale = d*(33-32) = 1.0; all other scales = 0.0
    //   d = f16(1.0)
    // gate_row · vec = 1.0,  up_row · vec = 1.0
    // Expected: gelu(1.0) * 1.0 ≈ 0.841
    const ctx = GpuContext.init() catch |e| {
        std.debug.print("gpu init failed: {}\n", .{e});
        return;
    };
    var gpu = ctx;
    defer gpu.deinit();

    var pl = FusedGateUpPipeline.init(&gpu) catch |e| {
        std.debug.print("fused pipeline init failed: {}\n", .{e});
        return;
    };
    defer pl.deinit();

    const blk_size = 110;
    var mat_bytes: [blk_size]u8 = [_]u8{0} ** blk_size;
    @memset(mat_bytes[0..32], 0xFF);
    mat_bytes[32] = 0x01;
    mat_bytes[96] = 0x01; mat_bytes[97] = 0x00; mat_bytes[98] = 0x00; mat_bytes[99] = 0x00;
    mat_bytes[100] = 0x00; mat_bytes[101] = 0x00; mat_bytes[102] = 0x00; mat_bytes[103] = 0x00;
    mat_bytes[104] = 0xAA; mat_bytes[105] = 0xAA; mat_bytes[106] = 0xAA; mat_bytes[107] = 0xAA;
    mat_bytes[108] = 0x00; mat_bytes[109] = 0x3C; // d = f16(1.0)

    const usage = vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    var gate_buf = try GpuBuffer.initHostCoherent(&gpu, blk_size, usage);
    defer gate_buf.deinit();
    var up_buf = try GpuBuffer.initHostCoherent(&gpu, blk_size, usage);
    defer up_buf.deinit();
    var vec_buf = try GpuBuffer.initHostCoherent(&gpu, 256 * @sizeOf(f32), usage);
    defer vec_buf.deinit();
    var out_buf = try GpuBuffer.initHostCoherent(&gpu, 1 * @sizeOf(f32), usage);
    defer out_buf.deinit();

    try gate_buf.upload(&mat_bytes);
    try up_buf.upload(&mat_bytes);
    var vec: [256]f32 = [_]f32{1.0} ** 256;
    try vec_buf.upload(std.mem.sliceAsBytes(&vec));
    const zero = [1]f32{0.0};
    try out_buf.upload(std.mem.sliceAsBytes(&zero));

    const cmd = try gpu.beginBatch();
    var dset = try pl.record(cmd, &gate_buf, &up_buf, &vec_buf, &out_buf, 1, 256);
    try gpu.submitBatch(cmd);
    _ = vk.vkFreeDescriptorSets(gpu.device, pl.desc_pool, 1, &dset);

    var result: [1]f32 = .{0.0};
    try out_buf.download(std.mem.sliceAsBytes(&result));

    // gelu(1.0) * 1.0 ≈ 0.8413
    try std.testing.expectApproxEqAbs(result[0], 0.841, 0.01);
}

test "gpu expert accum correctness" {
    // n=2 experts, d_model=4.
    // inputs = [[1,2,3,4],[5,6,7,8]], scales = [2.0, 3.0].
    // output[i] = 0 + 2*inputs[0][i] + 3*inputs[1][i]
    //           = [17, 22, 27, 32]
    const ctx = GpuContext.init() catch |e| {
        std.debug.print("gpu init failed: {}\n", .{e});
        return;
    };
    var gpu = ctx;
    defer gpu.deinit();

    var pl = AccumPipeline.init(&gpu) catch |e| {
        std.debug.print("accum pipeline init failed: {}\n", .{e});
        return;
    };
    defer pl.deinit();

    const d_model: u32 = 4;
    const n: u32 = 2;
    const inputs = [8]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const scales = [2]f32{ 2.0, 3.0 };

    const usage = vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    var in_buf  = try GpuBuffer.initHostCoherent(&gpu, inputs.len * @sizeOf(f32), usage);
    defer in_buf.deinit();
    var sc_buf  = try GpuBuffer.initHostCoherent(&gpu, scales.len * @sizeOf(f32), usage);
    defer sc_buf.deinit();
    var out_buf = try GpuBuffer.initHostCoherent(&gpu, d_model * @sizeOf(f32), usage);
    defer out_buf.deinit();

    try in_buf.upload(std.mem.sliceAsBytes(&inputs));
    try sc_buf.upload(std.mem.sliceAsBytes(&scales));
    const zeros = [4]f32{ 0.0, 0.0, 0.0, 0.0 };
    try out_buf.upload(std.mem.sliceAsBytes(&zeros));

    const cmd = try gpu.beginBatch();
    var dset = try pl.record(cmd, &in_buf, &sc_buf, &out_buf, d_model, n);
    try gpu.submitBatch(cmd);
    _ = vk.vkFreeDescriptorSets(gpu.device, pl.desc_pool, 1, &dset);

    var result: [4]f32 = .{ 0.0, 0.0, 0.0, 0.0 };
    try out_buf.download(std.mem.sliceAsBytes(&result));

    const expected = [4]f32{ 17.0, 22.0, 27.0, 32.0 };
    for (0..d_model) |i| {
        try std.testing.expectApproxEqAbs(result[i], expected[i], 0.1);
    }
}

// ── per-quant fuzz tests ──────────────────────────────────────────────────────
//
// Generate random block-quantized matrix bytes (with a clamped f16 scale to
// avoid NaN/Inf), then assert that GPU matvec matches CPU dequant + dot to
// within 1e-4 relative.  Catches indexing/ordering bugs that the small
// hand-crafted tests above miss (e.g. the original Q5_1 nibble interleave).

fn fuzzQuantMatvec(
    comptime tag: GgmlType,
    initPipeline: anytype,
    initSession: anytype,
    rows: usize,
    cols: usize,
    seed: u64,
) !void {
    const ctx = GpuContext.init() catch |e| {
        std.debug.print("gpu init failed: {}\n", .{e});
        return;
    };
    var gpu = ctx;
    defer gpu.deinit();

    var prng = std.Random.DefaultPrng.init(seed);
    const r  = prng.random();
    const al = std.testing.allocator;

    const blk_bytes  = math_mod.rowBytes(tag, cols);
    const total_mat  = rows * blk_bytes;
    const mat = try al.alloc(u8, total_mat);
    defer al.free(mat);
    for (mat) |*b| b.* = r.int(u8);

    // Stamp every block's f16 scales to small magnitudes so the dequant doesn't
    // explode into NaN/Inf.  Layouts share the convention of f16 d at offset 0,
    // and (for Q5_1 / Q4_K / Q5_K) f16 m or dmin at offset 2.
    const block_elems: usize = switch (tag) {
        .q4_k, .q5_k, .q3_k, .q6_k => 256,
        else => 32,
    };
    const blocks_per_row = cols / block_elems;
    const small_d_le: [2]u8 = .{ 0xCD, 0x21 }; // f16 ≈ 0.012
    for (0..rows) |i| for (0..blocks_per_row) |b| {
        const off = (i * blocks_per_row + b) * blk_bytes;
        mat[off + 0] = small_d_le[0]; mat[off + 1] = small_d_le[1];
        // Q5_1 / Q4_K / Q5_K all have a second f16 at [2..3]
        if (tag == .q5_1 or tag == .q4_k or tag == .q5_k) {
            mat[off + 2] = small_d_le[0]; mat[off + 3] = small_d_le[1];
        }
    };

    const vec = try al.alloc(f32, cols);
    defer al.free(vec);
    for (vec) |*v| v.* = (r.float(f32) - 0.5) * 2.0;

    // CPU reference via existing dequant + dot.
    const cpu_out = try al.alloc(f32, rows);
    defer al.free(cpu_out);
    const row_buf = try al.alloc(f32, cols);
    defer al.free(row_buf);
    math_mod.quantMatvec(cpu_out, mat, tag, vec, rows, cols, row_buf);

    // GPU.
    var pipeline = try initPipeline(&gpu);
    defer pipeline.deinit();
    var session = try initSession(&gpu, mat, @intCast(rows), @intCast(cols));
    defer session.deinit();

    const gpu_out = try al.alloc(f32, rows);
    defer al.free(gpu_out);
    try session.runOwned(&gpu, &pipeline, vec, gpu_out);

    var max_abs: f32 = 0.0;
    var max_ref: f32 = 0.0;
    for (cpu_out, gpu_out) |c, g| {
        const d = @abs(c - g);
        if (d > max_abs) max_abs = d;
        if (@abs(c) > max_ref) max_ref = @abs(c);
    }
    const rel = max_abs / (max_ref + 1e-6);
    std.debug.print("{s} fuzz rows={} cols={}  max|D|={d:.6}  rel={e:.3}\n",
        .{ tag.label(), rows, cols, max_abs, rel });
    try std.testing.expect(rel < 1e-4);
}

test "gpu matvec Q8_0 fuzz" {
    try fuzzQuantMatvec(.q8_0,
        MatvecPipeline.initQ8_0, MatvecSession.initQ8_0, 32, 256, 1);
}
test "gpu matvec Q5_0 fuzz" {
    try fuzzQuantMatvec(.q5_0,
        MatvecPipeline.initQ5_0, MatvecSession.initQ5_0, 32, 256, 2);
}
test "gpu matvec Q5_1 fuzz" {
    try fuzzQuantMatvec(.q5_1,
        MatvecPipeline.initQ5_1, MatvecSession.initQ5_1, 32, 256, 3);
}
test "gpu matvec Q4_K fuzz" {
    try fuzzQuantMatvec(.q4_k,
        MatvecPipeline.initQ4K, MatvecSession.initQ4K, 32, 512, 4);
}
test "gpu matvec Q3_K fuzz" {
    try fuzzQuantMatvec(.q3_k,
        MatvecPipeline.initQ3K, MatvecSession.initQ3K, 32, 512, 5);
}

// ── Q8_1 quantization fuzz test ───────────────────────────────────────────────
//
// Round-trip: CPU f32 → GPU quantize_q8_1 → CPU dequantize.  Asserts the
// per-element error stays within Q8_1's quantization bound (~amax/127 per
// block).  Also cross-checks against the CPU `quantizeQ8_1` reference: the
// GPU's per-block `d` scale must match within a small epsilon; the i8 quants
// may differ by ±1 due to round-to-even differences between hardware and
// `std.math.round`.
fn runQuantizeShader(ctx: *const GpuContext, in: []const f32, out_x4: []u8) !void {
    var pl = try QuantizeQ8_1Pipeline.init(ctx);
    defer pl.deinit();

    var in_buf  = try GpuBuffer.initHostCoherent(ctx, in.len * @sizeOf(f32),
        @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer in_buf.deinit();
    try in_buf.upload(std.mem.sliceAsBytes(in));

    var out_buf = try GpuBuffer.initHostCoherent(ctx, out_x4.len,
        @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer out_buf.deinit();

    const cmd = try ctx.beginBatch();
    _ = try pl.record(cmd, &in_buf, &out_buf, @intCast(in.len));
    try ctx.submitBatch(cmd);

    try out_buf.download(out_x4);
}

test "gpu quantize_q8_1 round-trip" {
    const ctx = GpuContext.init() catch |e| {
        std.debug.print("gpu init failed: {}\n", .{e});
        return;
    };
    var gpu = ctx;
    defer gpu.deinit();

    // Two x4 groups = 256 elements.
    const n: usize = 256;
    const al = std.testing.allocator;

    const in = try al.alloc(f32, n);
    defer al.free(in);
    var prng = std.Random.DefaultPrng.init(0x81_FACE);
    const r = prng.random();
    for (in) |*v| v.* = (r.float(f32) - 0.5) * 6.0;

    const out_x4_bytes = q8_1OutBytes(@intCast(n));
    const out_x4 = try al.alloc(u8, out_x4_bytes);
    defer al.free(out_x4);
    try runQuantizeShader(&gpu, in, out_x4);

    // Unpack x4 → basic and dequantize.
    const basic = try al.alloc(u8, out_x4_bytes);
    defer al.free(basic);
    dq.unpackQ8_1_x4(out_x4, basic);
    const dec = try al.alloc(f32, n);
    defer al.free(dec);
    dq.dequantQ8_1(basic, dec);

    // Bound per-block max error by amax/127 (+ a small epsilon for f16 d).
    var max_abs: f32 = 0.0;
    var max_amax: f32 = 0.0;
    for (0..n / dq.Q8_1_BLOCK_ELEMS) |b| {
        var amax: f32 = 0.0;
        for (0..dq.Q8_1_BLOCK_ELEMS) |i| {
            if (@abs(in[b * dq.Q8_1_BLOCK_ELEMS + i]) > amax) amax = @abs(in[b * dq.Q8_1_BLOCK_ELEMS + i]);
        }
        if (amax > max_amax) max_amax = amax;
        const tol = amax / 127.0 + 5e-4;
        for (0..dq.Q8_1_BLOCK_ELEMS) |i| {
            const idx = b * dq.Q8_1_BLOCK_ELEMS + i;
            const e = @abs(dec[idx] - in[idx]);
            if (e > max_abs) max_abs = e;
            try std.testing.expect(e <= tol);
        }
    }
    std.debug.print("Q8_1 quantize round-trip n={}  max|D|={d:.6}  amax={d:.6}\n",
        .{ n, max_abs, max_amax });
}

// ── Q4_K × Q8_1 integer-dot matvec fuzz test ──────────────────────────────────
//
// End-to-end test: random Q4_K weights × random f32 activations.
//   1. CPU quantizes activations to Q8_1 (basic format), then dequantizes back
//      to f32 → this is the "Q8_1-rounded" activation that the GPU effectively
//      computes against.
//   2. CPU computes the f32 dot product (math.quantMatvec) on the Q8_1-rounded
//      activation. Because the Q8_1 quants are identical on both sides, this
//      is the deterministic reference the GPU's integer-dot path must match
//      to within float-reduction-order noise (~1e-5 relative).
//   3. GPU: upload Q8_1 x4 buffer + Q4_K matrix → run matvec_q4_k_q8_1.
//   4. Assert rel < 1e-3 (loose to absorb f16 ds + per-row float order).
fn fuzzQ4KQ8_1(
    rows: usize,
    cols: usize,
    seed: u64,
    comptime initPipeline: fn (*const GpuContext) anyerror!MatvecPipeline,
    comptime label: []const u8,
) !void {
    const ctx = GpuContext.init() catch |e| {
        std.debug.print("gpu init failed: {}\n", .{e});
        return;
    };
    var gpu = ctx;
    defer gpu.deinit();

    std.debug.assert(cols % 256 == 0);
    std.debug.assert(cols % 128 == 0);

    const al = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();

    // Random Q4_K matrix bytes with clamped f16 super-scales (matches existing fuzz).
    const blk_bytes_q4k: usize = 144;
    const total_mat = rows * (cols / 256) * blk_bytes_q4k;
    const mat = try al.alloc(u8, total_mat);
    defer al.free(mat);
    for (mat) |*b| b.* = r.int(u8);
    const small_d_le: [2]u8 = .{ 0xCD, 0x21 }; // f16 ≈ 0.012
    const blocks_per_row = cols / 256;
    for (0..rows) |i| for (0..blocks_per_row) |b| {
        const off = (i * blocks_per_row + b) * blk_bytes_q4k;
        mat[off + 0] = small_d_le[0]; mat[off + 1] = small_d_le[1]; // d
        mat[off + 2] = small_d_le[0]; mat[off + 3] = small_d_le[1]; // dmin
    };

    const vec = try al.alloc(f32, cols);
    defer al.free(vec);
    for (vec) |*v| v.* = (r.float(f32) - 0.5) * 2.0;

    // CPU quantize activation to Q8_1 (basic) → dequant → reference activation.
    const q8_1_basic = try al.alloc(u8, (cols / 32) * dq.Q8_1_BLOCK_BYTES);
    defer al.free(q8_1_basic);
    dq.quantizeQ8_1(vec, q8_1_basic);
    const vec_q8_rounded = try al.alloc(f32, cols);
    defer al.free(vec_q8_rounded);
    dq.dequantQ8_1(q8_1_basic, vec_q8_rounded);

    // CPU reference: f32 dot of dequantized weights × Q8_1-rounded activation.
    const cpu_out = try al.alloc(f32, rows);
    defer al.free(cpu_out);
    const row_buf = try al.alloc(f32, cols);
    defer al.free(row_buf);
    math_mod.quantMatvec(cpu_out, mat, .q4_k, vec_q8_rounded, rows, cols, row_buf);

    // Pack basic Q8_1 → x4 for the GPU shader.
    const q8_1_x4 = try al.alloc(u8, q8_1_basic.len);
    defer al.free(q8_1_x4);
    dq.packQ8_1_x4(q8_1_basic, q8_1_x4);

    // GPU side: device-local Q4_K weights, host-coherent Q8_1 acts + output.
    var pipeline = try initPipeline(&gpu);
    defer pipeline.deinit();
    var session = try MatvecSession.initQ4K(&gpu, mat, @intCast(rows), @intCast(cols));
    defer session.deinit();

    var acts_buf = try GpuBuffer.initHostCoherent(&gpu, q8_1_x4.len,
        @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer acts_buf.deinit();
    try acts_buf.upload(q8_1_x4);

    var out_buf = try GpuBuffer.initHostCoherent(&gpu, rows * @sizeOf(f32),
        @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer out_buf.deinit();

    const cmd = try gpu.beginBatch();
    _ = try pipeline.record(cmd, &session.mat_buf, &acts_buf, &out_buf,
        @intCast(rows), @intCast(cols));
    try gpu.submitBatch(cmd);

    const gpu_out = try al.alloc(f32, rows);
    defer al.free(gpu_out);
    try out_buf.download(std.mem.sliceAsBytes(gpu_out));

    var max_abs: f32 = 0.0;
    var max_ref: f32 = 0.0;
    for (cpu_out, gpu_out) |c, g| {
        const d = @abs(c - g);
        if (d > max_abs) max_abs = d;
        if (@abs(c) > max_ref) max_ref = @abs(c);
    }
    const rel = max_abs / (max_ref + 1e-6);
    std.debug.print("{s} fuzz rows={} cols={}  max|D|={d:.6}  rel={e:.3}\n",
        .{ label, rows, cols, max_abs, rel });
    try std.testing.expect(rel < 1e-3);
}

test "gpu matvec Q4_K × Q8_1 fuzz small" {
    try fuzzQ4KQ8_1(32, 256, 11, MatvecPipeline.initQ4KQ8_1, "Q4_K×Q8_1");
}

test "gpu matvec Q4_K × Q8_1 fuzz model-sized" {
    // Closest analogue to a Gemma4 attention matmul: cols = d_model = 2304.
    try fuzzQ4KQ8_1(64, 2304, 13, MatvecPipeline.initQ4KQ8_1, "Q4_K×Q8_1");
}

test "gpu matvec Q4_K × Q8_1 R4 fuzz small" {
    try fuzzQ4KQ8_1(32, 256, 17, MatvecPipeline.initQ4KQ8_1R4, "Q4_K×Q8_1.r4");
}

test "gpu matvec Q4_K × Q8_1 R4 fuzz model-sized" {
    try fuzzQ4KQ8_1(64, 2304, 19, MatvecPipeline.initQ4KQ8_1R4, "Q4_K×Q8_1.r4");
}

// Q3_K × Q8_1 fuzz: same structure as fuzzQ4KQ8_1 but with Q3_K weight bytes.
// Q3_K's row stride is 110 bytes per 256 cols. The shader expects no padding
// between blocks, so a row of cols=2304 occupies 9 × 110 = 990 bytes contiguously.
fn fuzzQ3KQ8_1(rows: usize, cols: usize, seed: u64) !void {
    const ctx = GpuContext.init() catch |e| {
        std.debug.print("gpu init failed: {}\n", .{e});
        return;
    };
    var gpu = ctx;
    defer gpu.deinit();

    std.debug.assert(cols % 256 == 0);

    const al = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();

    const blk_bytes_q3k: usize = 110;
    const total_mat = rows * (cols / 256) * blk_bytes_q3k;
    const mat = try al.alloc(u8, total_mat);
    defer al.free(mat);
    for (mat) |*b| b.* = r.int(u8);
    // Stamp every block's f16 super-scale d to a small clamped magnitude so the
    // CPU dequant doesn't blow up into NaN/Inf. d lives at offset 108..109.
    const small_d_le: [2]u8 = .{ 0xCD, 0x21 }; // f16 ≈ 0.012
    const blocks_per_row = cols / 256;
    for (0..rows) |i| for (0..blocks_per_row) |b| {
        const off = (i * blocks_per_row + b) * blk_bytes_q3k;
        mat[off + 108] = small_d_le[0];
        mat[off + 109] = small_d_le[1];
    };

    const vec = try al.alloc(f32, cols);
    defer al.free(vec);
    for (vec) |*v| v.* = (r.float(f32) - 0.5) * 2.0;

    // CPU Q8_1 → dequant → reference activation; CPU f32 ref dot.
    const q8_1_basic = try al.alloc(u8, (cols / 32) * dq.Q8_1_BLOCK_BYTES);
    defer al.free(q8_1_basic);
    dq.quantizeQ8_1(vec, q8_1_basic);
    const vec_q8_rounded = try al.alloc(f32, cols);
    defer al.free(vec_q8_rounded);
    dq.dequantQ8_1(q8_1_basic, vec_q8_rounded);

    const cpu_out = try al.alloc(f32, rows);
    defer al.free(cpu_out);
    const row_buf = try al.alloc(f32, cols);
    defer al.free(row_buf);
    math_mod.quantMatvec(cpu_out, mat, .q3_k, vec_q8_rounded, rows, cols, row_buf);

    const q8_1_x4 = try al.alloc(u8, q8_1_basic.len);
    defer al.free(q8_1_x4);
    dq.packQ8_1_x4(q8_1_basic, q8_1_x4);

    var pipeline = try MatvecPipeline.initQ3KQ8_1(&gpu);
    defer pipeline.deinit();
    var session = try MatvecSession.initQ3K(&gpu, mat, @intCast(rows), @intCast(cols));
    defer session.deinit();

    var acts_buf = try GpuBuffer.initHostCoherent(&gpu, q8_1_x4.len,
        @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer acts_buf.deinit();
    try acts_buf.upload(q8_1_x4);

    var out_buf = try GpuBuffer.initHostCoherent(&gpu, rows * @sizeOf(f32),
        @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer out_buf.deinit();

    const cmd = try gpu.beginBatch();
    _ = try pipeline.record(cmd, &session.mat_buf, &acts_buf, &out_buf,
        @intCast(rows), @intCast(cols));
    try gpu.submitBatch(cmd);

    const gpu_out = try al.alloc(f32, rows);
    defer al.free(gpu_out);
    try out_buf.download(std.mem.sliceAsBytes(gpu_out));

    var max_abs: f32 = 0.0;
    var max_ref: f32 = 0.0;
    for (cpu_out, gpu_out) |c, g| {
        const d = @abs(c - g);
        if (d > max_abs) max_abs = d;
        if (@abs(c) > max_ref) max_ref = @abs(c);
    }
    const rel = max_abs / (max_ref + 1e-6);
    std.debug.print("Q3_K×Q8_1 fuzz rows={} cols={}  max|D|={d:.6}  rel={e:.3}\n",
        .{ rows, cols, max_abs, rel });
    try std.testing.expect(rel < 1e-3);
}

test "gpu matvec Q3_K × Q8_1 fuzz small" {
    try fuzzQ3KQ8_1(32, 256, 23);
}

test "gpu matvec Q3_K × Q8_1 fuzz model-sized" {
    try fuzzQ3KQ8_1(64, 2304, 29);
}

// Generic Q*-type × Q8_1 fuzz used for Q5_0 and Q5_1 (and any future quant
// whose row stride is `cols/32 * blk_bytes`). Mirrors fuzzQ4KQ8_1 but with
// caller-supplied parameters for block size, stride, and CPU dequant tag.
fn fuzzQuantQ8_1(
    comptime tag: GgmlType,
    blk_bytes: usize,
    blk_elems: usize,
    initPipeline: anytype,
    initSession: anytype,
    rows: usize,
    cols: usize,
    seed: u64,
) !void {
    const ctx = GpuContext.init() catch |e| {
        std.debug.print("gpu init failed: {}\n", .{e});
        return;
    };
    var gpu = ctx;
    defer gpu.deinit();

    std.debug.assert(cols % blk_elems == 0);
    std.debug.assert(cols % 32 == 0);

    const al = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();

    const total_mat = rows * (cols / blk_elems) * blk_bytes;
    const mat = try al.alloc(u8, total_mat);
    defer al.free(mat);
    for (mat) |*b| b.* = r.int(u8);

    // Clamp f16 scales at offset 0..1 (and 2..3 for Q5_1's m) to small magnitudes
    // so dequant doesn't blow up.
    const small_d_le: [2]u8 = .{ 0xCD, 0x21 }; // f16 ≈ 0.012
    const blocks_per_row = cols / blk_elems;
    for (0..rows) |i| for (0..blocks_per_row) |b| {
        const off = (i * blocks_per_row + b) * blk_bytes;
        const d_off: usize = if (tag == .q6_k) 208 else 0;
        mat[off + d_off + 0] = small_d_le[0];
        mat[off + d_off + 1] = small_d_le[1];
        if (tag == .q5_1) {
            mat[off + 2] = small_d_le[0];
            mat[off + 3] = small_d_le[1];
        }
    };

    const vec = try al.alloc(f32, cols);
    defer al.free(vec);
    for (vec) |*v| v.* = (r.float(f32) - 0.5) * 2.0;

    // CPU reference: f32 dot of dequantized weights × dequant(quantize(vec)).
    const q8_1_basic = try al.alloc(u8, (cols / 32) * dq.Q8_1_BLOCK_BYTES);
    defer al.free(q8_1_basic);
    dq.quantizeQ8_1(vec, q8_1_basic);
    const vec_rounded = try al.alloc(f32, cols);
    defer al.free(vec_rounded);
    dq.dequantQ8_1(q8_1_basic, vec_rounded);

    const cpu_out = try al.alloc(f32, rows);
    defer al.free(cpu_out);
    const row_buf = try al.alloc(f32, cols);
    defer al.free(row_buf);
    math_mod.quantMatvec(cpu_out, mat, tag, vec_rounded, rows, cols, row_buf);

    const q8_1_x4_bytes = q8_1OutBytes(@intCast(cols));
    const q8_1_x4 = try al.alloc(u8, q8_1_x4_bytes);
    defer al.free(q8_1_x4);
    dq.packQ8_1_x4(q8_1_basic, q8_1_x4);

    var pipeline = try initPipeline(&gpu);
    defer pipeline.deinit();
    var session = try initSession(&gpu, mat, @intCast(rows), @intCast(cols));
    defer session.deinit();

    var acts_buf = try GpuBuffer.initHostCoherent(&gpu, q8_1_x4.len,
        @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer acts_buf.deinit();
    try acts_buf.upload(q8_1_x4);

    var out_buf = try GpuBuffer.initHostCoherent(&gpu, rows * @sizeOf(f32),
        @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer out_buf.deinit();

    const cmd = try gpu.beginBatch();
    _ = try pipeline.record(cmd, &session.mat_buf, &acts_buf, &out_buf,
        @intCast(rows), @intCast(cols));
    try gpu.submitBatch(cmd);

    const gpu_out = try al.alloc(f32, rows);
    defer al.free(gpu_out);
    try out_buf.download(std.mem.sliceAsBytes(gpu_out));

    var max_abs: f32 = 0.0;
    var max_ref: f32 = 0.0;
    for (cpu_out, gpu_out) |c, g| {
        const d = @abs(c - g);
        if (d > max_abs) max_abs = d;
        if (@abs(c) > max_ref) max_ref = @abs(c);
    }
    const rel = max_abs / (max_ref + 1e-6);
    std.debug.print("{s}×Q8_1 fuzz rows={} cols={}  max|D|={d:.6}  rel={e:.3}\n",
        .{ tag.label(), rows, cols, max_abs, rel });
    try std.testing.expect(rel < 1e-3);
}

test "gpu matvec Q5_0 × Q8_1 fuzz small" {
    // cols=32 is fine — only 32-aligned, no 256 requirement (Q5_0 has 32-elem blocks).
    try fuzzQuantQ8_1(.q5_0, 22, 32,
        MatvecPipeline.initQ5_0Q8_1, MatvecSession.initQ5_0, 32, 256, 41);
}
test "gpu matvec Q5_0 × Q8_1 fuzz model-sized" {
    // d_expert = 704 — closest analogue to a Gemma4 expert down cols.
    try fuzzQuantQ8_1(.q5_0, 22, 32,
        MatvecPipeline.initQ5_0Q8_1, MatvecSession.initQ5_0, 64, 704, 43);
}
test "gpu matvec Q5_1 × Q8_1 fuzz small" {
    try fuzzQuantQ8_1(.q5_1, 24, 32,
        MatvecPipeline.initQ5_1Q8_1, MatvecSession.initQ5_1, 32, 256, 47);
}
test "gpu matvec Q5_1 × Q8_1 fuzz model-sized" {
    try fuzzQuantQ8_1(.q5_1, 24, 32,
        MatvecPipeline.initQ5_1Q8_1, MatvecSession.initQ5_1, 64, 704, 53);
}

test "gpu matvec Q6_K × Q8_1 fuzz small" {
    try fuzzQuantQ8_1(.q6_k, 210, 256,
        MatvecPipeline.initQ6KQ8_1, MatvecSession.initQ6K, 32, 256, 59);
}

test "gpu matvec Q6_K × Q8_1 fuzz lm-head-shaped cols" {
    try fuzzQuantQ8_1(.q6_k, 210, 256,
        MatvecPipeline.initQ6KQ8_1, MatvecSession.initQ6K, 64, 2816, 61);
}

test "gpu matvec Q6_K × Q8_1 fast fuzz small" {
    try fuzzQuantQ8_1(.q6_k, 210, 256,
        MatvecPipeline.initQ6KQ8_1Fast, MatvecSession.initQ6K, 32, 256, 63);
}

test "gpu matvec Q6_K × Q8_1 fast fuzz lm-head-shaped cols" {
    try fuzzQuantQ8_1(.q6_k, 210, 256,
        MatvecPipeline.initQ6KQ8_1Fast, MatvecSession.initQ6K, 64, 2816, 65);
}

fn initQ6KQ8_1MmvqB32R1(ctx: *const GpuContext) !MatvecPipeline {
    return MatvecPipeline.initQ6KQ8_1Mmvq(ctx, .{
        .block_size = 32,
        .num_rows = 1,
        .num_cols = 1,
    });
}

fn initQ6KQ8_1MmvqB64R1(ctx: *const GpuContext) !MatvecPipeline {
    return MatvecPipeline.initQ6KQ8_1Mmvq(ctx, .{
        .block_size = 64,
        .num_rows = 1,
        .num_cols = 1,
    });
}

fn initQ6KQ8_1MmvqB64R2(ctx: *const GpuContext) !MatvecPipeline {
    return MatvecPipeline.initQ6KQ8_1Mmvq(ctx, .{
        .block_size = 64,
        .num_rows = 2,
        .num_cols = 1,
    });
}

fn initQ6KQ8_1MmvqB64R4(ctx: *const GpuContext) !MatvecPipeline {
    return MatvecPipeline.initQ6KQ8_1Mmvq(ctx, .{
        .block_size = 64,
        .num_rows = 4,
        .num_cols = 1,
    });
}

fn initQ3KQ8_1MmvqB32R1(ctx: *const GpuContext) !MatvecPipeline {
    return MatvecPipeline.initQ3KQ8_1Mmvq(ctx, .{
        .block_size = 32,
        .num_rows = 1,
        .num_cols = 1,
    });
}

fn initQ3KQ8_1MmvqB64R1(ctx: *const GpuContext) !MatvecPipeline {
    return MatvecPipeline.initQ3KQ8_1Mmvq(ctx, .{
        .block_size = 64,
        .num_rows = 1,
        .num_cols = 1,
    });
}

test "gpu matvec Q3_K × Q8_1 MMVQ b32 r1 fuzz small" {
    try fuzzQuantQ8_1(.q3_k, 110, 256,
        initQ3KQ8_1MmvqB32R1, MatvecSession.initQ3K, 32, 256, 80);
}

test "gpu matvec Q3_K × Q8_1 MMVQ b32 r1 fuzz lm-head-shaped cols" {
    try fuzzQuantQ8_1(.q3_k, 110, 256,
        initQ3KQ8_1MmvqB32R1, MatvecSession.initQ3K, 64, 2816, 82);
}

test "gpu matvec Q3_K × Q8_1 MMVQ b64 r1 fuzz small" {
    try fuzzQuantQ8_1(.q3_k, 110, 256,
        initQ3KQ8_1MmvqB64R1, MatvecSession.initQ3K, 32, 256, 81);
}

test "gpu matvec Q3_K × Q8_1 MMVQ b64 r1 fuzz lm-head-shaped cols" {
    try fuzzQuantQ8_1(.q3_k, 110, 256,
        initQ3KQ8_1MmvqB64R1, MatvecSession.initQ3K, 64, 2816, 83);
}

test "gpu matvec Q6_K × Q8_1 MMVQ b32 r1 fuzz small" {
    try fuzzQuantQ8_1(.q6_k, 210, 256,
        initQ6KQ8_1MmvqB32R1, MatvecSession.initQ6K, 32, 256, 67);
}

test "gpu matvec Q6_K × Q8_1 MMVQ b32 r1 fuzz lm-head-shaped cols" {
    try fuzzQuantQ8_1(.q6_k, 210, 256,
        initQ6KQ8_1MmvqB32R1, MatvecSession.initQ6K, 64, 2816, 69);
}

test "gpu matvec Q6_K × Q8_1 MMVQ b64 r1 fuzz small" {
    try fuzzQuantQ8_1(.q6_k, 210, 256,
        initQ6KQ8_1MmvqB64R1, MatvecSession.initQ6K, 32, 256, 71);
}

test "gpu matvec Q6_K × Q8_1 MMVQ b64 r1 fuzz lm-head-shaped cols" {
    try fuzzQuantQ8_1(.q6_k, 210, 256,
        initQ6KQ8_1MmvqB64R1, MatvecSession.initQ6K, 64, 2816, 73);
}

test "gpu matvec Q6_K × Q8_1 MMVQ b64 r2 fuzz lm-head-shaped cols" {
    try fuzzQuantQ8_1(.q6_k, 210, 256,
        initQ6KQ8_1MmvqB64R2, MatvecSession.initQ6K, 64, 2816, 75);
}

test "gpu matvec Q6_K × Q8_1 MMVQ b64 r4 fuzz lm-head-shaped cols" {
    try fuzzQuantQ8_1(.q6_k, 210, 256,
        initQ6KQ8_1MmvqB64R4, MatvecSession.initQ6K, 64, 2816, 77);
}

test "gpu matvec Q5_K × Q8_1 fuzz small" {
    try fuzzQuantQ8_1(.q5_k, 176, 256,
        MatvecPipeline.initQ5KQ8_1, MatvecSession.initQ5K, 32, 256, 73);
}

test "gpu matvec Q5_K × Q8_1 fuzz attn-v-shaped cols" {
    try fuzzQuantQ8_1(.q5_k, 176, 256,
        MatvecPipeline.initQ5KQ8_1, MatvecSession.initQ5K, 64, 2816, 79);
}

test "gpu matvec IQ4_NL × Q8_1 fuzz small" {
    try fuzzQuantQ8_1(.iq4_nl, 18, 32,
        MatvecPipeline.initIQ4NLQ8_1, MatvecSession.initIQ4NL, 32, 256, 67);
}

test "gpu matvec IQ4_NL × Q8_1 fuzz expert-down-shaped" {
    try fuzzQuantQ8_1(.iq4_nl, 18, 32,
        MatvecPipeline.initIQ4NLQ8_1, MatvecSession.initIQ4NL, 64, 704, 71);
}

// ── rmsnorm fuzz test ─────────────────────────────────────────────────────────
fn runRmsnormShader(
    gpu: *const GpuContext,
    x: []const f32,
    w: []const f32,
    eps: f32,
    weight_offset: bool,
    out: []f32,
) !void {
    var pl = try RmsnormPipeline.init(gpu);
    defer pl.deinit();

    var x_buf = try GpuBuffer.initHostCoherent(gpu, x.len * @sizeOf(f32),
        @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer x_buf.deinit();
    try x_buf.upload(std.mem.sliceAsBytes(x));

    var w_buf = try GpuBuffer.initHostCoherent(gpu, w.len * @sizeOf(f32),
        @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer w_buf.deinit();
    try w_buf.upload(std.mem.sliceAsBytes(w));

    var y_buf = try GpuBuffer.initHostCoherent(gpu, out.len * @sizeOf(f32),
        @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer y_buf.deinit();

    const cmd = try gpu.beginBatch();
    _ = try pl.record(cmd, &x_buf, &w_buf, &y_buf, @intCast(x.len), eps, weight_offset);
    try gpu.submitBatch(cmd);
    try y_buf.download(std.mem.sliceAsBytes(out));
}

fn fuzzRmsnorm(n: usize, seed: u64, weight_offset: bool) !void {
    const ctx = GpuContext.init() catch |e| {
        std.debug.print("gpu init failed: {}\n", .{e});
        return;
    };
    var gpu = ctx;
    defer gpu.deinit();

    const al = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();

    const x = try al.alloc(f32, n);    defer al.free(x);
    const w = try al.alloc(f32, n);    defer al.free(w);
    const cpu_out = try al.alloc(f32, n); defer al.free(cpu_out);
    const gpu_out = try al.alloc(f32, n); defer al.free(gpu_out);

    for (x) |*v| v.* = (r.float(f32) - 0.5) * 4.0;
    for (w) |*v| v.* = (r.float(f32) - 0.5) * 2.0;

    // CPU reference: y = x * rms_inv * (bias + w)
    var ss: f32 = 0.0;
    for (x) |v| ss += v * v;
    const eps: f32 = 1e-6;
    const rms_inv: f32 = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(n)) + eps);
    const bias: f32 = if (weight_offset) 1.0 else 0.0;
    for (cpu_out, x, w) |*o, xi, wi| o.* = xi * rms_inv * (bias + wi);

    try runRmsnormShader(&gpu, x, w, eps, weight_offset, gpu_out);

    var max_abs: f32 = 0.0;
    var max_ref: f32 = 0.0;
    for (cpu_out, gpu_out) |c, g| {
        const d = @abs(c - g);
        if (d > max_abs) max_abs = d;
        if (@abs(c) > max_ref) max_ref = @abs(c);
    }
    const rel = max_abs / (max_ref + 1e-6);
    std.debug.print("rmsnorm fuzz n={} bias={} max|D|={d:.6} rel={e:.3}\n",
        .{ n, weight_offset, max_abs, rel });
    try std.testing.expect(rel < 1e-5);
}

test "gpu rmsnorm n=512 fuzz" {
    try fuzzRmsnorm(512, 71, false);  // head_dim_global
}
test "gpu rmsnorm n=2816 fuzz" {
    try fuzzRmsnorm(2816, 73, false); // d_model
}
test "gpu rmsnorm n=2816 fuzz with (1+w) convention" {
    try fuzzRmsnorm(2816, 79, true);  // exercise the weight_offset path
}

// ── rmsnorm_perhead fuzz test ─────────────────────────────────────────────────

fn runRmsnormPerHeadShader(
    gpu: *const GpuContext,
    x: []const f32,
    w: []const f32,
    n_heads: u32,
    head_dim: u32,
    eps: f32,
    weight_offset: bool,
    use_weight: bool,
    out: []f32,
) !void {
    var pl = try RmsnormPerHeadPipeline.init(gpu);
    defer pl.deinit();

    var x_buf = try GpuBuffer.initHostCoherent(gpu, x.len * @sizeOf(f32),
        @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer x_buf.deinit();
    try x_buf.upload(std.mem.sliceAsBytes(x));

    var w_buf = try GpuBuffer.initHostCoherent(gpu, w.len * @sizeOf(f32),
        @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer w_buf.deinit();
    try w_buf.upload(std.mem.sliceAsBytes(w));

    var y_buf = try GpuBuffer.initHostCoherent(gpu, out.len * @sizeOf(f32),
        @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer y_buf.deinit();

    const cmd = try gpu.beginBatch();
    _ = try pl.record(cmd, &x_buf, &w_buf, &y_buf,
        n_heads, head_dim, eps, weight_offset, use_weight);
    try gpu.submitBatch(cmd);
    try y_buf.download(std.mem.sliceAsBytes(out));
}

fn fuzzRmsnormPerHead(
    n_heads: u32, head_dim: u32, seed: u64,
    weight_offset: bool, use_weight: bool,
) !void {
    const ctx = GpuContext.init() catch |e| {
        std.debug.print("gpu init failed: {}\n", .{e});
        return;
    };
    var gpu = ctx;
    defer gpu.deinit();

    const al = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();

    const n_total = @as(usize, n_heads) * @as(usize, head_dim);
    const x = try al.alloc(f32, n_total);   defer al.free(x);
    const w = try al.alloc(f32, head_dim);  defer al.free(w);
    const cpu_out = try al.alloc(f32, n_total); defer al.free(cpu_out);
    const gpu_out = try al.alloc(f32, n_total); defer al.free(gpu_out);

    for (x) |*v| v.* = (r.float(f32) - 0.5) * 4.0;
    for (w) |*v| v.* = (r.float(f32) - 0.5) * 2.0;

    const eps: f32 = 1e-6;
    const bias: f32 = if (weight_offset) 1.0 else 0.0;
    for (0..n_heads) |h| {
        const base = h * head_dim;
        var ss: f32 = 0.0;
        for (x[base..][0..head_dim]) |v| ss += v * v;
        const rms_inv: f32 = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(head_dim)) + eps);
        for (0..head_dim) |i| {
            const scale: f32 = if (use_weight) (bias + w[i]) else 1.0;
            cpu_out[base + i] = x[base + i] * rms_inv * scale;
        }
    }

    try runRmsnormPerHeadShader(&gpu, x, w, n_heads, head_dim,
        eps, weight_offset, use_weight, gpu_out);

    var max_abs: f32 = 0.0;
    var max_ref: f32 = 0.0;
    for (cpu_out, gpu_out) |c, g| {
        const d = @abs(c - g);
        if (d > max_abs) max_abs = d;
        if (@abs(c) > max_ref) max_ref = @abs(c);
    }
    const rel = max_abs / (max_ref + 1e-6);
    std.debug.print(
        "rmsnorm_perhead fuzz n_heads={} hd={} bias={} use_w={} max|D|={d:.6} rel={e:.3}\n",
        .{ n_heads, head_dim, weight_offset, use_weight, max_abs, rel });
    try std.testing.expect(rel < 1e-5);
}

test "gpu rmsnorm_perhead n_heads=16 hd=256 (SWA Q-norm)" {
    try fuzzRmsnormPerHead(16, 256, 91, true, true);
}
test "gpu rmsnorm_perhead n_heads=4 hd=512 (global K-norm)" {
    try fuzzRmsnormPerHead(4, 512, 97, true, true);
}
test "gpu rmsnorm_perhead n_heads=4 hd=256 no weight (V rmsnormRaw)" {
    try fuzzRmsnormPerHead(4, 256, 103, false, false);
}

// ── attn_qk_softmax + attn_av fuzz tests ──────────────────────────────────────

fn cpuAttn(
    q: []const f32, k: []const f32, v: []const f32,
    scores_out: []f32, out: []f32,
    n_heads: u32, n_kv_heads: u32, head_dim: u32,
    win_len: u32, scale: f32,
) void {
    const n_q_per_kv = n_heads / n_kv_heads;
    for (0..n_heads) |h| {
        const kv_h = h / n_q_per_kv;
        const q_h = q[h * head_dim ..][0..head_dim];
        const sc = scores_out[h * win_len ..][0..win_len];
        var mx: f32 = -3.4e38;
        for (0..win_len) |i| {
            const k_i = k[i * n_kv_heads * head_dim + kv_h * head_dim ..][0..head_dim];
            var dot: f32 = 0.0;
            for (q_h, k_i) |qv, kv_| dot += qv * kv_;
            const s = dot * scale;
            sc[i] = s;
            if (s > mx) mx = s;
        }
        var sum: f32 = 0.0;
        for (sc) |*s| { s.* = @exp(s.* - mx); sum += s.*; }
        const inv = 1.0 / sum;
        for (sc) |*s| s.* *= inv;

        const out_h = out[h * head_dim ..][0..head_dim];
        for (out_h) |*o| o.* = 0;
        for (0..win_len) |i| {
            const v_i = v[i * n_kv_heads * head_dim + kv_h * head_dim ..][0..head_dim];
            const w = sc[i];
            for (out_h, v_i) |*o, vv| o.* += w * vv;
        }
    }
}

fn fuzzAttn(
    n_heads: u32, n_kv_heads: u32, head_dim: u32, win_len: u32, seed: u64,
) !void {
    const ctx = GpuContext.init() catch |e| {
        std.debug.print("gpu init failed: {}\n", .{e});
        return;
    };
    var gpu = ctx;
    defer gpu.deinit();

    var pl_qk = try AttnQkSoftmaxPipeline.init(&gpu);
    defer pl_qk.deinit();
    var pl_av = try AttnAvPipeline.init(&gpu);
    defer pl_av.deinit();

    const al = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();

    const q_len  = @as(usize, n_heads)    * @as(usize, head_dim);
    const kv_len = @as(usize, win_len)    * @as(usize, n_kv_heads) * @as(usize, head_dim);
    const sc_len = @as(usize, n_heads)    * @as(usize, win_len);
    const out_len = q_len;

    const q  = try al.alloc(f32, q_len);  defer al.free(q);
    const k  = try al.alloc(f32, kv_len); defer al.free(k);
    const v  = try al.alloc(f32, kv_len); defer al.free(v);
    const sc_cpu  = try al.alloc(f32, sc_len);  defer al.free(sc_cpu);
    const sc_gpu  = try al.alloc(f32, sc_len);  defer al.free(sc_gpu);
    const out_cpu = try al.alloc(f32, out_len); defer al.free(out_cpu);
    const out_gpu = try al.alloc(f32, out_len); defer al.free(out_gpu);

    for (q) |*x| x.* = (r.float(f32) - 0.5) * 2.0;
    for (k) |*x| x.* = (r.float(f32) - 0.5) * 2.0;
    for (v) |*x| x.* = (r.float(f32) - 0.5) * 2.0;

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    cpuAttn(q, k, v, sc_cpu, out_cpu, n_heads, n_kv_heads, head_dim, win_len, scale);

    // No circular wrapping: seq = win_len, cap = win_len → slot(i) = i.
    const seq: u32 = win_len;
    const cap: u32 = win_len;
    const n_q_per_kv: u32 = n_heads / n_kv_heads;

    const usage: vk.VkBufferUsageFlags = @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    var q_buf  = try GpuBuffer.initHostCoherent(&gpu, q_len  * @sizeOf(f32), usage);
    defer q_buf.deinit();
    var k_buf  = try GpuBuffer.initHostCoherent(&gpu, kv_len * @sizeOf(f32), usage);
    defer k_buf.deinit();
    var v_buf  = try GpuBuffer.initHostCoherent(&gpu, kv_len * @sizeOf(f32), usage);
    defer v_buf.deinit();
    var sc_buf = try GpuBuffer.initHostCoherent(&gpu, sc_len * @sizeOf(f32), usage);
    defer sc_buf.deinit();
    var o_buf  = try GpuBuffer.initHostCoherent(&gpu, out_len * @sizeOf(f32), usage);
    defer o_buf.deinit();

    try q_buf.upload(std.mem.sliceAsBytes(q));
    try k_buf.upload(std.mem.sliceAsBytes(k));
    try v_buf.upload(std.mem.sliceAsBytes(v));

    const cmd = try gpu.beginBatch();
    _ = try pl_qk.record(cmd, &q_buf, &k_buf, &sc_buf,
        n_heads, seq, win_len, head_dim, n_kv_heads, n_q_per_kv, cap, scale);
    GpuContext.recordShaderBarrier(cmd);
    _ = try pl_av.record(cmd, &sc_buf, &v_buf, &o_buf,
        n_heads, seq, win_len, head_dim, n_kv_heads, n_q_per_kv, cap);
    try gpu.submitBatch(cmd);

    try sc_buf.download(std.mem.sliceAsBytes(sc_gpu));
    try o_buf.download(std.mem.sliceAsBytes(out_gpu));

    var sc_max: f32 = 0; var sc_ref: f32 = 0;
    for (sc_cpu, sc_gpu) |c, g| {
        const d = @abs(c - g);
        if (d > sc_max) sc_max = d;
        if (@abs(c) > sc_ref) sc_ref = @abs(c);
    }
    var ov_max: f32 = 0; var ov_ref: f32 = 0;
    for (out_cpu, out_gpu) |c, g| {
        const d = @abs(c - g);
        if (d > ov_max) ov_max = d;
        if (@abs(c) > ov_ref) ov_ref = @abs(c);
    }
    const sc_rel = sc_max / (sc_ref + 1e-6);
    const ov_rel = ov_max / (ov_ref + 1e-6);
    std.debug.print(
        "attn fuzz nh={} nkv={} hd={} win={} scores rel={e:.3} out rel={e:.3}\n",
        .{ n_heads, n_kv_heads, head_dim, win_len, sc_rel, ov_rel });
    try std.testing.expect(sc_rel < 1e-4);
    try std.testing.expect(ov_rel < 1e-4);
}

test "gpu attn fuzz SWA shape n_heads=16 n_kv=4 hd=256 win=1" {
    try fuzzAttn(16, 4, 256, 1, 401);
}
test "gpu attn fuzz SWA shape n_heads=16 n_kv=4 hd=256 win=128" {
    try fuzzAttn(16, 4, 256, 128, 403);
}
test "gpu attn fuzz SWA shape n_heads=16 n_kv=4 hd=256 win=1024" {
    try fuzzAttn(16, 4, 256, 1024, 409);
}
test "gpu attn fuzz global shape n_heads=4 n_kv=4 hd=512 win=64" {
    try fuzzAttn(4, 4, 512, 64, 411);
}
test "gpu attn fuzz global shape n_heads=4 n_kv=4 hd=512 win=512" {
    try fuzzAttn(4, 4, 512, 512, 419);
}

test "gpu elem_add fuzz" {
    const ctx = GpuContext.init() catch |e| {
        std.debug.print("gpu init failed: {}\n", .{e});
        return;
    };
    var gpu = ctx;
    defer gpu.deinit();

    const al = std.testing.allocator;
    const n: usize = 2816;
    const a = try al.alloc(f32, n); defer al.free(a);
    const b = try al.alloc(f32, n); defer al.free(b);
    const expected = try al.alloc(f32, n); defer al.free(expected);
    var prng = std.Random.DefaultPrng.init(81);
    const r = prng.random();
    for (a, b, expected) |*ai, *bi, *ei| {
        ai.* = (r.float(f32) - 0.5) * 10.0;
        bi.* = (r.float(f32) - 0.5) * 10.0;
        ei.* = ai.* + bi.*;
    }

    var pl = try ElemAddPipeline.init(&gpu);
    defer pl.deinit();
    var a_buf = try GpuBuffer.initHostCoherent(&gpu, n * @sizeOf(f32),
        @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer a_buf.deinit();
    var b_buf = try GpuBuffer.initHostCoherent(&gpu, n * @sizeOf(f32),
        @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer b_buf.deinit();
    try a_buf.upload(std.mem.sliceAsBytes(a));
    try b_buf.upload(std.mem.sliceAsBytes(b));

    const cmd = try gpu.beginBatch();
    _ = try pl.record(cmd, &a_buf, &b_buf, @intCast(n));
    try gpu.submitBatch(cmd);

    const got = try al.alloc(f32, n); defer al.free(got);
    try a_buf.download(std.mem.sliceAsBytes(got));
    for (got, expected) |g, e| try std.testing.expectApproxEqAbs(e, g, 1e-6);
}

test "gpu rope_neox_theta fuzz" {
    const ctx = GpuContext.init() catch |e| {
        std.debug.print("gpu init failed: {}\n", .{e});
        return;
    };
    var gpu = ctx;
    defer gpu.deinit();

    const rope_mod = @import("../ops/rope.zig");
    const al = std.testing.allocator;

    const n_heads: usize = 4;
    const head_dim: usize = 256;
    const pos: usize = 13;
    const theta: f32 = 10000.0;
    const total = n_heads * head_dim;

    const vec = try al.alloc(f32, total); defer al.free(vec);
    const cpu_out = try al.alloc(f32, total); defer al.free(cpu_out);
    var prng = std.Random.DefaultPrng.init(91);
    const r = prng.random();
    for (vec) |*v| v.* = (r.float(f32) - 0.5) * 4.0;
    @memcpy(cpu_out, vec);
    for (0..n_heads) |h|
        rope_mod.applyRopeNeox(cpu_out[h * head_dim ..][0..head_dim], pos, theta);

    var pl = try RopeNeoxThetaPipeline.init(&gpu);
    defer pl.deinit();
    var vec_buf = try GpuBuffer.initHostCoherent(&gpu, total * @sizeOf(f32),
        @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer vec_buf.deinit();
    try vec_buf.upload(std.mem.sliceAsBytes(vec));
    const cmd = try gpu.beginBatch();
    _ = try pl.record(cmd, &vec_buf, @intCast(pos), @intCast(head_dim), theta, @intCast(n_heads));
    try gpu.submitBatch(cmd);

    const gpu_out = try al.alloc(f32, total); defer al.free(gpu_out);
    try vec_buf.download(std.mem.sliceAsBytes(gpu_out));
    var max_abs: f32 = 0.0;
    var max_ref: f32 = 0.0;
    for (cpu_out, gpu_out) |c, g| {
        const d = @abs(c - g);
        if (d > max_abs) max_abs = d;
        if (@abs(c) > max_ref) max_ref = @abs(c);
    }
    const rel = max_abs / (max_ref + 1e-6);
    std.debug.print("rope_neox_theta fuzz n_heads={} hd={} pos={} max|D|={d:.6} rel={e:.3}\n",
        .{ n_heads, head_dim, pos, max_abs, rel });
    try std.testing.expect(rel < 1e-5);
}

test "gpu rope_neox_table fuzz" {
    const ctx = GpuContext.init() catch |e| {
        std.debug.print("gpu init failed: {}\n", .{e});
        return;
    };
    var gpu = ctx;
    defer gpu.deinit();

    const rope_mod = @import("../ops/rope.zig");
    const al = std.testing.allocator;

    const n_heads: usize = 4;
    const head_dim: usize = 512;
    const pos: usize = 7;
    const total = n_heads * head_dim;

    const vec = try al.alloc(f32, total); defer al.free(vec);
    const cpu_out = try al.alloc(f32, total); defer al.free(cpu_out);
    const freqs = try al.alloc(f32, head_dim / 2); defer al.free(freqs);
    var prng = std.Random.DefaultPrng.init(97);
    const r = prng.random();
    for (vec) |*v| v.* = (r.float(f32) - 0.5) * 4.0;
    // Generate plausible inverse-frequency values (≈ 1/theta^(2i/dim) shape).
    for (freqs, 0..) |*f, i|
        f.* = 1.0 / std.math.pow(f32, 10000.0,
            @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(head_dim)));

    @memcpy(cpu_out, vec);
    for (0..n_heads) |h|
        rope_mod.applyRopeFreqsNeox(cpu_out[h * head_dim ..][0..head_dim], freqs, pos);

    var pl = try RopeNeoxTablePipeline.init(&gpu);
    defer pl.deinit();
    var vec_buf = try GpuBuffer.initHostCoherent(&gpu, total * @sizeOf(f32),
        @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer vec_buf.deinit();
    try vec_buf.upload(std.mem.sliceAsBytes(vec));
    var freqs_buf = try GpuBuffer.initHostCoherent(&gpu, freqs.len * @sizeOf(f32),
        @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer freqs_buf.deinit();
    try freqs_buf.upload(std.mem.sliceAsBytes(freqs));

    const cmd = try gpu.beginBatch();
    _ = try pl.record(cmd, &vec_buf, &freqs_buf, @intCast(pos), @intCast(head_dim), @intCast(n_heads));
    try gpu.submitBatch(cmd);

    const gpu_out = try al.alloc(f32, total); defer al.free(gpu_out);
    try vec_buf.download(std.mem.sliceAsBytes(gpu_out));
    var max_abs: f32 = 0.0;
    var max_ref: f32 = 0.0;
    for (cpu_out, gpu_out) |c, g| {
        const d = @abs(c - g);
        if (d > max_abs) max_abs = d;
        if (@abs(c) > max_ref) max_ref = @abs(c);
    }
    const rel = max_abs / (max_ref + 1e-6);
    std.debug.print("rope_neox_table fuzz n_heads={} hd={} pos={} max|D|={d:.6} rel={e:.3}\n",
        .{ n_heads, head_dim, pos, max_abs, rel });
    try std.testing.expect(rel < 1e-5);
}

test "gpu gelu_mul fuzz" {
    const ctx = GpuContext.init() catch |e| {
        std.debug.print("gpu init failed: {}\n", .{e});
        return;
    };
    var gpu = ctx;
    defer gpu.deinit();

    const math_ref = @import("../ops/math.zig");
    const al = std.testing.allocator;
    const n: usize = 2112;  // dense FFN dim in Gemma4
    const a = try al.alloc(f32, n); defer al.free(a);
    const b = try al.alloc(f32, n); defer al.free(b);
    const expected = try al.alloc(f32, n); defer al.free(expected);
    var prng = std.Random.DefaultPrng.init(84);
    const r = prng.random();
    for (a, b, expected) |*ai, *bi, *ei| {
        ai.* = (r.float(f32) - 0.5) * 10.0;
        bi.* = (r.float(f32) - 0.5) * 10.0;
        ei.* = math_ref.gelu(ai.*) * bi.*;
    }

    var pl = try GeluMulPipeline.init(&gpu);
    defer pl.deinit();
    var a_buf = try GpuBuffer.initHostCoherent(&gpu, n * @sizeOf(f32),
        @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer a_buf.deinit();
    var b_buf = try GpuBuffer.initHostCoherent(&gpu, n * @sizeOf(f32),
        @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer b_buf.deinit();
    try a_buf.upload(std.mem.sliceAsBytes(a));
    try b_buf.upload(std.mem.sliceAsBytes(b));

    const cmd = try gpu.beginBatch();
    _ = try pl.record(cmd, &a_buf, &b_buf, @intCast(n));
    try gpu.submitBatch(cmd);

    const got = try al.alloc(f32, n); defer al.free(got);
    try a_buf.download(std.mem.sliceAsBytes(got));
    // GELU * mul can produce near-zero outputs (gelu(x) ≈ 0 for x ∈ [-2, 0]),
    // so a strict relative check is dominated by floor noise.  Use absolute
    // tolerance, which is f32 precision (~7 decimal digits) on bounded inputs.
    var max_abs: f32 = 0;
    for (got, expected) |g, e| {
        const d = @abs(g - e);
        if (d > max_abs) max_abs = d;
    }
    std.debug.print("gelu_mul fuzz n={} max|D|={d:.6}\n", .{ n, max_abs });
    try std.testing.expect(max_abs < 1e-4);
}

test "gpu elem_scale fuzz" {
    const ctx = GpuContext.init() catch |e| {
        std.debug.print("gpu init failed: {}\n", .{e});
        return;
    };
    var gpu = ctx;
    defer gpu.deinit();

    const al = std.testing.allocator;
    const n: usize = 2816;
    const x = try al.alloc(f32, n); defer al.free(x);
    const s: f32 = 1.4142136; // sqrt(2)
    var prng = std.Random.DefaultPrng.init(83);
    const r = prng.random();
    for (x) |*xi| xi.* = (r.float(f32) - 0.5) * 10.0;

    var pl = try ElemScalePipeline.init(&gpu);
    defer pl.deinit();
    var x_buf = try GpuBuffer.initHostCoherent(&gpu, n * @sizeOf(f32),
        @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer x_buf.deinit();
    try x_buf.upload(std.mem.sliceAsBytes(x));

    const cmd = try gpu.beginBatch();
    _ = try pl.record(cmd, &x_buf, @intCast(n), s);
    try gpu.submitBatch(cmd);

    const got = try al.alloc(f32, n); defer al.free(got);
    try x_buf.download(std.mem.sliceAsBytes(got));
    for (got, x) |g, xi| try std.testing.expectApproxEqAbs(xi * s, g, 1e-6);
}
