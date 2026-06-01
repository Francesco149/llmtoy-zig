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

    pub fn initQ4KQ8_1Mmvq(ctx: *const GpuContext, spec: MmvqSpec) !MatvecPipeline {
        comptime std.debug.assert(shaders.matvec_q4_k_q8_1_mmvq.len % 4 == 0);
        return initFromSpvMmvq(ctx, &shaders.matvec_q4_k_q8_1_mmvq, spec);
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

    pub fn initQ5_0Q8_1Mmvq(ctx: *const GpuContext, spec: MmvqSpec) !MatvecPipeline {
        comptime std.debug.assert(shaders.matvec_q5_0_q8_1_mmvq.len % 4 == 0);
        return initFromSpvMmvq(ctx, &shaders.matvec_q5_0_q8_1_mmvq, spec);
    }

    pub fn initQ5_1Q8_1(ctx: *const GpuContext) !MatvecPipeline {
        comptime std.debug.assert(shaders.matvec_q5_1_q8_1.len % 4 == 0);
        return initFromSpv(ctx, &shaders.matvec_q5_1_q8_1, 1);
    }

    pub fn initQ5_1Q8_1Mmvq(ctx: *const GpuContext, spec: MmvqSpec) !MatvecPipeline {
        comptime std.debug.assert(shaders.matvec_q5_1_q8_1_mmvq.len % 4 == 0);
        return initFromSpvMmvq(ctx, &shaders.matvec_q5_1_q8_1_mmvq, spec);
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

    pub fn initQ5KQ8_1Mmvq(ctx: *const GpuContext, spec: MmvqSpec) !MatvecPipeline {
        comptime std.debug.assert(shaders.matvec_q5_k_q8_1_mmvq.len % 4 == 0);
        return initFromSpvMmvq(ctx, &shaders.matvec_q5_k_q8_1_mmvq, spec);
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
            .{ .constantID = 1, .offset = @offsetOf(MmvqSpec, "num_rows"), .size = @sizeOf(u32) },
            .{ .constantID = 2, .offset = @offsetOf(MmvqSpec, "num_cols"), .size = @sizeOf(u32) },
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
        try ctx.createComputePipeline(&pipeline_ci, &pipeline);
        errdefer vk.vkDestroyPipeline(dev, pipeline, null);

        // Persistent per-layer graph-shaped paths reuse descriptor sets across
        // tokens, so each quant pipeline needs room for attention, dense, final,
        // and occasional transient fallback bindings.
        const max_sets = 256;
        const pool_size = vk.VkDescriptorPoolSize{
            .type = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 3 * max_sets,
        };
        const pool_ci = vk.VkDescriptorPoolCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = null,
            .flags = vk.VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            .maxSets = max_sets,
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
    // Caller must free the returned set after the submitted command buffer has
    // completed.
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

    pub fn allocSet(
        self: *const MatvecPipeline,
        mat_buf: *const GpuBuffer,
        vec_buf: *const GpuBuffer,
        out_buf: *const GpuBuffer,
    ) !vk.VkDescriptorSet {
        const dset = try self.allocDescriptorSet();
        self.updateDescriptorSet(dset, mat_buf, vec_buf, out_buf);
        return dset;
    }

    pub fn recordDescriptor(
        self: *const MatvecPipeline,
        cmd: vk.VkCommandBuffer,
        dset: vk.VkDescriptorSet,
        rows: u32,
        cols: u32,
    ) void {
        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.layout, 0, 1, &dset, 0, null);

        const pc = PushConst{ .rows = rows, .cols = cols };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(PushConst), &pc);

        const groups = (rows + self.rows_per_workgroup - 1) / self.rows_per_workgroup;
        vk.vkCmdDispatch(cmd, groups, 1, 1);
    }

    pub fn recordWithSet(
        self: *const MatvecPipeline,
        cmd: vk.VkCommandBuffer,
        dset: vk.VkDescriptorSet,
        rows: u32,
        cols: u32,
    ) void {
        self.recordDescriptor(cmd, dset, rows, cols);
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
            .{ .buffer = mat_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = vec_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = out_handle, .offset = out_offset, .range = out_range },
        };
        const writes = [3]vk.VkWriteDescriptorSet{
            mkWrite(dset, 0, &buf_infos[0]),
            mkWrite(dset, 1, &buf_infos[1]),
            mkWrite(dset, 2, &buf_infos[2]),
        };
        vk.vkUpdateDescriptorSets(dev, writes.len, &writes, 0, null);

        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.layout, 0, 1, &dset, 0, null);

        const pc = PushConst{ .rows = rows, .cols = cols };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(PushConst), &pc);

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
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.layout, 0, 1, &dset, 0, null);

        const pc = PushConst{ .rows = rows, .cols = cols };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(PushConst), &pc);

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
            .f32 => try initBytes(ctx, mat_data, rows, cols),
            .q8_0 => try initQ8_0(ctx, mat_data, rows, cols),
            .q3_k => try initQ3K(ctx, mat_data, rows, cols),
            .q4_k => try initQ4K(ctx, mat_data, rows, cols),
            .q5_1 => try initQ5_1(ctx, mat_data, rows, cols),
            .q5_0 => try initQ5_0(ctx, mat_data, rows, cols),
            .q6_k => try initQ6K(ctx, mat_data, rows, cols),
            .q5_k => try initQ5K(ctx, mat_data, rows, cols),
            .iq4_nl => try initIQ4NL(ctx, mat_data, rows, cols),
            else => null,
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

        var mat_buf = try GpuBuffer.initDeviceLocal(ctx, mat_bytes.len, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
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
        var vb = try GpuBuffer.initHostCoherent(ctx, self.cols * @sizeOf(f32), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        defer vb.deinit();
        var ob = try GpuBuffer.initHostCoherent(ctx, self.rows * @sizeOf(f32), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
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
    pipeline: vk.VkPipeline,
    layout: vk.VkPipelineLayout,
    dset_layout: vk.VkDescriptorSetLayout,
    desc_pool: vk.VkDescriptorPool,
    device: vk.VkDevice,
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
        try ctx.createComputePipeline(&pipeline_ci, &pipeline);
        errdefer vk.vkDestroyPipeline(dev, pipeline, null);

        // 16 sets: one per active expert per layer (n_experts_used = 8 typical)
        const pool_size = vk.VkDescriptorPoolSize{
            .type = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 4 * 16,
        };
        const pool_ci = vk.VkDescriptorPoolCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = null,
            .flags = vk.VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            .maxSets = 16,
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

    // Record one fused gate-gelu-up dispatch into an open command buffer.
    // Returns the descriptor set; caller frees it after the submit completes.
    pub fn record(
        self: *const FusedGateUpPipeline,
        cmd: vk.VkCommandBuffer,
        gate_buf: *const GpuBuffer,
        up_buf: *const GpuBuffer,
        vec_buf: *const GpuBuffer,
        out_buf: *const GpuBuffer,
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

        const buf_infos = [4]vk.VkDescriptorBufferInfo{
            .{ .buffer = gate_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = up_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = vec_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = out_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        };
        const writes = [4]vk.VkWriteDescriptorSet{
            mkWrite(dset, 0, &buf_infos[0]), mkWrite(dset, 1, &buf_infos[1]),
            mkWrite(dset, 2, &buf_infos[2]), mkWrite(dset, 3, &buf_infos[3]),
        };
        vk.vkUpdateDescriptorSets(dev, writes.len, &writes, 0, null);

        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.layout, 0, 1, &dset, 0, null);
        const pc = PushConst{ .rows = rows, .cols = cols };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(PushConst), &pc);
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
    pipeline: vk.VkPipeline,
    layout: vk.VkPipelineLayout,
    dset_layout: vk.VkDescriptorSetLayout,
    desc_pool: vk.VkDescriptorPool,
    device: vk.VkDevice,

    pub fn init(ctx: *const GpuContext) !AccumPipeline {
        comptime std.debug.assert(shaders.expert_accum.len % 4 == 0);
        const dev = ctx.device;

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

        const spv = &shaders.expert_accum;
        std.debug.assert(@intFromPtr(spv) % 4 == 0);
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
        try ctx.createComputePipeline(&pipeline_ci, &pipeline);
        errdefer vk.vkDestroyPipeline(dev, pipeline, null);

        const pool_size = vk.VkDescriptorPoolSize{
            .type = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 3 * 4,
        };
        const pool_ci = vk.VkDescriptorPoolCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = null,
            .flags = vk.VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            .maxSets = 4,
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

    pub fn record(
        self: *const AccumPipeline,
        cmd: vk.VkCommandBuffer,
        inputs_buf: *const GpuBuffer,
        scales_buf: *const GpuBuffer,
        out_buf: *const GpuBuffer,
        d_model: u32,
        n: u32,
    ) !vk.VkDescriptorSet {
        const dset = try self.allocSet(inputs_buf, scales_buf, out_buf);
        self.recordWithSet(cmd, dset, d_model, n);
        return dset;
    }

    pub fn allocSet(
        self: *const AccumPipeline,
        inputs_buf: *const GpuBuffer,
        scales_buf: *const GpuBuffer,
        out_buf: *const GpuBuffer,
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
            .{ .buffer = inputs_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = scales_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = out_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        };
        const writes = [3]vk.VkWriteDescriptorSet{
            mkWrite(dset, 0, &buf_infos[0]),
            mkWrite(dset, 1, &buf_infos[1]),
            mkWrite(dset, 2, &buf_infos[2]),
        };
        vk.vkUpdateDescriptorSets(dev, writes.len, &writes, 0, null);

        return dset;
    }

    pub fn recordWithSet(
        self: *const AccumPipeline,
        cmd: vk.VkCommandBuffer,
        dset: vk.VkDescriptorSet,
        d_model: u32,
        n: u32,
    ) void {
        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.layout, 0, 1, &dset, 0, null);
        // PushConst: rows=d_model, cols=n (reuses the 2×u32 push constant struct)
        const pc = PushConst{ .rows = d_model, .cols = n };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(PushConst), &pc);
        const groups = (d_model + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        vk.vkCmdDispatch(cmd, groups, 1, 1);
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
const QuantizeBatchedPushConst = extern struct { ncols: u32, n_active: u32 };

pub const QuantizeQ8_1Pipeline = struct {
    pipeline: vk.VkPipeline,
    layout: vk.VkPipelineLayout,
    dset_layout: vk.VkDescriptorSetLayout,
    desc_pool: vk.VkDescriptorPool,
    device: vk.VkDevice,

    pub fn init(ctx: *const GpuContext) !QuantizeQ8_1Pipeline {
        comptime std.debug.assert(shaders.quantize_q8_1.len % 4 == 0);
        const dev = ctx.device;

        const bindings = [2]vk.VkDescriptorSetLayoutBinding{
            mkStorageBuf(0), mkStorageBuf(1),
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
            .size = @sizeOf(QuantizePushConst),
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

        const spv = &shaders.quantize_q8_1;
        std.debug.assert(@intFromPtr(spv) % 4 == 0);
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
        try ctx.createComputePipeline(&pipeline_ci, &pipeline);
        errdefer vk.vkDestroyPipeline(dev, pipeline, null);

        const pool_size = vk.VkDescriptorPoolSize{
            .type = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 2 * 16,
        };
        const pool_ci = vk.VkDescriptorPoolCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = null,
            .flags = vk.VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            .maxSets = 16,
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

    // Record a Q8_1 quantization dispatch into an already-recording command buffer.
    pub fn record(
        self: *const QuantizeQ8_1Pipeline,
        cmd: vk.VkCommandBuffer,
        in_buf: *const GpuBuffer,
        out_buf: *const GpuBuffer,
        ncols: u32,
    ) !vk.VkDescriptorSet {
        std.debug.assert(ncols % 4 == 0); // vec4-aligned reads
        const dset = try self.allocSet(in_buf, out_buf);
        self.recordWithSet(cmd, dset, ncols);
        return dset;
    }

    pub fn allocSet(
        self: *const QuantizeQ8_1Pipeline,
        in_buf: *const GpuBuffer,
        out_buf: *const GpuBuffer,
    ) !vk.VkDescriptorSet {
        return self.allocSetRangeToOffset(in_buf, 0, vk.VK_WHOLE_SIZE, out_buf, 0, vk.VK_WHOLE_SIZE);
    }

    pub fn allocSetToOffset(
        self: *const QuantizeQ8_1Pipeline,
        in_buf: *const GpuBuffer,
        out_buf: *const GpuBuffer,
        out_offset: u64,
        out_range: u64,
    ) !vk.VkDescriptorSet {
        return self.allocSetRangeToOffset(in_buf, 0, vk.VK_WHOLE_SIZE, out_buf, out_offset, out_range);
    }

    pub fn allocSetRangeToOffset(
        self: *const QuantizeQ8_1Pipeline,
        in_buf: *const GpuBuffer,
        in_offset: u64,
        in_range: u64,
        out_buf: *const GpuBuffer,
        out_offset: u64,
        out_range: u64,
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

        const buf_infos = [2]vk.VkDescriptorBufferInfo{
            .{ .buffer = in_buf.handle, .offset = in_offset, .range = in_range },
            .{ .buffer = out_buf.handle, .offset = out_offset, .range = out_range },
        };
        const writes = [2]vk.VkWriteDescriptorSet{
            mkWrite(dset, 0, &buf_infos[0]),
            mkWrite(dset, 1, &buf_infos[1]),
        };
        vk.vkUpdateDescriptorSets(dev, writes.len, &writes, 0, null);

        return dset;
    }

    pub fn recordWithSet(
        self: *const QuantizeQ8_1Pipeline,
        cmd: vk.VkCommandBuffer,
        dset: vk.VkDescriptorSet,
        ncols: u32,
    ) void {
        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.layout, 0, 1, &dset, 0, null);
        const pc = QuantizePushConst{ .ncols = ncols };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(QuantizePushConst), &pc);
        // One workgroup per 128-element x4 group (32 threads each).
        const groups = (ncols + 127) / 128;
        vk.vkCmdDispatch(cmd, groups, 1, 1);
    }

    pub fn recordToOffset(
        self: *const QuantizeQ8_1Pipeline,
        cmd: vk.VkCommandBuffer,
        in_buf: *const GpuBuffer,
        out_buf: *const GpuBuffer,
        out_offset: u64,
        out_range: u64,
        ncols: u32,
    ) !vk.VkDescriptorSet {
        std.debug.assert(ncols % 4 == 0);
        const dset = try self.allocSetToOffset(in_buf, out_buf, out_offset, out_range);
        self.recordWithSet(cmd, dset, ncols);
        return dset;
    }

    pub fn recordRangeToOffset(
        self: *const QuantizeQ8_1Pipeline,
        cmd: vk.VkCommandBuffer,
        in_buf: *const GpuBuffer,
        in_offset: u64,
        in_range: u64,
        out_buf: *const GpuBuffer,
        out_offset: u64,
        out_range: u64,
        ncols: u32,
    ) !vk.VkDescriptorSet {
        std.debug.assert(ncols % 4 == 0);
        const dset = try self.allocSetRangeToOffset(in_buf, in_offset, in_range, out_buf, out_offset, out_range);
        self.recordWithSet(cmd, dset, ncols);
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

pub const QuantizeQ8_1BatchedPipeline = struct {
    pipeline: vk.VkPipeline,
    layout: vk.VkPipelineLayout,
    dset_layout: vk.VkDescriptorSetLayout,
    desc_pool: vk.VkDescriptorPool,
    device: vk.VkDevice,

    pub fn init(ctx: *const GpuContext) !QuantizeQ8_1BatchedPipeline {
        comptime std.debug.assert(shaders.quantize_q8_1_batched.len % 4 == 0);
        const built = try buildSimplePipeline(ctx, &shaders.quantize_q8_1_batched, 2, @sizeOf(QuantizeBatchedPushConst), 16);
        return .{
            .pipeline = built.pipeline,
            .layout = built.layout,
            .dset_layout = built.dset_layout,
            .desc_pool = built.desc_pool,
            .device = ctx.device,
        };
    }

    pub fn record(
        self: *const QuantizeQ8_1BatchedPipeline,
        cmd: vk.VkCommandBuffer,
        in_buf: *const GpuBuffer,
        out_buf: *const GpuBuffer,
        ncols: u32,
        active: u32,
    ) !vk.VkDescriptorSet {
        std.debug.assert(ncols % 4 == 0);
        const dset = try self.allocSet(in_buf, out_buf);
        self.recordWithSet(cmd, dset, ncols, active);
        return dset;
    }

    pub fn allocSet(
        self: *const QuantizeQ8_1BatchedPipeline,
        in_buf: *const GpuBuffer,
        out_buf: *const GpuBuffer,
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

        const buf_infos = [2]vk.VkDescriptorBufferInfo{
            .{ .buffer = in_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = out_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        };
        const writes = [2]vk.VkWriteDescriptorSet{
            mkWrite(dset, 0, &buf_infos[0]),
            mkWrite(dset, 1, &buf_infos[1]),
        };
        vk.vkUpdateDescriptorSets(dev, writes.len, &writes, 0, null);

        return dset;
    }

    pub fn recordWithSet(
        self: *const QuantizeQ8_1BatchedPipeline,
        cmd: vk.VkCommandBuffer,
        dset: vk.VkDescriptorSet,
        ncols: u32,
        active: u32,
    ) void {
        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.layout, 0, 1, &dset, 0, null);
        const pc = QuantizeBatchedPushConst{ .ncols = ncols, .n_active = active };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(QuantizeBatchedPushConst), &pc);
        const groups_x = (ncols + 127) / 128;
        vk.vkCmdDispatch(cmd, groups_x, active, 1);
    }

    pub fn deinit(self: *QuantizeQ8_1BatchedPipeline) void {
        vk.vkDestroyDescriptorPool(self.device, self.desc_pool, null);
        vk.vkDestroyPipeline(self.device, self.pipeline, null);
        vk.vkDestroyPipelineLayout(self.device, self.layout, null);
        vk.vkDestroyDescriptorSetLayout(self.device, self.dset_layout, null);
    }
};

const ExpertGateUpIdPushConst = extern struct {
    rows: u32,
    cols: u32,
    n_active: u32,
};

pub const ExpertGateUpIdPipeline = struct {
    pipeline: vk.VkPipeline,
    layout: vk.VkPipelineLayout,
    dset_layout: vk.VkDescriptorSetLayout,
    desc_pool: vk.VkDescriptorPool,
    device: vk.VkDevice,
    rows_per_workgroup: u32,

    pub fn initQ3KQ8_1(ctx: *const GpuContext) !ExpertGateUpIdPipeline {
        comptime std.debug.assert(shaders.expert_gate_up_id_q3_k_q8_1.len % 4 == 0);
        return initFromSpv(ctx, &shaders.expert_gate_up_id_q3_k_q8_1, 1);
    }

    pub fn initQ3KQ8_1R2(ctx: *const GpuContext) !ExpertGateUpIdPipeline {
        comptime std.debug.assert(shaders.expert_gate_up_id_q3_k_q8_1_r2.len % 4 == 0);
        return initFromSpv(ctx, &shaders.expert_gate_up_id_q3_k_q8_1_r2, 2);
    }

    pub fn initQ3KQ8_1R4(ctx: *const GpuContext) !ExpertGateUpIdPipeline {
        comptime std.debug.assert(shaders.expert_gate_up_id_q3_k_q8_1_r4.len % 4 == 0);
        return initFromSpv(ctx, &shaders.expert_gate_up_id_q3_k_q8_1_r4, 4);
    }

    fn initFromSpv(ctx: *const GpuContext, spv: []align(4) const u8, rows_per_workgroup: u32) !ExpertGateUpIdPipeline {
        const built = try buildSimplePipeline(ctx, spv, 4, @sizeOf(ExpertGateUpIdPushConst), 64);
        return .{
            .pipeline = built.pipeline,
            .layout = built.layout,
            .dset_layout = built.dset_layout,
            .desc_pool = built.desc_pool,
            .device = ctx.device,
            .rows_per_workgroup = rows_per_workgroup,
        };
    }

    pub fn record(
        self: *const ExpertGateUpIdPipeline,
        cmd: vk.VkCommandBuffer,
        weights_buf: *const GpuBuffer,
        acts_buf: *const GpuBuffer,
        ids_buf: *const GpuBuffer,
        out_buf: *const GpuBuffer,
        rows: u32,
        cols: u32,
        active: u32,
    ) !vk.VkDescriptorSet {
        const dset = try self.allocSet(weights_buf, acts_buf, ids_buf, out_buf);
        self.recordWithSet(cmd, dset, rows, cols, active);
        return dset;
    }

    pub fn allocSet(
        self: *const ExpertGateUpIdPipeline,
        weights_buf: *const GpuBuffer,
        acts_buf: *const GpuBuffer,
        ids_buf: *const GpuBuffer,
        out_buf: *const GpuBuffer,
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

        const buf_infos = [4]vk.VkDescriptorBufferInfo{
            .{ .buffer = weights_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = acts_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = ids_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = out_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        };
        const writes = [4]vk.VkWriteDescriptorSet{
            mkWrite(dset, 0, &buf_infos[0]),
            mkWrite(dset, 1, &buf_infos[1]),
            mkWrite(dset, 2, &buf_infos[2]),
            mkWrite(dset, 3, &buf_infos[3]),
        };
        vk.vkUpdateDescriptorSets(dev, writes.len, &writes, 0, null);

        return dset;
    }

    pub fn recordWithSet(
        self: *const ExpertGateUpIdPipeline,
        cmd: vk.VkCommandBuffer,
        dset: vk.VkDescriptorSet,
        rows: u32,
        cols: u32,
        active: u32,
    ) void {
        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.layout, 0, 1, &dset, 0, null);
        const pc = ExpertGateUpIdPushConst{ .rows = rows, .cols = cols, .n_active = active };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(ExpertGateUpIdPushConst), &pc);
        const groups_x = (rows + self.rows_per_workgroup - 1) / self.rows_per_workgroup;
        vk.vkCmdDispatch(cmd, groups_x, active, 1);
    }

    pub fn deinit(self: *ExpertGateUpIdPipeline) void {
        vk.vkDestroyDescriptorPool(self.device, self.desc_pool, null);
        vk.vkDestroyPipeline(self.device, self.pipeline, null);
        vk.vkDestroyPipelineLayout(self.device, self.layout, null);
        vk.vkDestroyDescriptorSetLayout(self.device, self.dset_layout, null);
    }
};

const ExpertDownIdPushConst = extern struct {
    rows: u32,
    cols: u32,
    n_active: u32,
};

pub const ExpertDownIdPipeline = struct {
    pipeline: vk.VkPipeline,
    layout: vk.VkPipelineLayout,
    dset_layout: vk.VkDescriptorSetLayout,
    desc_pool: vk.VkDescriptorPool,
    device: vk.VkDevice,
    rows_per_workgroup: u32,
    dispatch_active_groups: bool,

    pub fn initQ5_0Q8_1(ctx: *const GpuContext) !ExpertDownIdPipeline {
        comptime std.debug.assert(shaders.expert_down_id_q5_0_q8_1.len % 4 == 0);
        return initFromSpv(ctx, &shaders.expert_down_id_q5_0_q8_1, 1, true);
    }

    pub fn initQ5_1Q8_1(ctx: *const GpuContext) !ExpertDownIdPipeline {
        comptime std.debug.assert(shaders.expert_down_id_q5_1_q8_1.len % 4 == 0);
        return initFromSpv(ctx, &shaders.expert_down_id_q5_1_q8_1, 1, true);
    }

    pub fn initIQ4NLQ8_1(ctx: *const GpuContext) !ExpertDownIdPipeline {
        comptime std.debug.assert(shaders.expert_down_id_iq4_nl_q8_1.len % 4 == 0);
        return initFromSpv(ctx, &shaders.expert_down_id_iq4_nl_q8_1, 1, true);
    }

    pub fn initIQ4NLQ8_1R2(ctx: *const GpuContext) !ExpertDownIdPipeline {
        comptime std.debug.assert(shaders.expert_down_id_iq4_nl_q8_1_r2.len % 4 == 0);
        return initFromSpv(ctx, &shaders.expert_down_id_iq4_nl_q8_1_r2, 2, true);
    }

    pub fn initIQ4NLQ8_1B16(ctx: *const GpuContext) !ExpertDownIdPipeline {
        comptime std.debug.assert(shaders.expert_down_id_iq4_nl_q8_1_b16.len % 4 == 0);
        return initFromSpv(ctx, &shaders.expert_down_id_iq4_nl_q8_1_b16, 1, true);
    }

    pub fn initIQ4NLQ8_1Iacc(ctx: *const GpuContext) !ExpertDownIdPipeline {
        comptime std.debug.assert(shaders.expert_down_id_iq4_nl_q8_1_iacc.len % 4 == 0);
        return initFromSpv(ctx, &shaders.expert_down_id_iq4_nl_q8_1_iacc, 1, true);
    }

    pub fn initIQ4NLQ8_1Sum(ctx: *const GpuContext) !ExpertDownIdPipeline {
        comptime std.debug.assert(shaders.expert_down_sum_id_iq4_nl_q8_1.len % 4 == 0);
        return initFromSpv(ctx, &shaders.expert_down_sum_id_iq4_nl_q8_1, 1, false);
    }

    fn initFromSpv(ctx: *const GpuContext, spv: anytype, rows_per_workgroup: u32, dispatch_active_groups: bool) !ExpertDownIdPipeline {
        const built = try buildSimplePipeline(ctx, spv, 5, @sizeOf(ExpertDownIdPushConst), 64);
        return .{
            .pipeline = built.pipeline,
            .layout = built.layout,
            .dset_layout = built.dset_layout,
            .desc_pool = built.desc_pool,
            .device = ctx.device,
            .rows_per_workgroup = rows_per_workgroup,
            .dispatch_active_groups = dispatch_active_groups,
        };
    }

    pub fn record(
        self: *const ExpertDownIdPipeline,
        cmd: vk.VkCommandBuffer,
        weights_buf: *const GpuBuffer,
        acts_buf: *const GpuBuffer,
        ids_buf: *const GpuBuffer,
        scales_buf: *const GpuBuffer,
        out_buf: *const GpuBuffer,
        rows: u32,
        cols: u32,
        active: u32,
    ) !vk.VkDescriptorSet {
        const dset = try self.allocSet(weights_buf, acts_buf, ids_buf, scales_buf, out_buf);
        self.recordWithSet(cmd, dset, rows, cols, active);
        return dset;
    }

    pub fn allocSet(
        self: *const ExpertDownIdPipeline,
        weights_buf: *const GpuBuffer,
        acts_buf: *const GpuBuffer,
        ids_buf: *const GpuBuffer,
        scales_buf: *const GpuBuffer,
        out_buf: *const GpuBuffer,
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

        const buf_infos = [5]vk.VkDescriptorBufferInfo{
            .{ .buffer = weights_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = acts_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = ids_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = scales_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = out_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        };
        const writes = [5]vk.VkWriteDescriptorSet{
            mkWrite(dset, 0, &buf_infos[0]),
            mkWrite(dset, 1, &buf_infos[1]),
            mkWrite(dset, 2, &buf_infos[2]),
            mkWrite(dset, 3, &buf_infos[3]),
            mkWrite(dset, 4, &buf_infos[4]),
        };
        vk.vkUpdateDescriptorSets(dev, writes.len, &writes, 0, null);

        return dset;
    }

    pub fn recordWithSet(
        self: *const ExpertDownIdPipeline,
        cmd: vk.VkCommandBuffer,
        dset: vk.VkDescriptorSet,
        rows: u32,
        cols: u32,
        active: u32,
    ) void {
        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.layout, 0, 1, &dset, 0, null);
        const pc = ExpertDownIdPushConst{ .rows = rows, .cols = cols, .n_active = active };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(ExpertDownIdPushConst), &pc);
        const groups_x = (rows + self.rows_per_workgroup - 1) / self.rows_per_workgroup;
        const groups_y = if (self.dispatch_active_groups) active else 1;
        vk.vkCmdDispatch(cmd, groups_x, groups_y, 1);
    }

    pub fn deinit(self: *ExpertDownIdPipeline) void {
        vk.vkDestroyDescriptorPool(self.device, self.desc_pool, null);
        vk.vkDestroyPipeline(self.device, self.pipeline, null);
        vk.vkDestroyPipelineLayout(self.device, self.layout, null);
        vk.vkDestroyDescriptorSetLayout(self.device, self.dset_layout, null);
    }
};

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
    pipeline: vk.VkPipeline,
    layout: vk.VkPipelineLayout,
    dset_layout: vk.VkDescriptorSetLayout,
    desc_pool: vk.VkDescriptorPool,
    device: vk.VkDevice,
    workgroup_size: u32,

    pub fn init(ctx: *const GpuContext) !RmsnormPipeline {
        comptime std.debug.assert(shaders.rmsnorm.len % 4 == 0);
        return initFromSpv(ctx, &shaders.rmsnorm, 256);
    }

    pub fn initR128(ctx: *const GpuContext) !RmsnormPipeline {
        comptime std.debug.assert(shaders.rmsnorm_128.len % 4 == 0);
        return initFromSpv(ctx, &shaders.rmsnorm_128, 128);
    }

    fn initFromSpv(ctx: *const GpuContext, spv: []align(4) const u8, workgroup_size: u32) !RmsnormPipeline {
        const dev = ctx.device;

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
            .size = @sizeOf(RmsnormPushConst),
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
        try ctx.createComputePipeline(&pipeline_ci, &pipeline);
        errdefer vk.vkDestroyPipeline(dev, pipeline, null);

        const pool_size = vk.VkDescriptorPoolSize{
            .type = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 3 * 256,
        };
        const pool_ci = vk.VkDescriptorPoolCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = null,
            .flags = vk.VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            .maxSets = 256,
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
            .workgroup_size = workgroup_size,
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
        return self.recordWithMode(cmd, x_buf, w_buf, y_buf, n, eps, weight_offset, false);
    }

    pub fn recordPrecise(
        self: *const RmsnormPipeline,
        cmd: vk.VkCommandBuffer,
        x_buf: *const GpuBuffer,
        w_buf: *const GpuBuffer,
        y_buf: *const GpuBuffer,
        n: u32,
        eps: f32,
        weight_offset: bool,
    ) !vk.VkDescriptorSet {
        return self.recordWithMode(cmd, x_buf, w_buf, y_buf, n, eps, weight_offset, true);
    }

    fn recordWithMode(
        self: *const RmsnormPipeline,
        cmd: vk.VkCommandBuffer,
        x_buf: *const GpuBuffer,
        w_buf: *const GpuBuffer,
        y_buf: *const GpuBuffer,
        n: u32,
        eps: f32,
        weight_offset: bool,
        precise_sum: bool,
    ) !vk.VkDescriptorSet {
        std.debug.assert(n % self.workgroup_size == 0);
        const dset = try self.allocSet(x_buf, w_buf, y_buf);
        self.recordWithSetMode(cmd, dset, n, eps, weight_offset, precise_sum);
        return dset;
    }

    pub fn allocSet(
        self: *const RmsnormPipeline,
        x_buf: *const GpuBuffer,
        w_buf: *const GpuBuffer,
        y_buf: *const GpuBuffer,
    ) !vk.VkDescriptorSet {
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
        vk.vkUpdateDescriptorSets(self.device, writes.len, &writes, 0, null);
        return dset;
    }

    pub fn recordWithSet(
        self: *const RmsnormPipeline,
        cmd: vk.VkCommandBuffer,
        dset: vk.VkDescriptorSet,
        n: u32,
        eps: f32,
        weight_offset: bool,
    ) void {
        self.recordWithSetMode(cmd, dset, n, eps, weight_offset, false);
    }

    pub fn recordPreciseWithSet(
        self: *const RmsnormPipeline,
        cmd: vk.VkCommandBuffer,
        dset: vk.VkDescriptorSet,
        n: u32,
        eps: f32,
        weight_offset: bool,
    ) void {
        self.recordWithSetMode(cmd, dset, n, eps, weight_offset, true);
    }

    fn recordWithSetMode(
        self: *const RmsnormPipeline,
        cmd: vk.VkCommandBuffer,
        dset: vk.VkDescriptorSet,
        n: u32,
        eps: f32,
        weight_offset: bool,
        precise_sum: bool,
    ) void {
        std.debug.assert(n % self.workgroup_size == 0);
        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.layout, 0, 1, &dset, 0, null);

        const pc = RmsnormPushConst{
            .n = n,
            .eps = eps,
            .weight_offset = (if (weight_offset) @as(u32, 1) else 0) |
                (if (precise_sum) @as(u32, 2) else 0),
        };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(RmsnormPushConst), &pc);
        vk.vkCmdDispatch(cmd, 1, 1, 1);
    }

    pub fn deinit(self: *RmsnormPipeline) void {
        vk.vkDestroyDescriptorPool(self.device, self.desc_pool, null);
        vk.vkDestroyPipeline(self.device, self.pipeline, null);
        vk.vkDestroyPipelineLayout(self.device, self.layout, null);
        vk.vkDestroyDescriptorSetLayout(self.device, self.dset_layout, null);
    }
};

// ── AddRmsnormPipeline ────────────────────────────────────────────────────────
//
// Fuses an elementwise add with the following RMSNorm:
//   y[i] = (a[i] + b[i]) * rms_inv * (bias + w[i])
// This is used by the MoE tail to replace combine + post_ffw_norm.

pub const AddRmsnormPipeline = struct {
    pipeline: vk.VkPipeline,
    layout: vk.VkPipelineLayout,
    dset_layout: vk.VkDescriptorSetLayout,
    desc_pool: vk.VkDescriptorPool,
    device: vk.VkDevice,
    workgroup_size: u32,

    pub fn init(ctx: *const GpuContext) !AddRmsnormPipeline {
        comptime std.debug.assert(shaders.add_rmsnorm.len % 4 == 0);
        return initFromSpv(ctx, &shaders.add_rmsnorm, 256);
    }

    pub fn initR128(ctx: *const GpuContext) !AddRmsnormPipeline {
        comptime std.debug.assert(shaders.add_rmsnorm_128.len % 4 == 0);
        return initFromSpv(ctx, &shaders.add_rmsnorm_128, 128);
    }

    fn initFromSpv(ctx: *const GpuContext, spv: []align(4) const u8, workgroup_size: u32) !AddRmsnormPipeline {
        const built = try buildSimplePipeline(ctx, spv, 4, @sizeOf(RmsnormPushConst), 64);
        return .{
            .pipeline = built.pipeline,
            .layout = built.layout,
            .dset_layout = built.dset_layout,
            .desc_pool = built.desc_pool,
            .device = ctx.device,
            .workgroup_size = workgroup_size,
        };
    }

    pub fn allocSet(
        self: *const AddRmsnormPipeline,
        a_buf: *const GpuBuffer,
        b_buf: *const GpuBuffer,
        w_buf: *const GpuBuffer,
        y_buf: *const GpuBuffer,
    ) !vk.VkDescriptorSet {
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

        const buf_infos = [4]vk.VkDescriptorBufferInfo{
            .{ .buffer = a_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = b_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = w_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = y_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        };
        const writes = [4]vk.VkWriteDescriptorSet{
            mkWrite(dset, 0, &buf_infos[0]),
            mkWrite(dset, 1, &buf_infos[1]),
            mkWrite(dset, 2, &buf_infos[2]),
            mkWrite(dset, 3, &buf_infos[3]),
        };
        vk.vkUpdateDescriptorSets(self.device, writes.len, &writes, 0, null);
        return dset;
    }

    pub fn recordWithSet(
        self: *const AddRmsnormPipeline,
        cmd: vk.VkCommandBuffer,
        dset: vk.VkDescriptorSet,
        n: u32,
        eps: f32,
        weight_offset: bool,
        precise_sum: bool,
    ) void {
        std.debug.assert(n % self.workgroup_size == 0);
        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.layout, 0, 1, &dset, 0, null);
        const pc = RmsnormPushConst{
            .n = n,
            .eps = eps,
            .weight_offset = (if (weight_offset) @as(u32, 1) else 0) |
                (if (precise_sum) @as(u32, 2) else 0),
        };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(RmsnormPushConst), &pc);
        vk.vkCmdDispatch(cmd, 1, 1, 1);
    }

    pub fn deinit(self: *AddRmsnormPipeline) void {
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
    pipeline: vk.VkPipeline,
    layout: vk.VkPipelineLayout,
    dset_layout: vk.VkDescriptorSetLayout,
    desc_pool: vk.VkDescriptorPool,
    device: vk.VkDevice,

    pub fn init(ctx: *const GpuContext) !RmsnormPerHeadPipeline {
        comptime std.debug.assert(shaders.rmsnorm_perhead.len % 4 == 0);
        const dev = ctx.device;

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
            .size = @sizeOf(RmsnormPerHeadPushConst),
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

        const spv = &shaders.rmsnorm_perhead;
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
        try ctx.createComputePipeline(&pipeline_ci, &pipeline);
        errdefer vk.vkDestroyPipeline(dev, pipeline, null);

        const pool_size = vk.VkDescriptorPoolSize{
            .type = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 3 * 128,
        };
        const pool_ci = vk.VkDescriptorPoolCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = null,
            .flags = vk.VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            .maxSets = 128,
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

    pub fn record(
        self: *const RmsnormPerHeadPipeline,
        cmd: vk.VkCommandBuffer,
        x_buf: *const GpuBuffer,
        w_buf: *const GpuBuffer, // may equal x_buf when use_weight==false (unread)
        y_buf: *const GpuBuffer,
        n_heads: u32,
        head_dim: u32,
        eps: f32,
        weight_offset: bool,
        use_weight: bool,
    ) !vk.VkDescriptorSet {
        std.debug.assert(head_dim % 256 == 0);
        const dset = try self.allocSet(x_buf, w_buf, y_buf);
        self.recordWithSet(cmd, dset, n_heads, head_dim, eps, weight_offset, use_weight);
        return dset;
    }

    pub fn allocSet(
        self: *const RmsnormPerHeadPipeline,
        x_buf: *const GpuBuffer,
        w_buf: *const GpuBuffer,
        y_buf: *const GpuBuffer,
    ) !vk.VkDescriptorSet {
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
        vk.vkUpdateDescriptorSets(self.device, writes.len, &writes, 0, null);
        return dset;
    }

    pub fn recordWithSet(
        self: *const RmsnormPerHeadPipeline,
        cmd: vk.VkCommandBuffer,
        dset: vk.VkDescriptorSet,
        n_heads: u32,
        head_dim: u32,
        eps: f32,
        weight_offset: bool,
        use_weight: bool,
    ) void {
        std.debug.assert(head_dim % 256 == 0);
        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.layout, 0, 1, &dset, 0, null);

        const pc = RmsnormPerHeadPushConst{
            .head_dim = head_dim,
            .eps = eps,
            .weight_offset = if (weight_offset) 1 else 0,
            .use_weight = if (use_weight) 1 else 0,
        };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(RmsnormPerHeadPushConst), &pc);
        vk.vkCmdDispatch(cmd, n_heads, 1, 1);
    }

    pub fn deinit(self: *RmsnormPerHeadPipeline) void {
        vk.vkDestroyDescriptorPool(self.device, self.desc_pool, null);
        vk.vkDestroyPipeline(self.device, self.pipeline, null);
        vk.vkDestroyPipelineLayout(self.device, self.layout, null);
        vk.vkDestroyDescriptorSetLayout(self.device, self.dset_layout, null);
    }
};

// ── RmsnormPerHeadRope pipelines ─────────────────────────────────────────────
//
// llama.cpp fuses RMSNorm + multiply + RoPE when the graph edge allows it.
// Gemma4 attention has that exact shape for Q and K, so these variants keep
// the normalized head in shared memory and write only the rotated result.

const RmsnormPerHeadRopeTablePushConst = extern struct { head_dim: u32, eps: f32, pos: u32 };
const RmsnormPerHeadRopeThetaPushConst = extern struct { head_dim: u32, eps: f32, pos: u32, theta: f32 };

pub const RmsnormPerHeadRopeTablePipeline = struct {
    pipeline: vk.VkPipeline,
    layout: vk.VkPipelineLayout,
    dset_layout: vk.VkDescriptorSetLayout,
    desc_pool: vk.VkDescriptorPool,
    device: vk.VkDevice,

    pub fn init(ctx: *const GpuContext) !RmsnormPerHeadRopeTablePipeline {
        comptime std.debug.assert(shaders.rmsnorm_perhead_rope_table.len % 4 == 0);
        const built = try buildSimplePipeline(ctx, &shaders.rmsnorm_perhead_rope_table, 4, @sizeOf(RmsnormPerHeadRopeTablePushConst), 128);
        return .{ .pipeline = built.pipeline, .layout = built.layout, .dset_layout = built.dset_layout, .desc_pool = built.desc_pool, .device = ctx.device };
    }

    pub fn allocSet(self: *const RmsnormPerHeadRopeTablePipeline, x: *const GpuBuffer, w: *const GpuBuffer, freqs: *const GpuBuffer, y: *const GpuBuffer) !vk.VkDescriptorSet {
        return allocSimpleSet(self.device, self.desc_pool, self.dset_layout, &.{ x, w, freqs, y });
    }

    pub fn recordWithSet(self: *const RmsnormPerHeadRopeTablePipeline, cmd: vk.VkCommandBuffer, dset: vk.VkDescriptorSet, n_heads: u32, head_dim: u32, eps: f32, pos: u32) void {
        std.debug.assert(head_dim <= 512 and head_dim % 256 == 0);
        bindSimple(cmd, self.pipeline, self.layout, dset);
        const pc = RmsnormPerHeadRopeTablePushConst{ .head_dim = head_dim, .eps = eps, .pos = pos };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(@TypeOf(pc)), &pc);
        vk.vkCmdDispatch(cmd, n_heads, 1, 1);
    }

    pub fn deinit(self: *RmsnormPerHeadRopeTablePipeline) void {
        deinitSimplePipeline(self.device, self.desc_pool, self.pipeline, self.layout, self.dset_layout);
    }
};

pub const RmsnormPerHeadRopeThetaPipeline = struct {
    pipeline: vk.VkPipeline,
    layout: vk.VkPipelineLayout,
    dset_layout: vk.VkDescriptorSetLayout,
    desc_pool: vk.VkDescriptorPool,
    device: vk.VkDevice,

    pub fn init(ctx: *const GpuContext) !RmsnormPerHeadRopeThetaPipeline {
        comptime std.debug.assert(shaders.rmsnorm_perhead_rope_theta.len % 4 == 0);
        const built = try buildSimplePipeline(ctx, &shaders.rmsnorm_perhead_rope_theta, 3, @sizeOf(RmsnormPerHeadRopeThetaPushConst), 128);
        return .{ .pipeline = built.pipeline, .layout = built.layout, .dset_layout = built.dset_layout, .desc_pool = built.desc_pool, .device = ctx.device };
    }

    pub fn allocSet(self: *const RmsnormPerHeadRopeThetaPipeline, x: *const GpuBuffer, w: *const GpuBuffer, y: *const GpuBuffer) !vk.VkDescriptorSet {
        return allocSimpleSet(self.device, self.desc_pool, self.dset_layout, &.{ x, w, y });
    }

    pub fn recordWithSet(self: *const RmsnormPerHeadRopeThetaPipeline, cmd: vk.VkCommandBuffer, dset: vk.VkDescriptorSet, n_heads: u32, head_dim: u32, eps: f32, pos: u32, theta: f32) void {
        std.debug.assert(head_dim <= 512 and head_dim % 256 == 0);
        bindSimple(cmd, self.pipeline, self.layout, dset);
        const pc = RmsnormPerHeadRopeThetaPushConst{ .head_dim = head_dim, .eps = eps, .pos = pos, .theta = theta };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(@TypeOf(pc)), &pc);
        vk.vkCmdDispatch(cmd, n_heads, 1, 1);
    }

    pub fn deinit(self: *RmsnormPerHeadRopeThetaPipeline) void {
        deinitSimplePipeline(self.device, self.desc_pool, self.pipeline, self.layout, self.dset_layout);
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

    // Variable-size descriptor binding list. Expert-id kernels need five
    // buffers: weights, activations, ids, scales, output.
    std.debug.assert(n_bindings >= 1 and n_bindings <= 5);
    var bindings_buf: [5]vk.VkDescriptorSetLayoutBinding = undefined;
    var b: u32 = 0;
    while (b < n_bindings) : (b += 1) bindings_buf[b] = mkStorageBuf(b);
    const dsl_ci = vk.VkDescriptorSetLayoutCreateInfo{
        .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .bindingCount = n_bindings,
        .pBindings = &bindings_buf,
    };
    var dset_layout: vk.VkDescriptorSetLayout = null;
    if (vk.vkCreateDescriptorSetLayout(dev, &dsl_ci, null, &dset_layout) != vk.VK_SUCCESS)
        return error.VkDescSetLayoutFailed;
    errdefer vk.vkDestroyDescriptorSetLayout(dev, dset_layout, null);

    const pc_range = vk.VkPushConstantRange{
        .stageFlags = vk.VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = pc_size,
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
    try ctx.createComputePipeline(&pipeline_ci, &pipeline);
    errdefer vk.vkDestroyPipeline(dev, pipeline, null);

    const pool_size = vk.VkDescriptorPoolSize{
        .type = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = n_bindings * max_sets,
    };
    const pool_ci = vk.VkDescriptorPoolCreateInfo{
        .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = null,
        .flags = vk.VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = max_sets,
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
    };
}

fn allocSimpleSet(
    device: vk.VkDevice,
    desc_pool: vk.VkDescriptorPool,
    dset_layout: vk.VkDescriptorSetLayout,
    buffers: []const *const GpuBuffer,
) !vk.VkDescriptorSet {
    std.debug.assert(buffers.len >= 1 and buffers.len <= 5);
    const alloc_ci = vk.VkDescriptorSetAllocateInfo{
        .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = null,
        .descriptorPool = desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &dset_layout,
    };
    var dset: vk.VkDescriptorSet = null;
    if (vk.vkAllocateDescriptorSets(device, &alloc_ci, &dset) != vk.VK_SUCCESS)
        return error.VkDescriptorSetAllocFailed;

    var infos: [5]vk.VkDescriptorBufferInfo = undefined;
    var writes: [5]vk.VkWriteDescriptorSet = undefined;
    for (buffers, 0..) |buf, i| {
        infos[i] = .{ .buffer = buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE };
        writes[i] = mkWrite(dset, @intCast(i), &infos[i]);
    }
    vk.vkUpdateDescriptorSets(device, @intCast(buffers.len), &writes, 0, null);
    return dset;
}

fn bindSimple(cmd: vk.VkCommandBuffer, pipeline: vk.VkPipeline, layout: vk.VkPipelineLayout, dset: vk.VkDescriptorSet) void {
    vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &dset, 0, null);
}

fn deinitSimplePipeline(device: vk.VkDevice, desc_pool: vk.VkDescriptorPool, pipeline: vk.VkPipeline, layout: vk.VkPipelineLayout, dset_layout: vk.VkDescriptorSetLayout) void {
    vk.vkDestroyDescriptorPool(device, desc_pool, null);
    vk.vkDestroyPipeline(device, pipeline, null);
    vk.vkDestroyPipelineLayout(device, layout, null);
    vk.vkDestroyDescriptorSetLayout(device, dset_layout, null);
}

pub const ElemAddPipeline = struct {
    pipeline: vk.VkPipeline,
    layout: vk.VkPipelineLayout,
    dset_layout: vk.VkDescriptorSetLayout,
    desc_pool: vk.VkDescriptorPool,
    device: vk.VkDevice,

    pub fn init(ctx: *const GpuContext) !ElemAddPipeline {
        comptime std.debug.assert(shaders.elem_add.len % 4 == 0);
        const built = try buildSimplePipeline(ctx, &shaders.elem_add, 2, @sizeOf(ElemPushConst), 64);
        return .{
            .pipeline = built.pipeline,
            .layout = built.layout,
            .dset_layout = built.dset_layout,
            .desc_pool = built.desc_pool,
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
        const dset = try self.allocSet(a_buf, b_buf);
        self.recordWithSet(cmd, dset, n);
        return dset;
    }

    pub fn allocSet(
        self: *const ElemAddPipeline,
        a_buf: *const GpuBuffer,
        b_buf: *const GpuBuffer,
    ) !vk.VkDescriptorSet {
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

        const buf_infos = [2]vk.VkDescriptorBufferInfo{
            .{ .buffer = a_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = b_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        };
        const writes = [2]vk.VkWriteDescriptorSet{
            mkWrite(dset, 0, &buf_infos[0]),
            mkWrite(dset, 1, &buf_infos[1]),
        };
        vk.vkUpdateDescriptorSets(self.device, writes.len, &writes, 0, null);
        return dset;
    }

    pub fn recordWithSet(
        self: *const ElemAddPipeline,
        cmd: vk.VkCommandBuffer,
        dset: vk.VkDescriptorSet,
        n: u32,
    ) void {
        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.layout, 0, 1, &dset, 0, null);
        const pc = ElemPushConst{ .n = n };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(ElemPushConst), &pc);
        const groups = (n + 255) / 256;
        vk.vkCmdDispatch(cmd, groups, 1, 1);
    }

    pub fn deinit(self: *ElemAddPipeline) void {
        vk.vkDestroyDescriptorPool(self.device, self.desc_pool, null);
        vk.vkDestroyPipeline(self.device, self.pipeline, null);
        vk.vkDestroyPipelineLayout(self.device, self.layout, null);
        vk.vkDestroyDescriptorSetLayout(self.device, self.dset_layout, null);
    }
};

pub const ElemScalePipeline = struct {
    pipeline: vk.VkPipeline,
    layout: vk.VkPipelineLayout,
    dset_layout: vk.VkDescriptorSetLayout,
    desc_pool: vk.VkDescriptorPool,
    device: vk.VkDevice,

    pub fn init(ctx: *const GpuContext) !ElemScalePipeline {
        comptime std.debug.assert(shaders.elem_scale.len % 4 == 0);
        const built = try buildSimplePipeline(ctx, &shaders.elem_scale, 1, @sizeOf(ElemScalePushConst), 64);
        return .{
            .pipeline = built.pipeline,
            .layout = built.layout,
            .dset_layout = built.dset_layout,
            .desc_pool = built.desc_pool,
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
        const dset = try self.allocSet(x_buf);
        self.recordWithSet(cmd, dset, n, s);
        return dset;
    }

    pub fn allocSet(self: *const ElemScalePipeline, x_buf: *const GpuBuffer) !vk.VkDescriptorSet {
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

        const buf_info = vk.VkDescriptorBufferInfo{
            .buffer = x_buf.handle,
            .offset = 0,
            .range = vk.VK_WHOLE_SIZE,
        };
        const write = mkWrite(dset, 0, &buf_info);
        vk.vkUpdateDescriptorSets(self.device, 1, &write, 0, null);
        return dset;
    }

    pub fn recordWithSet(
        self: *const ElemScalePipeline,
        cmd: vk.VkCommandBuffer,
        dset: vk.VkDescriptorSet,
        n: u32,
        s: f32,
    ) void {
        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.layout, 0, 1, &dset, 0, null);
        const pc = ElemScalePushConst{ .n = n, .s = s };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(ElemScalePushConst), &pc);
        const groups = (n + 255) / 256;
        vk.vkCmdDispatch(cmd, groups, 1, 1);
    }

    pub fn deinit(self: *ElemScalePipeline) void {
        vk.vkDestroyDescriptorPool(self.device, self.desc_pool, null);
        vk.vkDestroyPipeline(self.device, self.pipeline, null);
        vk.vkDestroyPipelineLayout(self.device, self.layout, null);
        vk.vkDestroyDescriptorSetLayout(self.device, self.dset_layout, null);
    }
};

pub const LogitSoftcapPipeline = struct {
    pipeline: vk.VkPipeline,
    layout: vk.VkPipelineLayout,
    dset_layout: vk.VkDescriptorSetLayout,
    desc_pool: vk.VkDescriptorPool,
    device: vk.VkDevice,

    pub fn init(ctx: *const GpuContext) !LogitSoftcapPipeline {
        comptime std.debug.assert(shaders.logit_softcap.len % 4 == 0);
        const built = try buildSimplePipeline(ctx, &shaders.logit_softcap, 1, @sizeOf(ElemScalePushConst), 8);
        return .{
            .pipeline = built.pipeline,
            .layout = built.layout,
            .dset_layout = built.dset_layout,
            .desc_pool = built.desc_pool,
            .device = ctx.device,
        };
    }

    pub fn record(
        self: *const LogitSoftcapPipeline,
        cmd: vk.VkCommandBuffer,
        x_buf: *const GpuBuffer,
        n: u32,
        cap: f32,
    ) !vk.VkDescriptorSet {
        const dset = try self.allocSet(x_buf);
        self.recordWithSet(cmd, dset, n, cap);
        return dset;
    }

    pub fn allocSet(self: *const LogitSoftcapPipeline, x_buf: *const GpuBuffer) !vk.VkDescriptorSet {
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

        const buf_info = vk.VkDescriptorBufferInfo{
            .buffer = x_buf.handle,
            .offset = 0,
            .range = vk.VK_WHOLE_SIZE,
        };
        const write = mkWrite(dset, 0, &buf_info);
        vk.vkUpdateDescriptorSets(self.device, 1, &write, 0, null);
        return dset;
    }

    pub fn recordWithSet(
        self: *const LogitSoftcapPipeline,
        cmd: vk.VkCommandBuffer,
        dset: vk.VkDescriptorSet,
        n: u32,
        cap: f32,
    ) void {
        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.layout, 0, 1, &dset, 0, null);
        const pc = ElemScalePushConst{ .n = n, .s = cap };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(ElemScalePushConst), &pc);
        const groups = (n + 255) / 256;
        vk.vkCmdDispatch(cmd, groups, 1, 1);
    }

    pub fn deinit(self: *LogitSoftcapPipeline) void {
        vk.vkDestroyDescriptorPool(self.device, self.desc_pool, null);
        vk.vkDestroyPipeline(self.device, self.pipeline, null);
        vk.vkDestroyPipelineLayout(self.device, self.layout, null);
        vk.vkDestroyDescriptorSetLayout(self.device, self.dset_layout, null);
    }
};

const ArgmaxPushConst = extern struct {
    n: u32,
    n_groups: u32,
    pass: u32,
};

const argmax_workgroup_size: u32 = 256;
const argmax_max_groups: u32 = 256;

pub fn argmaxScratchBytes() usize {
    return argmax_max_groups * 2 * @sizeOf(u32);
}

pub const ArgmaxPipeline = struct {
    pipeline: vk.VkPipeline,
    layout: vk.VkPipelineLayout,
    dset_layout: vk.VkDescriptorSetLayout,
    desc_pool: vk.VkDescriptorPool,
    device: vk.VkDevice,

    pub fn init(ctx: *const GpuContext) !ArgmaxPipeline {
        comptime std.debug.assert(shaders.argmax.len % 4 == 0);
        const built = try buildSimplePipeline(ctx, &shaders.argmax, 3, @sizeOf(ArgmaxPushConst), 8);
        return .{
            .pipeline = built.pipeline,
            .layout = built.layout,
            .dset_layout = built.dset_layout,
            .desc_pool = built.desc_pool,
            .device = ctx.device,
        };
    }

    pub fn allocSet(self: *const ArgmaxPipeline, x_buf: *const GpuBuffer, scratch_buf: *const GpuBuffer, out_buf: *const GpuBuffer) !vk.VkDescriptorSet {
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

        const buf_infos = [3]vk.VkDescriptorBufferInfo{
            .{ .buffer = x_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = scratch_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = out_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        };
        const writes = [3]vk.VkWriteDescriptorSet{
            mkWrite(dset, 0, &buf_infos[0]),
            mkWrite(dset, 1, &buf_infos[1]),
            mkWrite(dset, 2, &buf_infos[2]),
        };
        vk.vkUpdateDescriptorSets(self.device, writes.len, &writes, 0, null);
        return dset;
    }

    pub fn recordWithSet(self: *const ArgmaxPipeline, cmd: vk.VkCommandBuffer, dset: vk.VkDescriptorSet, n: u32) void {
        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.layout, 0, 1, &dset, 0, null);
        const groups = @min(argmax_max_groups, (n + argmax_workgroup_size - 1) / argmax_workgroup_size);
        const partial_pc = ArgmaxPushConst{ .n = n, .n_groups = groups, .pass = 0 };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(ArgmaxPushConst), &partial_pc);
        vk.vkCmdDispatch(cmd, groups, 1, 1);
        GpuContext.recordShaderBarrier(cmd);
        const final_pc = ArgmaxPushConst{ .n = groups, .n_groups = 1, .pass = 1 };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(ArgmaxPushConst), &final_pc);
        vk.vkCmdDispatch(cmd, 1, 1, 1);
    }

    pub fn deinit(self: *ArgmaxPipeline) void {
        vk.vkDestroyDescriptorPool(self.device, self.desc_pool, null);
        vk.vkDestroyPipeline(self.device, self.pipeline, null);
        vk.vkDestroyPipelineLayout(self.device, self.layout, null);
        vk.vkDestroyDescriptorSetLayout(self.device, self.dset_layout, null);
    }
};

pub const ElemAddScalePipeline = struct {
    pipeline: vk.VkPipeline,
    layout: vk.VkPipelineLayout,
    dset_layout: vk.VkDescriptorSetLayout,
    desc_pool: vk.VkDescriptorPool,
    device: vk.VkDevice,

    pub fn init(ctx: *const GpuContext) !ElemAddScalePipeline {
        comptime std.debug.assert(shaders.elem_add_scale.len % 4 == 0);
        const built = try buildSimplePipeline(ctx, &shaders.elem_add_scale, 2, @sizeOf(ElemScalePushConst), 64);
        return .{
            .pipeline = built.pipeline,
            .layout = built.layout,
            .dset_layout = built.dset_layout,
            .desc_pool = built.desc_pool,
            .device = ctx.device,
        };
    }

    pub fn record(
        self: *const ElemAddScalePipeline,
        cmd: vk.VkCommandBuffer,
        a_buf: *const GpuBuffer,
        b_buf: *const GpuBuffer,
        n: u32,
        s: f32,
    ) !vk.VkDescriptorSet {
        const dset = try self.allocSet(a_buf, b_buf);
        self.recordWithSet(cmd, dset, n, s);
        return dset;
    }

    pub fn allocSet(
        self: *const ElemAddScalePipeline,
        a_buf: *const GpuBuffer,
        b_buf: *const GpuBuffer,
    ) !vk.VkDescriptorSet {
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

        const buf_infos = [2]vk.VkDescriptorBufferInfo{
            .{ .buffer = a_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = b_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        };
        const writes = [2]vk.VkWriteDescriptorSet{
            mkWrite(dset, 0, &buf_infos[0]),
            mkWrite(dset, 1, &buf_infos[1]),
        };
        vk.vkUpdateDescriptorSets(self.device, writes.len, &writes, 0, null);
        return dset;
    }

    pub fn recordWithSet(
        self: *const ElemAddScalePipeline,
        cmd: vk.VkCommandBuffer,
        dset: vk.VkDescriptorSet,
        n: u32,
        s: f32,
    ) void {
        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.layout, 0, 1, &dset, 0, null);
        const pc = ElemScalePushConst{ .n = n, .s = s };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(ElemScalePushConst), &pc);
        const groups = (n + 255) / 256;
        vk.vkCmdDispatch(cmd, groups, 1, 1);
    }

    pub fn deinit(self: *ElemAddScalePipeline) void {
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
    pipeline: vk.VkPipeline,
    layout: vk.VkPipelineLayout,
    dset_layout: vk.VkDescriptorSetLayout,
    desc_pool: vk.VkDescriptorPool,
    device: vk.VkDevice,

    pub fn init(ctx: *const GpuContext) !GeluMulPipeline {
        comptime std.debug.assert(shaders.gelu_mul.len % 4 == 0);
        const built = try buildSimplePipeline(ctx, &shaders.gelu_mul, 2, @sizeOf(ElemPushConst), 32);
        return .{
            .pipeline = built.pipeline,
            .layout = built.layout,
            .dset_layout = built.dset_layout,
            .desc_pool = built.desc_pool,
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
        const dset = try self.allocSet(a_buf, b_buf);
        self.recordWithSet(cmd, dset, n);
        return dset;
    }

    pub fn allocSet(
        self: *const GeluMulPipeline,
        a_buf: *const GpuBuffer,
        b_buf: *const GpuBuffer,
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

        const buf_infos = [2]vk.VkDescriptorBufferInfo{
            .{ .buffer = a_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = b_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        };
        const writes = [2]vk.VkWriteDescriptorSet{
            mkWrite(dset, 0, &buf_infos[0]),
            mkWrite(dset, 1, &buf_infos[1]),
        };
        vk.vkUpdateDescriptorSets(dev, writes.len, &writes, 0, null);
        return dset;
    }

    pub fn recordWithSet(
        self: *const GeluMulPipeline,
        cmd: vk.VkCommandBuffer,
        dset: vk.VkDescriptorSet,
        n: u32,
    ) void {
        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.layout, 0, 1, &dset, 0, null);
        const pc = ElemPushConst{ .n = n };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(ElemPushConst), &pc);
        const groups = (n + 255) / 256;
        vk.vkCmdDispatch(cmd, groups, 1, 1);
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
    pipeline: vk.VkPipeline,
    layout: vk.VkPipelineLayout,
    dset_layout: vk.VkDescriptorSetLayout,
    desc_pool: vk.VkDescriptorPool,
    device: vk.VkDevice,

    pub fn init(ctx: *const GpuContext) !RopeNeoxTablePipeline {
        comptime std.debug.assert(shaders.rope_neox_table.len % 4 == 0);
        const built = try buildSimplePipeline(ctx, &shaders.rope_neox_table, 2, @sizeOf(RopeTablePushConst), 64);
        return .{
            .pipeline = built.pipeline,
            .layout = built.layout,
            .dset_layout = built.dset_layout,
            .desc_pool = built.desc_pool,
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
        const dset = try self.allocSet(vec_buf, freqs_buf);
        self.recordWithSet(cmd, dset, pos, head_dim, n_heads);
        return dset;
    }

    pub fn allocSet(
        self: *const RopeNeoxTablePipeline,
        vec_buf: *const GpuBuffer,
        freqs_buf: *const GpuBuffer,
    ) !vk.VkDescriptorSet {
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
        const buf_infos = [2]vk.VkDescriptorBufferInfo{
            .{ .buffer = vec_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = freqs_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        };
        const writes = [2]vk.VkWriteDescriptorSet{
            mkWrite(dset, 0, &buf_infos[0]),
            mkWrite(dset, 1, &buf_infos[1]),
        };
        vk.vkUpdateDescriptorSets(self.device, writes.len, &writes, 0, null);
        return dset;
    }

    pub fn recordWithSet(
        self: *const RopeNeoxTablePipeline,
        cmd: vk.VkCommandBuffer,
        dset: vk.VkDescriptorSet,
        pos: u32,
        head_dim: u32,
        n_heads: u32,
    ) void {
        std.debug.assert(head_dim <= 512);
        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.layout, 0, 1, &dset, 0, null);
        const pc = RopeTablePushConst{ .pos = pos, .head_dim = head_dim };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(RopeTablePushConst), &pc);
        vk.vkCmdDispatch(cmd, n_heads, 1, 1);
    }

    pub fn deinit(self: *RopeNeoxTablePipeline) void {
        vk.vkDestroyDescriptorPool(self.device, self.desc_pool, null);
        vk.vkDestroyPipeline(self.device, self.pipeline, null);
        vk.vkDestroyPipelineLayout(self.device, self.layout, null);
        vk.vkDestroyDescriptorSetLayout(self.device, self.dset_layout, null);
    }
};

pub const RopeNeoxThetaPipeline = struct {
    pipeline: vk.VkPipeline,
    layout: vk.VkPipelineLayout,
    dset_layout: vk.VkDescriptorSetLayout,
    desc_pool: vk.VkDescriptorPool,
    device: vk.VkDevice,

    pub fn init(ctx: *const GpuContext) !RopeNeoxThetaPipeline {
        comptime std.debug.assert(shaders.rope_neox_theta.len % 4 == 0);
        const built = try buildSimplePipeline(ctx, &shaders.rope_neox_theta, 1, @sizeOf(RopeThetaPushConst), 64);
        return .{
            .pipeline = built.pipeline,
            .layout = built.layout,
            .dset_layout = built.dset_layout,
            .desc_pool = built.desc_pool,
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
        const dset = try self.allocSet(vec_buf);
        self.recordWithSet(cmd, dset, pos, head_dim, theta, n_heads);
        return dset;
    }

    pub fn allocSet(
        self: *const RopeNeoxThetaPipeline,
        vec_buf: *const GpuBuffer,
    ) !vk.VkDescriptorSet {
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
        const buf_info = vk.VkDescriptorBufferInfo{
            .buffer = vec_buf.handle,
            .offset = 0,
            .range = vk.VK_WHOLE_SIZE,
        };
        const write = mkWrite(dset, 0, &buf_info);
        vk.vkUpdateDescriptorSets(self.device, 1, &write, 0, null);
        return dset;
    }

    pub fn recordWithSet(
        self: *const RopeNeoxThetaPipeline,
        cmd: vk.VkCommandBuffer,
        dset: vk.VkDescriptorSet,
        pos: u32,
        head_dim: u32,
        theta: f32,
        n_heads: u32,
    ) void {
        std.debug.assert(head_dim <= 512);
        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.layout, 0, 1, &dset, 0, null);
        const pc = RopeThetaPushConst{ .pos = pos, .head_dim = head_dim, .theta = theta };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(RopeThetaPushConst), &pc);
        vk.vkCmdDispatch(cmd, n_heads, 1, 1);
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
    seq: u32,
    win_len: u32,
    head_dim: u32,
    n_kv_heads: u32,
    n_q_per_kv: u32,
    cap: u32,
    scale: f32,
};

const AttnAvPushConst = extern struct {
    seq: u32,
    win_len: u32,
    head_dim: u32,
    n_kv_heads: u32,
    n_q_per_kv: u32,
    cap: u32,
};

pub const AttnQkSoftmaxPipeline = struct {
    pipeline: vk.VkPipeline,
    layout: vk.VkPipelineLayout,
    dset_layout: vk.VkDescriptorSetLayout,
    desc_pool: vk.VkDescriptorPool,
    device: vk.VkDevice,

    pub fn init(ctx: *const GpuContext) !AttnQkSoftmaxPipeline {
        comptime std.debug.assert(shaders.attn_qk_softmax.len % 4 == 0);
        const built = try buildSimplePipeline(ctx, &shaders.attn_qk_softmax, 3, @sizeOf(AttnQkSoftmaxPushConst), 32);
        return .{
            .pipeline = built.pipeline,
            .layout = built.layout,
            .dset_layout = built.dset_layout,
            .desc_pool = built.desc_pool,
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
        seq: u32,
        win_len: u32,
        head_dim: u32,
        n_kv_heads: u32,
        n_q_per_kv: u32,
        cap: u32,
        scale: f32,
    ) !vk.VkDescriptorSet {
        const dset = try self.allocSet(q_buf, k_buf, scores_buf);
        self.recordWithSet(cmd, dset, n_heads, seq, win_len, head_dim, n_kv_heads, n_q_per_kv, cap, scale);
        return dset;
    }

    pub fn allocSet(
        self: *const AttnQkSoftmaxPipeline,
        q_buf: *const GpuBuffer,
        k_buf: *const GpuBuffer,
        scores_buf: *const GpuBuffer,
    ) !vk.VkDescriptorSet {
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

        const buf_infos = [3]vk.VkDescriptorBufferInfo{
            .{ .buffer = q_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = k_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = scores_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        };
        const writes = [3]vk.VkWriteDescriptorSet{
            mkWrite(dset, 0, &buf_infos[0]),
            mkWrite(dset, 1, &buf_infos[1]),
            mkWrite(dset, 2, &buf_infos[2]),
        };
        vk.vkUpdateDescriptorSets(self.device, writes.len, &writes, 0, null);
        return dset;
    }

    pub fn recordWithSet(
        self: *const AttnQkSoftmaxPipeline,
        cmd: vk.VkCommandBuffer,
        dset: vk.VkDescriptorSet,
        n_heads: u32,
        seq: u32,
        win_len: u32,
        head_dim: u32,
        n_kv_heads: u32,
        n_q_per_kv: u32,
        cap: u32,
        scale: f32,
    ) void {
        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.layout, 0, 1, &dset, 0, null);
        const pc = AttnQkSoftmaxPushConst{
            .seq = seq,
            .win_len = win_len,
            .head_dim = head_dim,
            .n_kv_heads = n_kv_heads,
            .n_q_per_kv = n_q_per_kv,
            .cap = cap,
            .scale = scale,
        };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(AttnQkSoftmaxPushConst), &pc);
        vk.vkCmdDispatch(cmd, n_heads, 1, 1);
    }

    pub fn deinit(self: *AttnQkSoftmaxPipeline) void {
        vk.vkDestroyDescriptorPool(self.device, self.desc_pool, null);
        vk.vkDestroyPipeline(self.device, self.pipeline, null);
        vk.vkDestroyPipelineLayout(self.device, self.layout, null);
        vk.vkDestroyDescriptorSetLayout(self.device, self.dset_layout, null);
    }
};

pub const AttnAvPipeline = struct {
    pipeline: vk.VkPipeline,
    layout: vk.VkPipelineLayout,
    dset_layout: vk.VkDescriptorSetLayout,
    desc_pool: vk.VkDescriptorPool,
    device: vk.VkDevice,

    pub fn init(ctx: *const GpuContext) !AttnAvPipeline {
        comptime std.debug.assert(shaders.attn_av.len % 4 == 0);
        const built = try buildSimplePipeline(ctx, &shaders.attn_av, 3, @sizeOf(AttnAvPushConst), 32);
        return .{
            .pipeline = built.pipeline,
            .layout = built.layout,
            .dset_layout = built.dset_layout,
            .desc_pool = built.desc_pool,
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
        seq: u32,
        win_len: u32,
        head_dim: u32,
        n_kv_heads: u32,
        n_q_per_kv: u32,
        cap: u32,
    ) !vk.VkDescriptorSet {
        const dset = try self.allocSet(scores_buf, v_buf, out_buf);
        self.recordWithSet(cmd, dset, n_heads, seq, win_len, head_dim, n_kv_heads, n_q_per_kv, cap);
        return dset;
    }

    pub fn allocSet(
        self: *const AttnAvPipeline,
        scores_buf: *const GpuBuffer,
        v_buf: *const GpuBuffer,
        out_buf: *const GpuBuffer,
    ) !vk.VkDescriptorSet {
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

        const buf_infos = [3]vk.VkDescriptorBufferInfo{
            .{ .buffer = scores_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = v_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = out_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        };
        const writes = [3]vk.VkWriteDescriptorSet{
            mkWrite(dset, 0, &buf_infos[0]),
            mkWrite(dset, 1, &buf_infos[1]),
            mkWrite(dset, 2, &buf_infos[2]),
        };
        vk.vkUpdateDescriptorSets(self.device, writes.len, &writes, 0, null);
        return dset;
    }

    pub fn recordWithSet(
        self: *const AttnAvPipeline,
        cmd: vk.VkCommandBuffer,
        dset: vk.VkDescriptorSet,
        n_heads: u32,
        seq: u32,
        win_len: u32,
        head_dim: u32,
        n_kv_heads: u32,
        n_q_per_kv: u32,
        cap: u32,
    ) void {
        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.layout, 0, 1, &dset, 0, null);
        const pc = AttnAvPushConst{
            .seq = seq,
            .win_len = win_len,
            .head_dim = head_dim,
            .n_kv_heads = n_kv_heads,
            .n_q_per_kv = n_q_per_kv,
            .cap = cap,
        };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(AttnAvPushConst), &pc);
        vk.vkCmdDispatch(cmd, n_heads, 1, 1);
    }

    pub fn deinit(self: *AttnAvPipeline) void {
        vk.vkDestroyDescriptorPool(self.device, self.desc_pool, null);
        vk.vkDestroyPipeline(self.device, self.pipeline, null);
        vk.vkDestroyPipelineLayout(self.device, self.layout, null);
        vk.vkDestroyDescriptorSetLayout(self.device, self.dset_layout, null);
    }
};

pub const AttnFusedSmallPipeline = struct {
    pipeline: vk.VkPipeline,
    layout: vk.VkPipelineLayout,
    dset_layout: vk.VkDescriptorSetLayout,
    desc_pool: vk.VkDescriptorPool,
    device: vk.VkDevice,

    pub fn init(ctx: *const GpuContext) !AttnFusedSmallPipeline {
        comptime std.debug.assert(shaders.attn_fused_small.len % 4 == 0);
        const built = try buildSimplePipeline(ctx, &shaders.attn_fused_small, 4, @sizeOf(AttnQkSoftmaxPushConst), 32);
        return .{
            .pipeline = built.pipeline,
            .layout = built.layout,
            .dset_layout = built.dset_layout,
            .desc_pool = built.desc_pool,
            .device = ctx.device,
        };
    }

    pub fn record(
        self: *const AttnFusedSmallPipeline,
        cmd: vk.VkCommandBuffer,
        q_buf: *const GpuBuffer,
        k_buf: *const GpuBuffer,
        v_buf: *const GpuBuffer,
        out_buf: *const GpuBuffer,
        n_heads: u32,
        seq: u32,
        win_len: u32,
        head_dim: u32,
        n_kv_heads: u32,
        n_q_per_kv: u32,
        cap: u32,
        scale: f32,
    ) !vk.VkDescriptorSet {
        const dset = try self.allocSet(q_buf, k_buf, v_buf, out_buf);
        self.recordWithSet(cmd, dset, n_heads, seq, win_len, head_dim, n_kv_heads, n_q_per_kv, cap, scale);
        return dset;
    }

    pub fn allocSet(
        self: *const AttnFusedSmallPipeline,
        q_buf: *const GpuBuffer,
        k_buf: *const GpuBuffer,
        v_buf: *const GpuBuffer,
        out_buf: *const GpuBuffer,
    ) !vk.VkDescriptorSet {
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

        const buf_infos = [4]vk.VkDescriptorBufferInfo{
            .{ .buffer = q_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = k_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = v_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = out_buf.handle, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        };
        const writes = [4]vk.VkWriteDescriptorSet{
            mkWrite(dset, 0, &buf_infos[0]),
            mkWrite(dset, 1, &buf_infos[1]),
            mkWrite(dset, 2, &buf_infos[2]),
            mkWrite(dset, 3, &buf_infos[3]),
        };
        vk.vkUpdateDescriptorSets(self.device, writes.len, &writes, 0, null);
        return dset;
    }

    pub fn recordWithSet(
        self: *const AttnFusedSmallPipeline,
        cmd: vk.VkCommandBuffer,
        dset: vk.VkDescriptorSet,
        n_heads: u32,
        seq: u32,
        win_len: u32,
        head_dim: u32,
        n_kv_heads: u32,
        n_q_per_kv: u32,
        cap: u32,
        scale: f32,
    ) void {
        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.layout, 0, 1, &dset, 0, null);
        const pc = AttnQkSoftmaxPushConst{
            .seq = seq,
            .win_len = win_len,
            .head_dim = head_dim,
            .n_kv_heads = n_kv_heads,
            .n_q_per_kv = n_q_per_kv,
            .cap = cap,
            .scale = scale,
        };
        vk.vkCmdPushConstants(cmd, self.layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(AttnQkSoftmaxPushConst), &pc);
        vk.vkCmdDispatch(cmd, n_heads, 1, 1);
    }

    pub fn deinit(self: *AttnFusedSmallPipeline) void {
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

test "gpu argmax returns lowest maximum index" {
    const ctx = GpuContext.init() catch |e| {
        std.debug.print("gpu init failed: {}\n", .{e});
        return;
    };
    var gpu = ctx;
    defer gpu.deinit();

    var pipeline = try ArgmaxPipeline.init(&gpu);
    defer pipeline.deinit();

    var values = [_]f32{ -4.0, 3.0, 9.0, 2.0, 9.0, 1.0 };
    var in_buf = try GpuBuffer.initHostCoherent(&gpu, @sizeOf(@TypeOf(values)), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer in_buf.deinit();
    try in_buf.upload(std.mem.sliceAsBytes(&values));
    var out_buf = try GpuBuffer.initHostCoherent(&gpu, @sizeOf(u32), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer out_buf.deinit();
    var scratch_buf = try GpuBuffer.initDeviceLocal(&gpu, argmaxScratchBytes(), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer scratch_buf.deinit();

    const cmd = try gpu.beginBatch();
    const dset = try pipeline.allocSet(&in_buf, &scratch_buf, &out_buf);
    defer {
        var tmp = dset;
        _ = vk.vkFreeDescriptorSets(gpu.device, pipeline.desc_pool, 1, &tmp);
    }
    pipeline.recordWithSet(cmd, dset, values.len);
    try gpu.submitBatch(cmd);

    var idx: u32 = undefined;
    try out_buf.download(std.mem.asBytes(&idx));
    try std.testing.expectEqual(@as(u32, 2), idx);
}

test "gpu argmax reduces partial winners across workgroups" {
    const ctx = GpuContext.init() catch |e| {
        std.debug.print("gpu init failed: {}\n", .{e});
        return;
    };
    var gpu = ctx;
    defer gpu.deinit();

    var pipeline = try ArgmaxPipeline.init(&gpu);
    defer pipeline.deinit();

    var values: [1024]f32 = @splat(-4.0);
    values[17] = 9.0;
    values[900] = 9.0;
    var in_buf = try GpuBuffer.initHostCoherent(&gpu, @sizeOf(@TypeOf(values)), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer in_buf.deinit();
    try in_buf.upload(std.mem.sliceAsBytes(&values));
    var out_buf = try GpuBuffer.initHostCoherent(&gpu, @sizeOf(u32), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer out_buf.deinit();
    var scratch_buf = try GpuBuffer.initDeviceLocal(&gpu, argmaxScratchBytes(), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer scratch_buf.deinit();

    const cmd = try gpu.beginBatch();
    const dset = try pipeline.allocSet(&in_buf, &scratch_buf, &out_buf);
    defer {
        var tmp = dset;
        _ = vk.vkFreeDescriptorSets(gpu.device, pipeline.desc_pool, 1, &tmp);
    }
    pipeline.recordWithSet(cmd, dset, values.len);
    try gpu.submitBatch(cmd);

    var idx: u32 = undefined;
    try out_buf.download(std.mem.asBytes(&idx));
    try std.testing.expectEqual(@as(u32, 17), idx);
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
    @memset(mat_bytes[0..32], 0xFF); // hmask: all hi bits set
    mat_bytes[32] = 0x01; // qs[0]: lo2=1 at shift=0 for element 0
    // scales: sc[0]=33, sc[1..15]=32
    mat_bytes[96] = 0x01;
    mat_bytes[97] = 0x00;
    mat_bytes[98] = 0x00;
    mat_bytes[99] = 0x00;
    mat_bytes[100] = 0x00;
    mat_bytes[101] = 0x00;
    mat_bytes[102] = 0x00;
    mat_bytes[103] = 0x00;
    mat_bytes[104] = 0xAA;
    mat_bytes[105] = 0xAA;
    mat_bytes[106] = 0xAA;
    mat_bytes[107] = 0xAA;
    mat_bytes[108] = 0x00;
    mat_bytes[109] = 0x3C; // d = f16(1.0)

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
    mat_bytes[96] = 0x11;
    mat_bytes[97] = 0x11;
    mat_bytes[98] = 0x11;
    mat_bytes[99] = 0x11;
    mat_bytes[100] = 0x11;
    mat_bytes[101] = 0x11;
    mat_bytes[102] = 0x11;
    mat_bytes[103] = 0x11;
    mat_bytes[104] = 0xAA;
    mat_bytes[105] = 0xAA;
    mat_bytes[106] = 0xAA;
    mat_bytes[107] = 0xAA;
    mat_bytes[108] = 0x00;
    mat_bytes[109] = 0x3C; // d = f16(1.0)

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
    mat_bytes[0] = 0x00;
    mat_bytes[1] = 0x3C; // d = f16(1.0)
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
    mat_bytes[2] = 0x00;
    mat_bytes[3] = 0x3C; // dmin = f16(1.0)
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
    mat_bytes[0] = 0x00;
    mat_bytes[1] = 0x3C; // d = f16(1.0)
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
    mat_bytes[2] = 0x00;
    mat_bytes[3] = 0x3C; // m = f16(1.0)

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
    mat_bytes[0] = 0x00;
    mat_bytes[1] = 0x3C; // d = f16(1.0)
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
    mat_bytes[0] = 0x00;
    mat_bytes[1] = 0x3C; // d = f16(1.0)

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
    mat_bytes[0] = 0x00;
    mat_bytes[1] = 0x3C; // d = f16(1.0)
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
    mat_bytes[0] = 0x00;
    mat_bytes[1] = 0x3C; // d = f16(1.0)
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
    mat_bytes[96] = 0x01;
    mat_bytes[97] = 0x00;
    mat_bytes[98] = 0x00;
    mat_bytes[99] = 0x00;
    mat_bytes[100] = 0x00;
    mat_bytes[101] = 0x00;
    mat_bytes[102] = 0x00;
    mat_bytes[103] = 0x00;
    mat_bytes[104] = 0xAA;
    mat_bytes[105] = 0xAA;
    mat_bytes[106] = 0xAA;
    mat_bytes[107] = 0xAA;
    mat_bytes[108] = 0x00;
    mat_bytes[109] = 0x3C; // d = f16(1.0)

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

test "gpu expert-id gate-up Q3_K x Q8_1 correctness" {
    const ctx = GpuContext.init() catch |e| {
        std.debug.print("gpu init failed: {}\n", .{e});
        return;
    };
    var gpu = ctx;
    defer gpu.deinit();

    var pl = ExpertGateUpIdPipeline.initQ3KQ8_1(&gpu) catch |e| {
        std.debug.print("expert gate-up id pipeline init failed: {}\n", .{e});
        return;
    };
    defer pl.deinit();
    var quant = try QuantizeQ8_1Pipeline.init(&gpu);
    defer quant.deinit();

    const blk_size = 110;
    const n_experts = 3;
    const rows = 1;
    const cols = 256;
    const flat_size = n_experts * 2 * rows * blk_size;
    var mat_bytes: [flat_size]u8 = [_]u8{0} ** flat_size;
    for (0..n_experts) |e| for (0..2) |which| {
        const off = (e * 2 + which) * blk_size;
        @memset(mat_bytes[off..][0..32], 0xFF);
        mat_bytes[off + 32] = 0x01;
        mat_bytes[off + 96] = 0x01;
        mat_bytes[off + 104] = 0xAA;
        mat_bytes[off + 105] = 0xAA;
        mat_bytes[off + 106] = 0xAA;
        mat_bytes[off + 107] = 0xAA;
        mat_bytes[off + 108] = 0x00;
        mat_bytes[off + 109] = 0x3C; // d = f16(1.0)
    };

    const usage = vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    var weight_buf = try GpuBuffer.initHostCoherent(&gpu, flat_size, usage);
    defer weight_buf.deinit();
    try weight_buf.upload(&mat_bytes);

    const ids = [2]u32{ 2, 0 };
    var ids_buf = try GpuBuffer.initHostCoherent(&gpu, ids.len * @sizeOf(u32), usage);
    defer ids_buf.deinit();
    try ids_buf.upload(std.mem.sliceAsBytes(&ids));

    var vec_buf = try GpuBuffer.initHostCoherent(&gpu, cols * @sizeOf(f32), usage);
    defer vec_buf.deinit();
    var vec: [cols]f32 = [_]f32{1.0} ** cols;
    try vec_buf.upload(std.mem.sliceAsBytes(&vec));

    var q8_buf = try GpuBuffer.initHostCoherent(&gpu, q8_1OutBytes(cols), usage);
    defer q8_buf.deinit();
    var out_buf = try GpuBuffer.initHostCoherent(&gpu, ids.len * rows * @sizeOf(f32), usage);
    defer out_buf.deinit();

    {
        const cmd = try gpu.beginBatch();
        var qds = try quant.record(cmd, &vec_buf, &q8_buf, cols);
        GpuContext.recordShaderBarrier(cmd);
        var ds = try pl.record(cmd, &weight_buf, &q8_buf, &ids_buf, &out_buf, rows, cols, ids.len);
        try gpu.submitBatch(cmd);
        _ = vk.vkFreeDescriptorSets(gpu.device, quant.desc_pool, 1, &qds);
        _ = vk.vkFreeDescriptorSets(gpu.device, pl.desc_pool, 1, &ds);
    }

    var result: [ids.len]f32 = .{ 0.0, 0.0 };
    try out_buf.download(std.mem.sliceAsBytes(&result));
    for (result) |v| {
        try std.testing.expectApproxEqAbs(v, 0.841, 0.01);
    }
}

test "gpu expert-id gate-up Q3_K x Q8_1 fuzz" {
    const ctx = GpuContext.init() catch |e| {
        std.debug.print("gpu init failed: {}\n", .{e});
        return;
    };
    var gpu = ctx;
    defer gpu.deinit();

    const al = std.testing.allocator;
    const n_experts: usize = 128;
    const active: usize = 8;
    const rows: usize = 4;
    const cols: usize = 2816;
    const row_bytes = math_mod.rowBytes(.q3_k, cols);
    const one_mat_bytes = rows * row_bytes;
    const flat_bytes = n_experts * 2 * one_mat_bytes;

    var prng = std.Random.DefaultPrng.init(0xE1D5_0002);
    const r = prng.random();

    const mat = try al.alloc(u8, flat_bytes);
    defer al.free(mat);
    for (mat) |*b| b.* = r.int(u8);
    const small_d_le: [2]u8 = .{ 0xCD, 0x21 };
    for (0..n_experts * 2 * rows) |i| {
        const off = i * row_bytes;
        mat[off + 108] = small_d_le[0];
        mat[off + 109] = small_d_le[1];
    }

    const vec = try al.alloc(f32, cols);
    defer al.free(vec);
    for (vec) |*v| v.* = (r.float(f32) - 0.5) * 2.0;

    const q8_1_basic = try al.alloc(u8, (cols / 32) * dq.Q8_1_BLOCK_BYTES);
    defer al.free(q8_1_basic);
    dq.quantizeQ8_1(vec, q8_1_basic);

    const vec_q8_rounded = try al.alloc(f32, cols);
    defer al.free(vec_q8_rounded);
    dq.dequantQ8_1(q8_1_basic, vec_q8_rounded);

    const q8_1_x4 = try al.alloc(u8, q8_1_basic.len);
    defer al.free(q8_1_x4);
    dq.packQ8_1_x4(q8_1_basic, q8_1_x4);

    const ids = [_]u32{ 122, 53, 51, 119, 79, 0, 102, 73 };
    const cpu_out = try al.alloc(f32, active * rows);
    defer al.free(cpu_out);
    const gate_tmp = try al.alloc(f32, rows);
    defer al.free(gate_tmp);
    const up_tmp = try al.alloc(f32, rows);
    defer al.free(up_tmp);
    const row_buf = try al.alloc(f32, cols);
    defer al.free(row_buf);
    for (ids, 0..) |expert_id, slot| {
        const e: usize = @intCast(expert_id);
        const expert_base = e * 2 * one_mat_bytes;
        const gate_mat = mat[expert_base..][0..one_mat_bytes];
        const up_mat = mat[expert_base + one_mat_bytes ..][0..one_mat_bytes];
        math_mod.quantMatvec(gate_tmp, gate_mat, .q3_k, vec_q8_rounded, rows, cols, row_buf);
        math_mod.quantMatvec(up_tmp, up_mat, .q3_k, vec_q8_rounded, rows, cols, row_buf);
        for (0..rows) |row| {
            cpu_out[slot * rows + row] = math_mod.gelu(gate_tmp[row]) * up_tmp[row];
        }
    }

    var pipeline = try ExpertGateUpIdPipeline.initQ3KQ8_1(&gpu);
    defer pipeline.deinit();
    var pipeline_r2 = try ExpertGateUpIdPipeline.initQ3KQ8_1R2(&gpu);
    defer pipeline_r2.deinit();
    var pipeline_r4 = try ExpertGateUpIdPipeline.initQ3KQ8_1R4(&gpu);
    defer pipeline_r4.deinit();

    var weights_buf = try GpuBuffer.initHostCoherent(&gpu, flat_bytes, @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer weights_buf.deinit();
    try weights_buf.upload(mat);

    var acts_buf = try GpuBuffer.initHostCoherent(&gpu, q8_1_x4.len, @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer acts_buf.deinit();
    try acts_buf.upload(q8_1_x4);

    var ids_buf = try GpuBuffer.initHostCoherent(&gpu, ids.len * @sizeOf(u32), @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer ids_buf.deinit();
    try ids_buf.upload(std.mem.sliceAsBytes(&ids));

    var out_buf = try GpuBuffer.initHostCoherent(&gpu, active * rows * @sizeOf(f32), @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer out_buf.deinit();

    const cmd = try gpu.beginBatch();
    var ds = try pipeline.record(cmd, &weights_buf, &acts_buf, &ids_buf, &out_buf, @intCast(rows), @intCast(cols), @intCast(active));
    try gpu.submitBatch(cmd);
    _ = vk.vkFreeDescriptorSets(gpu.device, pipeline.desc_pool, 1, &ds);

    const gpu_out = try al.alloc(f32, active * rows);
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
    std.debug.print("expert-id Q3_K×Q8_1 GU fuzz active={} rows={} cols={} max|D|={d:.6} rel={e:.3}\n", .{ active, rows, cols, max_abs, rel });
    try std.testing.expect(rel < 1e-4);

    {
        const zeroes = try al.alloc(f32, active * rows);
        defer al.free(zeroes);
        @memset(zeroes, 0.0);
        try out_buf.upload(std.mem.sliceAsBytes(zeroes));

        const cmd_r2 = try gpu.beginBatch();
        var ds_r2 = try pipeline_r2.record(cmd_r2, &weights_buf, &acts_buf, &ids_buf, &out_buf, @intCast(rows), @intCast(cols), @intCast(active));
        try gpu.submitBatch(cmd_r2);
        _ = vk.vkFreeDescriptorSets(gpu.device, pipeline_r2.desc_pool, 1, &ds_r2);

        const gpu_out_r2 = try al.alloc(f32, active * rows);
        defer al.free(gpu_out_r2);
        try out_buf.download(std.mem.sliceAsBytes(gpu_out_r2));

        var max_abs_r2: f32 = 0.0;
        var max_ref_r2: f32 = 0.0;
        for (cpu_out, gpu_out_r2) |c, g| {
            const d = @abs(c - g);
            if (d > max_abs_r2) max_abs_r2 = d;
            if (@abs(c) > max_ref_r2) max_ref_r2 = @abs(c);
        }
        const rel_r2 = max_abs_r2 / (max_ref_r2 + 1e-6);
        std.debug.print("expert-id Q3_K×Q8_1 GU r2 fuzz active={} rows={} cols={} max|D|={d:.6} rel={e:.3}\n", .{ active, rows, cols, max_abs_r2, rel_r2 });
        try std.testing.expect(rel_r2 < 1e-4);
    }

    {
        const zeroes = try al.alloc(f32, active * rows);
        defer al.free(zeroes);
        @memset(zeroes, 0.0);
        try out_buf.upload(std.mem.sliceAsBytes(zeroes));

        const cmd_r4 = try gpu.beginBatch();
        var ds_r4 = try pipeline_r4.record(cmd_r4, &weights_buf, &acts_buf, &ids_buf, &out_buf, @intCast(rows), @intCast(cols), @intCast(active));
        try gpu.submitBatch(cmd_r4);
        _ = vk.vkFreeDescriptorSets(gpu.device, pipeline_r4.desc_pool, 1, &ds_r4);

        const gpu_out_r4 = try al.alloc(f32, active * rows);
        defer al.free(gpu_out_r4);
        try out_buf.download(std.mem.sliceAsBytes(gpu_out_r4));

        var max_abs_r4: f32 = 0.0;
        var max_ref_r4: f32 = 0.0;
        for (cpu_out, gpu_out_r4) |c, g| {
            const d = @abs(c - g);
            if (d > max_abs_r4) max_abs_r4 = d;
            if (@abs(c) > max_ref_r4) max_ref_r4 = @abs(c);
        }
        const rel_r4 = max_abs_r4 / (max_ref_r4 + 1e-6);
        std.debug.print("expert-id Q3_K×Q8_1 GU r4 fuzz active={} rows={} cols={} max|D|={d:.6} rel={e:.3}\n", .{ active, rows, cols, max_abs_r4, rel_r4 });
        try std.testing.expect(rel_r4 < 1e-4);
    }
}

test "gpu quantize Q8_1 batched round-trip" {
    const ctx = GpuContext.init() catch |e| {
        std.debug.print("gpu init failed: {}\n", .{e});
        return;
    };
    var gpu = ctx;
    defer gpu.deinit();

    const al = std.testing.allocator;
    const ncols: usize = 704;
    const active: usize = 8;
    const in_len = active * ncols;
    const out_slot_bytes = q8_1OutBytes(@intCast(ncols));
    const out_len = active * out_slot_bytes;

    var prng = std.Random.DefaultPrng.init(0xE1D5_0003);
    const r = prng.random();
    const input = try al.alloc(f32, in_len);
    defer al.free(input);
    for (input) |*v| v.* = (r.float(f32) - 0.5) * 12.0;

    var pl_batched = try QuantizeQ8_1BatchedPipeline.init(&gpu);
    defer pl_batched.deinit();

    var in_buf = try GpuBuffer.initHostCoherent(&gpu, input.len * @sizeOf(f32), @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer in_buf.deinit();
    try in_buf.upload(std.mem.sliceAsBytes(input));

    var batched_out = try GpuBuffer.initHostCoherent(&gpu, out_len, @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer batched_out.deinit();

    {
        const cmd = try gpu.beginBatch();
        var batched_set = try pl_batched.record(cmd, &in_buf, &batched_out, @intCast(ncols), @intCast(active));
        try gpu.submitBatch(cmd);
        _ = vk.vkFreeDescriptorSets(gpu.device, pl_batched.desc_pool, 1, &batched_set);
    }

    const batched_bytes = try al.alloc(u8, out_len);
    defer al.free(batched_bytes);
    try batched_out.download(batched_bytes);

    const basic = try al.alloc(u8, out_slot_bytes);
    defer al.free(basic);
    const deq = try al.alloc(f32, ncols);
    defer al.free(deq);
    for (0..active) |slot| {
        dq.unpackQ8_1_x4(batched_bytes[slot * out_slot_bytes ..][0..out_slot_bytes], basic);
        dq.dequantQ8_1(basic[0 .. (ncols / 32) * dq.Q8_1_BLOCK_BYTES], deq);
        var max_abs: f32 = 0.0;
        var amax: f32 = 0.0;
        for (input[slot * ncols ..][0..ncols], deq) |c, g| {
            const d = @abs(c - g);
            if (d > max_abs) max_abs = d;
            if (@abs(c) > amax) amax = @abs(c);
        }
        try std.testing.expect(max_abs <= amax / 120.0 + 1e-5);
    }
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
    var in_buf = try GpuBuffer.initHostCoherent(&gpu, inputs.len * @sizeOf(f32), usage);
    defer in_buf.deinit();
    var sc_buf = try GpuBuffer.initHostCoherent(&gpu, scales.len * @sizeOf(f32), usage);
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
    const r = prng.random();
    const al = std.testing.allocator;

    const blk_bytes = math_mod.rowBytes(tag, cols);
    const total_mat = rows * blk_bytes;
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
        mat[off + 0] = small_d_le[0];
        mat[off + 1] = small_d_le[1];
        // Q5_1 / Q4_K / Q5_K all have a second f16 at [2..3]
        if (tag == .q5_1 or tag == .q4_k or tag == .q5_k) {
            mat[off + 2] = small_d_le[0];
            mat[off + 3] = small_d_le[1];
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
    std.debug.print("{s} fuzz rows={} cols={}  max|D|={d:.6}  rel={e:.3}\n", .{ tag.label(), rows, cols, max_abs, rel });
    try std.testing.expect(rel < 1e-4);
}

test "gpu matvec Q8_0 fuzz" {
    try fuzzQuantMatvec(.q8_0, MatvecPipeline.initQ8_0, MatvecSession.initQ8_0, 32, 256, 1);
}
test "gpu matvec Q5_0 fuzz" {
    try fuzzQuantMatvec(.q5_0, MatvecPipeline.initQ5_0, MatvecSession.initQ5_0, 32, 256, 2);
}
test "gpu matvec Q5_1 fuzz" {
    try fuzzQuantMatvec(.q5_1, MatvecPipeline.initQ5_1, MatvecSession.initQ5_1, 32, 256, 3);
}

test "gpu expert-id down Q5_1 x Q8_1 fuzz" {
    const ctx = GpuContext.init() catch |e| {
        std.debug.print("gpu init failed: {}\n", .{e});
        return;
    };
    var gpu = ctx;
    defer gpu.deinit();

    const al = std.testing.allocator;
    const n_experts: usize = 3;
    const active: usize = 2;
    const rows: usize = 8;
    const cols: usize = 256;
    const row_bytes = math_mod.rowBytes(.q5_1, cols);

    var prng = std.Random.DefaultPrng.init(0xE1D5_0001);
    const r = prng.random();

    const mat = try al.alloc(u8, n_experts * rows * row_bytes);
    defer al.free(mat);
    for (mat) |*b| b.* = r.int(u8);

    const small_d_le: [2]u8 = .{ 0xCD, 0x21 };
    const blocks_per_row = cols / 32;
    for (0..n_experts * rows) |i| for (0..blocks_per_row) |b| {
        const off = (i * blocks_per_row + b) * 24;
        mat[off + 0] = small_d_le[0];
        mat[off + 1] = small_d_le[1];
        mat[off + 2] = small_d_le[0];
        mat[off + 3] = small_d_le[1];
    };

    const vec = try al.alloc(f32, cols);
    defer al.free(vec);
    for (vec) |*v| v.* = (r.float(f32) - 0.5) * 2.0;

    const q8_1_basic = try al.alloc(u8, (cols / 32) * dq.Q8_1_BLOCK_BYTES);
    defer al.free(q8_1_basic);
    dq.quantizeQ8_1(vec, q8_1_basic);

    const vec_q8_rounded = try al.alloc(f32, cols);
    defer al.free(vec_q8_rounded);
    dq.dequantQ8_1(q8_1_basic, vec_q8_rounded);

    const q8_1_x4 = try al.alloc(u8, q8_1_basic.len);
    defer al.free(q8_1_x4);
    dq.packQ8_1_x4(q8_1_basic, q8_1_x4);
    const q8_1_x4_all = try al.alloc(u8, active * q8_1_x4.len);
    defer al.free(q8_1_x4_all);
    for (0..active) |slot| {
        @memcpy(q8_1_x4_all[slot * q8_1_x4.len ..][0..q8_1_x4.len], q8_1_x4);
    }

    const ids = [_]u32{ 2, 0 };
    const scales = [_]f32{ 0.25, -0.75 };

    const cpu_out = try al.alloc(f32, active * rows);
    defer al.free(cpu_out);
    const row_buf = try al.alloc(f32, cols);
    defer al.free(row_buf);
    for (ids, 0..) |expert_id, slot| {
        const e: usize = @intCast(expert_id);
        const expert_mat = mat[e * rows * row_bytes ..][0 .. rows * row_bytes];
        math_mod.quantMatvec(cpu_out[slot * rows ..][0..rows], expert_mat, .q5_1, vec_q8_rounded, rows, cols, row_buf);
        for (cpu_out[slot * rows ..][0..rows]) |*v| v.* *= scales[slot];
    }

    var pipeline = try ExpertDownIdPipeline.initQ5_1Q8_1(&gpu);
    defer pipeline.deinit();
    var session = try MatvecSession.initQ5_1(&gpu, mat, @intCast(n_experts * rows), @intCast(cols));
    defer session.deinit();

    var acts_buf = try GpuBuffer.initHostCoherent(&gpu, q8_1_x4_all.len, @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer acts_buf.deinit();
    try acts_buf.upload(q8_1_x4_all);

    var ids_buf = try GpuBuffer.initHostCoherent(&gpu, ids.len * @sizeOf(u32), @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer ids_buf.deinit();
    try ids_buf.upload(std.mem.sliceAsBytes(&ids));

    var scales_buf = try GpuBuffer.initHostCoherent(&gpu, scales.len * @sizeOf(f32), @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer scales_buf.deinit();
    try scales_buf.upload(std.mem.sliceAsBytes(&scales));

    var out_buf = try GpuBuffer.initHostCoherent(&gpu, active * rows * @sizeOf(f32), @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer out_buf.deinit();

    const cmd = try gpu.beginBatch();
    const ds = try pipeline.record(cmd, &session.mat_buf, &acts_buf, &ids_buf, &scales_buf, &out_buf, @intCast(rows), @intCast(cols), @intCast(active));
    try gpu.submitBatch(cmd);
    var ds_mut = ds;
    _ = vk.vkFreeDescriptorSets(gpu.device, pipeline.desc_pool, 1, &ds_mut);

    const gpu_out = try al.alloc(f32, active * rows);
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
    std.debug.print("expert-id Q5_1×Q8_1 fuzz active={} rows={} cols={} max|D|={d:.6} rel={e:.3}\n", .{ active, rows, cols, max_abs, rel });
    try std.testing.expect(rel < 1e-3);
}

test "gpu expert-id down Q5_0 x Q8_1 fuzz" {
    const ctx = GpuContext.init() catch |e| {
        std.debug.print("gpu init failed: {}\n", .{e});
        return;
    };
    var gpu = ctx;
    defer gpu.deinit();

    const al = std.testing.allocator;
    const n_experts: usize = 3;
    const active: usize = 2;
    const rows: usize = 8;
    const cols: usize = 256;
    const row_bytes = math_mod.rowBytes(.q5_0, cols);

    var prng = std.Random.DefaultPrng.init(0xE1D5_0003);
    const r = prng.random();

    const mat = try al.alloc(u8, n_experts * rows * row_bytes);
    defer al.free(mat);
    for (mat) |*b| b.* = r.int(u8);

    const small_d_le: [2]u8 = .{ 0xCD, 0x21 };
    const blocks_per_row = cols / 32;
    for (0..n_experts * rows) |i| for (0..blocks_per_row) |b| {
        const off = (i * blocks_per_row + b) * 22;
        mat[off + 0] = small_d_le[0];
        mat[off + 1] = small_d_le[1];
    };

    const vec = try al.alloc(f32, cols);
    defer al.free(vec);
    for (vec) |*v| v.* = (r.float(f32) - 0.5) * 2.0;

    const q8_1_basic = try al.alloc(u8, (cols / 32) * dq.Q8_1_BLOCK_BYTES);
    defer al.free(q8_1_basic);
    dq.quantizeQ8_1(vec, q8_1_basic);

    const vec_q8_rounded = try al.alloc(f32, cols);
    defer al.free(vec_q8_rounded);
    dq.dequantQ8_1(q8_1_basic, vec_q8_rounded);

    const q8_1_x4 = try al.alloc(u8, q8_1_basic.len);
    defer al.free(q8_1_x4);
    dq.packQ8_1_x4(q8_1_basic, q8_1_x4);
    const q8_1_x4_all = try al.alloc(u8, active * q8_1_x4.len);
    defer al.free(q8_1_x4_all);
    for (0..active) |slot| {
        @memcpy(q8_1_x4_all[slot * q8_1_x4.len ..][0..q8_1_x4.len], q8_1_x4);
    }

    const ids = [_]u32{ 2, 0 };
    const scales = [_]f32{ 0.25, -0.75 };

    const cpu_out = try al.alloc(f32, active * rows);
    defer al.free(cpu_out);
    const row_buf = try al.alloc(f32, cols);
    defer al.free(row_buf);
    for (ids, 0..) |expert_id, slot| {
        const e: usize = @intCast(expert_id);
        const expert_mat = mat[e * rows * row_bytes ..][0 .. rows * row_bytes];
        math_mod.quantMatvec(cpu_out[slot * rows ..][0..rows], expert_mat, .q5_0, vec_q8_rounded, rows, cols, row_buf);
        for (cpu_out[slot * rows ..][0..rows]) |*v| v.* *= scales[slot];
    }

    var pipeline = try ExpertDownIdPipeline.initQ5_0Q8_1(&gpu);
    defer pipeline.deinit();
    var session = try MatvecSession.initQ5_0(&gpu, mat, @intCast(n_experts * rows), @intCast(cols));
    defer session.deinit();

    var acts_buf = try GpuBuffer.initHostCoherent(&gpu, q8_1_x4_all.len, @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer acts_buf.deinit();
    try acts_buf.upload(q8_1_x4_all);

    var ids_buf = try GpuBuffer.initHostCoherent(&gpu, ids.len * @sizeOf(u32), @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer ids_buf.deinit();
    try ids_buf.upload(std.mem.sliceAsBytes(&ids));

    var scales_buf = try GpuBuffer.initHostCoherent(&gpu, scales.len * @sizeOf(f32), @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer scales_buf.deinit();
    try scales_buf.upload(std.mem.sliceAsBytes(&scales));

    var out_buf = try GpuBuffer.initHostCoherent(&gpu, active * rows * @sizeOf(f32), @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer out_buf.deinit();

    const cmd = try gpu.beginBatch();
    const ds = try pipeline.record(cmd, &session.mat_buf, &acts_buf, &ids_buf, &scales_buf, &out_buf, @intCast(rows), @intCast(cols), @intCast(active));
    try gpu.submitBatch(cmd);
    var ds_mut = ds;
    _ = vk.vkFreeDescriptorSets(gpu.device, pipeline.desc_pool, 1, &ds_mut);

    const gpu_out = try al.alloc(f32, active * rows);
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
    std.debug.print("expert-id Q5_0×Q8_1 fuzz active={} rows={} cols={} max|D|={d:.6} rel={e:.3}\n", .{ active, rows, cols, max_abs, rel });
    try std.testing.expect(rel < 1e-3);
}

test "gpu expert-id down IQ4_NL x Q8_1 fuzz" {
    const ctx = GpuContext.init() catch |e| {
        std.debug.print("gpu init failed: {}\n", .{e});
        return;
    };
    var gpu = ctx;
    defer gpu.deinit();

    const al = std.testing.allocator;
    const n_experts: usize = 3;
    const active: usize = 2;
    const rows: usize = 8;
    const cols: usize = 256;
    const row_bytes = math_mod.rowBytes(.iq4_nl, cols);

    var prng = std.Random.DefaultPrng.init(0xE1D5_0004);
    const r = prng.random();

    const mat = try al.alloc(u8, n_experts * rows * row_bytes);
    defer al.free(mat);
    for (mat) |*b| b.* = r.int(u8);

    const small_d_le: [2]u8 = .{ 0xCD, 0x21 };
    const blocks_per_row = cols / 32;
    for (0..n_experts * rows) |i| for (0..blocks_per_row) |b| {
        const off = (i * blocks_per_row + b) * 18;
        mat[off + 0] = small_d_le[0];
        mat[off + 1] = small_d_le[1];
    };

    const vec = try al.alloc(f32, cols);
    defer al.free(vec);
    for (vec) |*v| v.* = (r.float(f32) - 0.5) * 2.0;

    const q8_1_basic = try al.alloc(u8, (cols / 32) * dq.Q8_1_BLOCK_BYTES);
    defer al.free(q8_1_basic);
    dq.quantizeQ8_1(vec, q8_1_basic);

    const vec_q8_rounded = try al.alloc(f32, cols);
    defer al.free(vec_q8_rounded);
    dq.dequantQ8_1(q8_1_basic, vec_q8_rounded);

    const q8_1_x4 = try al.alloc(u8, q8_1_basic.len);
    defer al.free(q8_1_x4);
    dq.packQ8_1_x4(q8_1_basic, q8_1_x4);
    const q8_1_x4_all = try al.alloc(u8, active * q8_1_x4.len);
    defer al.free(q8_1_x4_all);
    for (0..active) |slot| {
        @memcpy(q8_1_x4_all[slot * q8_1_x4.len ..][0..q8_1_x4.len], q8_1_x4);
    }

    const ids = [_]u32{ 2, 0 };
    const scales = [_]f32{ 0.25, -0.75 };

    const cpu_out = try al.alloc(f32, active * rows);
    defer al.free(cpu_out);
    const row_buf = try al.alloc(f32, cols);
    defer al.free(row_buf);
    for (ids, 0..) |expert_id, slot| {
        const e: usize = @intCast(expert_id);
        const expert_mat = mat[e * rows * row_bytes ..][0 .. rows * row_bytes];
        math_mod.quantMatvec(cpu_out[slot * rows ..][0..rows], expert_mat, .iq4_nl, vec_q8_rounded, rows, cols, row_buf);
        for (cpu_out[slot * rows ..][0..rows]) |*v| v.* *= scales[slot];
    }

    var pipeline = try ExpertDownIdPipeline.initIQ4NLQ8_1(&gpu);
    defer pipeline.deinit();
    var session = try MatvecSession.initIQ4NL(&gpu, mat, @intCast(n_experts * rows), @intCast(cols));
    defer session.deinit();

    var acts_buf = try GpuBuffer.initHostCoherent(&gpu, q8_1_x4_all.len, @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer acts_buf.deinit();
    try acts_buf.upload(q8_1_x4_all);

    var ids_buf = try GpuBuffer.initHostCoherent(&gpu, ids.len * @sizeOf(u32), @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer ids_buf.deinit();
    try ids_buf.upload(std.mem.sliceAsBytes(&ids));

    var scales_buf = try GpuBuffer.initHostCoherent(&gpu, scales.len * @sizeOf(f32), @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer scales_buf.deinit();
    try scales_buf.upload(std.mem.sliceAsBytes(&scales));

    var out_buf = try GpuBuffer.initHostCoherent(&gpu, active * rows * @sizeOf(f32), @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer out_buf.deinit();

    const cmd = try gpu.beginBatch();
    const ds = try pipeline.record(cmd, &session.mat_buf, &acts_buf, &ids_buf, &scales_buf, &out_buf, @intCast(rows), @intCast(cols), @intCast(active));
    try gpu.submitBatch(cmd);
    var ds_mut = ds;
    _ = vk.vkFreeDescriptorSets(gpu.device, pipeline.desc_pool, 1, &ds_mut);

    const gpu_out = try al.alloc(f32, active * rows);
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
    std.debug.print("expert-id IQ4_NL×Q8_1 fuzz active={} rows={} cols={} max|D|={d:.6} rel={e:.3}\n", .{ active, rows, cols, max_abs, rel });
    try std.testing.expect(rel < 1e-3);

    var pipeline_r2 = try ExpertDownIdPipeline.initIQ4NLQ8_1R2(&gpu);
    defer pipeline_r2.deinit();

    const cmd_r2 = try gpu.beginBatch();
    const ds_r2 = try pipeline_r2.record(cmd_r2, &session.mat_buf, &acts_buf, &ids_buf, &scales_buf, &out_buf, @intCast(rows), @intCast(cols), @intCast(active));
    try gpu.submitBatch(cmd_r2);
    var ds_r2_mut = ds_r2;
    _ = vk.vkFreeDescriptorSets(gpu.device, pipeline_r2.desc_pool, 1, &ds_r2_mut);

    const gpu_out_r2 = try al.alloc(f32, active * rows);
    defer al.free(gpu_out_r2);
    try out_buf.download(std.mem.sliceAsBytes(gpu_out_r2));

    var max_abs_r2: f32 = 0.0;
    var max_ref_r2: f32 = 0.0;
    for (cpu_out, gpu_out_r2) |c, g| {
        const d = @abs(c - g);
        if (d > max_abs_r2) max_abs_r2 = d;
        if (@abs(c) > max_ref_r2) max_ref_r2 = @abs(c);
    }
    const rel_r2 = max_abs_r2 / (max_ref_r2 + 1e-6);
    std.debug.print("expert-id IQ4_NL×Q8_1.r2 fuzz active={} rows={} cols={} max|D|={d:.6} rel={e:.3}\n", .{ active, rows, cols, max_abs_r2, rel_r2 });
    try std.testing.expect(rel_r2 < 1e-3);

    var pipeline_b16 = try ExpertDownIdPipeline.initIQ4NLQ8_1B16(&gpu);
    defer pipeline_b16.deinit();

    const cmd_b16 = try gpu.beginBatch();
    const ds_b16 = try pipeline_b16.record(cmd_b16, &session.mat_buf, &acts_buf, &ids_buf, &scales_buf, &out_buf, @intCast(rows), @intCast(cols), @intCast(active));
    try gpu.submitBatch(cmd_b16);
    var ds_b16_mut = ds_b16;
    _ = vk.vkFreeDescriptorSets(gpu.device, pipeline_b16.desc_pool, 1, &ds_b16_mut);

    const gpu_out_b16 = try al.alloc(f32, active * rows);
    defer al.free(gpu_out_b16);
    try out_buf.download(std.mem.sliceAsBytes(gpu_out_b16));

    var max_abs_b16: f32 = 0.0;
    var max_ref_b16: f32 = 0.0;
    for (cpu_out, gpu_out_b16) |c, g| {
        const d = @abs(c - g);
        if (d > max_abs_b16) max_abs_b16 = d;
        if (@abs(c) > max_ref_b16) max_ref_b16 = @abs(c);
    }
    const rel_b16 = max_abs_b16 / (max_ref_b16 + 1e-6);
    std.debug.print("expert-id IQ4_NL×Q8_1.b16 fuzz active={} rows={} cols={} max|D|={d:.6} rel={e:.3}\n", .{ active, rows, cols, max_abs_b16, rel_b16 });
    try std.testing.expect(rel_b16 < 1e-3);

    var pipeline_iacc = try ExpertDownIdPipeline.initIQ4NLQ8_1Iacc(&gpu);
    defer pipeline_iacc.deinit();

    const cmd_iacc = try gpu.beginBatch();
    const ds_iacc = try pipeline_iacc.record(cmd_iacc, &session.mat_buf, &acts_buf, &ids_buf, &scales_buf, &out_buf, @intCast(rows), @intCast(cols), @intCast(active));
    try gpu.submitBatch(cmd_iacc);
    var ds_iacc_mut = ds_iacc;
    _ = vk.vkFreeDescriptorSets(gpu.device, pipeline_iacc.desc_pool, 1, &ds_iacc_mut);

    const gpu_out_iacc = try al.alloc(f32, active * rows);
    defer al.free(gpu_out_iacc);
    try out_buf.download(std.mem.sliceAsBytes(gpu_out_iacc));

    var max_abs_iacc: f32 = 0.0;
    var max_ref_iacc: f32 = 0.0;
    for (cpu_out, gpu_out_iacc) |c, g| {
        const d = @abs(c - g);
        if (d > max_abs_iacc) max_abs_iacc = d;
        if (@abs(c) > max_ref_iacc) max_ref_iacc = @abs(c);
    }
    const rel_iacc = max_abs_iacc / (max_ref_iacc + 1e-6);
    std.debug.print("expert-id IQ4_NL×Q8_1.iacc fuzz active={} rows={} cols={} max|D|={d:.6} rel={e:.3}\n", .{ active, rows, cols, max_abs_iacc, rel_iacc });
    try std.testing.expect(rel_iacc < 1e-3);

    const cpu_sum = try al.alloc(f32, rows);
    defer al.free(cpu_sum);
    @memset(cpu_sum, 0.0);
    for (0..active) |slot| {
        for (0..rows) |row| cpu_sum[row] += cpu_out[slot * rows + row];
    }

    var pipeline_sum = try ExpertDownIdPipeline.initIQ4NLQ8_1Sum(&gpu);
    defer pipeline_sum.deinit();

    const cmd_sum = try gpu.beginBatch();
    const ds_sum = try pipeline_sum.record(cmd_sum, &session.mat_buf, &acts_buf, &ids_buf, &scales_buf, &out_buf, @intCast(rows), @intCast(cols), @intCast(active));
    try gpu.submitBatch(cmd_sum);
    var ds_sum_mut = ds_sum;
    _ = vk.vkFreeDescriptorSets(gpu.device, pipeline_sum.desc_pool, 1, &ds_sum_mut);

    const gpu_sum = try al.alloc(f32, rows);
    defer al.free(gpu_sum);
    try out_buf.download(std.mem.sliceAsBytes(gpu_sum));

    var max_abs_sum: f32 = 0.0;
    var max_ref_sum: f32 = 0.0;
    for (cpu_sum, gpu_sum) |c, g| {
        const d = @abs(c - g);
        if (d > max_abs_sum) max_abs_sum = d;
        if (@abs(c) > max_ref_sum) max_ref_sum = @abs(c);
    }
    const rel_sum = max_abs_sum / (max_ref_sum + 1e-6);
    std.debug.print("expert-id IQ4_NL×Q8_1.sum fuzz active={} rows={} cols={} max|D|={d:.6} rel={e:.3}\n", .{ active, rows, cols, max_abs_sum, rel_sum });
    try std.testing.expect(rel_sum < 1e-3);
}

test "gpu matvec Q4_K fuzz" {
    try fuzzQuantMatvec(.q4_k, MatvecPipeline.initQ4K, MatvecSession.initQ4K, 32, 512, 4);
}
test "gpu matvec Q3_K fuzz" {
    try fuzzQuantMatvec(.q3_k, MatvecPipeline.initQ3K, MatvecSession.initQ3K, 32, 512, 5);
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

    var in_buf = try GpuBuffer.initHostCoherent(ctx, in.len * @sizeOf(f32), @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer in_buf.deinit();
    try in_buf.upload(std.mem.sliceAsBytes(in));

    var out_buf = try GpuBuffer.initHostCoherent(ctx, out_x4.len, @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
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
    std.debug.print("Q8_1 quantize round-trip n={}  max|D|={d:.6}  amax={d:.6}\n", .{ n, max_abs, max_amax });
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
        mat[off + 0] = small_d_le[0];
        mat[off + 1] = small_d_le[1]; // d
        mat[off + 2] = small_d_le[0];
        mat[off + 3] = small_d_le[1]; // dmin
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

    var acts_buf = try GpuBuffer.initHostCoherent(&gpu, q8_1_x4.len, @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer acts_buf.deinit();
    try acts_buf.upload(q8_1_x4);

    var out_buf = try GpuBuffer.initHostCoherent(&gpu, rows * @sizeOf(f32), @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer out_buf.deinit();

    const cmd = try gpu.beginBatch();
    _ = try pipeline.record(cmd, &session.mat_buf, &acts_buf, &out_buf, @intCast(rows), @intCast(cols));
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
    std.debug.print("{s} fuzz rows={} cols={}  max|D|={d:.6}  rel={e:.3}\n", .{ label, rows, cols, max_abs, rel });
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

fn initQ4KQ8_1MmvqB32R1(ctx: *const GpuContext) !MatvecPipeline {
    return MatvecPipeline.initQ4KQ8_1Mmvq(ctx, .{
        .block_size = 32,
        .num_rows = 1,
        .num_cols = 1,
    });
}

fn initQ4KQ8_1MmvqB64R1(ctx: *const GpuContext) !MatvecPipeline {
    return MatvecPipeline.initQ4KQ8_1Mmvq(ctx, .{
        .block_size = 64,
        .num_rows = 1,
        .num_cols = 1,
    });
}

fn initQ4KQ8_1MmvqB64R2(ctx: *const GpuContext) !MatvecPipeline {
    return MatvecPipeline.initQ4KQ8_1Mmvq(ctx, .{
        .block_size = 64,
        .num_rows = 2,
        .num_cols = 1,
    });
}

fn initQ4KQ8_1MmvqB64R4(ctx: *const GpuContext) !MatvecPipeline {
    return MatvecPipeline.initQ4KQ8_1Mmvq(ctx, .{
        .block_size = 64,
        .num_rows = 4,
        .num_cols = 1,
    });
}

test "gpu matvec Q4_K × Q8_1 MMVQ b32 r1 fuzz small" {
    try fuzzQ4KQ8_1(32, 256, 21, initQ4KQ8_1MmvqB32R1, "Q4_K×Q8_1.mmvq.b32.r1");
}

test "gpu matvec Q4_K × Q8_1 MMVQ b32 r1 fuzz model-sized" {
    try fuzzQ4KQ8_1(64, 2304, 23, initQ4KQ8_1MmvqB32R1, "Q4_K×Q8_1.mmvq.b32.r1");
}

test "gpu matvec Q4_K × Q8_1 MMVQ b64 r1 fuzz small" {
    try fuzzQ4KQ8_1(32, 256, 25, initQ4KQ8_1MmvqB64R1, "Q4_K×Q8_1.mmvq.b64.r1");
}

test "gpu matvec Q4_K × Q8_1 MMVQ b64 r1 fuzz model-sized" {
    try fuzzQ4KQ8_1(64, 2304, 27, initQ4KQ8_1MmvqB64R1, "Q4_K×Q8_1.mmvq.b64.r1");
}

test "gpu matvec Q4_K × Q8_1 MMVQ b64 r2 fuzz model-sized" {
    try fuzzQ4KQ8_1(64, 2304, 29, initQ4KQ8_1MmvqB64R2, "Q4_K×Q8_1.mmvq.b64.r2");
}

test "gpu matvec Q4_K × Q8_1 MMVQ b64 r4 fuzz model-sized" {
    try fuzzQ4KQ8_1(64, 2304, 31, initQ4KQ8_1MmvqB64R4, "Q4_K×Q8_1.mmvq.b64.r4");
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

    var acts_buf = try GpuBuffer.initHostCoherent(&gpu, q8_1_x4.len, @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer acts_buf.deinit();
    try acts_buf.upload(q8_1_x4);

    var out_buf = try GpuBuffer.initHostCoherent(&gpu, rows * @sizeOf(f32), @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer out_buf.deinit();

    const cmd = try gpu.beginBatch();
    _ = try pipeline.record(cmd, &session.mat_buf, &acts_buf, &out_buf, @intCast(rows), @intCast(cols));
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
    std.debug.print("Q3_K×Q8_1 fuzz rows={} cols={}  max|D|={d:.6}  rel={e:.3}\n", .{ rows, cols, max_abs, rel });
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

    var acts_buf = try GpuBuffer.initHostCoherent(&gpu, q8_1_x4.len, @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer acts_buf.deinit();
    try acts_buf.upload(q8_1_x4);

    var out_buf = try GpuBuffer.initHostCoherent(&gpu, rows * @sizeOf(f32), @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer out_buf.deinit();

    const cmd = try gpu.beginBatch();
    _ = try pipeline.record(cmd, &session.mat_buf, &acts_buf, &out_buf, @intCast(rows), @intCast(cols));
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
    std.debug.print("{s}×Q8_1 fuzz rows={} cols={}  max|D|={d:.6}  rel={e:.3}\n", .{ tag.label(), rows, cols, max_abs, rel });
    try std.testing.expect(rel < 1e-3);
}

test "gpu matvec Q5_0 × Q8_1 fuzz small" {
    // cols=32 is fine — only 32-aligned, no 256 requirement (Q5_0 has 32-elem blocks).
    try fuzzQuantQ8_1(.q5_0, 22, 32, MatvecPipeline.initQ5_0Q8_1, MatvecSession.initQ5_0, 32, 256, 41);
}
test "gpu matvec Q5_0 × Q8_1 fuzz model-sized" {
    // d_expert = 704 — closest analogue to a Gemma4 expert down cols.
    try fuzzQuantQ8_1(.q5_0, 22, 32, MatvecPipeline.initQ5_0Q8_1, MatvecSession.initQ5_0, 64, 704, 43);
}
test "gpu matvec Q5_1 × Q8_1 fuzz small" {
    try fuzzQuantQ8_1(.q5_1, 24, 32, MatvecPipeline.initQ5_1Q8_1, MatvecSession.initQ5_1, 32, 256, 47);
}
test "gpu matvec Q5_1 × Q8_1 fuzz model-sized" {
    try fuzzQuantQ8_1(.q5_1, 24, 32, MatvecPipeline.initQ5_1Q8_1, MatvecSession.initQ5_1, 64, 704, 53);
}

fn initQ5_0Q8_1MmvqB64R1(ctx: *const GpuContext) !MatvecPipeline {
    return MatvecPipeline.initQ5_0Q8_1Mmvq(ctx, .{
        .block_size = 64,
        .num_rows = 1,
        .num_cols = 1,
    });
}

fn initQ5_0Q8_1MmvqB64R2(ctx: *const GpuContext) !MatvecPipeline {
    return MatvecPipeline.initQ5_0Q8_1Mmvq(ctx, .{
        .block_size = 64,
        .num_rows = 2,
        .num_cols = 1,
    });
}

fn initQ5_0Q8_1MmvqB64R4(ctx: *const GpuContext) !MatvecPipeline {
    return MatvecPipeline.initQ5_0Q8_1Mmvq(ctx, .{
        .block_size = 64,
        .num_rows = 4,
        .num_cols = 1,
    });
}

fn initQ5_1Q8_1MmvqB64R1(ctx: *const GpuContext) !MatvecPipeline {
    return MatvecPipeline.initQ5_1Q8_1Mmvq(ctx, .{
        .block_size = 64,
        .num_rows = 1,
        .num_cols = 1,
    });
}

fn initQ5_1Q8_1MmvqB64R2(ctx: *const GpuContext) !MatvecPipeline {
    return MatvecPipeline.initQ5_1Q8_1Mmvq(ctx, .{
        .block_size = 64,
        .num_rows = 2,
        .num_cols = 1,
    });
}

fn initQ5_1Q8_1MmvqB64R4(ctx: *const GpuContext) !MatvecPipeline {
    return MatvecPipeline.initQ5_1Q8_1Mmvq(ctx, .{
        .block_size = 64,
        .num_rows = 4,
        .num_cols = 1,
    });
}

test "gpu matvec Q5_0 × Q8_1 MMVQ b64 r1 fuzz small" {
    try fuzzQuantQ8_1(.q5_0, 22, 32, initQ5_0Q8_1MmvqB64R1, MatvecSession.initQ5_0, 32, 256, 90);
}

test "gpu matvec Q5_0 × Q8_1 MMVQ b64 r1 fuzz model-sized" {
    try fuzzQuantQ8_1(.q5_0, 22, 32, initQ5_0Q8_1MmvqB64R1, MatvecSession.initQ5_0, 64, 704, 91);
}

test "gpu matvec Q5_0 × Q8_1 MMVQ b64 r2 fuzz model-sized" {
    try fuzzQuantQ8_1(.q5_0, 22, 32, initQ5_0Q8_1MmvqB64R2, MatvecSession.initQ5_0, 64, 704, 94);
}

test "gpu matvec Q5_0 × Q8_1 MMVQ b64 r4 fuzz model-sized" {
    try fuzzQuantQ8_1(.q5_0, 22, 32, initQ5_0Q8_1MmvqB64R4, MatvecSession.initQ5_0, 64, 704, 95);
}

test "gpu matvec Q5_1 × Q8_1 MMVQ b64 r1 fuzz small" {
    try fuzzQuantQ8_1(.q5_1, 24, 32, initQ5_1Q8_1MmvqB64R1, MatvecSession.initQ5_1, 32, 256, 92);
}

test "gpu matvec Q5_1 × Q8_1 MMVQ b64 r1 fuzz model-sized" {
    try fuzzQuantQ8_1(.q5_1, 24, 32, initQ5_1Q8_1MmvqB64R1, MatvecSession.initQ5_1, 64, 704, 93);
}

test "gpu matvec Q5_1 × Q8_1 MMVQ b64 r2 fuzz model-sized" {
    try fuzzQuantQ8_1(.q5_1, 24, 32, initQ5_1Q8_1MmvqB64R2, MatvecSession.initQ5_1, 64, 704, 96);
}

test "gpu matvec Q5_1 × Q8_1 MMVQ b64 r4 fuzz model-sized" {
    try fuzzQuantQ8_1(.q5_1, 24, 32, initQ5_1Q8_1MmvqB64R4, MatvecSession.initQ5_1, 64, 704, 97);
}

test "gpu matvec Q6_K × Q8_1 fuzz small" {
    try fuzzQuantQ8_1(.q6_k, 210, 256, MatvecPipeline.initQ6KQ8_1, MatvecSession.initQ6K, 32, 256, 59);
}

test "gpu matvec Q6_K × Q8_1 fuzz lm-head-shaped cols" {
    try fuzzQuantQ8_1(.q6_k, 210, 256, MatvecPipeline.initQ6KQ8_1, MatvecSession.initQ6K, 64, 2816, 61);
}

test "gpu matvec Q6_K × Q8_1 fast fuzz small" {
    try fuzzQuantQ8_1(.q6_k, 210, 256, MatvecPipeline.initQ6KQ8_1Fast, MatvecSession.initQ6K, 32, 256, 63);
}

test "gpu matvec Q6_K × Q8_1 fast fuzz lm-head-shaped cols" {
    try fuzzQuantQ8_1(.q6_k, 210, 256, MatvecPipeline.initQ6KQ8_1Fast, MatvecSession.initQ6K, 64, 2816, 65);
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
    try fuzzQuantQ8_1(.q3_k, 110, 256, initQ3KQ8_1MmvqB32R1, MatvecSession.initQ3K, 32, 256, 80);
}

test "gpu matvec Q3_K × Q8_1 MMVQ b32 r1 fuzz lm-head-shaped cols" {
    try fuzzQuantQ8_1(.q3_k, 110, 256, initQ3KQ8_1MmvqB32R1, MatvecSession.initQ3K, 64, 2816, 82);
}

test "gpu matvec Q3_K × Q8_1 MMVQ b64 r1 fuzz small" {
    try fuzzQuantQ8_1(.q3_k, 110, 256, initQ3KQ8_1MmvqB64R1, MatvecSession.initQ3K, 32, 256, 81);
}

test "gpu matvec Q3_K × Q8_1 MMVQ b64 r1 fuzz lm-head-shaped cols" {
    try fuzzQuantQ8_1(.q3_k, 110, 256, initQ3KQ8_1MmvqB64R1, MatvecSession.initQ3K, 64, 2816, 83);
}

test "gpu matvec Q6_K × Q8_1 MMVQ b32 r1 fuzz small" {
    try fuzzQuantQ8_1(.q6_k, 210, 256, initQ6KQ8_1MmvqB32R1, MatvecSession.initQ6K, 32, 256, 67);
}

test "gpu matvec Q6_K × Q8_1 MMVQ b32 r1 fuzz lm-head-shaped cols" {
    try fuzzQuantQ8_1(.q6_k, 210, 256, initQ6KQ8_1MmvqB32R1, MatvecSession.initQ6K, 64, 2816, 69);
}

test "gpu matvec Q6_K × Q8_1 MMVQ b64 r1 fuzz small" {
    try fuzzQuantQ8_1(.q6_k, 210, 256, initQ6KQ8_1MmvqB64R1, MatvecSession.initQ6K, 32, 256, 71);
}

test "gpu matvec Q6_K × Q8_1 MMVQ b64 r1 fuzz lm-head-shaped cols" {
    try fuzzQuantQ8_1(.q6_k, 210, 256, initQ6KQ8_1MmvqB64R1, MatvecSession.initQ6K, 64, 2816, 73);
}

test "gpu matvec Q6_K × Q8_1 MMVQ b64 r2 fuzz lm-head-shaped cols" {
    try fuzzQuantQ8_1(.q6_k, 210, 256, initQ6KQ8_1MmvqB64R2, MatvecSession.initQ6K, 64, 2816, 75);
}

test "gpu matvec Q6_K × Q8_1 MMVQ b64 r4 fuzz lm-head-shaped cols" {
    try fuzzQuantQ8_1(.q6_k, 210, 256, initQ6KQ8_1MmvqB64R4, MatvecSession.initQ6K, 64, 2816, 77);
}

test "gpu matvec Q5_K × Q8_1 fuzz small" {
    try fuzzQuantQ8_1(.q5_k, 176, 256, MatvecPipeline.initQ5KQ8_1, MatvecSession.initQ5K, 32, 256, 73);
}

test "gpu matvec Q5_K × Q8_1 fuzz attn-v-shaped cols" {
    try fuzzQuantQ8_1(.q5_k, 176, 256, MatvecPipeline.initQ5KQ8_1, MatvecSession.initQ5K, 64, 2816, 79);
}

fn initQ5KQ8_1MmvqB64R1(ctx: *const GpuContext) !MatvecPipeline {
    return MatvecPipeline.initQ5KQ8_1Mmvq(ctx, .{
        .block_size = 64,
        .num_rows = 1,
        .num_cols = 1,
    });
}

fn initQ5KQ8_1MmvqB64R2(ctx: *const GpuContext) !MatvecPipeline {
    return MatvecPipeline.initQ5KQ8_1Mmvq(ctx, .{
        .block_size = 64,
        .num_rows = 2,
        .num_cols = 1,
    });
}

fn initQ5KQ8_1MmvqB64R4(ctx: *const GpuContext) !MatvecPipeline {
    return MatvecPipeline.initQ5KQ8_1Mmvq(ctx, .{
        .block_size = 64,
        .num_rows = 4,
        .num_cols = 1,
    });
}

test "gpu matvec Q5_K × Q8_1 MMVQ b64 r1 fuzz small" {
    try fuzzQuantQ8_1(.q5_k, 176, 256, initQ5KQ8_1MmvqB64R1, MatvecSession.initQ5K, 32, 256, 101);
}

test "gpu matvec Q5_K × Q8_1 MMVQ b64 r1 fuzz attn-v-shaped cols" {
    try fuzzQuantQ8_1(.q5_k, 176, 256, initQ5KQ8_1MmvqB64R1, MatvecSession.initQ5K, 64, 2816, 103);
}

test "gpu matvec Q5_K × Q8_1 MMVQ b64 r2 fuzz attn-v-shaped cols" {
    try fuzzQuantQ8_1(.q5_k, 176, 256, initQ5KQ8_1MmvqB64R2, MatvecSession.initQ5K, 64, 2816, 105);
}

test "gpu matvec Q5_K × Q8_1 MMVQ b64 r4 fuzz attn-v-shaped cols" {
    try fuzzQuantQ8_1(.q5_k, 176, 256, initQ5KQ8_1MmvqB64R4, MatvecSession.initQ5K, 64, 2816, 107);
}

test "gpu matvec IQ4_NL × Q8_1 fuzz small" {
    try fuzzQuantQ8_1(.iq4_nl, 18, 32, MatvecPipeline.initIQ4NLQ8_1, MatvecSession.initIQ4NL, 32, 256, 67);
}

test "gpu matvec IQ4_NL × Q8_1 fuzz expert-down-shaped" {
    try fuzzQuantQ8_1(.iq4_nl, 18, 32, MatvecPipeline.initIQ4NLQ8_1, MatvecSession.initIQ4NL, 64, 704, 71);
}

// ── rmsnorm fuzz test ─────────────────────────────────────────────────────────
fn runRmsnormShader(
    gpu: *const GpuContext,
    x: []const f32,
    w: []const f32,
    eps: f32,
    weight_offset: bool,
    precise_sum: bool,
    use_r128: bool,
    out: []f32,
) !void {
    var pl = if (use_r128)
        try RmsnormPipeline.initR128(gpu)
    else
        try RmsnormPipeline.init(gpu);
    defer pl.deinit();

    var x_buf = try GpuBuffer.initHostCoherent(gpu, x.len * @sizeOf(f32), @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer x_buf.deinit();
    try x_buf.upload(std.mem.sliceAsBytes(x));

    var w_buf = try GpuBuffer.initHostCoherent(gpu, w.len * @sizeOf(f32), @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer w_buf.deinit();
    try w_buf.upload(std.mem.sliceAsBytes(w));

    var y_buf = try GpuBuffer.initHostCoherent(gpu, out.len * @sizeOf(f32), @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer y_buf.deinit();

    const cmd = try gpu.beginBatch();
    _ = if (precise_sum)
        try pl.recordPrecise(cmd, &x_buf, &w_buf, &y_buf, @intCast(x.len), eps, weight_offset)
    else
        try pl.record(cmd, &x_buf, &w_buf, &y_buf, @intCast(x.len), eps, weight_offset);
    try gpu.submitBatch(cmd);
    try y_buf.download(std.mem.sliceAsBytes(out));
}

fn fuzzRmsnorm(n: usize, seed: u64, weight_offset: bool, precise_sum: bool) !void {
    const ctx = GpuContext.init() catch |e| {
        std.debug.print("gpu init failed: {}\n", .{e});
        return;
    };
    var gpu = ctx;
    defer gpu.deinit();

    const al = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();

    const x = try al.alloc(f32, n);
    defer al.free(x);
    const w = try al.alloc(f32, n);
    defer al.free(w);
    const cpu_out = try al.alloc(f32, n);
    defer al.free(cpu_out);
    const gpu_out = try al.alloc(f32, n);
    defer al.free(gpu_out);
    const gpu_out_r128 = try al.alloc(f32, n);
    defer al.free(gpu_out_r128);

    for (x) |*v| v.* = (r.float(f32) - 0.5) * 4.0;
    for (w) |*v| v.* = (r.float(f32) - 0.5) * 2.0;

    // CPU reference: y = x * rms_inv * (bias + w)
    var ss: f32 = 0.0;
    for (x) |v| ss += v * v;
    const eps: f32 = 1e-6;
    const rms_inv: f32 = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(n)) + eps);
    const bias: f32 = if (weight_offset) 1.0 else 0.0;
    for (cpu_out, x, w) |*o, xi, wi| o.* = xi * rms_inv * (bias + wi);

    try runRmsnormShader(&gpu, x, w, eps, weight_offset, precise_sum, false, gpu_out);
    try runRmsnormShader(&gpu, x, w, eps, weight_offset, precise_sum, true, gpu_out_r128);

    var max_abs: f32 = 0.0;
    var max_ref: f32 = 0.0;
    var max_abs_r128: f32 = 0.0;
    for (cpu_out, gpu_out) |c, g| {
        const d = @abs(c - g);
        if (d > max_abs) max_abs = d;
        if (@abs(c) > max_ref) max_ref = @abs(c);
    }
    for (cpu_out, gpu_out_r128) |c, g| {
        const d = @abs(c - g);
        if (d > max_abs_r128) max_abs_r128 = d;
    }
    const rel = max_abs / (max_ref + 1e-6);
    const rel_r128 = max_abs_r128 / (max_ref + 1e-6);
    std.debug.print("rmsnorm fuzz n={} bias={} precise={} max|D|={d:.6} rel={e:.3} r128_rel={e:.3}\n", .{ n, weight_offset, precise_sum, max_abs, rel, rel_r128 });
    try std.testing.expect(rel < 1e-5);
    try std.testing.expect(rel_r128 < 1e-5);
}

test "gpu rmsnorm n=512 fuzz" {
    try fuzzRmsnorm(512, 71, false, false); // head_dim_global
}
test "gpu rmsnorm n=2816 fuzz" {
    try fuzzRmsnorm(2816, 73, false, false); // d_model
}
test "gpu rmsnorm n=2816 fuzz with (1+w) convention" {
    try fuzzRmsnorm(2816, 79, true, false); // exercise the weight_offset path
}
test "gpu rmsnorm n=2816 precise-sum fuzz" {
    try fuzzRmsnorm(2816, 83, false, true);
}

// ── add_rmsnorm fuzz test ─────────────────────────────────────────────────────

fn runAddRmsnormShader(
    gpu: *const GpuContext,
    a: []const f32,
    b: []const f32,
    w: []const f32,
    eps: f32,
    weight_offset: bool,
    precise_sum: bool,
    use_r128: bool,
    out: []f32,
) !void {
    var pl = if (use_r128)
        try AddRmsnormPipeline.initR128(gpu)
    else
        try AddRmsnormPipeline.init(gpu);
    defer pl.deinit();

    var a_buf = try GpuBuffer.initHostCoherent(gpu, a.len * @sizeOf(f32), @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer a_buf.deinit();
    try a_buf.upload(std.mem.sliceAsBytes(a));

    var b_buf = try GpuBuffer.initHostCoherent(gpu, b.len * @sizeOf(f32), @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer b_buf.deinit();
    try b_buf.upload(std.mem.sliceAsBytes(b));

    var w_buf = try GpuBuffer.initHostCoherent(gpu, w.len * @sizeOf(f32), @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer w_buf.deinit();
    try w_buf.upload(std.mem.sliceAsBytes(w));

    var y_buf = try GpuBuffer.initHostCoherent(gpu, out.len * @sizeOf(f32), @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer y_buf.deinit();

    const dset = try pl.allocSet(&a_buf, &b_buf, &w_buf, &y_buf);
    const cmd = try gpu.beginBatch();
    pl.recordWithSet(cmd, dset, @intCast(a.len), eps, weight_offset, precise_sum);
    try gpu.submitBatch(cmd);
    try y_buf.download(std.mem.sliceAsBytes(out));
}

fn fuzzAddRmsnorm(n: usize, seed: u64, weight_offset: bool, precise_sum: bool) !void {
    const ctx = GpuContext.init() catch |e| {
        std.debug.print("gpu init failed: {}\n", .{e});
        return;
    };
    var gpu = ctx;
    defer gpu.deinit();

    const al = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();

    const a = try al.alloc(f32, n);
    defer al.free(a);
    const b = try al.alloc(f32, n);
    defer al.free(b);
    const w = try al.alloc(f32, n);
    defer al.free(w);
    const cpu_out = try al.alloc(f32, n);
    defer al.free(cpu_out);
    const gpu_out = try al.alloc(f32, n);
    defer al.free(gpu_out);
    const gpu_out_r128 = try al.alloc(f32, n);
    defer al.free(gpu_out_r128);

    for (a) |*v| v.* = (r.float(f32) - 0.5) * 4.0;
    for (b) |*v| v.* = (r.float(f32) - 0.5) * 4.0;
    for (w) |*v| v.* = (r.float(f32) - 0.5) * 2.0;

    var ss: f32 = 0.0;
    for (a, b) |av, bv| {
        const v = av + bv;
        ss += v * v;
    }
    const eps: f32 = 1e-6;
    const rms_inv: f32 = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(n)) + eps);
    const bias: f32 = if (weight_offset) 1.0 else 0.0;
    for (cpu_out, a, b, w) |*o, av, bv, wi| o.* = (av + bv) * rms_inv * (bias + wi);

    try runAddRmsnormShader(&gpu, a, b, w, eps, weight_offset, precise_sum, false, gpu_out);
    try runAddRmsnormShader(&gpu, a, b, w, eps, weight_offset, precise_sum, true, gpu_out_r128);

    var max_abs: f32 = 0.0;
    var max_ref: f32 = 0.0;
    var max_abs_r128: f32 = 0.0;
    for (cpu_out, gpu_out) |c, g| {
        const d = @abs(c - g);
        if (d > max_abs) max_abs = d;
        if (@abs(c) > max_ref) max_ref = @abs(c);
    }
    for (cpu_out, gpu_out_r128) |c, g| {
        const d = @abs(c - g);
        if (d > max_abs_r128) max_abs_r128 = d;
    }
    const rel = max_abs / (max_ref + 1e-6);
    const rel_r128 = max_abs_r128 / (max_ref + 1e-6);
    std.debug.print("add_rmsnorm fuzz n={} bias={} precise={} max|D|={d:.6} rel={e:.3} r128_rel={e:.3}\n", .{ n, weight_offset, precise_sum, max_abs, rel, rel_r128 });
    try std.testing.expect(rel < 1e-5);
    try std.testing.expect(rel_r128 < 1e-5);
}

test "gpu add_rmsnorm n=2816 fuzz" {
    try fuzzAddRmsnorm(2816, 89, false, false);
}
test "gpu add_rmsnorm n=2816 precise-sum fuzz" {
    try fuzzAddRmsnorm(2816, 97, false, true);
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

    var x_buf = try GpuBuffer.initHostCoherent(gpu, x.len * @sizeOf(f32), @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer x_buf.deinit();
    try x_buf.upload(std.mem.sliceAsBytes(x));

    var w_buf = try GpuBuffer.initHostCoherent(gpu, w.len * @sizeOf(f32), @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer w_buf.deinit();
    try w_buf.upload(std.mem.sliceAsBytes(w));

    var y_buf = try GpuBuffer.initHostCoherent(gpu, out.len * @sizeOf(f32), @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer y_buf.deinit();

    const cmd = try gpu.beginBatch();
    _ = try pl.record(cmd, &x_buf, &w_buf, &y_buf, n_heads, head_dim, eps, weight_offset, use_weight);
    try gpu.submitBatch(cmd);
    try y_buf.download(std.mem.sliceAsBytes(out));
}

fn fuzzRmsnormPerHead(
    n_heads: u32,
    head_dim: u32,
    seed: u64,
    weight_offset: bool,
    use_weight: bool,
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
    const x = try al.alloc(f32, n_total);
    defer al.free(x);
    const w = try al.alloc(f32, head_dim);
    defer al.free(w);
    const cpu_out = try al.alloc(f32, n_total);
    defer al.free(cpu_out);
    const gpu_out = try al.alloc(f32, n_total);
    defer al.free(gpu_out);

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

    try runRmsnormPerHeadShader(&gpu, x, w, n_heads, head_dim, eps, weight_offset, use_weight, gpu_out);

    var max_abs: f32 = 0.0;
    var max_ref: f32 = 0.0;
    for (cpu_out, gpu_out) |c, g| {
        const d = @abs(c - g);
        if (d > max_abs) max_abs = d;
        if (@abs(c) > max_ref) max_ref = @abs(c);
    }
    const rel = max_abs / (max_ref + 1e-6);
    std.debug.print("rmsnorm_perhead fuzz n_heads={} hd={} bias={} use_w={} max|D|={d:.6} rel={e:.3}\n", .{ n_heads, head_dim, weight_offset, use_weight, max_abs, rel });
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

fn fuzzRmsnormPerHeadRope(use_table: bool, n_heads: u32, head_dim: u32, seed: u64) !void {
    const ctx = GpuContext.init() catch |e| {
        std.debug.print("gpu init failed: {}\n", .{e});
        return;
    };
    var gpu = ctx;
    defer gpu.deinit();

    const rope_mod = @import("../ops/rope.zig");
    const al = std.testing.allocator;
    const total = @as(usize, n_heads) * @as(usize, head_dim);
    const pos: usize = 13;
    const theta: f32 = 10000.0;
    const eps: f32 = 1e-6;
    const x = try al.alloc(f32, total);
    defer al.free(x);
    const w = try al.alloc(f32, head_dim);
    defer al.free(w);
    const freqs = try al.alloc(f32, head_dim / 2);
    defer al.free(freqs);
    const cpu_out = try al.alloc(f32, total);
    defer al.free(cpu_out);
    const gpu_out = try al.alloc(f32, total);
    defer al.free(gpu_out);

    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    for (x) |*v| v.* = (r.float(f32) - 0.5) * 4.0;
    for (w) |*v| v.* = (r.float(f32) - 0.5) * 2.0;
    for (freqs, 0..) |*f, i|
        f.* = 1.0 / std.math.pow(f32, theta, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(head_dim)));

    for (0..n_heads) |h| {
        const head = cpu_out[h * head_dim ..][0..head_dim];
        const src = x[h * head_dim ..][0..head_dim];
        var ss: f32 = 0.0;
        for (src) |v| ss += v * v;
        const rms_inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(head_dim)) + eps);
        for (head, src, w) |*o, v, weight| o.* = v * rms_inv * (1.0 + weight);
        if (use_table)
            rope_mod.applyRopeFreqsNeox(head, freqs, pos)
        else
            rope_mod.applyRopeNeox(head, pos, theta);
    }

    const usage: vk.VkBufferUsageFlags = @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    var x_buf = try GpuBuffer.initHostCoherent(&gpu, total * @sizeOf(f32), usage);
    defer x_buf.deinit();
    var w_buf = try GpuBuffer.initHostCoherent(&gpu, w.len * @sizeOf(f32), usage);
    defer w_buf.deinit();
    var freqs_buf = try GpuBuffer.initHostCoherent(&gpu, freqs.len * @sizeOf(f32), usage);
    defer freqs_buf.deinit();
    try x_buf.upload(std.mem.sliceAsBytes(x));
    try w_buf.upload(std.mem.sliceAsBytes(w));
    try freqs_buf.upload(std.mem.sliceAsBytes(freqs));

    var table_pl = try RmsnormPerHeadRopeTablePipeline.init(&gpu);
    defer table_pl.deinit();
    var theta_pl = try RmsnormPerHeadRopeThetaPipeline.init(&gpu);
    defer theta_pl.deinit();
    const cmd = try gpu.beginBatch();
    if (use_table) {
        const dset = try table_pl.allocSet(&x_buf, &w_buf, &freqs_buf, &x_buf);
        table_pl.recordWithSet(cmd, dset, n_heads, head_dim, eps, @intCast(pos));
    } else {
        const dset = try theta_pl.allocSet(&x_buf, &w_buf, &x_buf);
        theta_pl.recordWithSet(cmd, dset, n_heads, head_dim, eps, @intCast(pos), theta);
    }
    try gpu.submitBatch(cmd);
    try x_buf.download(std.mem.sliceAsBytes(gpu_out));

    var max_abs: f32 = 0.0;
    var max_ref: f32 = 0.0;
    for (cpu_out, gpu_out) |c, g| {
        max_abs = @max(max_abs, @abs(c - g));
        max_ref = @max(max_ref, @abs(c));
    }
    const rel = max_abs / (max_ref + 1e-6);
    std.debug.print("rmsnorm_perhead_rope fuzz table={} nh={} hd={} rel={e:.3}\n", .{ use_table, n_heads, head_dim, rel });
    try std.testing.expect(rel < 1e-5);
}

test "gpu rmsnorm_perhead_rope theta SWA shape fuzz" {
    try fuzzRmsnormPerHeadRope(false, 16, 256, 107);
}
test "gpu rmsnorm_perhead_rope table global shape fuzz" {
    try fuzzRmsnormPerHeadRope(true, 4, 512, 109);
}

// ── attn_qk_softmax + attn_av fuzz tests ──────────────────────────────────────

fn cpuAttn(
    q: []const f32,
    k: []const f32,
    v: []const f32,
    scores_out: []f32,
    out: []f32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    win_len: u32,
    scale: f32,
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
        for (sc) |*s| {
            s.* = @exp(s.* - mx);
            sum += s.*;
        }
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
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    win_len: u32,
    seed: u64,
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
    var pl_fused = try AttnFusedSmallPipeline.init(&gpu);
    defer pl_fused.deinit();

    const al = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();

    const q_len = @as(usize, n_heads) * @as(usize, head_dim);
    const kv_len = @as(usize, win_len) * @as(usize, n_kv_heads) * @as(usize, head_dim);
    const sc_len = @as(usize, n_heads) * @as(usize, win_len);
    const out_len = q_len;

    const q = try al.alloc(f32, q_len);
    defer al.free(q);
    const k = try al.alloc(f32, kv_len);
    defer al.free(k);
    const v = try al.alloc(f32, kv_len);
    defer al.free(v);
    const sc_cpu = try al.alloc(f32, sc_len);
    defer al.free(sc_cpu);
    const sc_gpu = try al.alloc(f32, sc_len);
    defer al.free(sc_gpu);
    const out_cpu = try al.alloc(f32, out_len);
    defer al.free(out_cpu);
    const out_gpu = try al.alloc(f32, out_len);
    defer al.free(out_gpu);
    const out_fused = try al.alloc(f32, out_len);
    defer al.free(out_fused);

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
    var q_buf = try GpuBuffer.initHostCoherent(&gpu, q_len * @sizeOf(f32), usage);
    defer q_buf.deinit();
    var k_buf = try GpuBuffer.initHostCoherent(&gpu, kv_len * @sizeOf(f32), usage);
    defer k_buf.deinit();
    var v_buf = try GpuBuffer.initHostCoherent(&gpu, kv_len * @sizeOf(f32), usage);
    defer v_buf.deinit();
    var sc_buf = try GpuBuffer.initHostCoherent(&gpu, sc_len * @sizeOf(f32), usage);
    defer sc_buf.deinit();
    var o_buf = try GpuBuffer.initHostCoherent(&gpu, out_len * @sizeOf(f32), usage);
    defer o_buf.deinit();
    var fused_o_buf = try GpuBuffer.initHostCoherent(&gpu, out_len * @sizeOf(f32), usage);
    defer fused_o_buf.deinit();

    try q_buf.upload(std.mem.sliceAsBytes(q));
    try k_buf.upload(std.mem.sliceAsBytes(k));
    try v_buf.upload(std.mem.sliceAsBytes(v));

    const cmd = try gpu.beginBatch();
    _ = try pl_qk.record(cmd, &q_buf, &k_buf, &sc_buf, n_heads, seq, win_len, head_dim, n_kv_heads, n_q_per_kv, cap, scale);
    GpuContext.recordShaderBarrier(cmd);
    _ = try pl_av.record(cmd, &sc_buf, &v_buf, &o_buf, n_heads, seq, win_len, head_dim, n_kv_heads, n_q_per_kv, cap);
    try gpu.submitBatch(cmd);

    try sc_buf.download(std.mem.sliceAsBytes(sc_gpu));
    try o_buf.download(std.mem.sliceAsBytes(out_gpu));

    const fused_cmd = try gpu.beginBatch();
    _ = try pl_fused.record(fused_cmd, &q_buf, &k_buf, &v_buf, &fused_o_buf, n_heads, seq, win_len, head_dim, n_kv_heads, n_q_per_kv, cap, scale);
    try gpu.submitBatch(fused_cmd);
    try fused_o_buf.download(std.mem.sliceAsBytes(out_fused));

    var sc_max: f32 = 0;
    var sc_ref: f32 = 0;
    for (sc_cpu, sc_gpu) |c, g| {
        const d = @abs(c - g);
        if (d > sc_max) sc_max = d;
        if (@abs(c) > sc_ref) sc_ref = @abs(c);
    }
    var ov_max: f32 = 0;
    var ov_ref: f32 = 0;
    for (out_cpu, out_gpu) |c, g| {
        const d = @abs(c - g);
        if (d > ov_max) ov_max = d;
        if (@abs(c) > ov_ref) ov_ref = @abs(c);
    }
    var fused_max: f32 = 0;
    var fused_ref: f32 = 0;
    for (out_cpu, out_fused) |c, g| {
        const d = @abs(c - g);
        if (d > fused_max) fused_max = d;
        if (@abs(c) > fused_ref) fused_ref = @abs(c);
    }
    const sc_rel = sc_max / (sc_ref + 1e-6);
    const ov_rel = ov_max / (ov_ref + 1e-6);
    const fused_rel = fused_max / (fused_ref + 1e-6);
    std.debug.print("attn fuzz nh={} nkv={} hd={} win={} scores rel={e:.3} out rel={e:.3} fused rel={e:.3}\n", .{ n_heads, n_kv_heads, head_dim, win_len, sc_rel, ov_rel, fused_rel });
    try std.testing.expect(sc_rel < 1e-4);
    try std.testing.expect(ov_rel < 1e-4);
    try std.testing.expect(fused_rel < 1e-4);
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
    const a = try al.alloc(f32, n);
    defer al.free(a);
    const b = try al.alloc(f32, n);
    defer al.free(b);
    const expected = try al.alloc(f32, n);
    defer al.free(expected);
    var prng = std.Random.DefaultPrng.init(81);
    const r = prng.random();
    for (a, b, expected) |*ai, *bi, *ei| {
        ai.* = (r.float(f32) - 0.5) * 10.0;
        bi.* = (r.float(f32) - 0.5) * 10.0;
        ei.* = ai.* + bi.*;
    }

    var pl = try ElemAddPipeline.init(&gpu);
    defer pl.deinit();
    var a_buf = try GpuBuffer.initHostCoherent(&gpu, n * @sizeOf(f32), @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer a_buf.deinit();
    var b_buf = try GpuBuffer.initHostCoherent(&gpu, n * @sizeOf(f32), @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer b_buf.deinit();
    try a_buf.upload(std.mem.sliceAsBytes(a));
    try b_buf.upload(std.mem.sliceAsBytes(b));

    const cmd = try gpu.beginBatch();
    _ = try pl.record(cmd, &a_buf, &b_buf, @intCast(n));
    try gpu.submitBatch(cmd);

    const got = try al.alloc(f32, n);
    defer al.free(got);
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

    const vec = try al.alloc(f32, total);
    defer al.free(vec);
    const cpu_out = try al.alloc(f32, total);
    defer al.free(cpu_out);
    var prng = std.Random.DefaultPrng.init(91);
    const r = prng.random();
    for (vec) |*v| v.* = (r.float(f32) - 0.5) * 4.0;
    @memcpy(cpu_out, vec);
    for (0..n_heads) |h|
        rope_mod.applyRopeNeox(cpu_out[h * head_dim ..][0..head_dim], pos, theta);

    var pl = try RopeNeoxThetaPipeline.init(&gpu);
    defer pl.deinit();
    var vec_buf = try GpuBuffer.initHostCoherent(&gpu, total * @sizeOf(f32), @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer vec_buf.deinit();
    try vec_buf.upload(std.mem.sliceAsBytes(vec));
    const cmd = try gpu.beginBatch();
    _ = try pl.record(cmd, &vec_buf, @intCast(pos), @intCast(head_dim), theta, @intCast(n_heads));
    try gpu.submitBatch(cmd);

    const gpu_out = try al.alloc(f32, total);
    defer al.free(gpu_out);
    try vec_buf.download(std.mem.sliceAsBytes(gpu_out));
    var max_abs: f32 = 0.0;
    var max_ref: f32 = 0.0;
    for (cpu_out, gpu_out) |c, g| {
        const d = @abs(c - g);
        if (d > max_abs) max_abs = d;
        if (@abs(c) > max_ref) max_ref = @abs(c);
    }
    const rel = max_abs / (max_ref + 1e-6);
    std.debug.print("rope_neox_theta fuzz n_heads={} hd={} pos={} max|D|={d:.6} rel={e:.3}\n", .{ n_heads, head_dim, pos, max_abs, rel });
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

    const vec = try al.alloc(f32, total);
    defer al.free(vec);
    const cpu_out = try al.alloc(f32, total);
    defer al.free(cpu_out);
    const freqs = try al.alloc(f32, head_dim / 2);
    defer al.free(freqs);
    var prng = std.Random.DefaultPrng.init(97);
    const r = prng.random();
    for (vec) |*v| v.* = (r.float(f32) - 0.5) * 4.0;
    // Generate plausible inverse-frequency values (≈ 1/theta^(2i/dim) shape).
    for (freqs, 0..) |*f, i|
        f.* = 1.0 / std.math.pow(f32, 10000.0, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(head_dim)));

    @memcpy(cpu_out, vec);
    for (0..n_heads) |h|
        rope_mod.applyRopeFreqsNeox(cpu_out[h * head_dim ..][0..head_dim], freqs, pos);

    var pl = try RopeNeoxTablePipeline.init(&gpu);
    defer pl.deinit();
    var vec_buf = try GpuBuffer.initHostCoherent(&gpu, total * @sizeOf(f32), @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer vec_buf.deinit();
    try vec_buf.upload(std.mem.sliceAsBytes(vec));
    var freqs_buf = try GpuBuffer.initHostCoherent(&gpu, freqs.len * @sizeOf(f32), @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer freqs_buf.deinit();
    try freqs_buf.upload(std.mem.sliceAsBytes(freqs));

    const cmd = try gpu.beginBatch();
    _ = try pl.record(cmd, &vec_buf, &freqs_buf, @intCast(pos), @intCast(head_dim), @intCast(n_heads));
    try gpu.submitBatch(cmd);

    const gpu_out = try al.alloc(f32, total);
    defer al.free(gpu_out);
    try vec_buf.download(std.mem.sliceAsBytes(gpu_out));
    var max_abs: f32 = 0.0;
    var max_ref: f32 = 0.0;
    for (cpu_out, gpu_out) |c, g| {
        const d = @abs(c - g);
        if (d > max_abs) max_abs = d;
        if (@abs(c) > max_ref) max_ref = @abs(c);
    }
    const rel = max_abs / (max_ref + 1e-6);
    std.debug.print("rope_neox_table fuzz n_heads={} hd={} pos={} max|D|={d:.6} rel={e:.3}\n", .{ n_heads, head_dim, pos, max_abs, rel });
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
    const n: usize = 2112; // dense FFN dim in Gemma4
    const a = try al.alloc(f32, n);
    defer al.free(a);
    const b = try al.alloc(f32, n);
    defer al.free(b);
    const expected = try al.alloc(f32, n);
    defer al.free(expected);
    var prng = std.Random.DefaultPrng.init(84);
    const r = prng.random();
    for (a, b, expected) |*ai, *bi, *ei| {
        ai.* = (r.float(f32) - 0.5) * 10.0;
        bi.* = (r.float(f32) - 0.5) * 10.0;
        ei.* = math_ref.gelu(ai.*) * bi.*;
    }

    var pl = try GeluMulPipeline.init(&gpu);
    defer pl.deinit();
    var a_buf = try GpuBuffer.initHostCoherent(&gpu, n * @sizeOf(f32), @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer a_buf.deinit();
    var b_buf = try GpuBuffer.initHostCoherent(&gpu, n * @sizeOf(f32), @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer b_buf.deinit();
    try a_buf.upload(std.mem.sliceAsBytes(a));
    try b_buf.upload(std.mem.sliceAsBytes(b));

    const cmd = try gpu.beginBatch();
    _ = try pl.record(cmd, &a_buf, &b_buf, @intCast(n));
    try gpu.submitBatch(cmd);

    const got = try al.alloc(f32, n);
    defer al.free(got);
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
    const x = try al.alloc(f32, n);
    defer al.free(x);
    const s: f32 = 1.4142136; // sqrt(2)
    var prng = std.Random.DefaultPrng.init(83);
    const r = prng.random();
    for (x) |*xi| xi.* = (r.float(f32) - 0.5) * 10.0;

    var pl = try ElemScalePipeline.init(&gpu);
    defer pl.deinit();
    var x_buf = try GpuBuffer.initHostCoherent(&gpu, n * @sizeOf(f32), @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer x_buf.deinit();
    try x_buf.upload(std.mem.sliceAsBytes(x));

    const cmd = try gpu.beginBatch();
    _ = try pl.record(cmd, &x_buf, @intCast(n), s);
    try gpu.submitBatch(cmd);

    const got = try al.alloc(f32, n);
    defer al.free(got);
    try x_buf.download(std.mem.sliceAsBytes(got));
    for (got, x) |g, xi| try std.testing.expectApproxEqAbs(xi * s, g, 1e-6);
}

test "gpu logit_softcap fuzz" {
    const ctx = GpuContext.init() catch |e| {
        std.debug.print("gpu init failed: {}\n", .{e});
        return;
    };
    var gpu = ctx;
    defer gpu.deinit();

    const al = std.testing.allocator;
    const n: usize = 262144;
    const cap: f32 = 30.0;
    const x = try al.alloc(f32, n);
    defer al.free(x);
    var prng = std.Random.DefaultPrng.init(101);
    const r = prng.random();
    for (x) |*xi| xi.* = (r.float(f32) - 0.5) * 160.0;

    var pl = try LogitSoftcapPipeline.init(&gpu);
    defer pl.deinit();
    var x_buf = try GpuBuffer.initHostCoherent(&gpu, n * @sizeOf(f32), @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    defer x_buf.deinit();
    try x_buf.upload(std.mem.sliceAsBytes(x));

    const cmd = try gpu.beginBatch();
    _ = try pl.record(cmd, &x_buf, @intCast(n), cap);
    try gpu.submitBatch(cmd);

    const got = try al.alloc(f32, n);
    defer al.free(got);
    try x_buf.download(std.mem.sliceAsBytes(got));

    var max_abs: f32 = 0.0;
    for (got, x) |g, xi| {
        const expected = std.math.tanh(xi / cap) * cap;
        const d = @abs(expected - g);
        if (d > max_abs) max_abs = d;
    }
    std.debug.print("logit_softcap fuzz n={} max|D|={d:.6}\n", .{ n, max_abs });
    try std.testing.expect(max_abs < 2e-5);
}
