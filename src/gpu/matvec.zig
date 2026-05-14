// GPU fp32 matrix-vector multiply.
// Wraps pipeline creation, descriptor management, and dispatch into a simple API.
const std = @import("std");
const vk = @import("vk.zig").vk;
const GpuContext = @import("context.zig").GpuContext;
const GpuBuffer = @import("buffer.zig").GpuBuffer;
const shaders = @import("gpu_shaders");

const WORKGROUP_SIZE: u32 = 64;

// Push-constant layout must match the GLSL shader.
const PushConst = extern struct { rows: u32, cols: u32 };

pub const MatvecPipeline = struct {
    pipeline: vk.VkPipeline,
    layout: vk.VkPipelineLayout,
    dset_layout: vk.VkDescriptorSetLayout,
    desc_pool: vk.VkDescriptorPool,
    device: vk.VkDevice,

    pub fn init(ctx: *const GpuContext) !MatvecPipeline {
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

        // Compile embedded SPIR-V into a shader module.
        // VkShaderModuleCreateInfo.pCode requires 4-byte alignment.
        // Allocate []u32 (naturally aligned to 4 bytes) and copy the bytes in.
        const spv_bytes: []const u8 = shaders.matvec_f32;
        const spv_words = try std.heap.page_allocator.alloc(u32, spv_bytes.len / 4);
        defer std.heap.page_allocator.free(spv_words);
        @memcpy(std.mem.sliceAsBytes(spv_words), spv_bytes);

        const shader_ci = vk.VkShaderModuleCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .codeSize = spv_bytes.len,
            .pCode = spv_words.ptr,
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
            .basePipelineHandle = null, // no derivative pipeline
            .basePipelineIndex = -1,
        };
        var pipeline: vk.VkPipeline = null;
        // null pipeline cache: no caching for now
        if (vk.vkCreateComputePipelines(dev, null, 1, &pipeline_ci, null, &pipeline) != vk.VK_SUCCESS)
            return error.VkComputePipelineFailed;
        errdefer vk.vkDestroyPipeline(dev, pipeline, null);

        // Descriptor pool: capacity for a few concurrent sets
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

    // Compute out = mat * vec_in where mat is (rows x cols) row-major f32.
    // All buffers must be host-coherent GpuBuffers.
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

        // Allocate and update a descriptor set for this call
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

        // Record and submit a one-shot command buffer
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

        // One workgroup per 64 rows
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
        // null fence: block via vkQueueWaitIdle instead
        if (vk.vkQueueSubmit(ctx.queue, 1, &submit, null) != vk.VK_SUCCESS)
            return error.VkQueueSubmitFailed;
        _ = vk.vkQueueWaitIdle(ctx.queue);
    }
};

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

// Convenience: allocate host-coherent buffers, upload data, run, download result.
// mat is row-major f32[rows * cols], vec is f32[cols], out receives f32[rows].
pub fn matvecF32(
    ctx: *const GpuContext,
    pipeline: *const MatvecPipeline,
    mat: []const f32,
    vec: []const f32,
    out: []f32,
    rows: u32,
    cols: u32,
    allocator: std.mem.Allocator,
) !void {
    _ = allocator;
    var mat_buf = try GpuBuffer.init(ctx, mat.len * @sizeOf(f32),
        vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer mat_buf.deinit();
    var vec_buf = try GpuBuffer.init(ctx, vec.len * @sizeOf(f32),
        vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer vec_buf.deinit();
    var out_buf = try GpuBuffer.init(ctx, out.len * @sizeOf(f32),
        vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer out_buf.deinit();

    try mat_buf.upload(std.mem.sliceAsBytes(mat));
    try vec_buf.upload(std.mem.sliceAsBytes(vec));
    try pipeline.run(ctx, &mat_buf, &vec_buf, &out_buf, rows, cols);
    try out_buf.download(std.mem.sliceAsBytes(out));
}

test "gpu matvec f32 correctness" {
    const ctx = GpuContext.init() catch |e| {
        std.debug.print("gpu init failed: {}\n", .{e});
        return; // skip if no GPU
    };
    var gpu = ctx;
    defer gpu.deinit();

    var pipeline = try MatvecPipeline.init(&gpu);
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

    try matvecF32(&gpu, &pipeline, &mat, &vec, &out, rows, cols, std.testing.allocator);

    try std.testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
    try std.testing.expectApproxEqAbs(out[1], 2.0, 1e-5);
    try std.testing.expectApproxEqAbs(out[2], 3.0, 1e-5);
    try std.testing.expectApproxEqAbs(out[3], 4.0, 1e-5);
}
