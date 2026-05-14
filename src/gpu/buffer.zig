const std = @import("std");
const vk = @import("vk.zig").vk;
const GpuContext = @import("context.zig").GpuContext;

pub const GpuBuffer = struct {
    handle: vk.VkBuffer,
    memory: vk.VkDeviceMemory,
    size: vk.VkDeviceSize,
    device: vk.VkDevice,

    // Host-coherent: CPU writes are immediately visible to the GPU.
    // Used for per-token activations and for staging uploads/downloads.
    // Simple but limited by PCIe bandwidth (≈ 64 GB/s peak for PCIe 4.0 x16).
    pub fn initHostCoherent(ctx: *const GpuContext, size: usize, usage: vk.VkBufferUsageFlags) !GpuBuffer {
        const props: vk.VkMemoryPropertyFlags = @intCast(
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        return initMem(ctx, size, usage, props);
    }

    // Device-local: lives in GPU VRAM; fastest for compute (RX 7800 XT ≈ 432 GB/s).
    // Cannot be mapped by the CPU. Requires a staging buffer for uploads/downloads.
    // Use for weight matrices that are uploaded once and read many times per token.
    pub fn initDeviceLocal(ctx: *const GpuContext, size: usize, usage: vk.VkBufferUsageFlags) !GpuBuffer {
        const props: vk.VkMemoryPropertyFlags = @intCast(vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        return initMem(ctx, size,
            usage | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            props);
    }

    // Staging buffer: host-coherent, used only for transfer source/dest.
    pub fn initStaging(ctx: *const GpuContext, size: usize) !GpuBuffer {
        const props: vk.VkMemoryPropertyFlags = @intCast(
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        const usage: vk.VkBufferUsageFlags = @intCast(
            vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        return initMem(ctx, size, usage, props);
    }

    fn initMem(ctx: *const GpuContext, size: usize, usage: vk.VkBufferUsageFlags, props: vk.VkMemoryPropertyFlags) !GpuBuffer {
        const buf_ci = vk.VkBufferCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .size = size,
            .usage = usage,
            .sharingMode = vk.VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = null,
        };
        var buf: vk.VkBuffer = null;
        if (vk.vkCreateBuffer(ctx.device, &buf_ci, null, &buf) != vk.VK_SUCCESS)
            return error.VkBufferCreateFailed;
        errdefer vk.vkDestroyBuffer(ctx.device, buf, null);

        var req: vk.VkMemoryRequirements = undefined;
        vk.vkGetBufferMemoryRequirements(ctx.device, buf, &req);
        const mem_type = try ctx.findMemoryType(req.memoryTypeBits, props);

        const alloc_ci = vk.VkMemoryAllocateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = null,
            .allocationSize = req.size,
            .memoryTypeIndex = mem_type,
        };
        var memory: vk.VkDeviceMemory = null;
        if (vk.vkAllocateMemory(ctx.device, &alloc_ci, null, &memory) != vk.VK_SUCCESS)
            return error.VkAllocateMemoryFailed;
        errdefer vk.vkFreeMemory(ctx.device, memory, null);

        if (vk.vkBindBufferMemory(ctx.device, buf, memory, 0) != vk.VK_SUCCESS)
            return error.VkBindBufferMemoryFailed;

        return .{ .handle = buf, .memory = memory, .size = size, .device = ctx.device };
    }

    pub fn deinit(self: *GpuBuffer) void {
        vk.vkDestroyBuffer(self.device, self.handle, null);
        vk.vkFreeMemory(self.device, self.memory, null);
    }

    // Direct CPU→GPU write. Only valid for host-coherent buffers.
    pub fn upload(self: *const GpuBuffer, data: []const u8) !void {
        std.debug.assert(data.len <= self.size);
        var ptr: ?*anyopaque = null;
        if (vk.vkMapMemory(self.device, self.memory, 0, vk.VK_WHOLE_SIZE, 0, &ptr) != vk.VK_SUCCESS)
            return error.VkMapFailed;
        defer vk.vkUnmapMemory(self.device, self.memory);
        const dst: [*]u8 = @ptrCast(ptr.?);
        @memcpy(dst[0..data.len], data);
    }

    // Direct GPU→CPU read. Only valid for host-coherent buffers.
    pub fn download(self: *const GpuBuffer, dest: []u8) !void {
        std.debug.assert(dest.len <= self.size);
        var ptr: ?*anyopaque = null;
        if (vk.vkMapMemory(self.device, self.memory, 0, vk.VK_WHOLE_SIZE, 0, &ptr) != vk.VK_SUCCESS)
            return error.VkMapFailed;
        defer vk.vkUnmapMemory(self.device, self.memory);
        const src: [*]const u8 = @ptrCast(ptr.?);
        @memcpy(dest, src[0..dest.len]);
    }

    // Map host-coherent buffer for direct CPU read/write. Call unmap() when done.
    // Safe to read immediately after vkQueueWaitIdle; writes visible to GPU after unmap.
    pub fn mapSlice(self: *const GpuBuffer, comptime T: type, count: usize) ![]T {
        std.debug.assert(count * @sizeOf(T) <= self.size);
        var ptr: ?*anyopaque = null;
        if (vk.vkMapMemory(self.device, self.memory, 0, count * @sizeOf(T), 0, &ptr) != vk.VK_SUCCESS)
            return error.VkMapFailed;
        const typed: [*]T = @alignCast(@ptrCast(ptr.?));
        return typed[0..count];
    }

    pub fn unmap(self: *const GpuBuffer) void {
        vk.vkUnmapMemory(self.device, self.memory);
    }
};
