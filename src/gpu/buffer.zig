const std = @import("std");
const vk = @import("vk.zig").vk;
const GpuContext = @import("context.zig").GpuContext;

// A single Vulkan buffer backed by host-coherent device memory.
// Simple for the first pass — no staging, no device-local optimization.
pub const GpuBuffer = struct {
    handle: vk.VkBuffer,
    memory: vk.VkDeviceMemory,
    size: vk.VkDeviceSize,
    device: vk.VkDevice,

    pub fn init(ctx: *const GpuContext, size: usize, usage: vk.VkBufferUsageFlags) !GpuBuffer {
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

        // Host-coherent so CPU writes are immediately visible to GPU (no explicit flush)
        const host_props = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        const mem_type = try ctx.findMemoryType(req.memoryTypeBits, @intCast(host_props));

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

    pub fn upload(self: *const GpuBuffer, data: []const u8) !void {
        std.debug.assert(data.len <= self.size);
        var ptr: ?*anyopaque = null;
        if (vk.vkMapMemory(self.device, self.memory, 0, vk.VK_WHOLE_SIZE, 0, &ptr) != vk.VK_SUCCESS)
            return error.VkMapFailed;
        defer vk.vkUnmapMemory(self.device, self.memory);
        const dst: [*]u8 = @ptrCast(ptr.?);
        @memcpy(dst[0..data.len], data);
    }

    pub fn download(self: *const GpuBuffer, dest: []u8) !void {
        std.debug.assert(dest.len <= self.size);
        var ptr: ?*anyopaque = null;
        if (vk.vkMapMemory(self.device, self.memory, 0, vk.VK_WHOLE_SIZE, 0, &ptr) != vk.VK_SUCCESS)
            return error.VkMapFailed;
        defer vk.vkUnmapMemory(self.device, self.memory);
        const src: [*]const u8 = @ptrCast(ptr.?);
        @memcpy(dest, src[0..dest.len]);
    }
};
