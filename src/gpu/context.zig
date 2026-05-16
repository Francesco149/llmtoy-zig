const std = @import("std");
const vk = @import("vk.zig").vk;

pub const GpuContext = struct {
    instance: vk.VkInstance,
    phys_dev: vk.VkPhysicalDevice,
    device: vk.VkDevice,
    queue: vk.VkQueue,
    queue_family: u32,
    cmd_pool: vk.VkCommandPool,

    pub fn init() !GpuContext {
        // We require Vulkan 1.3 for shaderIntegerDotProduct (core in 1.3) plus
        // shaderInt8 / shaderFloat16 / 8-bit + 16-bit storage (core in 1.2/1.1).
        // These are the foundation of the Q8_1-activation matvec path.
        const app_info = vk.VkApplicationInfo{
            .sType = vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pNext = null,
            .pApplicationName = "llmtoy",
            .applicationVersion = vk.VK_MAKE_VERSION(0, 7, 0),
            .pEngineName = "llmtoy",
            .engineVersion = vk.VK_MAKE_VERSION(0, 7, 0),
            .apiVersion = vk.VK_API_VERSION_1_3,
        };
        const inst_ci = vk.VkInstanceCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .pApplicationInfo = &app_info,
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = null,
            .enabledExtensionCount = 0,
            .ppEnabledExtensionNames = null,
        };
        var instance: vk.VkInstance = undefined;
        if (vk.vkCreateInstance(&inst_ci, null, &instance) != vk.VK_SUCCESS)
            return error.VkInstanceCreateFailed;
        errdefer vk.vkDestroyInstance(instance, null);

        // Pick physical device with a compute queue
        var dev_count: u32 = 0;
        _ = vk.vkEnumeratePhysicalDevices(instance, &dev_count, null);
        if (dev_count == 0) return error.NoVulkanDevice;

        var phys_devs: [8]vk.VkPhysicalDevice = undefined;
        dev_count = @min(dev_count, 8);
        _ = vk.vkEnumeratePhysicalDevices(instance, &dev_count, &phys_devs);

        var phys_dev: vk.VkPhysicalDevice = undefined;
        var queue_family: u32 = std.math.maxInt(u32);

        outer: for (phys_devs[0..dev_count]) |pdev| {
            var qfam_count: u32 = 0;
            vk.vkGetPhysicalDeviceQueueFamilyProperties(pdev, &qfam_count, null);
            var qfams: [16]vk.VkQueueFamilyProperties = undefined;
            qfam_count = @min(qfam_count, 16);
            vk.vkGetPhysicalDeviceQueueFamilyProperties(pdev, &qfam_count, &qfams);
            for (qfams[0..qfam_count], 0..) |qf, i| {
                if (qf.queueFlags & vk.VK_QUEUE_COMPUTE_BIT != 0) {
                    phys_dev = pdev;
                    queue_family = @intCast(i);
                    break :outer;
                }
            }
        }
        if (queue_family == std.math.maxInt(u32)) return error.NoComputeQueue;

        // Logical device
        const priority: f32 = 1.0;
        const queue_ci = vk.VkDeviceQueueCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .queueFamilyIndex = queue_family,
            .queueCount = 1,
            .pQueuePriorities = &priority,
        };

        // Feature chain: V13 → V12 → V11 → Features2.  Everything we need for the
        // Q8_1 + integer-dot matvec path lives in core Vulkan 1.3.  RX 7800 XT
        // (RADV) reports all of these as true; an unsupported device will fail
        // device creation with a clear error.
        var v13 = std.mem.zeroes(vk.VkPhysicalDeviceVulkan13Features);
        v13.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        v13.shaderIntegerDotProduct = vk.VK_TRUE;        // dotPacked4x8EXT

        var v12 = std.mem.zeroes(vk.VkPhysicalDeviceVulkan12Features);
        v12.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        v12.pNext = &v13;
        v12.shaderFloat16 = vk.VK_TRUE;                  // f16vec2 ds, dm
        v12.shaderInt8 = vk.VK_TRUE;                     // int8_t arithmetic
        v12.storageBuffer8BitAccess = vk.VK_TRUE;        // int8_t qs[32] loads
        v12.shaderSubgroupExtendedTypes = vk.VK_TRUE;    // subgroupAdd(int32_t) etc.
        v12.uniformAndStorageBuffer8BitAccess = vk.VK_TRUE;

        var v11 = std.mem.zeroes(vk.VkPhysicalDeviceVulkan11Features);
        v11.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        v11.pNext = &v12;
        v11.storageBuffer16BitAccess = vk.VK_TRUE;       // f16vec2 / int16 loads
        v11.uniformAndStorageBuffer16BitAccess = vk.VK_TRUE;

        var feats2 = std.mem.zeroes(vk.VkPhysicalDeviceFeatures2);
        feats2.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        feats2.pNext = &v11;
        feats2.features.shaderInt16 = vk.VK_TRUE;        // i16vec2 unpack in Q5_0/Q5_1
        feats2.features.shaderInt64 = vk.VK_TRUE;

        const dev_ci = vk.VkDeviceCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext = &feats2,
            .flags = 0,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queue_ci,
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = null,
            .enabledExtensionCount = 0,
            .ppEnabledExtensionNames = null,
            .pEnabledFeatures = null,                    // using Features2 in pNext instead
        };
        var device: vk.VkDevice = undefined;
        if (vk.vkCreateDevice(phys_dev, &dev_ci, null, &device) != vk.VK_SUCCESS)
            return error.VkDeviceCreateFailed;
        errdefer vk.vkDestroyDevice(device, null);

        var queue: vk.VkQueue = undefined;
        vk.vkGetDeviceQueue(device, queue_family, 0, &queue);

        // Command pool (reset-able)
        const pool_ci = vk.VkCommandPoolCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext = null,
            .flags = vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = queue_family,
        };
        var cmd_pool: vk.VkCommandPool = null;
        if (vk.vkCreateCommandPool(device, &pool_ci, null, &cmd_pool) != vk.VK_SUCCESS)
            return error.VkCommandPoolCreateFailed;

        return .{
            .instance = instance,
            .phys_dev = phys_dev,
            .device = device,
            .queue = queue,
            .queue_family = queue_family,
            .cmd_pool = cmd_pool,
        };
    }

    pub fn deinit(self: *GpuContext) void {
        vk.vkDestroyCommandPool(self.device, self.cmd_pool, null);
        vk.vkDestroyDevice(self.device, null);
        vk.vkDestroyInstance(self.instance, null);
    }

    pub fn deviceName(self: *const GpuContext) [vk.VK_MAX_PHYSICAL_DEVICE_NAME_SIZE]u8 {
        var props: vk.VkPhysicalDeviceProperties = undefined;
        vk.vkGetPhysicalDeviceProperties(self.phys_dev, &props);
        return props.deviceName;
    }

    pub fn findMemoryType(self: *const GpuContext, type_bits: u32, props: vk.VkMemoryPropertyFlags) !u32 {
        var mem_props: vk.VkPhysicalDeviceMemoryProperties = undefined;
        vk.vkGetPhysicalDeviceMemoryProperties(self.phys_dev, &mem_props);
        for (0..mem_props.memoryTypeCount) |i| {
            if (type_bits & (@as(u32, 1) << @intCast(i)) != 0 and
                mem_props.memoryTypes[i].propertyFlags & props == props)
                return @intCast(i);
        }
        return error.NoSuitableMemoryType;
    }

    // Open a command buffer for recording any mix of GPU commands (copies, dispatches).
    // Call recordCopy() or MatvecPipeline.record() for each command, then submitBatch() once.
    // One submission = one GPU power-state wakeup.
    pub fn beginBatch(self: *const GpuContext) !vk.VkCommandBuffer {
        return self.beginBatchCopy();
    }

    pub fn submitBatch(self: *const GpuContext, cmd: vk.VkCommandBuffer) !void {
        return self.submitBatchCopy(cmd);
    }

    // Open a command buffer for recording multiple buffer copies.
    // Call recordCopy() for each pair, then submitBatchCopy() once.
    // One submission = one GPU power-state wakeup; much faster than copyBuffer per matrix.
    pub fn beginBatchCopy(self: *const GpuContext) !vk.VkCommandBuffer {
        const ci = vk.VkCommandBufferAllocateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = null,
            .commandPool = self.cmd_pool,
            .level = vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        var cmd: vk.VkCommandBuffer = undefined;
        if (vk.vkAllocateCommandBuffers(self.device, &ci, &cmd) != vk.VK_SUCCESS)
            return error.VkCommandBufferAllocFailed;
        const begin_ci = vk.VkCommandBufferBeginInfo{
            .sType = vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = null,
            .flags = vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = null,
        };
        _ = vk.vkBeginCommandBuffer(cmd, &begin_ci);
        return cmd;
    }

    pub fn recordCopy(cmd: vk.VkCommandBuffer, src: vk.VkBuffer, dst: vk.VkBuffer, size: vk.VkDeviceSize) void {
        const region = vk.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = size };
        vk.vkCmdCopyBuffer(cmd, src, dst, 1, &region);
    }

    // Compute → compute pipeline barrier.
    // Ensures all shader writes before this point are visible to shader reads after it.
    // Use between fused gate-gelu-up dispatches and the down matmul dispatches.
    pub fn recordShaderBarrier(cmd: vk.VkCommandBuffer) void {
        const barrier = vk.VkMemoryBarrier{
            .sType = vk.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = null,
            .srcAccessMask = vk.VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = vk.VK_ACCESS_SHADER_READ_BIT,
        };
        vk.vkCmdPipelineBarrier(
            cmd,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &barrier, 0, null, 0, null,
        );
    }

    // End recording, submit all recorded copies, wait for completion, free command buffer.
    pub fn submitBatchCopy(self: *const GpuContext, cmd: vk.VkCommandBuffer) !void {
        _ = vk.vkEndCommandBuffer(cmd);
        defer vk.vkFreeCommandBuffers(self.device, self.cmd_pool, 1, &cmd);
        const submit = vk.VkSubmitInfo{
            .sType = vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = null,
            .waitSemaphoreCount = 0, .pWaitSemaphores = null, .pWaitDstStageMask = null,
            .commandBufferCount = 1, .pCommandBuffers = &cmd,
            .signalSemaphoreCount = 0, .pSignalSemaphores = null,
        };
        if (vk.vkQueueSubmit(self.queue, 1, &submit, null) != vk.VK_SUCCESS)
            return error.VkQueueSubmitFailed;
        _ = vk.vkQueueWaitIdle(self.queue);
    }

    // Copy `size` bytes from src[src_offset..] to dst[dst_offset..] in a
    // one-shot command buffer. Blocks until complete. Use this when the source
    // or destination is a region inside a larger buffer (e.g. a per-position
    // slot in a KV cache buffer).
    pub fn copyBufferRegion(
        self: *const GpuContext,
        src: vk.VkBuffer, dst: vk.VkBuffer,
        src_offset: vk.VkDeviceSize, dst_offset: vk.VkDeviceSize,
        size: vk.VkDeviceSize,
    ) !void {
        const alloc_ci = vk.VkCommandBufferAllocateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = null,
            .commandPool = self.cmd_pool,
            .level = vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        var cmd: vk.VkCommandBuffer = undefined;
        if (vk.vkAllocateCommandBuffers(self.device, &alloc_ci, &cmd) != vk.VK_SUCCESS)
            return error.VkCommandBufferAllocFailed;
        defer vk.vkFreeCommandBuffers(self.device, self.cmd_pool, 1, &cmd);

        const begin_ci = vk.VkCommandBufferBeginInfo{
            .sType = vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = null,
            .flags = vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = null,
        };
        _ = vk.vkBeginCommandBuffer(cmd, &begin_ci);
        const region = vk.VkBufferCopy{
            .srcOffset = src_offset, .dstOffset = dst_offset, .size = size };
        vk.vkCmdCopyBuffer(cmd, src, dst, 1, &region);
        _ = vk.vkEndCommandBuffer(cmd);

        const submit = vk.VkSubmitInfo{
            .sType = vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = null,
            .waitSemaphoreCount = 0, .pWaitSemaphores = null, .pWaitDstStageMask = null,
            .commandBufferCount = 1, .pCommandBuffers = &cmd,
            .signalSemaphoreCount = 0, .pSignalSemaphores = null,
        };
        if (vk.vkQueueSubmit(self.queue, 1, &submit, null) != vk.VK_SUCCESS)
            return error.VkQueueSubmitFailed;
        _ = vk.vkQueueWaitIdle(self.queue);
    }

    // Copy `size` bytes from src to dst using a one-shot command buffer.
    // Blocks until the transfer completes. Used for staging uploads/downloads.
    pub fn copyBuffer(self: *const GpuContext, src: vk.VkBuffer, dst: vk.VkBuffer, size: vk.VkDeviceSize) !void {
        const alloc_ci = vk.VkCommandBufferAllocateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = null,
            .commandPool = self.cmd_pool,
            .level = vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        var cmd: vk.VkCommandBuffer = undefined;
        if (vk.vkAllocateCommandBuffers(self.device, &alloc_ci, &cmd) != vk.VK_SUCCESS)
            return error.VkCommandBufferAllocFailed;
        defer vk.vkFreeCommandBuffers(self.device, self.cmd_pool, 1, &cmd);

        const begin_ci = vk.VkCommandBufferBeginInfo{
            .sType = vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = null,
            .flags = vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = null,
        };
        _ = vk.vkBeginCommandBuffer(cmd, &begin_ci);
        const region = vk.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = size };
        vk.vkCmdCopyBuffer(cmd, src, dst, 1, &region);
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
        if (vk.vkQueueSubmit(self.queue, 1, &submit, null) != vk.VK_SUCCESS)
            return error.VkQueueSubmitFailed;
        _ = vk.vkQueueWaitIdle(self.queue);
    }
};
