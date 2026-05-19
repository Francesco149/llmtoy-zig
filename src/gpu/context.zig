const std = @import("std");
const vk = @import("vk.zig").vk;

const max_profile_events = 4096;
const max_profile_labels = 256;
const profile_label_len = 72;

const ProfileEvent = struct {
    label: [profile_label_len]u8 = [_]u8{0} ** profile_label_len,
    label_len: u8 = 0,
    start_query: u32 = 0,
    end_query: u32 = 0,
};

const ProfileAggregate = struct {
    label: [profile_label_len]u8 = [_]u8{0} ** profile_label_len,
    label_len: u8 = 0,
    count: u64 = 0,
    total_ns: u64 = 0,
    min_ns: u64 = std.math.maxInt(u64),
    max_ns: u64 = 0,
};

pub const ProfileStats = struct {
    count: u64 = 0,
    total_ns: u64 = 0,
    min_ns: u64 = 0,
    max_ns: u64 = 0,
};

pub const GpuProfiler = struct {
    query_pool: vk.VkQueryPool,
    timestamp_period: f32,
    events: [max_profile_events]ProfileEvent = undefined,
    event_count: u32 = 0,
    query_count: u32 = 0,
    timestamps: [max_profile_events * 2]u64 = undefined,
    aggregates: [max_profile_labels]ProfileAggregate = undefined,
    aggregate_count: u32 = 0,
    dropped_events: u64 = 0,

    pub fn init(device: vk.VkDevice, timestamp_period: f32) !GpuProfiler {
        const ci = vk.VkQueryPoolCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .queryType = vk.VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = max_profile_events * 2,
            .pipelineStatistics = 0,
        };
        var pool: vk.VkQueryPool = null;
        if (vk.vkCreateQueryPool(device, &ci, null, &pool) != vk.VK_SUCCESS)
            return error.VkQueryPoolCreateFailed;

        return .{
            .query_pool = pool,
            .timestamp_period = timestamp_period,
        };
    }

    pub fn deinit(self: *GpuProfiler, device: vk.VkDevice) void {
        vk.vkDestroyQueryPool(device, self.query_pool, null);
    }

    fn resetBatch(self: *GpuProfiler, cmd: vk.VkCommandBuffer) void {
        self.event_count = 0;
        self.query_count = 0;
        vk.vkCmdResetQueryPool(cmd, self.query_pool, 0, max_profile_events * 2);
    }

    fn begin(self: *GpuProfiler, cmd: vk.VkCommandBuffer, label: []const u8) u32 {
        if (self.event_count >= max_profile_events or self.query_count + 2 > max_profile_events * 2) {
            self.dropped_events += 1;
            return std.math.maxInt(u32);
        }

        const event_id = self.event_count;
        self.event_count += 1;
        const start_query = self.query_count;
        self.query_count += 1;

        var ev = &self.events[event_id];
        const n = @min(label.len, profile_label_len);
        @memset(&ev.label, 0);
        @memcpy(ev.label[0..n], label[0..n]);
        ev.label_len = @intCast(n);
        ev.start_query = start_query;
        ev.end_query = std.math.maxInt(u32);

        vk.vkCmdWriteTimestamp(cmd, vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, self.query_pool, start_query);
        return event_id;
    }

    fn end(self: *GpuProfiler, cmd: vk.VkCommandBuffer, event_id: u32) void {
        if (event_id == std.math.maxInt(u32) or event_id >= self.event_count) return;
        if (self.query_count >= max_profile_events * 2) {
            self.dropped_events += 1;
            return;
        }
        const end_query = self.query_count;
        self.query_count += 1;
        self.events[event_id].end_query = end_query;
        vk.vkCmdWriteTimestamp(cmd, vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, self.query_pool, end_query);
    }

    fn collectBatch(self: *GpuProfiler, device: vk.VkDevice) void {
        if (self.query_count == 0) return;
        const rc = vk.vkGetQueryPoolResults(
            device,
            self.query_pool,
            0,
            self.query_count,
            self.query_count * @sizeOf(u64),
            @ptrCast(&self.timestamps),
            @sizeOf(u64),
            vk.VK_QUERY_RESULT_64_BIT | vk.VK_QUERY_RESULT_WAIT_BIT,
        );
        if (rc != vk.VK_SUCCESS) {
            self.dropped_events += self.event_count;
            return;
        }

        for (self.events[0..self.event_count]) |ev| {
            if (ev.end_query == std.math.maxInt(u32)) continue;
            const start = self.timestamps[ev.start_query];
            const end_ts = self.timestamps[ev.end_query];
            if (end_ts < start) continue;
            const ns_f = @as(f64, @floatFromInt(end_ts - start)) *
                @as(f64, @floatCast(self.timestamp_period));
            const ns: u64 = @intFromFloat(ns_f);
            self.addAggregate(ev.label[0..ev.label_len], ns);
        }
    }

    fn addAggregate(self: *GpuProfiler, label: []const u8, ns: u64) void {
        for (self.aggregates[0..self.aggregate_count]) |*agg| {
            if (agg.label_len == label.len and
                std.mem.eql(u8, agg.label[0..agg.label_len], label))
            {
                agg.count += 1;
                agg.total_ns += ns;
                agg.min_ns = @min(agg.min_ns, ns);
                agg.max_ns = @max(agg.max_ns, ns);
                return;
            }
        }

        if (self.aggregate_count >= max_profile_labels) {
            self.dropped_events += 1;
            return;
        }

        const idx = self.aggregate_count;
        self.aggregate_count += 1;
        var agg = &self.aggregates[idx];
        @memset(&agg.label, 0);
        @memcpy(agg.label[0..label.len], label);
        agg.label_len = @intCast(label.len);
        agg.count = 1;
        agg.total_ns = ns;
        agg.min_ns = ns;
        agg.max_ns = ns;
    }

    pub fn statsFor(self: *const GpuProfiler, label: []const u8) ProfileStats {
        for (self.aggregates[0..self.aggregate_count]) |agg| {
            if (agg.label_len == label.len and
                std.mem.eql(u8, agg.label[0..agg.label_len], label))
            {
                return .{
                    .count = agg.count,
                    .total_ns = agg.total_ns,
                    .min_ns = agg.min_ns,
                    .max_ns = agg.max_ns,
                };
            }
        }
        return .{};
    }

    pub fn print(self: *const GpuProfiler) void {
        if (self.aggregate_count == 0 and self.dropped_events == 0) return;

        var total_ns: u64 = 0;
        for (self.aggregates[0..self.aggregate_count]) |agg| total_ns += agg.total_ns;

        var order: [max_profile_labels]u32 = undefined;
        for (order[0..self.aggregate_count], 0..) |*idx, i| idx.* = @intCast(i);
        std.mem.sort(u32, order[0..self.aggregate_count], self, struct {
            fn lessThan(prof: *const GpuProfiler, lhs: u32, rhs: u32) bool {
                return prof.aggregates[lhs].total_ns > prof.aggregates[rhs].total_ns;
            }
        }.lessThan);

        std.debug.print("\nGPU profile (timestamp queries):\n", .{});
        std.debug.print("{s: <36} {s: >8} {s: >12} {s: >10} {s: >10} {s: >10} {s: >7}\n", .{ "label", "count", "total ms", "avg us", "min us", "max us", "%" });

        for (order[0..self.aggregate_count]) |agg_idx| {
            const agg = self.aggregates[agg_idx];
            const total_ms = @as(f64, @floatFromInt(agg.total_ns)) / 1_000_000.0;
            const avg_us = @as(f64, @floatFromInt(agg.total_ns)) /
                @as(f64, @floatFromInt(agg.count)) / 1_000.0;
            const min_us = @as(f64, @floatFromInt(agg.min_ns)) / 1_000.0;
            const max_us = @as(f64, @floatFromInt(agg.max_ns)) / 1_000.0;
            const pct = if (total_ns == 0) 0.0 else 100.0 * @as(f64, @floatFromInt(agg.total_ns)) /
                @as(f64, @floatFromInt(total_ns));
            std.debug.print("{s: <36} {d: >8} {d: >12.3} {d: >10.2} {d: >10.2} {d: >10.2} {d: >6.1}\n", .{ agg.label[0..agg.label_len], agg.count, total_ms, avg_us, min_us, max_us, pct });
        }

        if (self.dropped_events != 0) {
            std.debug.print("GPU profile dropped events: {}\n", .{self.dropped_events});
        }
    }
};

pub const GpuContext = struct {
    instance: vk.VkInstance,
    phys_dev: vk.VkPhysicalDevice,
    device: vk.VkDevice,
    queue: vk.VkQueue,
    queue_family: u32,
    cmd_pool: vk.VkCommandPool,
    profiler: ?*GpuProfiler,
    subgroup_size: u32,
    subgroup_supported_stages: vk.VkShaderStageFlags,
    subgroup_supported_operations: vk.VkSubgroupFeatureFlags,

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

        var subgroup_props = std.mem.zeroes(vk.VkPhysicalDeviceSubgroupProperties);
        subgroup_props.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;

        var props2 = std.mem.zeroes(vk.VkPhysicalDeviceProperties2);
        props2.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &subgroup_props;
        vk.vkGetPhysicalDeviceProperties2(phys_dev, &props2);

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
        v13.shaderIntegerDotProduct = vk.VK_TRUE; // dotPacked4x8EXT

        var v12 = std.mem.zeroes(vk.VkPhysicalDeviceVulkan12Features);
        v12.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        v12.pNext = &v13;
        v12.shaderFloat16 = vk.VK_TRUE; // f16vec2 ds, dm
        v12.shaderInt8 = vk.VK_TRUE; // int8_t arithmetic
        v12.storageBuffer8BitAccess = vk.VK_TRUE; // int8_t qs[32] loads
        v12.shaderSubgroupExtendedTypes = vk.VK_TRUE; // subgroupAdd(int32_t) etc.
        v12.uniformAndStorageBuffer8BitAccess = vk.VK_TRUE;

        var v11 = std.mem.zeroes(vk.VkPhysicalDeviceVulkan11Features);
        v11.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        v11.pNext = &v12;
        v11.storageBuffer16BitAccess = vk.VK_TRUE; // f16vec2 / int16 loads
        v11.uniformAndStorageBuffer16BitAccess = vk.VK_TRUE;

        var feats2 = std.mem.zeroes(vk.VkPhysicalDeviceFeatures2);
        feats2.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        feats2.pNext = &v11;
        feats2.features.shaderInt16 = vk.VK_TRUE; // i16vec2 unpack in Q5_0/Q5_1
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
            .pEnabledFeatures = null, // using Features2 in pNext instead
        };
        var device: vk.VkDevice = undefined;
        if (vk.vkCreateDevice(phys_dev, &dev_ci, null, &device) != vk.VK_SUCCESS)
            return error.VkDeviceCreateFailed;
        errdefer vk.vkDestroyDevice(device, null);

        var queue: vk.VkQueue = undefined;
        vk.vkGetDeviceQueue(device, queue_family, 0, &queue);

        var props: vk.VkPhysicalDeviceProperties = undefined;
        vk.vkGetPhysicalDeviceProperties(phys_dev, &props);

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

        var profiler: ?*GpuProfiler = null;
        if (std.c.getenv("LLMTOY_GPU_PROFILE") != null) {
            profiler = try std.heap.page_allocator.create(GpuProfiler);
            profiler.?.* = try GpuProfiler.init(device, props.limits.timestampPeriod);
        }

        return .{
            .instance = instance,
            .phys_dev = phys_dev,
            .device = device,
            .queue = queue,
            .queue_family = queue_family,
            .cmd_pool = cmd_pool,
            .profiler = profiler,
            .subgroup_size = subgroup_props.subgroupSize,
            .subgroup_supported_stages = subgroup_props.supportedStages,
            .subgroup_supported_operations = subgroup_props.supportedOperations,
        };
    }

    pub fn deinit(self: *GpuContext) void {
        if (self.profiler) |p| {
            p.print();
            p.deinit(self.device);
            std.heap.page_allocator.destroy(p);
        }
        vk.vkDestroyCommandPool(self.device, self.cmd_pool, null);
        vk.vkDestroyDevice(self.device, null);
        vk.vkDestroyInstance(self.instance, null);
    }

    pub fn deviceName(self: *const GpuContext) [vk.VK_MAX_PHYSICAL_DEVICE_NAME_SIZE]u8 {
        var props: vk.VkPhysicalDeviceProperties = undefined;
        vk.vkGetPhysicalDeviceProperties(self.phys_dev, &props);
        return props.deviceName;
    }

    pub fn hasSubgroupArithmetic(self: *const GpuContext) bool {
        return self.subgroup_supported_operations & vk.VK_SUBGROUP_FEATURE_ARITHMETIC_BIT != 0;
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

    pub fn allocReusableCommandBuffer(self: *const GpuContext) !vk.VkCommandBuffer {
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
        return cmd;
    }

    pub fn freeReusableCommandBuffer(self: *const GpuContext, cmd: vk.VkCommandBuffer) void {
        vk.vkFreeCommandBuffers(self.device, self.cmd_pool, 1, &cmd);
    }

    pub fn beginReusableBatch(self: *const GpuContext, cmd: vk.VkCommandBuffer) !vk.VkCommandBuffer {
        _ = vk.vkResetCommandBuffer(cmd, 0);
        const begin_ci = vk.VkCommandBufferBeginInfo{
            .sType = vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = null,
            .flags = vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = null,
        };
        _ = vk.vkBeginCommandBuffer(cmd, &begin_ci);
        if (self.profiler) |p| p.resetBatch(cmd);
        return cmd;
    }

    pub fn submitReusableBatch(self: *const GpuContext, cmd: vk.VkCommandBuffer) !void {
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
        if (self.profiler) |p| p.collectBatch(self.device);
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
        if (self.profiler) |p| p.resetBatch(cmd);
        return cmd;
    }

    pub fn profileBegin(self: *const GpuContext, cmd: vk.VkCommandBuffer, label: []const u8) u32 {
        if (self.profiler) |p| return p.begin(cmd, label);
        return std.math.maxInt(u32);
    }

    pub fn profileBeginFmt(
        self: *const GpuContext,
        cmd: vk.VkCommandBuffer,
        comptime fmt: []const u8,
        args: anytype,
    ) u32 {
        if (self.profiler) |p| {
            var buf: [profile_label_len]u8 = undefined;
            const label = std.fmt.bufPrint(&buf, fmt, args) catch "profile.label_truncated";
            return p.begin(cmd, label);
        }
        return std.math.maxInt(u32);
    }

    pub fn profileEnd(self: *const GpuContext, cmd: vk.VkCommandBuffer, event_id: u32) void {
        if (self.profiler) |p| p.end(cmd, event_id);
    }

    pub fn profileStats(self: *const GpuContext, label: []const u8) ProfileStats {
        if (self.profiler) |p| return p.statsFor(label);
        return .{};
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
            0,
            1,
            &barrier,
            0,
            null,
            0,
            null,
        );
    }

    // Compute → transfer pipeline barrier.
    // Ensures all shader writes before this point are visible to vkCmdCopyBuffer
    // (or any other transfer op) reads after it. Required when a copy reads a
    // buffer that was just written by a compute dispatch in the same command
    // buffer (e.g. 7l.1 GPU-rope output → KV cache append copy).
    pub fn recordShaderToTransferBarrier(cmd: vk.VkCommandBuffer) void {
        const barrier = vk.VkMemoryBarrier{
            .sType = vk.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = null,
            .srcAccessMask = vk.VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = vk.VK_ACCESS_TRANSFER_READ_BIT,
        };
        vk.vkCmdPipelineBarrier(
            cmd,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            1,
            &barrier,
            0,
            null,
            0,
            null,
        );
    }

    // Transfer → compute pipeline barrier.
    // Ensures vkCmdCopyBuffer writes before this point are visible to shader
    // reads/writes after it. Used when a compute-produced buffer is copied into
    // another compute input within the same command buffer.
    pub fn recordTransferToShaderBarrier(cmd: vk.VkCommandBuffer) void {
        const barrier = vk.VkMemoryBarrier{
            .sType = vk.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = null,
            .srcAccessMask = vk.VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
        };
        vk.vkCmdPipelineBarrier(
            cmd,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            1,
            &barrier,
            0,
            null,
            0,
            null,
        );
    }

    // vkCmdCopyBuffer with explicit offsets, recorded into an existing command
    // buffer. Use this for in-submit copies (e.g. appending one slot of K/V
    // into the per-layer VRAM cache). For one-shot copies see copyBufferRegion.
    pub fn recordCopyRegion(
        cmd: vk.VkCommandBuffer,
        src: vk.VkBuffer,
        dst: vk.VkBuffer,
        src_offset: vk.VkDeviceSize,
        dst_offset: vk.VkDeviceSize,
        size: vk.VkDeviceSize,
    ) void {
        const region = vk.VkBufferCopy{ .srcOffset = src_offset, .dstOffset = dst_offset, .size = size };
        vk.vkCmdCopyBuffer(cmd, src, dst, 1, &region);
    }

    // End recording, submit all recorded copies, wait for completion, free command buffer.
    pub fn submitBatchCopy(self: *const GpuContext, cmd: vk.VkCommandBuffer) !void {
        _ = vk.vkEndCommandBuffer(cmd);
        defer vk.vkFreeCommandBuffers(self.device, self.cmd_pool, 1, &cmd);
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
        if (self.profiler) |p| p.collectBatch(self.device);
    }

    // Copy `size` bytes from src[src_offset..] to dst[dst_offset..] in a
    // one-shot command buffer. Blocks until complete. Use this when the source
    // or destination is a region inside a larger buffer (e.g. a per-position
    // slot in a KV cache buffer).
    pub fn copyBufferRegion(
        self: *const GpuContext,
        src: vk.VkBuffer,
        dst: vk.VkBuffer,
        src_offset: vk.VkDeviceSize,
        dst_offset: vk.VkDeviceSize,
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
        const region = vk.VkBufferCopy{ .srcOffset = src_offset, .dstOffset = dst_offset, .size = size };
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
