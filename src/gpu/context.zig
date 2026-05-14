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
        // Instance
        const app_info = vk.VkApplicationInfo{
            .sType = vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pNext = null,
            .pApplicationName = "llmtoy",
            .applicationVersion = vk.VK_MAKE_VERSION(0, 7, 0),
            .pEngineName = "llmtoy",
            .engineVersion = vk.VK_MAKE_VERSION(0, 7, 0),
            .apiVersion = vk.VK_API_VERSION_1_1,
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
        const dev_ci = vk.VkDeviceCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queue_ci,
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = null,
            .enabledExtensionCount = 0,
            .ppEnabledExtensionNames = null,
            .pEnabledFeatures = null,
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
};
