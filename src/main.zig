const std = @import("std");
const gguf_reader = @import("gguf/reader.zig");
const vocab_mod = @import("tokenizer/vocab.zig");
const bpe = @import("tokenizer/bpe.zig");
const chat_tmpl = @import("tokenizer/chat_template.zig");
const loader = @import("model/loader.zig");
const fwd = @import("model/forward.zig");
const kv_mod = @import("model/kv_cache.zig");
const sample_mod = @import("model/sample.zig");
const tp = @import("ops/thread_pool.zig");
const g4_loader = @import("model/gemma4/loader.zig");
const g4_fwd = @import("model/gemma4/forward.zig");
const g4_kv = @import("model/gemma4/kv_cache.zig");
const g4_gpu = @import("model/gemma4/gpu_weights.zig");
const g4_weights = @import("model/gemma4/weights.zig");
const gpu_ctx = @import("gpu/context.zig");
const gpu_matvec = @import("gpu/matvec.zig");
const gpu_buffer = @import("gpu/buffer.zig");
const vk = @import("gpu/vk.zig").vk;
const math_mod = @import("ops/math.zig");
const gguf_types = @import("gguf/types.zig");

// Pull tests from sub-modules into the test binary.
comptime {
    _ = @import("gguf/reader.zig");
    _ = @import("tokenizer/bpe.zig");
    _ = @import("tokenizer/chat_template.zig");
    _ = @import("ops/math.zig");
    _ = @import("ops/attn.zig");
    _ = @import("ops/rope.zig");
    _ = @import("ops/thread_pool.zig");
    _ = @import("quant/dequant.zig");
    _ = @import("model/forward.zig");
    _ = @import("model/sample.zig");
    _ = @import("model/gemma4/forward.zig");
    _ = @import("model/gemma4/gpu_weights.zig");
    _ = @import("gpu/matvec.zig");
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = init.gpa;
    const arena = init.arena.allocator();

    const args = try init.minimal.args.toSlice(arena);

    var out_buf: [65536]u8 = undefined;
    var out_fw = std.Io.File.stdout().writer(io, &out_buf);
    const out = &out_fw.interface;
    defer out.flush() catch {};

    if (args.len < 2) {
        try usagePrint(out);
        return;
    }

    if (std.mem.eql(u8, args[1], "gpu-info")) {
        var verbose = false;
        for (args[2..]) |arg| {
            if (std.mem.eql(u8, arg, "--verbose")) verbose = true;
        }
        try cmdGpuInfo(out, verbose);
    } else if (std.mem.eql(u8, args[1], "info")) {
        if (args.len < 3) {
            std.debug.print("usage: llmtoy info <model.gguf>\n", .{});
            return error.MissingArg;
        }
        try cmdInfo(out, args[2], io, gpa);
    } else if (std.mem.eql(u8, args[1], "tokenize")) {
        if (args.len < 4) {
            std.debug.print("usage: llmtoy tokenize <model.gguf> <text>\n", .{});
            return error.MissingArg;
        }
        try cmdTokenize(out, args[2], args[3], io, gpa);
    } else if (std.mem.eql(u8, args[1], "bench-matvec")) {
        if (args.len < 3) {
            std.debug.print("usage: llmtoy bench-matvec <model.gguf> [--iters N] [--target NAME] [--reuse-descriptor]\n", .{});
            return error.MissingArg;
        }
        var opts = MatvecBenchOptions{};
        var i: usize = 3;
        while (i < args.len) {
            const flag = args[i];
            if (std.mem.eql(u8, flag, "--reuse-descriptor")) {
                opts.reuse_descriptor = true;
                i += 1;
                continue;
            }
            if (i + 1 >= args.len) break;
            const val = args[i + 1];
            if (std.mem.eql(u8, flag, "--iters")) opts.iters = try std.fmt.parseInt(u32, val, 10);
            if (std.mem.eql(u8, flag, "--target")) opts.target = val;
            i += 2;
        }
        try cmdBenchMatvec(out, args[2], opts, io, gpa);
    } else if (std.mem.eql(u8, args[1], "bench-moe")) {
        if (args.len < 3) {
            std.debug.print("usage: llmtoy bench-moe <model.gguf> [--iters N] [--layer N] [--skip-readback] [--tail]\n", .{});
            return error.MissingArg;
        }
        var opts = MoeBenchOptions{};
        var i: usize = 3;
        while (i < args.len) {
            const flag = args[i];
            if (std.mem.eql(u8, flag, "--skip-readback")) {
                opts.skip_readback = true;
                i += 1;
                continue;
            }
            if (std.mem.eql(u8, flag, "--tail")) {
                opts.tail = true;
                opts.skip_readback = true;
                i += 1;
                continue;
            }
            if (i + 1 >= args.len) break;
            const val = args[i + 1];
            if (std.mem.eql(u8, flag, "--iters")) opts.iters = try std.fmt.parseInt(u32, val, 10);
            if (std.mem.eql(u8, flag, "--layer")) opts.layer = try std.fmt.parseInt(usize, val, 10);
            i += 2;
        }
        try cmdBenchMoe(out, args[2], opts, io, gpa);
    } else if (std.mem.eql(u8, args[1], "bench-rmsnorm")) {
        var opts = RmsnormBenchOptions{};
        var i: usize = 2;
        while (i < args.len) {
            const flag = args[i];
            if (i + 1 >= args.len) break;
            const val = args[i + 1];
            if (std.mem.eql(u8, flag, "--iters")) opts.iters = try std.fmt.parseInt(u32, val, 10);
            if (std.mem.eql(u8, flag, "--n")) opts.n = try std.fmt.parseInt(u32, val, 10);
            i += 2;
        }
        try cmdBenchRmsnorm(out, opts, io, gpa);
    } else if (std.mem.eql(u8, args[1], "generate")) {
        if (args.len < 4) {
            std.debug.print(
                "usage: llmtoy generate <model.gguf> <prompt> [--chat] [--max-tokens N] [--temperature T] [--top-p P] [--top-k K] [--seed S] [--threads N]\n",
                .{},
            );
            return error.MissingArg;
        }
        const model_path = args[2];
        const prompt = args[3];

        // Parse optional flags.
        var max_tokens: u32 = 256;
        var temperature: f32 = 0.8;
        var top_p: f32 = 0.9;
        var top_k: u32 = 40;
        var seed: u64 = 42;
        var threads: u32 = 0; // 0 = auto (getCpuCount)
        var chat: bool = false;
        var use_gpu: bool = false;
        var stop_token: ?[]const u8 = null;
        var gpu_layer_range: ?[2]usize = null;
        var i: usize = 4;
        while (i < args.len) {
            const flag = args[i];
            if (std.mem.eql(u8, flag, "--chat")) {
                chat = true;
                i += 1;
                continue;
            }
            if (std.mem.eql(u8, flag, "--gpu")) {
                use_gpu = true;
                i += 1;
                continue;
            }
            if (i + 1 >= args.len) break;
            const val = args[i + 1];
            if (std.mem.eql(u8, flag, "--max-tokens")) max_tokens = try std.fmt.parseInt(u32, val, 10);
            if (std.mem.eql(u8, flag, "--temperature")) temperature = try std.fmt.parseFloat(f32, val);
            if (std.mem.eql(u8, flag, "--top-p")) top_p = try std.fmt.parseFloat(f32, val);
            if (std.mem.eql(u8, flag, "--top-k")) top_k = try std.fmt.parseInt(u32, val, 10);
            if (std.mem.eql(u8, flag, "--seed")) seed = try std.fmt.parseInt(u64, val, 10);
            if (std.mem.eql(u8, flag, "--threads")) threads = try std.fmt.parseInt(u32, val, 10);
            if (std.mem.eql(u8, flag, "--stop-token")) stop_token = val;
            if (std.mem.eql(u8, flag, "--gpu-layers")) gpu_layer_range = try parseLayerRange(val);
            i += 2;
        }

        try cmdGenerate(out, model_path, prompt, .{
            .max_tokens = max_tokens,
            .temperature = temperature,
            .top_p = top_p,
            .top_k = top_k,
            .seed = seed,
            .threads = threads,
            .chat = chat,
            .gpu = use_gpu,
            .stop_token = stop_token,
            .gpu_layer_range = gpu_layer_range,
        }, io, gpa);
    } else if (std.mem.eql(u8, args[1], "compare")) {
        if (args.len < 4) {
            std.debug.print(
                "usage: llmtoy compare <model.gguf> <prompt> [--chat] [--threads N] [--gpu-layers L0:L1]\n",
                .{},
            );
            return error.MissingArg;
        }
        const model_path = args[2];
        const prompt = args[3];
        var threads: u32 = 0;
        var chat: bool = false;
        var gpu_layer_range: ?[2]usize = null;
        var i: usize = 4;
        while (i < args.len) {
            const flag = args[i];
            if (std.mem.eql(u8, flag, "--chat")) {
                chat = true;
                i += 1;
                continue;
            }
            if (i + 1 >= args.len) break;
            const val = args[i + 1];
            if (std.mem.eql(u8, flag, "--threads")) threads = try std.fmt.parseInt(u32, val, 10);
            if (std.mem.eql(u8, flag, "--gpu-layers")) gpu_layer_range = try parseLayerRange(val);
            i += 2;
        }
        try cmdCompare(out, model_path, prompt, .{
            .threads = threads,
            .chat = chat,
            .gpu_layer_range = gpu_layer_range,
        }, io, gpa);
    } else {
        try usagePrint(out);
    }
}

fn usagePrint(out: *std.Io.Writer) !void {
    try out.writeAll(
        \\llmtoy-zig  —  educational LLM inference
        \\
        \\  llmtoy gpu-info [--verbose]            list Vulkan device and run a matvec smoke test
        \\  llmtoy info <model.gguf>               print model metadata and tensor summary
        \\  llmtoy tokenize <model.gguf> <text>    BPE-encode text, print IDs and decoded tokens
        \\  llmtoy bench-matvec <model.gguf> [--iters N] [--target NAME] [--reuse-descriptor]
        \\  llmtoy bench-moe <model.gguf> [--iters N] [--layer N] [--skip-readback] [--tail]
        \\  llmtoy bench-rmsnorm [--iters N] [--n N]
        \\  llmtoy generate <model.gguf> <prompt> [--chat] [--gpu] [--max-tokens N] [--temperature T] [--top-p P] [--top-k K] [--seed S] [--threads N] [--stop-token TOKEN] [--gpu-layers L0:L1]
        \\  llmtoy compare  <model.gguf> <prompt> [--chat] [--threads N] [--gpu-layers L0:L1]
        \\
        \\  --gpu:        offload attention + dense-FFN matmuls to the GPU via Vulkan (Gemma4 only)
        \\  --stop-token: stop generation when this token is sampled (default: auto-detect from vocab)
        \\  --gpu-layers: restrict GPU to layers L0..L1 inclusive (e.g. 0:14); others use CPU
        \\  compare:      run one CPU and one GPU forward pass, print per-layer residual divergence
        \\  bench-matvec: benchmark current Q8_1 matvec kernels on representative real tensors
        \\  bench-moe:    benchmark the current top-k expert batch path on real Gemma4 tensors
        \\  bench-rmsnorm: benchmark single-row RMSNorm and add+RMSNorm kernel shapes
        \\
    );
}

fn cmdGpuInfo(out: *std.Io.Writer, verbose: bool) !void {
    var ctx = gpu_ctx.GpuContext.init() catch |e| {
        try out.print("gpu init failed: {}\n", .{e});
        return;
    };
    defer ctx.deinit();

    const name = ctx.deviceName();
    try out.print("GPU device: {s}\n", .{std.mem.sliceTo(&name, 0)});
    try out.print("subgroup: size={} compute={} arithmetic={}\n", .{
        ctx.subgroup_size,
        ctx.subgroup_supported_stages & vk.VK_SHADER_STAGE_COMPUTE_BIT != 0,
        ctx.hasSubgroupArithmetic(),
    });
    if (verbose) try printGpuVerbose(out, &ctx);

    // Smoke test: 4×4 identity * [1,2,3,4] = [1,2,3,4]
    var pipeline = gpu_matvec.MatvecPipeline.initF32(&ctx) catch |e| {
        try out.print("pipeline init failed: {}\n", .{e});
        return;
    };
    defer pipeline.deinit();

    const mat = [16]f32{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
    const vec = [4]f32{ 1, 2, 3, 4 };
    var result = [4]f32{ 0, 0, 0, 0 };
    try gpu_matvec.matvecF32(&ctx, &pipeline, &mat, &vec, &result, 4, 4);

    try out.print("matvec smoke test: [{d:.0}, {d:.0}, {d:.0}, {d:.0}] (expect [1, 2, 3, 4])\n", .{ result[0], result[1], result[2], result[3] });
}

fn vkBool(v: vk.VkBool32) bool {
    return v != vk.VK_FALSE;
}

fn printGpuVerbose(out: *std.Io.Writer, ctx: *const gpu_ctx.GpuContext) !void {
    var idot_props = std.mem.zeroes(vk.VkPhysicalDeviceShaderIntegerDotProductProperties);
    idot_props.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES;

    var subgroup_size_props = std.mem.zeroes(vk.VkPhysicalDeviceSubgroupSizeControlProperties);
    subgroup_size_props.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES;
    subgroup_size_props.pNext = &idot_props;

    var subgroup_props = std.mem.zeroes(vk.VkPhysicalDeviceSubgroupProperties);
    subgroup_props.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    subgroup_props.pNext = &subgroup_size_props;

    var props2 = std.mem.zeroes(vk.VkPhysicalDeviceProperties2);
    props2.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &subgroup_props;
    vk.vkGetPhysicalDeviceProperties2(ctx.phys_dev, &props2);

    var idot_feats = std.mem.zeroes(vk.VkPhysicalDeviceShaderIntegerDotProductFeatures);
    idot_feats.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;

    var subgroup_size_feats = std.mem.zeroes(vk.VkPhysicalDeviceSubgroupSizeControlFeatures);
    subgroup_size_feats.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES;
    subgroup_size_feats.pNext = &idot_feats;

    var feats2 = std.mem.zeroes(vk.VkPhysicalDeviceFeatures2);
    feats2.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    feats2.pNext = &subgroup_size_feats;
    vk.vkGetPhysicalDeviceFeatures2(ctx.phys_dev, &feats2);

    try out.print("api: {}.{}.{}  driver=0x{x}  vendor=0x{x}  device=0x{x}\n", .{
        vk.VK_VERSION_MAJOR(props2.properties.apiVersion),
        vk.VK_VERSION_MINOR(props2.properties.apiVersion),
        vk.VK_VERSION_PATCH(props2.properties.apiVersion),
        props2.properties.driverVersion,
        props2.properties.vendorID,
        props2.properties.deviceID,
    });
    try out.print("device type: {s}\n", .{deviceTypeName(props2.properties.deviceType)});
    try out.print("limits: maxComputeWorkGroupInvocations={} timestampPeriod={d:.3} ns\n", .{
        props2.properties.limits.maxComputeWorkGroupInvocations,
        props2.properties.limits.timestampPeriod,
    });

    try out.print("subgroup stages: {s}\n", .{shaderStageFlags(ctx.subgroup_supported_stages)});
    try out.print("subgroup ops: {s}\n", .{subgroupOpFlags(ctx.subgroup_supported_operations)});
    try out.print("subgroup size control: feature={} fullSubgroups={} min={} max={} requiredStages={s}\n", .{
        vkBool(subgroup_size_feats.subgroupSizeControl),
        vkBool(subgroup_size_feats.computeFullSubgroups),
        subgroup_size_props.minSubgroupSize,
        subgroup_size_props.maxSubgroupSize,
        shaderStageFlags(subgroup_size_props.requiredSubgroupSizeStages),
    });
    try out.print("integer dot: feature={} s8x4_accel={} u8x4_accel={} mixed_s8x4_accel={} mixed_u8x4_accel={}\n", .{
        vkBool(idot_feats.shaderIntegerDotProduct),
        vkBool(idot_props.integerDotProduct4x8BitPackedSignedAccelerated),
        vkBool(idot_props.integerDotProduct4x8BitPackedUnsignedAccelerated),
        vkBool(idot_props.integerDotProduct4x8BitPackedMixedSignednessAccelerated),
        vkBool(idot_props.integerDotProduct4x8BitPackedMixedSignednessAccelerated),
    });

    try printExtensionSupport(out, ctx);
    try printMemoryInfo(out, ctx.phys_dev);
}

fn printExtensionSupport(out: *std.Io.Writer, ctx: *const gpu_ctx.GpuContext) !void {
    const exts = [_][]const u8{
        "VK_KHR_pipeline_executable_properties",
        "VK_KHR_cooperative_matrix",
        "VK_KHR_shader_subgroup_uniform_control_flow",
    };
    var count: u32 = 0;
    _ = vk.vkEnumerateDeviceExtensionProperties(ctx.phys_dev, null, &count, null);
    var props: [256]vk.VkExtensionProperties = undefined;
    const n = @min(count, props.len);
    count = @intCast(n);
    _ = vk.vkEnumerateDeviceExtensionProperties(ctx.phys_dev, null, &count, &props);

    try out.writeAll("extensions:");
    for (exts) |name| {
        var supported = false;
        for (props[0..count]) |prop| {
            if (std.mem.eql(u8, std.mem.sliceTo(&prop.extensionName, 0), name)) {
                supported = true;
                break;
            }
        }
        try out.print(" {s}={}", .{ name, supported });
    }
    try out.writeByte('\n');
    try printCooperativeMatrixShapes(out, ctx);
}

fn printCooperativeMatrixShapes(out: *std.Io.Writer, ctx: *const gpu_ctx.GpuContext) !void {
    const raw_fn = vk.vkGetInstanceProcAddr(ctx.instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR") orelse {
        try out.writeAll("cooperative matrix shapes: unavailable\n");
        return;
    };
    const get_props: vk.PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR = @ptrCast(raw_fn);

    var count: u32 = 0;
    var rc = get_props.?(ctx.phys_dev, &count, null);
    if (rc != vk.VK_SUCCESS or count == 0) {
        try out.writeAll("cooperative matrix shapes: unavailable\n");
        return;
    }

    var props: [64]vk.VkCooperativeMatrixPropertiesKHR = undefined;
    const n = @min(count, props.len);
    count = @intCast(n);
    for (props[0..count]) |*prop| {
        prop.* = std.mem.zeroes(vk.VkCooperativeMatrixPropertiesKHR);
        prop.sType = vk.VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    }
    rc = get_props.?(ctx.phys_dev, &count, &props);
    if (rc != vk.VK_SUCCESS) {
        try out.writeAll("cooperative matrix shapes: query failed\n");
        return;
    }

    try out.print("cooperative matrix shapes: count={}\n", .{count});
    for (props[0..count]) |prop| {
        try out.print("  {}x{}x{} scope={s} A={s} B={s} C={s} R={s} saturating={}\n", .{
            prop.MSize,
            prop.NSize,
            prop.KSize,
            cooperativeMatrixScopeName(prop.scope),
            componentTypeName(prop.AType),
            componentTypeName(prop.BType),
            componentTypeName(prop.CType),
            componentTypeName(prop.ResultType),
            vkBool(prop.saturatingAccumulation),
        });
    }
}

fn printMemoryInfo(out: *std.Io.Writer, phys_dev: vk.VkPhysicalDevice) !void {
    var mem: vk.VkPhysicalDeviceMemoryProperties = undefined;
    vk.vkGetPhysicalDeviceMemoryProperties(phys_dev, &mem);

    try out.writeAll("memory heaps:\n");
    for (0..mem.memoryHeapCount) |i| {
        const heap = mem.memoryHeaps[i];
        try out.print("  heap {}: {d:.2} GiB flags={s}\n", .{
            i,
            @as(f64, @floatFromInt(heap.size)) / (1024.0 * 1024.0 * 1024.0),
            memoryHeapFlags(heap.flags),
        });
    }

    try out.writeAll("memory types:\n");
    for (0..mem.memoryTypeCount) |i| {
        const mt = mem.memoryTypes[i];
        try out.print("  type {}: heap={} flags={s}\n", .{ i, mt.heapIndex, memoryPropertyFlags(mt.propertyFlags) });
    }
}

fn deviceTypeName(t: vk.VkPhysicalDeviceType) []const u8 {
    return switch (t) {
        vk.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU => "integrated-gpu",
        vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU => "discrete-gpu",
        vk.VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU => "virtual-gpu",
        vk.VK_PHYSICAL_DEVICE_TYPE_CPU => "cpu",
        else => "other",
    };
}

fn shaderStageFlags(flags: vk.VkShaderStageFlags) []const u8 {
    if (flags & vk.VK_SHADER_STAGE_COMPUTE_BIT != 0) return "compute";
    if (flags == 0) return "none";
    return "non-compute";
}

fn subgroupOpFlags(flags: vk.VkSubgroupFeatureFlags) []const u8 {
    const arithmetic = flags & vk.VK_SUBGROUP_FEATURE_ARITHMETIC_BIT != 0;
    const clustered = flags & vk.VK_SUBGROUP_FEATURE_CLUSTERED_BIT != 0;
    const ballot = flags & vk.VK_SUBGROUP_FEATURE_BALLOT_BIT != 0;
    if (arithmetic and clustered and ballot) return "arithmetic,clustered,ballot";
    if (arithmetic and clustered) return "arithmetic,clustered";
    if (arithmetic) return "arithmetic";
    if (flags == 0) return "none";
    return "other";
}

fn memoryHeapFlags(flags: vk.VkMemoryHeapFlags) []const u8 {
    if (flags & vk.VK_MEMORY_HEAP_DEVICE_LOCAL_BIT != 0) return "device-local";
    if (flags == 0) return "none";
    return "other";
}

fn memoryPropertyFlags(flags: vk.VkMemoryPropertyFlags) []const u8 {
    const device = flags & vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT != 0;
    const host_visible = flags & vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT != 0;
    const coherent = flags & vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT != 0;
    const cached = flags & vk.VK_MEMORY_PROPERTY_HOST_CACHED_BIT != 0;
    if (device and host_visible and coherent and cached) return "device-local,host-visible,host-coherent,host-cached";
    if (device and host_visible and coherent) return "device-local,host-visible,host-coherent";
    if (host_visible and coherent and cached) return "host-visible,host-coherent,host-cached";
    if (host_visible and coherent) return "host-visible,host-coherent";
    if (device) return "device-local";
    if (flags == 0) return "none";
    return "other";
}

fn cooperativeMatrixScopeName(scope: vk.VkScopeKHR) []const u8 {
    return switch (scope) {
        vk.VK_SCOPE_DEVICE_KHR => "device",
        vk.VK_SCOPE_WORKGROUP_KHR => "workgroup",
        vk.VK_SCOPE_SUBGROUP_KHR => "subgroup",
        vk.VK_SCOPE_QUEUE_FAMILY_KHR => "queue-family",
        else => "other",
    };
}

fn componentTypeName(t: vk.VkComponentTypeKHR) []const u8 {
    return switch (t) {
        vk.VK_COMPONENT_TYPE_FLOAT16_KHR => "f16",
        vk.VK_COMPONENT_TYPE_FLOAT32_KHR => "f32",
        vk.VK_COMPONENT_TYPE_SINT8_KHR => "i8",
        vk.VK_COMPONENT_TYPE_SINT16_KHR => "i16",
        vk.VK_COMPONENT_TYPE_SINT32_KHR => "i32",
        vk.VK_COMPONENT_TYPE_UINT8_KHR => "u8",
        vk.VK_COMPONENT_TYPE_UINT16_KHR => "u16",
        vk.VK_COMPONENT_TYPE_UINT32_KHR => "u32",
        else => "other",
    };
}

// ── commands ──────────────────────────────────────────────────────────────────

fn cmdInfo(out: *std.Io.Writer, path: []const u8, io: std.Io, gpa: std.mem.Allocator) !void {
    var reader = try gguf_reader.GgufReader.open(path, io, gpa);
    defer reader.deinit();

    const size_gib = @as(f64, @floatFromInt(reader.mmap.len)) / (1024 * 1024 * 1024);

    try out.print("file:      {s}\n", .{path});
    try out.print("version:   GGUF v{}\n", .{reader.data.version});
    try out.print("size:      {d:.2} GiB\n", .{size_gib});
    try out.print("tensors:   {}\n", .{reader.data.tensors.len});
    try out.print("metadata:  {} entries\n", .{reader.data.metadata.count()});

    if (reader.metaString("general.architecture")) |arch| {
        try out.print("arch:      {s}\n", .{arch});

        const ctx_key = try std.fmt.allocPrint(gpa, "{s}.context_length", .{arch});
        defer gpa.free(ctx_key);
        const emb_key = try std.fmt.allocPrint(gpa, "{s}.embedding_length", .{arch});
        defer gpa.free(emb_key);
        const blk_key = try std.fmt.allocPrint(gpa, "{s}.block_count", .{arch});
        defer gpa.free(blk_key);
        const head_key = try std.fmt.allocPrint(gpa, "{s}.attention.head_count", .{arch});
        defer gpa.free(head_key);

        if (reader.metaU32(ctx_key)) |v| try out.print("ctx_len:   {}\n", .{v});
        if (reader.metaU32(emb_key)) |v| try out.print("emb_dim:   {}\n", .{v});
        if (reader.metaU32(blk_key)) |v| try out.print("n_layers:  {}\n", .{v});
        if (reader.metaU32(head_key)) |v| try out.print("n_heads:   {}\n", .{v});
    }

    if (reader.metaString("general.name")) |name| {
        try out.print("name:      {s}\n", .{name});
    }
    if (reader.metaString("tokenizer.chat_template")) |tmpl| {
        const first_line_end = std.mem.indexOfScalar(u8, tmpl, '\n') orelse tmpl.len;
        try out.print("chat_tmpl: present ({} bytes), first line: {s}\n", .{ tmpl.len, tmpl[0..first_line_end] });
    }
    if (reader.metaU64("gemma4.embedding_length_per_layer_input")) |v| {
        try out.print("per_layer_input: {}\n", .{v});
    }
    if (reader.metaU64("gemma4.attention.shared_kv_layers")) |v| {
        try out.print("shared_kv_layers: {}\n", .{v});
    }
    if (reader.metaU32("tokenizer.ggml.bos_token_id")) |v| try out.print("bos_token: {}\n", .{v});
    if (reader.metaU32("tokenizer.ggml.eos_token_id")) |v| try out.print("eos_token: {}\n", .{v});

    try out.print("\nquant distribution:\n", .{});
    var counts = std.AutoHashMap(u32, u32).init(gpa);
    defer counts.deinit();
    for (reader.data.tensors) |tensor| {
        const key = @intFromEnum(tensor.type_);
        const entry = try counts.getOrPutValue(key, 0);
        entry.value_ptr.* += 1;
    }
    var it = counts.iterator();
    while (it.next()) |entry| {
        const gtype: @import("gguf/types.zig").GgmlType = @enumFromInt(entry.key_ptr.*);
        try out.print("  {s}: {}\n", .{ gtype.label(), entry.value_ptr.* });
    }

    try out.print("\nunsupported GPU quant tensors:\n", .{});
    var unsupported_count: usize = 0;
    for (reader.data.tensors) |tensor| {
        if (isGpuQuantSupported(tensor.type_)) continue;
        if (tensor.type_ == .f32) continue;
        unsupported_count += 1;
        try out.print("  [{s}] {s}  dims={any}\n", .{
            tensor.type_.label(), tensor.name, tensor.dims,
        });
    }
    if (unsupported_count == 0) {
        try out.print("  none\n", .{});
    }

    try out.print("\nfirst 16 tensors:\n", .{});
    for (reader.data.tensors[0..@min(16, reader.data.tensors.len)]) |tensor| {
        try out.print("  [{s}] {s}  dims={any}\n", .{
            tensor.type_.label(), tensor.name, tensor.dims,
        });
    }
    // Print blk.0 and blk.5 tensors for architecture inspection
    for ([_][]const u8{ "blk.0.", "blk.5." }) |prefix| {
        try out.print("\n{s}* tensors:\n", .{prefix});
        for (reader.data.tensors) |tensor| {
            if (std.mem.startsWith(u8, tensor.name, prefix)) {
                try out.print("  [{s}] {s}  dims={any}\n", .{
                    tensor.type_.label(), tensor.name, tensor.dims,
                });
            }
        }
    }
}

fn isGpuQuantSupported(t: @import("gguf/types.zig").GgmlType) bool {
    return switch (t) {
        .f32, .q8_0, .q3_k, .q4_k, .q5_0, .q5_1, .q6_k, .q5_k, .iq4_nl => true,
        else => false,
    };
}

const MatvecBenchOptions = struct {
    iters: u32 = 64,
    target: ?[]const u8 = null,
    reuse_descriptor: bool = false,
};

const MoeBenchOptions = struct {
    iters: u32 = 64,
    layer: usize = 0,
    skip_readback: bool = false,
    tail: bool = false,
};

const RmsnormBenchOptions = struct {
    iters: u32 = 256,
    n: u32 = 2816,
};

const BenchPipelines = struct {
    q3_k: gpu_matvec.MatvecPipeline,
    q3_k_mmvq_b32_r1: gpu_matvec.MatvecPipeline,
    q3_k_mmvq_b64_r1: gpu_matvec.MatvecPipeline,
    q4_k: gpu_matvec.MatvecPipeline,
    q4_k_mmvq_b32_r1: gpu_matvec.MatvecPipeline,
    q4_k_mmvq_b64_r1: gpu_matvec.MatvecPipeline,
    q4_k_mmvq_b64_r2: gpu_matvec.MatvecPipeline,
    q4_k_mmvq_b64_r4: gpu_matvec.MatvecPipeline,
    q4_k_r4: gpu_matvec.MatvecPipeline,
    q5_0: gpu_matvec.MatvecPipeline,
    q5_0_mmvq_b64_r1: gpu_matvec.MatvecPipeline,
    q5_0_mmvq_b64_r2: gpu_matvec.MatvecPipeline,
    q5_0_mmvq_b64_r4: gpu_matvec.MatvecPipeline,
    q5_1: gpu_matvec.MatvecPipeline,
    q5_1_mmvq_b64_r1: gpu_matvec.MatvecPipeline,
    q5_1_mmvq_b64_r2: gpu_matvec.MatvecPipeline,
    q5_1_mmvq_b64_r4: gpu_matvec.MatvecPipeline,
    q5_k: gpu_matvec.MatvecPipeline,
    q5_k_mmvq_b64_r1: gpu_matvec.MatvecPipeline,
    q5_k_mmvq_b64_r2: gpu_matvec.MatvecPipeline,
    q5_k_mmvq_b64_r4: gpu_matvec.MatvecPipeline,
    q6_k: gpu_matvec.MatvecPipeline,
    q6_k_fast: gpu_matvec.MatvecPipeline,
    q6_k_mmvq_b32_r1: gpu_matvec.MatvecPipeline,
    q6_k_mmvq_b64_r1: gpu_matvec.MatvecPipeline,
    q6_k_mmvq_b64_r2: gpu_matvec.MatvecPipeline,
    q6_k_mmvq_b64_r4: gpu_matvec.MatvecPipeline,
    iq4_nl: gpu_matvec.MatvecPipeline,
    quant: gpu_matvec.QuantizeQ8_1Pipeline,

    fn init(ctx: *const gpu_ctx.GpuContext) !BenchPipelines {
        var q3_k = try gpu_matvec.MatvecPipeline.initQ3KQ8_1(ctx);
        errdefer q3_k.deinit();
        var q3_k_mmvq_b32_r1 = try gpu_matvec.MatvecPipeline.initQ3KQ8_1Mmvq(ctx, .{
            .block_size = 32,
            .num_rows = 1,
            .num_cols = 1,
        });
        errdefer q3_k_mmvq_b32_r1.deinit();
        var q3_k_mmvq_b64_r1 = try gpu_matvec.MatvecPipeline.initQ3KQ8_1Mmvq(ctx, .{
            .block_size = 64,
            .num_rows = 1,
            .num_cols = 1,
        });
        errdefer q3_k_mmvq_b64_r1.deinit();
        var q4_k = try gpu_matvec.MatvecPipeline.initQ4KQ8_1(ctx);
        errdefer q4_k.deinit();
        var q4_k_mmvq_b32_r1 = try gpu_matvec.MatvecPipeline.initQ4KQ8_1Mmvq(ctx, .{
            .block_size = 32,
            .num_rows = 1,
            .num_cols = 1,
        });
        errdefer q4_k_mmvq_b32_r1.deinit();
        var q4_k_mmvq_b64_r1 = try gpu_matvec.MatvecPipeline.initQ4KQ8_1Mmvq(ctx, .{
            .block_size = 64,
            .num_rows = 1,
            .num_cols = 1,
        });
        errdefer q4_k_mmvq_b64_r1.deinit();
        var q4_k_mmvq_b64_r2 = try gpu_matvec.MatvecPipeline.initQ4KQ8_1Mmvq(ctx, .{
            .block_size = 64,
            .num_rows = 2,
            .num_cols = 1,
        });
        errdefer q4_k_mmvq_b64_r2.deinit();
        var q4_k_mmvq_b64_r4 = try gpu_matvec.MatvecPipeline.initQ4KQ8_1Mmvq(ctx, .{
            .block_size = 64,
            .num_rows = 4,
            .num_cols = 1,
        });
        errdefer q4_k_mmvq_b64_r4.deinit();
        var q4_k_r4 = try gpu_matvec.MatvecPipeline.initQ4KQ8_1R4(ctx);
        errdefer q4_k_r4.deinit();
        var q5_0 = try gpu_matvec.MatvecPipeline.initQ5_0Q8_1(ctx);
        errdefer q5_0.deinit();
        var q5_0_mmvq_b64_r1 = try gpu_matvec.MatvecPipeline.initQ5_0Q8_1Mmvq(ctx, .{
            .block_size = 64,
            .num_rows = 1,
            .num_cols = 1,
        });
        errdefer q5_0_mmvq_b64_r1.deinit();
        var q5_0_mmvq_b64_r2 = try gpu_matvec.MatvecPipeline.initQ5_0Q8_1Mmvq(ctx, .{
            .block_size = 64,
            .num_rows = 2,
            .num_cols = 1,
        });
        errdefer q5_0_mmvq_b64_r2.deinit();
        var q5_0_mmvq_b64_r4 = try gpu_matvec.MatvecPipeline.initQ5_0Q8_1Mmvq(ctx, .{
            .block_size = 64,
            .num_rows = 4,
            .num_cols = 1,
        });
        errdefer q5_0_mmvq_b64_r4.deinit();
        var q5_1 = try gpu_matvec.MatvecPipeline.initQ5_1Q8_1(ctx);
        errdefer q5_1.deinit();
        var q5_1_mmvq_b64_r1 = try gpu_matvec.MatvecPipeline.initQ5_1Q8_1Mmvq(ctx, .{
            .block_size = 64,
            .num_rows = 1,
            .num_cols = 1,
        });
        errdefer q5_1_mmvq_b64_r1.deinit();
        var q5_1_mmvq_b64_r2 = try gpu_matvec.MatvecPipeline.initQ5_1Q8_1Mmvq(ctx, .{
            .block_size = 64,
            .num_rows = 2,
            .num_cols = 1,
        });
        errdefer q5_1_mmvq_b64_r2.deinit();
        var q5_1_mmvq_b64_r4 = try gpu_matvec.MatvecPipeline.initQ5_1Q8_1Mmvq(ctx, .{
            .block_size = 64,
            .num_rows = 4,
            .num_cols = 1,
        });
        errdefer q5_1_mmvq_b64_r4.deinit();
        var q5_k = try gpu_matvec.MatvecPipeline.initQ5KQ8_1(ctx);
        errdefer q5_k.deinit();
        var q5_k_mmvq_b64_r1 = try gpu_matvec.MatvecPipeline.initQ5KQ8_1Mmvq(ctx, .{
            .block_size = 64,
            .num_rows = 1,
            .num_cols = 1,
        });
        errdefer q5_k_mmvq_b64_r1.deinit();
        var q5_k_mmvq_b64_r2 = try gpu_matvec.MatvecPipeline.initQ5KQ8_1Mmvq(ctx, .{
            .block_size = 64,
            .num_rows = 2,
            .num_cols = 1,
        });
        errdefer q5_k_mmvq_b64_r2.deinit();
        var q5_k_mmvq_b64_r4 = try gpu_matvec.MatvecPipeline.initQ5KQ8_1Mmvq(ctx, .{
            .block_size = 64,
            .num_rows = 4,
            .num_cols = 1,
        });
        errdefer q5_k_mmvq_b64_r4.deinit();
        var q6_k = try gpu_matvec.MatvecPipeline.initQ6KQ8_1(ctx);
        errdefer q6_k.deinit();
        var q6_k_fast = try gpu_matvec.MatvecPipeline.initQ6KQ8_1Fast(ctx);
        errdefer q6_k_fast.deinit();
        var q6_k_mmvq_b32_r1 = try gpu_matvec.MatvecPipeline.initQ6KQ8_1Mmvq(ctx, .{
            .block_size = 32,
            .num_rows = 1,
            .num_cols = 1,
        });
        errdefer q6_k_mmvq_b32_r1.deinit();
        var q6_k_mmvq_b64_r1 = try gpu_matvec.MatvecPipeline.initQ6KQ8_1Mmvq(ctx, .{
            .block_size = 64,
            .num_rows = 1,
            .num_cols = 1,
        });
        errdefer q6_k_mmvq_b64_r1.deinit();
        var q6_k_mmvq_b64_r2 = try gpu_matvec.MatvecPipeline.initQ6KQ8_1Mmvq(ctx, .{
            .block_size = 64,
            .num_rows = 2,
            .num_cols = 1,
        });
        errdefer q6_k_mmvq_b64_r2.deinit();
        var q6_k_mmvq_b64_r4 = try gpu_matvec.MatvecPipeline.initQ6KQ8_1Mmvq(ctx, .{
            .block_size = 64,
            .num_rows = 4,
            .num_cols = 1,
        });
        errdefer q6_k_mmvq_b64_r4.deinit();
        var iq4_nl = try gpu_matvec.MatvecPipeline.initIQ4NLQ8_1(ctx);
        errdefer iq4_nl.deinit();
        var quant = try gpu_matvec.QuantizeQ8_1Pipeline.init(ctx);
        errdefer quant.deinit();
        return .{
            .q3_k = q3_k,
            .q3_k_mmvq_b32_r1 = q3_k_mmvq_b32_r1,
            .q3_k_mmvq_b64_r1 = q3_k_mmvq_b64_r1,
            .q4_k = q4_k,
            .q4_k_mmvq_b32_r1 = q4_k_mmvq_b32_r1,
            .q4_k_mmvq_b64_r1 = q4_k_mmvq_b64_r1,
            .q4_k_mmvq_b64_r2 = q4_k_mmvq_b64_r2,
            .q4_k_mmvq_b64_r4 = q4_k_mmvq_b64_r4,
            .q4_k_r4 = q4_k_r4,
            .q5_0 = q5_0,
            .q5_0_mmvq_b64_r1 = q5_0_mmvq_b64_r1,
            .q5_0_mmvq_b64_r2 = q5_0_mmvq_b64_r2,
            .q5_0_mmvq_b64_r4 = q5_0_mmvq_b64_r4,
            .q5_1 = q5_1,
            .q5_1_mmvq_b64_r1 = q5_1_mmvq_b64_r1,
            .q5_1_mmvq_b64_r2 = q5_1_mmvq_b64_r2,
            .q5_1_mmvq_b64_r4 = q5_1_mmvq_b64_r4,
            .q5_k = q5_k,
            .q5_k_mmvq_b64_r1 = q5_k_mmvq_b64_r1,
            .q5_k_mmvq_b64_r2 = q5_k_mmvq_b64_r2,
            .q5_k_mmvq_b64_r4 = q5_k_mmvq_b64_r4,
            .q6_k = q6_k,
            .q6_k_fast = q6_k_fast,
            .q6_k_mmvq_b32_r1 = q6_k_mmvq_b32_r1,
            .q6_k_mmvq_b64_r1 = q6_k_mmvq_b64_r1,
            .q6_k_mmvq_b64_r2 = q6_k_mmvq_b64_r2,
            .q6_k_mmvq_b64_r4 = q6_k_mmvq_b64_r4,
            .iq4_nl = iq4_nl,
            .quant = quant,
        };
    }

    fn deinit(self: *BenchPipelines) void {
        self.quant.deinit();
        self.iq4_nl.deinit();
        self.q6_k_mmvq_b64_r4.deinit();
        self.q6_k_mmvq_b64_r2.deinit();
        self.q6_k_mmvq_b64_r1.deinit();
        self.q6_k_mmvq_b32_r1.deinit();
        self.q6_k_fast.deinit();
        self.q6_k.deinit();
        self.q5_k_mmvq_b64_r4.deinit();
        self.q5_k_mmvq_b64_r2.deinit();
        self.q5_k_mmvq_b64_r1.deinit();
        self.q5_k.deinit();
        self.q5_1_mmvq_b64_r4.deinit();
        self.q5_1_mmvq_b64_r2.deinit();
        self.q5_1_mmvq_b64_r1.deinit();
        self.q5_1.deinit();
        self.q5_0_mmvq_b64_r4.deinit();
        self.q5_0_mmvq_b64_r2.deinit();
        self.q5_0_mmvq_b64_r1.deinit();
        self.q5_0.deinit();
        self.q4_k_r4.deinit();
        self.q4_k_mmvq_b64_r4.deinit();
        self.q4_k_mmvq_b64_r2.deinit();
        self.q4_k_mmvq_b64_r1.deinit();
        self.q4_k_mmvq_b32_r1.deinit();
        self.q4_k.deinit();
        self.q3_k_mmvq_b64_r1.deinit();
        self.q3_k_mmvq_b32_r1.deinit();
        self.q3_k.deinit();
    }

    fn pipelineFor(self: *const BenchPipelines, t: gguf_types.GgmlType) ?*const gpu_matvec.MatvecPipeline {
        return switch (t) {
            .q3_k => &self.q3_k,
            .q4_k => &self.q4_k,
            .q5_0 => &self.q5_0,
            .q5_1 => &self.q5_1,
            .q5_k => &self.q5_k,
            .q6_k => &self.q6_k_fast,
            .iq4_nl => &self.iq4_nl,
            else => null,
        };
    }
};

fn cmdBenchRmsnorm(out: *std.Io.Writer, opts: RmsnormBenchOptions, io: std.Io, gpa: std.mem.Allocator) !void {
    if (opts.n % 256 != 0) return error.InvalidRmsnormShape;

    var ctx = try gpu_ctx.GpuContext.init();
    defer ctx.deinit();

    var rms_256 = try gpu_matvec.RmsnormPipeline.init(&ctx);
    defer rms_256.deinit();
    var rms_128 = try gpu_matvec.RmsnormPipeline.initR128(&ctx);
    defer rms_128.deinit();
    var add_256 = try gpu_matvec.AddRmsnormPipeline.init(&ctx);
    defer add_256.deinit();
    var add_128 = try gpu_matvec.AddRmsnormPipeline.initR128(&ctx);
    defer add_128.deinit();

    const n: usize = opts.n;
    const x = try gpa.alloc(f32, n);
    defer gpa.free(x);
    const b = try gpa.alloc(f32, n);
    defer gpa.free(b);
    const w = try gpa.alloc(f32, n);
    defer gpa.free(w);
    for (x, 0..) |*v, i| {
        const centered: i32 = @as(i32, @intCast(i % 41)) - 20;
        v.* = @as(f32, @floatFromInt(centered)) / 13.0;
    }
    for (b, 0..) |*v, i| {
        const centered: i32 = @as(i32, @intCast(i % 37)) - 18;
        v.* = @as(f32, @floatFromInt(centered)) / 17.0;
    }
    for (w, 0..) |*v, i| {
        const centered: i32 = @as(i32, @intCast(i % 29)) - 14;
        v.* = 1.0 + @as(f32, @floatFromInt(centered)) / 64.0;
    }

    var x_buf = try uploadDeviceLocalBytes(&ctx, std.mem.sliceAsBytes(x), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer x_buf.deinit();
    var b_buf = try uploadDeviceLocalBytes(&ctx, std.mem.sliceAsBytes(b), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer b_buf.deinit();
    var w_buf = try uploadDeviceLocalBytes(&ctx, std.mem.sliceAsBytes(w), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer w_buf.deinit();
    var y_buf = try gpu_buffer.GpuBuffer.initDeviceLocal(&ctx, n * @sizeOf(f32), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer y_buf.deinit();

    var rms_256_set = try rms_256.allocSet(&x_buf, &w_buf, &y_buf);
    defer _ = vk.vkFreeDescriptorSets(ctx.device, rms_256.desc_pool, 1, &rms_256_set);
    var rms_128_set = try rms_128.allocSet(&x_buf, &w_buf, &y_buf);
    defer _ = vk.vkFreeDescriptorSets(ctx.device, rms_128.desc_pool, 1, &rms_128_set);
    var add_256_set = try add_256.allocSet(&x_buf, &b_buf, &w_buf, &y_buf);
    defer _ = vk.vkFreeDescriptorSets(ctx.device, add_256.desc_pool, 1, &add_256_set);
    var add_128_set = try add_128.allocSet(&x_buf, &b_buf, &w_buf, &y_buf);
    defer _ = vk.vkFreeDescriptorSets(ctx.device, add_128.desc_pool, 1, &add_128_set);

    try out.print("GPU RMSNorm microbench: iters={} n={}\n", .{ opts.iters, opts.n });
    try out.print("{s:20} {s:>10} {s:>10} {s:>10}\n", .{ "name", "wall_us", "gpu_us", "host_us" });
    try benchRmsnormPipeline(out, &ctx, &rms_256, rms_256_set, opts.n, opts.iters, "rmsnorm.256", io);
    try benchRmsnormPipeline(out, &ctx, &rms_128, rms_128_set, opts.n, opts.iters, "rmsnorm.128", io);
    try benchAddRmsnormPipeline(out, &ctx, &add_256, add_256_set, opts.n, opts.iters, "add_rmsnorm.256", io);
    try benchAddRmsnormPipeline(out, &ctx, &add_128, add_128_set, opts.n, opts.iters, "add_rmsnorm.128", io);
}

fn benchRmsnormPipeline(
    out: *std.Io.Writer,
    ctx: *const gpu_ctx.GpuContext,
    pipeline: *const gpu_matvec.RmsnormPipeline,
    dset: vk.VkDescriptorSet,
    n: u32,
    iters: u32,
    label: []const u8,
    io: std.Io,
) !void {
    {
        const cmd = try ctx.beginBatch();
        const p = ctx.profileBegin(cmd, label);
        pipeline.recordWithSet(cmd, dset, n, 1e-6, false);
        ctx.profileEnd(cmd, p);
        try ctx.submitBatch(cmd);
    }
    const before = ctx.profileStats(label);
    const clk = std.Io.Clock.real;
    const t0 = clk.now(io);
    for (0..iters) |_| {
        const cmd = try ctx.beginBatch();
        const p = ctx.profileBegin(cmd, label);
        pipeline.recordWithSet(cmd, dset, n, 1e-6, false);
        ctx.profileEnd(cmd, p);
        try ctx.submitBatch(cmd);
    }
    const t1 = clk.now(io);
    const after = ctx.profileStats(label);
    try printGpuBenchLine(out, label, t0.durationTo(t1).nanoseconds, before, after, iters);
}

fn benchAddRmsnormPipeline(
    out: *std.Io.Writer,
    ctx: *const gpu_ctx.GpuContext,
    pipeline: *const gpu_matvec.AddRmsnormPipeline,
    dset: vk.VkDescriptorSet,
    n: u32,
    iters: u32,
    label: []const u8,
    io: std.Io,
) !void {
    {
        const cmd = try ctx.beginBatch();
        const p = ctx.profileBegin(cmd, label);
        pipeline.recordWithSet(cmd, dset, n, 1e-6, false, false);
        ctx.profileEnd(cmd, p);
        try ctx.submitBatch(cmd);
    }
    const before = ctx.profileStats(label);
    const clk = std.Io.Clock.real;
    const t0 = clk.now(io);
    for (0..iters) |_| {
        const cmd = try ctx.beginBatch();
        const p = ctx.profileBegin(cmd, label);
        pipeline.recordWithSet(cmd, dset, n, 1e-6, false, false);
        ctx.profileEnd(cmd, p);
        try ctx.submitBatch(cmd);
    }
    const t1 = clk.now(io);
    const after = ctx.profileStats(label);
    try printGpuBenchLine(out, label, t0.durationTo(t1).nanoseconds, before, after, iters);
}

fn printGpuBenchLine(
    out: *std.Io.Writer,
    label: []const u8,
    wall_ns: i128,
    before: gpu_ctx.ProfileStats,
    after: gpu_ctx.ProfileStats,
    iters: u32,
) !void {
    const wall_us = @as(f64, @floatFromInt(wall_ns)) / @as(f64, @floatFromInt(iters)) / 1000.0;
    const count_delta = after.count - before.count;
    const gpu_ns = after.total_ns - before.total_ns;
    const gpu_us = if (count_delta == 0)
        0.0
    else
        @as(f64, @floatFromInt(gpu_ns)) / @as(f64, @floatFromInt(count_delta)) / 1000.0;
    try out.print("{s:20} {d:10.2} {d:10.2} {d:10.2}\n", .{ label, wall_us, gpu_us, @max(0.0, wall_us - gpu_us) });
}

fn cmdBenchMatvec(
    out: *std.Io.Writer,
    path: []const u8,
    opts: MatvecBenchOptions,
    io: std.Io,
    gpa: std.mem.Allocator,
) !void {
    var reader = try gguf_reader.GgufReader.open(path, io, gpa);
    defer reader.deinit();
    if (!g4_loader.isGemma4(&reader)) return error.UnsupportedModel;

    const cfg = try g4_loader.configFromGguf(&reader, gpa);
    var weights = try g4_loader.loadWeights(&reader, cfg, gpa);
    defer weights.deinit();

    var ctx = try gpu_ctx.GpuContext.init();
    defer ctx.deinit();
    var pipes = try BenchPipelines.init(&ctx);
    defer pipes.deinit();

    try out.print("GPU matvec microbench: iters={} target={s} reuse_descriptor={}\n", .{
        opts.iters, opts.target orelse "all", opts.reuse_descriptor,
    });
    try out.print("{s:32} {s:7} {s:>9} {s:>9} {s:>10} {s:>10} {s:>10} {s:>10}\n", .{
        "name", "type", "rows", "cols", "wall_us", "gpu_us", "cpu_us", "GB/s",
    });

    var ran: usize = 0;
    try benchIfSelected(out, &ctx, &pipes, opts, &ran, "lm_head", weights.lm_head, gpa, io);
    try benchWithPipelineIfSelected(out, &ctx, &pipes.q6_k, &pipes.quant, opts, &ran, "lm_head.q6_old", weights.lm_head, .q6_k, gpa, io);
    try benchWithPipelineIfSelected(out, &ctx, &pipes.q6_k_fast, &pipes.quant, opts, &ran, "lm_head.fast", weights.lm_head, .q6_k, gpa, io);
    try benchWithPipelineIfSelected(out, &ctx, &pipes.q6_k_mmvq_b32_r1, &pipes.quant, opts, &ran, "lm_head.mmvq.b32.r1", weights.lm_head, .q6_k, gpa, io);
    try benchWithPipelineIfSelected(out, &ctx, &pipes.q6_k_mmvq_b64_r1, &pipes.quant, opts, &ran, "lm_head.mmvq.b64.r1", weights.lm_head, .q6_k, gpa, io);
    try benchWithPipelineIfSelected(out, &ctx, &pipes.q6_k_mmvq_b64_r2, &pipes.quant, opts, &ran, "lm_head.mmvq.b64.r2", weights.lm_head, .q6_k, gpa, io);
    try benchWithPipelineIfSelected(out, &ctx, &pipes.q6_k_mmvq_b64_r4, &pipes.quant, opts, &ran, "lm_head.mmvq.b64.r4", weights.lm_head, .q6_k, gpa, io);
    try benchIfSelected(out, &ctx, &pipes, opts, &ran, "L0.attn_q", weights.layers[0].wq, gpa, io);
    try benchWithPipelineIfSelected(out, &ctx, &pipes.q4_k_mmvq_b32_r1, &pipes.quant, opts, &ran, "L0.attn_q.mmvq.b32.r1", weights.layers[0].wq, .q4_k, gpa, io);
    try benchWithPipelineIfSelected(out, &ctx, &pipes.q4_k_mmvq_b64_r1, &pipes.quant, opts, &ran, "L0.attn_q.mmvq.b64.r1", weights.layers[0].wq, .q4_k, gpa, io);
    try benchWithPipelineIfSelected(out, &ctx, &pipes.q4_k_mmvq_b64_r2, &pipes.quant, opts, &ran, "L0.attn_q.mmvq.b64.r2", weights.layers[0].wq, .q4_k, gpa, io);
    try benchWithPipelineIfSelected(out, &ctx, &pipes.q4_k_mmvq_b64_r4, &pipes.quant, opts, &ran, "L0.attn_q.mmvq.b64.r4", weights.layers[0].wq, .q4_k, gpa, io);
    try benchWithPipelineIfSelected(out, &ctx, &pipes.q4_k_r4, &pipes.quant, opts, &ran, "L0.attn_q.r4", weights.layers[0].wq, .q4_k, gpa, io);
    try benchIfSelected(out, &ctx, &pipes, opts, &ran, "L0.attn_v", weights.layers[0].wv.?, gpa, io);
    try benchWithPipelineIfSelected(out, &ctx, &pipes.q4_k_mmvq_b32_r1, &pipes.quant, opts, &ran, "L0.attn_v.mmvq.b32.r1", weights.layers[0].wv.?, .q4_k, gpa, io);
    try benchWithPipelineIfSelected(out, &ctx, &pipes.q4_k_mmvq_b64_r1, &pipes.quant, opts, &ran, "L0.attn_v.mmvq.b64.r1", weights.layers[0].wv.?, .q4_k, gpa, io);
    try benchWithPipelineIfSelected(out, &ctx, &pipes.q4_k_mmvq_b64_r2, &pipes.quant, opts, &ran, "L0.attn_v.mmvq.b64.r2", weights.layers[0].wv.?, .q4_k, gpa, io);
    try benchWithPipelineIfSelected(out, &ctx, &pipes.q4_k_mmvq_b64_r4, &pipes.quant, opts, &ran, "L0.attn_v.mmvq.b64.r4", weights.layers[0].wv.?, .q4_k, gpa, io);
    try benchWithPipelineIfSelected(out, &ctx, &pipes.q4_k_r4, &pipes.quant, opts, &ran, "L0.attn_v.r4", weights.layers[0].wv.?, .q4_k, gpa, io);
    try benchIfSelected(out, &ctx, &pipes, opts, &ran, "L3.attn_v", weights.layers[3].wv.?, gpa, io);
    try benchWithPipelineIfSelected(out, &ctx, &pipes.q5_k_mmvq_b64_r1, &pipes.quant, opts, &ran, "L3.attn_v.mmvq.b64.r1", weights.layers[3].wv.?, .q5_k, gpa, io);
    try benchWithPipelineIfSelected(out, &ctx, &pipes.q5_k_mmvq_b64_r2, &pipes.quant, opts, &ran, "L3.attn_v.mmvq.b64.r2", weights.layers[3].wv.?, .q5_k, gpa, io);
    try benchWithPipelineIfSelected(out, &ctx, &pipes.q5_k_mmvq_b64_r4, &pipes.quant, opts, &ran, "L3.attn_v.mmvq.b64.r4", weights.layers[3].wv.?, .q5_k, gpa, io);
    try benchIfSelected(out, &ctx, &pipes, opts, &ran, "L5.attn_q", weights.layers[5].wq, gpa, io);
    try benchWithPipelineIfSelected(out, &ctx, &pipes.q3_k_mmvq_b32_r1, &pipes.quant, opts, &ran, "L5.attn_q.mmvq.b32.r1", weights.layers[5].wq, .q3_k, gpa, io);
    try benchWithPipelineIfSelected(out, &ctx, &pipes.q3_k_mmvq_b64_r1, &pipes.quant, opts, &ran, "L5.attn_q.mmvq.b64.r1", weights.layers[5].wq, .q3_k, gpa, io);
    try benchIfSelected(out, &ctx, &pipes, opts, &ran, "L0.dense_down", weights.layers[0].w_down, gpa, io);
    try benchWithPipelineIfSelected(out, &ctx, &pipes.q5_1_mmvq_b64_r1, &pipes.quant, opts, &ran, "L0.dense_down.mmvq.b64.r1", weights.layers[0].w_down, .q5_1, gpa, io);
    try benchWithPipelineIfSelected(out, &ctx, &pipes.q5_1_mmvq_b64_r2, &pipes.quant, opts, &ran, "L0.dense_down.mmvq.b64.r2", weights.layers[0].w_down, .q5_1, gpa, io);
    try benchWithPipelineIfSelected(out, &ctx, &pipes.q5_1_mmvq_b64_r4, &pipes.quant, opts, &ran, "L0.dense_down.mmvq.b64.r4", weights.layers[0].w_down, .q5_1, gpa, io);
    try benchIfSelected(out, &ctx, &pipes, opts, &ran, "L5.dense_down", weights.layers[5].w_down, gpa, io);
    try benchWithPipelineIfSelected(out, &ctx, &pipes.q5_0_mmvq_b64_r1, &pipes.quant, opts, &ran, "L5.dense_down.mmvq.b64.r1", weights.layers[5].w_down, .q5_0, gpa, io);
    try benchWithPipelineIfSelected(out, &ctx, &pipes.q5_0_mmvq_b64_r2, &pipes.quant, opts, &ran, "L5.dense_down.mmvq.b64.r2", weights.layers[5].w_down, .q5_0, gpa, io);
    try benchWithPipelineIfSelected(out, &ctx, &pipes.q5_0_mmvq_b64_r4, &pipes.quant, opts, &ran, "L5.dense_down.mmvq.b64.r4", weights.layers[5].w_down, .q5_0, gpa, io);
    try benchExpertDownIfSelected(out, &ctx, &pipes, opts, &ran, "L0.expert_down", weights.layers[0].down_exps, 0, cfg.d_model, cfg.d_expert, gpa, io);
    try benchExpertDownWithPipelineIfSelected(out, &ctx, &pipes.q5_1_mmvq_b64_r1, &pipes.quant, opts, &ran, "L0.expert_down.mmvq.b64.r1", weights.layers[0].down_exps, 0, cfg.d_model, cfg.d_expert, .q5_1, gpa, io);
    try benchExpertDownWithPipelineIfSelected(out, &ctx, &pipes.q5_1_mmvq_b64_r2, &pipes.quant, opts, &ran, "L0.expert_down.mmvq.b64.r2", weights.layers[0].down_exps, 0, cfg.d_model, cfg.d_expert, .q5_1, gpa, io);
    try benchExpertDownWithPipelineIfSelected(out, &ctx, &pipes.q5_1_mmvq_b64_r4, &pipes.quant, opts, &ran, "L0.expert_down.mmvq.b64.r4", weights.layers[0].down_exps, 0, cfg.d_model, cfg.d_expert, .q5_1, gpa, io);
    try benchExpertDownIfSelected(out, &ctx, &pipes, opts, &ran, "L10.expert_down", weights.layers[10].down_exps, 0, cfg.d_model, cfg.d_expert, gpa, io);

    if (ran == 0) {
        try out.print("no matching benchmark target\n", .{});
    }
}

fn cmdBenchMoe(
    out: *std.Io.Writer,
    path: []const u8,
    opts: MoeBenchOptions,
    io: std.Io,
    gpa: std.mem.Allocator,
) !void {
    var reader = try gguf_reader.GgufReader.open(path, io, gpa);
    defer reader.deinit();
    if (!g4_loader.isGemma4(&reader)) return error.UnsupportedModel;

    const cfg = try g4_loader.configFromGguf(&reader, gpa);
    if (opts.layer >= cfg.n_layers) return error.InvalidLayer;

    var weights = try g4_loader.loadWeights(&reader, cfg, gpa);
    defer weights.deinit();

    setOomAdj(500);
    std.debug.print("uploading weights to GPU VRAM...\n", .{});
    var gpu_weights = try g4_gpu.GpuWeights.init(&weights, cfg, gpa);
    defer gpu_weights.deinit();

    const layer = opts.layer;
    const lw = weights.layers[layer];
    const top_n = cfg.n_experts_used;

    const moe_in = try gpa.alloc(f32, cfg.d_model);
    defer gpa.free(moe_in);
    const router_out = try gpa.alloc(f32, cfg.n_experts);
    defer gpa.free(router_out);
    const moe_buf = try gpa.alloc(f32, cfg.d_model);
    defer gpa.free(moe_buf);
    const x = try gpa.alloc(f32, cfg.d_model);
    defer gpa.free(x);
    const dense_ffn = try gpa.alloc(f32, cfg.d_model);
    defer gpa.free(dense_ffn);
    const top_idx = try gpa.alloc(usize, top_n);
    defer gpa.free(top_idx);

    for (moe_in, 0..) |*v, i| {
        const centered: i32 = @as(i32, @intCast(i % 31)) - 15;
        v.* = @as(f32, @floatFromInt(centered)) / 16.0;
    }
    for (x, dense_ffn, 0..) |*xv, *dv, i| {
        const x_centered: i32 = @as(i32, @intCast(i % 43)) - 21;
        const d_centered: i32 = @as(i32, @intCast(i % 37)) - 18;
        xv.* = @as(f32, @floatFromInt(x_centered)) / 32.0;
        dv.* = @as(f32, @floatFromInt(d_centered)) / 32.0;
    }
    @memset(router_out, 0.0);
    for (top_idx, 0..) |*e, i| {
        e.* = i;
        router_out[i] = 1.0 / @as(f32, @floatFromInt(top_n));
    }

    try out.print("GPU MoE batch microbench: layer={} iters={} experts_used={}/{} d_model={} d_expert={}\n", .{
        layer, opts.iters, top_n, cfg.n_experts, cfg.d_model, cfg.d_expert,
    });
    try out.print("  gate_up_type={s} down_type={s} skip_readback={} tail={}\n", .{
        lw.gate_up_exps.type_.label(), lw.down_exps.type_.label(), opts.skip_readback, opts.tail,
    });

    if (opts.tail) {
        try gpu_weights.shared_vec.?.upload(std.mem.sliceAsBytes(x));
        try gpu_weights.dense_ffn_out_buf.?.upload(std.mem.sliceAsBytes(dense_ffn));
    }

    @memset(moe_buf, 0.0);
    const tail_params: ?g4_gpu.MoeTailParams = if (opts.tail) .{
        .eps = cfg.eps,
        .x = x,
        .dense_ffn = dense_ffn,
        .layer_output_scale = lw.layer_output_scale,
        .x_buf_current = true,
        .x_vram_current = false,
        .dense_buf_current = true,
    } else null;
    try gpu_weights.runExpertBatch(layer, top_idx, lw.gate_up_exps.type_, lw.down_exps.type_, lw.down_exps_scale, moe_in, router_out, moe_buf, opts.skip_readback, tail_params, null);

    const labels = [_][]const u8{
        "moe.quantize_input",
        "moe.fused_gate_up",
        "moe.quantize_mid",
        "moe.down",
        "moe.down_sum",
        "moe.accum",
        "moe.post_norm",
        "ffn_moe.add_post_norm",
        "ffn_moe.combine",
        "ffn_moe.post_norm",
        "ffn_moe.residual_add_scale",
        "ffn_moe.residual_add",
        "ffn_moe.layer_scale",
    };
    var before: [labels.len]gpu_ctx.ProfileStats = undefined;
    for (labels, 0..) |label, i| before[i] = gpu_weights.ctx.profileStats(label);

    const clk = std.Io.Clock.real;
    const t0 = clk.now(io);
    for (0..opts.iters) |_| {
        @memset(moe_buf, 0.0);
        try gpu_weights.runExpertBatch(layer, top_idx, lw.gate_up_exps.type_, lw.down_exps.type_, lw.down_exps_scale, moe_in, router_out, moe_buf, opts.skip_readback, tail_params, null);
    }
    const t1 = clk.now(io);

    var after: [labels.len]gpu_ctx.ProfileStats = undefined;
    for (labels, 0..) |label, i| after[i] = gpu_weights.ctx.profileStats(label);

    const ns = t0.durationTo(t1).nanoseconds;
    const wall_us = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(opts.iters)) / 1000.0;
    try out.print("{s:28} {s:>10} {s:>12} {s:>12}\n", .{
        "phase", "dispatches", "avg_us/disp", "avg_us/iter",
    });

    var gpu_total_ns: u64 = 0;
    for (labels, 0..) |label, i| {
        const count_delta = after[i].count - before[i].count;
        const ns_delta = after[i].total_ns - before[i].total_ns;
        gpu_total_ns += ns_delta;
        const per_dispatch = if (count_delta == 0)
            0.0
        else
            @as(f64, @floatFromInt(ns_delta)) / @as(f64, @floatFromInt(count_delta)) / 1000.0;
        const per_iter = @as(f64, @floatFromInt(ns_delta)) / @as(f64, @floatFromInt(opts.iters)) / 1000.0;
        try out.print("{s:28} {d:10} {d:12.2} {d:12.2}\n", .{
            label, count_delta, per_dispatch, per_iter,
        });
    }

    const gpu_us = @as(f64, @floatFromInt(gpu_total_ns)) / @as(f64, @floatFromInt(opts.iters)) / 1000.0;
    try out.print("total wall_us/iter={d:.2} gpu_phase_us/iter={d:.2} host_submit_us/iter={d:.2}\n", .{
        wall_us, gpu_us, @max(0.0, wall_us - gpu_us),
    });

    try benchExpertGateUpIdShape(out, &gpu_weights, &lw, cfg, layer, top_idx, opts.iters, io);
    try benchExpertDownIdShape(out, &gpu_weights, &lw, cfg, layer, top_idx, opts.iters, io);
}

fn benchExpertGateUpIdShape(
    out: *std.Io.Writer,
    gpu_weights: *g4_gpu.GpuWeights,
    lw: *const g4_weights.Gemma4LayerWeights,
    cfg: g4_loader.Gemma4Config,
    layer: usize,
    top_idx: []const usize,
    iters: u32,
    io: std.Io,
) !void {
    if (lw.gate_up_exps.type_ != .q3_k) {
        try out.print("expert-id gate-up shape: unsupported layer={} gate/up type {s}\n", .{
            layer, lw.gate_up_exps.type_.label(),
        });
        return;
    }

    var pipeline = try gpu_matvec.ExpertGateUpIdPipeline.initQ3KQ8_1(&gpu_weights.ctx);
    defer pipeline.deinit();
    var pipeline_r2 = try gpu_matvec.ExpertGateUpIdPipeline.initQ3KQ8_1R2(&gpu_weights.ctx);
    defer pipeline_r2.deinit();
    var pipeline_r4 = try gpu_matvec.ExpertGateUpIdPipeline.initQ3KQ8_1R4(&gpu_weights.ctx);
    defer pipeline_r4.deinit();

    const flat_session_opt = try gpu_matvec.MatvecSession.initFromRaw(
        &gpu_weights.ctx,
        lw.gate_up_exps.data,
        lw.gate_up_exps.type_,
        @intCast(cfg.n_experts * 2 * cfg.d_expert),
        @intCast(cfg.d_model),
    );
    var flat_session = flat_session_opt orelse return error.UnsupportedModel;
    defer flat_session.deinit();

    var ids_host: [16]u32 = undefined;
    for (top_idx, 0..) |idx, i| ids_host[i] = @intCast(idx);
    var ids_buf = try uploadDeviceLocalBytes(&gpu_weights.ctx, std.mem.sliceAsBytes(ids_host[0..top_idx.len]), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer ids_buf.deinit();

    const act_f32_buf = try gpu_buffer.GpuBuffer.initHostCoherent(&gpu_weights.ctx, cfg.d_model * @sizeOf(f32), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer {
        var b = act_f32_buf;
        b.unmap();
        b.deinit();
    }
    const act_f32 = try act_f32_buf.mapSlice(f32, cfg.d_model);
    for (act_f32, 0..) |*v, i| {
        const centered: i32 = @as(i32, @intCast(i % 31)) - 15;
        v.* = @as(f32, @floatFromInt(centered)) / 16.0;
    }

    var act_q8_buf = try gpu_buffer.GpuBuffer.initDeviceLocal(&gpu_weights.ctx, gpu_matvec.q8_1OutBytes(@intCast(cfg.d_model)), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer act_q8_buf.deinit();

    var out_buf = try gpu_buffer.GpuBuffer.initDeviceLocal(&gpu_weights.ctx, top_idx.len * cfg.d_expert * @sizeOf(f32), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer out_buf.deinit();

    {
        const cmd = try gpu_weights.ctx.beginBatch();
        const ds = try gpu_weights.pl_quantize_q8_1.record(cmd, &act_f32_buf, &act_q8_buf, @intCast(cfg.d_model));
        try gpu_weights.ctx.submitBatch(cmd);
        var ds_mut = ds;
        _ = vk.vkFreeDescriptorSets(gpu_weights.ctx.device, gpu_weights.pl_quantize_q8_1.desc_pool, 1, &ds_mut);
    }

    try warmExpertGateUpId(&gpu_weights.ctx, &pipeline, &flat_session.mat_buf, &act_q8_buf, &ids_buf, &out_buf, @intCast(cfg.d_expert), @intCast(cfg.d_model), @intCast(top_idx.len), "moe.gate_up_id");
    try warmExpertGateUpId(&gpu_weights.ctx, &pipeline_r2, &flat_session.mat_buf, &act_q8_buf, &ids_buf, &out_buf, @intCast(cfg.d_expert), @intCast(cfg.d_model), @intCast(top_idx.len), "moe.gate_up_id.r2");
    try warmExpertGateUpId(&gpu_weights.ctx, &pipeline_r4, &flat_session.mat_buf, &act_q8_buf, &ids_buf, &out_buf, @intCast(cfg.d_expert), @intCast(cfg.d_model), @intCast(top_idx.len), "moe.gate_up_id.r4");

    const before = gpu_weights.ctx.profileStats("moe.gate_up_id");
    const clk = std.Io.Clock.real;
    const t0 = clk.now(io);
    for (0..iters) |_| {
        const cmd = try gpu_weights.ctx.beginBatch();
        const p_gu = gpu_weights.ctx.profileBegin(cmd, "moe.gate_up_id");
        const ds = try pipeline.record(cmd, &flat_session.mat_buf, &act_q8_buf, &ids_buf, &out_buf, @intCast(cfg.d_expert), @intCast(cfg.d_model), @intCast(top_idx.len));
        gpu_weights.ctx.profileEnd(cmd, p_gu);
        try gpu_weights.ctx.submitBatch(cmd);
        var ds_mut = ds;
        _ = vk.vkFreeDescriptorSets(gpu_weights.ctx.device, pipeline.desc_pool, 1, &ds_mut);
    }
    const t1 = clk.now(io);
    const after = gpu_weights.ctx.profileStats("moe.gate_up_id");

    const wall_ns = t0.durationTo(t1).nanoseconds;
    const wall_us = @as(f64, @floatFromInt(wall_ns)) / @as(f64, @floatFromInt(iters)) / 1000.0;
    const count_delta = after.count - before.count;
    const gpu_ns = after.total_ns - before.total_ns;
    const gpu_us = if (count_delta == 0)
        0.0
    else
        @as(f64, @floatFromInt(gpu_ns)) / @as(f64, @floatFromInt(count_delta)) / 1000.0;

    try out.print("expert-id gate-up shape: type={s} active={} wall_us={d:.2} gpu_us={d:.2} host_us={d:.2}\n", .{
        lw.gate_up_exps.type_.label(), top_idx.len, wall_us, gpu_us, @max(0.0, wall_us - gpu_us),
    });

    const before_r2 = gpu_weights.ctx.profileStats("moe.gate_up_id.r2");
    const t2 = clk.now(io);
    for (0..iters) |_| {
        const cmd = try gpu_weights.ctx.beginBatch();
        const p_gu = gpu_weights.ctx.profileBegin(cmd, "moe.gate_up_id.r2");
        const ds = try pipeline_r2.record(cmd, &flat_session.mat_buf, &act_q8_buf, &ids_buf, &out_buf, @intCast(cfg.d_expert), @intCast(cfg.d_model), @intCast(top_idx.len));
        gpu_weights.ctx.profileEnd(cmd, p_gu);
        try gpu_weights.ctx.submitBatch(cmd);
        var ds_mut = ds;
        _ = vk.vkFreeDescriptorSets(gpu_weights.ctx.device, pipeline_r2.desc_pool, 1, &ds_mut);
    }
    const t3 = clk.now(io);
    const after_r2 = gpu_weights.ctx.profileStats("moe.gate_up_id.r2");

    const wall_r2_ns = t2.durationTo(t3).nanoseconds;
    const wall_r2_us = @as(f64, @floatFromInt(wall_r2_ns)) / @as(f64, @floatFromInt(iters)) / 1000.0;
    const count_r2_delta = after_r2.count - before_r2.count;
    const gpu_r2_ns = after_r2.total_ns - before_r2.total_ns;
    const gpu_r2_us = if (count_r2_delta == 0)
        0.0
    else
        @as(f64, @floatFromInt(gpu_r2_ns)) / @as(f64, @floatFromInt(count_r2_delta)) / 1000.0;

    try out.print("expert-id gate-up shape r2: type={s} active={} wall_us={d:.2} gpu_us={d:.2} host_us={d:.2}\n", .{
        lw.gate_up_exps.type_.label(), top_idx.len, wall_r2_us, gpu_r2_us, @max(0.0, wall_r2_us - gpu_r2_us),
    });

    const before_r4 = gpu_weights.ctx.profileStats("moe.gate_up_id.r4");
    const t4 = clk.now(io);
    for (0..iters) |_| {
        const cmd = try gpu_weights.ctx.beginBatch();
        const p_gu = gpu_weights.ctx.profileBegin(cmd, "moe.gate_up_id.r4");
        const ds = try pipeline_r4.record(cmd, &flat_session.mat_buf, &act_q8_buf, &ids_buf, &out_buf, @intCast(cfg.d_expert), @intCast(cfg.d_model), @intCast(top_idx.len));
        gpu_weights.ctx.profileEnd(cmd, p_gu);
        try gpu_weights.ctx.submitBatch(cmd);
        var ds_mut = ds;
        _ = vk.vkFreeDescriptorSets(gpu_weights.ctx.device, pipeline_r4.desc_pool, 1, &ds_mut);
    }
    const t5 = clk.now(io);
    const after_r4 = gpu_weights.ctx.profileStats("moe.gate_up_id.r4");

    const wall_r4_ns = t4.durationTo(t5).nanoseconds;
    const wall_r4_us = @as(f64, @floatFromInt(wall_r4_ns)) / @as(f64, @floatFromInt(iters)) / 1000.0;
    const count_r4_delta = after_r4.count - before_r4.count;
    const gpu_r4_ns = after_r4.total_ns - before_r4.total_ns;
    const gpu_r4_us = if (count_r4_delta == 0)
        0.0
    else
        @as(f64, @floatFromInt(gpu_r4_ns)) / @as(f64, @floatFromInt(count_r4_delta)) / 1000.0;

    try out.print("expert-id gate-up shape r4: type={s} active={} wall_us={d:.2} gpu_us={d:.2} host_us={d:.2}\n", .{
        lw.gate_up_exps.type_.label(), top_idx.len, wall_r4_us, gpu_r4_us, @max(0.0, wall_r4_us - gpu_r4_us),
    });
}

fn warmExpertGateUpId(
    ctx: *const gpu_ctx.GpuContext,
    pipeline: *const gpu_matvec.ExpertGateUpIdPipeline,
    weights_buf: *const gpu_buffer.GpuBuffer,
    acts_buf: *const gpu_buffer.GpuBuffer,
    ids_buf: *const gpu_buffer.GpuBuffer,
    out_buf: *const gpu_buffer.GpuBuffer,
    rows: u32,
    cols: u32,
    active: u32,
    label: []const u8,
) !void {
    const cmd = try ctx.beginBatch();
    const p_gu = ctx.profileBegin(cmd, label);
    const ds = try pipeline.record(cmd, weights_buf, acts_buf, ids_buf, out_buf, rows, cols, active);
    ctx.profileEnd(cmd, p_gu);
    try ctx.submitBatch(cmd);
    var ds_mut = ds;
    _ = vk.vkFreeDescriptorSets(ctx.device, pipeline.desc_pool, 1, &ds_mut);
}

fn benchExpertDownIdShape(
    out: *std.Io.Writer,
    gpu_weights: *g4_gpu.GpuWeights,
    lw: *const g4_weights.Gemma4LayerWeights,
    cfg: g4_loader.Gemma4Config,
    layer: usize,
    top_idx: []const usize,
    iters: u32,
    io: std.Io,
) !void {
    const down_type_id: u32 = @intFromEnum(lw.down_exps.type_);
    const down_type_label = lw.down_exps.type_.label();
    var use_iq4 = layer == 10 or down_type_id == 20 or std.mem.eql(u8, down_type_label, "IQ4_NL");
    var use_q5_0 = down_type_id == 6 or std.mem.eql(u8, down_type_label, "Q5_0");
    var use_q5_1 = layer == 0 or down_type_id == 7 or std.mem.eql(u8, down_type_label, "Q5_1");
    if (down_type_id == 20) use_iq4 = true;
    if (down_type_id == 6) use_q5_0 = true;
    if (down_type_id == 7) use_q5_1 = true;
    var pipeline: gpu_matvec.ExpertDownIdPipeline = undefined;
    if (use_iq4) {
        if (envFlagEnabled("LLMTOY_EXPERT_DN_IQ4_R2", false)) {
            pipeline = try gpu_matvec.ExpertDownIdPipeline.initIQ4NLQ8_1R2(&gpu_weights.ctx);
        } else if (envFlagEnabled("LLMTOY_EXPERT_DN_IQ4_B16", false)) {
            pipeline = try gpu_matvec.ExpertDownIdPipeline.initIQ4NLQ8_1B16(&gpu_weights.ctx);
        } else if (envFlagEnabled("LLMTOY_EXPERT_DN_IQ4_IACC", false)) {
            pipeline = try gpu_matvec.ExpertDownIdPipeline.initIQ4NLQ8_1Iacc(&gpu_weights.ctx);
        } else {
            pipeline = try gpu_matvec.ExpertDownIdPipeline.initIQ4NLQ8_1(&gpu_weights.ctx);
        }
    } else if (use_q5_0) {
        pipeline = try gpu_matvec.ExpertDownIdPipeline.initQ5_0Q8_1(&gpu_weights.ctx);
    } else if (use_q5_1) {
        pipeline = try gpu_matvec.ExpertDownIdPipeline.initQ5_1Q8_1(&gpu_weights.ctx);
    } else {
        try out.print("expert-id down shape: unsupported layer={} down type {s} ({}) q5_0={} q5_1={} iq4={}\n", .{
            layer, lw.down_exps.type_.label(), down_type_id, use_q5_0, use_q5_1, use_iq4,
        });
        return;
    }
    defer pipeline.deinit();

    const flat_session_opt = try gpu_matvec.MatvecSession.initFromRaw(
        &gpu_weights.ctx,
        lw.down_exps.data,
        lw.down_exps.type_,
        @intCast(cfg.n_experts * cfg.d_model),
        @intCast(cfg.d_expert),
    );
    var flat_session = flat_session_opt orelse return error.UnsupportedModel;
    defer flat_session.deinit();

    var ids_host: [16]u32 = undefined;
    for (top_idx, 0..) |idx, i| ids_host[i] = @intCast(idx);
    var ids_buf = try uploadDeviceLocalBytes(&gpu_weights.ctx, std.mem.sliceAsBytes(ids_host[0..top_idx.len]), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer ids_buf.deinit();

    var scales_host: [16]f32 = undefined;
    for (top_idx, 0..) |idx, i| {
        scales_host[i] = lw.down_exps_scale[idx] / @as(f32, @floatFromInt(top_idx.len));
    }
    var scales_buf = try uploadDeviceLocalBytes(&gpu_weights.ctx, std.mem.sliceAsBytes(scales_host[0..top_idx.len]), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer scales_buf.deinit();

    const act_f32_buf = try gpu_buffer.GpuBuffer.initHostCoherent(&gpu_weights.ctx, cfg.d_expert * @sizeOf(f32), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer {
        var b = act_f32_buf;
        b.unmap();
        b.deinit();
    }
    const act_f32 = try act_f32_buf.mapSlice(f32, cfg.d_expert);
    for (act_f32, 0..) |*v, i| {
        const centered: i32 = @as(i32, @intCast(i % 29)) - 14;
        v.* = @as(f32, @floatFromInt(centered)) / 15.0;
    }

    const q8_slot_bytes = gpu_matvec.q8_1OutBytes(@intCast(cfg.d_expert));
    var act_q8_buf = try gpu_buffer.GpuBuffer.initDeviceLocal(&gpu_weights.ctx, top_idx.len * q8_slot_bytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer act_q8_buf.deinit();

    var out_buf = try gpu_buffer.GpuBuffer.initDeviceLocal(&gpu_weights.ctx, top_idx.len * cfg.d_model * @sizeOf(f32), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer out_buf.deinit();

    {
        const cmd = try gpu_weights.ctx.beginBatch();
        var qsets: [16]vk.VkDescriptorSet = undefined;
        for (0..top_idx.len) |slot| {
            qsets[slot] = try gpu_weights.pl_quantize_q8_1.recordToOffset(cmd, &act_f32_buf, &act_q8_buf, slot * q8_slot_bytes, q8_slot_bytes, @intCast(cfg.d_expert));
        }
        try gpu_weights.ctx.submitBatch(cmd);
        for (qsets[0..top_idx.len]) |*ds|
            _ = vk.vkFreeDescriptorSets(gpu_weights.ctx.device, gpu_weights.pl_quantize_q8_1.desc_pool, 1, ds);
    }

    {
        const cmd = try gpu_weights.ctx.beginBatch();
        const p_down = gpu_weights.ctx.profileBegin(cmd, "moe.down_id");
        const ds = try pipeline.record(cmd, &flat_session.mat_buf, &act_q8_buf, &ids_buf, &scales_buf, &out_buf, @intCast(cfg.d_model), @intCast(cfg.d_expert), @intCast(top_idx.len));
        gpu_weights.ctx.profileEnd(cmd, p_down);
        try gpu_weights.ctx.submitBatch(cmd);
        var ds_mut = ds;
        _ = vk.vkFreeDescriptorSets(gpu_weights.ctx.device, pipeline.desc_pool, 1, &ds_mut);
    }

    const before = gpu_weights.ctx.profileStats("moe.down_id");
    const clk = std.Io.Clock.real;
    const t0 = clk.now(io);
    for (0..iters) |_| {
        const cmd = try gpu_weights.ctx.beginBatch();
        const p_down = gpu_weights.ctx.profileBegin(cmd, "moe.down_id");
        const ds = try pipeline.record(cmd, &flat_session.mat_buf, &act_q8_buf, &ids_buf, &scales_buf, &out_buf, @intCast(cfg.d_model), @intCast(cfg.d_expert), @intCast(top_idx.len));
        gpu_weights.ctx.profileEnd(cmd, p_down);
        try gpu_weights.ctx.submitBatch(cmd);
        var ds_mut = ds;
        _ = vk.vkFreeDescriptorSets(gpu_weights.ctx.device, pipeline.desc_pool, 1, &ds_mut);
    }
    const t1 = clk.now(io);
    const after = gpu_weights.ctx.profileStats("moe.down_id");

    const wall_ns = t0.durationTo(t1).nanoseconds;
    const wall_us = @as(f64, @floatFromInt(wall_ns)) / @as(f64, @floatFromInt(iters)) / 1000.0;
    const count_delta = after.count - before.count;
    const gpu_ns = after.total_ns - before.total_ns;
    const gpu_us = if (count_delta == 0)
        0.0
    else
        @as(f64, @floatFromInt(gpu_ns)) / @as(f64, @floatFromInt(count_delta)) / 1000.0;

    try out.print("expert-id down shape: type={s} active={} wall_us={d:.2} gpu_us={d:.2} host_us={d:.2}\n", .{
        lw.down_exps.type_.label(), top_idx.len, wall_us, gpu_us, @max(0.0, wall_us - gpu_us),
    });
}

fn uploadDeviceLocalBytes(
    ctx: *const gpu_ctx.GpuContext,
    bytes: []const u8,
    usage: vk.VkBufferUsageFlags,
) !gpu_buffer.GpuBuffer {
    var staging = try gpu_buffer.GpuBuffer.initStaging(ctx, bytes.len);
    defer staging.deinit();
    try staging.upload(bytes);

    var dst = try gpu_buffer.GpuBuffer.initDeviceLocal(ctx, bytes.len, usage);
    errdefer dst.deinit();

    const cmd = try ctx.beginBatchCopy();
    const region = vk.VkBufferCopy{
        .srcOffset = 0,
        .dstOffset = 0,
        .size = bytes.len,
    };
    vk.vkCmdCopyBuffer(cmd, staging.handle, dst.handle, 1, &region);
    try ctx.submitBatchCopy(cmd);

    return dst;
}

fn benchIfSelected(
    out: *std.Io.Writer,
    ctx: *const gpu_ctx.GpuContext,
    pipes: *const BenchPipelines,
    opts: MatvecBenchOptions,
    ran: *usize,
    name: []const u8,
    mat: g4_weights.RawMatrix,
    gpa: std.mem.Allocator,
    io: std.Io,
) !void {
    if (!benchTargetSelected(opts.target, name)) return;
    try benchOneMatvec(out, ctx, pipes, name, mat, opts.iters, opts.reuse_descriptor, gpa, io);
    ran.* += 1;
}

fn benchWithPipelineIfSelected(
    out: *std.Io.Writer,
    ctx: *const gpu_ctx.GpuContext,
    pipeline: *const gpu_matvec.MatvecPipeline,
    quant: *const gpu_matvec.QuantizeQ8_1Pipeline,
    opts: MatvecBenchOptions,
    ran: *usize,
    name: []const u8,
    mat: g4_weights.RawMatrix,
    expected_type: gguf_types.GgmlType,
    gpa: std.mem.Allocator,
    io: std.Io,
) !void {
    if (mat.type_ != expected_type) return;
    if (!benchTargetSelected(opts.target, name)) return;
    try benchOneMatvecWithPipeline(out, ctx, pipeline, quant, name, mat, opts.iters, opts.reuse_descriptor, gpa, io);
    ran.* += 1;
}

fn benchExpertDownIfSelected(
    out: *std.Io.Writer,
    ctx: *const gpu_ctx.GpuContext,
    pipes: *const BenchPipelines,
    opts: MatvecBenchOptions,
    ran: *usize,
    name: []const u8,
    flat: g4_weights.RawMatrix,
    expert: usize,
    rows: usize,
    cols: usize,
    gpa: std.mem.Allocator,
    io: std.Io,
) !void {
    if (!benchTargetSelected(opts.target, name)) return;
    const row_bytes = math_mod.rowBytes(flat.type_, cols);
    const per_expert = rows * row_bytes;
    const mat = g4_weights.RawMatrix{
        .data = flat.data[expert * per_expert ..][0..per_expert],
        .type_ = flat.type_,
        .rows = rows,
        .cols = cols,
    };
    try benchOneMatvec(out, ctx, pipes, name, mat, opts.iters, opts.reuse_descriptor, gpa, io);
    ran.* += 1;
}

fn benchExpertDownWithPipelineIfSelected(
    out: *std.Io.Writer,
    ctx: *const gpu_ctx.GpuContext,
    pipeline: *const gpu_matvec.MatvecPipeline,
    quant: *const gpu_matvec.QuantizeQ8_1Pipeline,
    opts: MatvecBenchOptions,
    ran: *usize,
    name: []const u8,
    flat: g4_weights.RawMatrix,
    expert: usize,
    rows: usize,
    cols: usize,
    expected_type: gguf_types.GgmlType,
    gpa: std.mem.Allocator,
    io: std.Io,
) !void {
    if (flat.type_ != expected_type) return;
    if (!benchTargetSelected(opts.target, name)) return;
    const row_bytes = math_mod.rowBytes(flat.type_, cols);
    const per_expert = rows * row_bytes;
    const mat = g4_weights.RawMatrix{
        .data = flat.data[expert * per_expert ..][0..per_expert],
        .type_ = flat.type_,
        .rows = rows,
        .cols = cols,
    };
    try benchOneMatvecWithPipeline(out, ctx, pipeline, quant, name, mat, opts.iters, opts.reuse_descriptor, gpa, io);
    ran.* += 1;
}

fn benchTargetSelected(target: ?[]const u8, name: []const u8) bool {
    const t = target orelse return true;
    return std.mem.eql(u8, t, "all") or
        std.mem.eql(u8, t, name) or
        std.mem.startsWith(u8, name, t);
}

fn benchOneMatvec(
    out: *std.Io.Writer,
    ctx: *const gpu_ctx.GpuContext,
    pipes: *const BenchPipelines,
    name: []const u8,
    mat: g4_weights.RawMatrix,
    iters: u32,
    reuse_descriptor: bool,
    gpa: std.mem.Allocator,
    io: std.Io,
) !void {
    const pl = pipes.pipelineFor(mat.type_) orelse {
        try out.print("{s:32} {s:7} unsupported\n", .{ name, mat.type_.label() });
        return;
    };
    try benchOneMatvecWithPipeline(out, ctx, pl, &pipes.quant, name, mat, iters, reuse_descriptor, gpa, io);
}

fn benchOneMatvecWithPipeline(
    out: *std.Io.Writer,
    ctx: *const gpu_ctx.GpuContext,
    pl: *const gpu_matvec.MatvecPipeline,
    quant: *const gpu_matvec.QuantizeQ8_1Pipeline,
    name: []const u8,
    mat: g4_weights.RawMatrix,
    iters: u32,
    reuse_descriptor: bool,
    gpa: std.mem.Allocator,
    io: std.Io,
) !void {
    if (mat.cols % 32 != 0) return error.InvalidMatvecShape;

    const session_opt = try gpu_matvec.MatvecSession.initFromRaw(ctx, mat.data, mat.type_, @intCast(mat.rows), @intCast(mat.cols));
    var session = session_opt orelse return error.UnsupportedModel;
    defer session.deinit();

    const vec = try gpa.alloc(f32, mat.cols);
    defer gpa.free(vec);
    for (vec, 0..) |*v, i| {
        const x = @as(f32, @floatFromInt((i % 31) + 1));
        v.* = (x - 16.0) / 16.0;
    }

    var vec_buf = try gpu_buffer.GpuBuffer.initHostCoherent(ctx, mat.cols * @sizeOf(f32), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer vec_buf.deinit();
    var acts_buf = try gpu_buffer.GpuBuffer.initDeviceLocal(ctx, gpu_matvec.q8_1OutBytes(@intCast(mat.cols)), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer acts_buf.deinit();
    var out_buf = try gpu_buffer.GpuBuffer.initHostCoherent(ctx, mat.rows * @sizeOf(f32), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    defer out_buf.deinit();

    try vec_buf.upload(std.mem.sliceAsBytes(vec));

    {
        const cmd = try ctx.beginBatch();
        const q_ds = try quant.record(cmd, &vec_buf, &acts_buf, @intCast(mat.cols));
        gpu_ctx.GpuContext.recordShaderBarrier(cmd);
        try ctx.submitBatch(cmd);
        var ds = q_ds;
        _ = vk.vkFreeDescriptorSets(ctx.device, quant.desc_pool, 1, &ds);
    }

    {
        const cmd = try ctx.beginBatch();
        const ds = try pl.record(cmd, &session.mat_buf, &acts_buf, &out_buf, @intCast(mat.rows), @intCast(mat.cols));
        try ctx.submitBatch(cmd);
        var ds_mut = ds;
        _ = vk.vkFreeDescriptorSets(ctx.device, pl.desc_pool, 1, &ds_mut);
    }

    const clk = std.Io.Clock.real;
    const profile_before = ctx.profileStats(name);
    const t0 = clk.now(io);
    var persistent_dset: ?vk.VkDescriptorSet = null;
    defer if (persistent_dset) |*ds| pl.freeDescriptorSet(ds);
    if (iters > 0 and reuse_descriptor) {
        const ds = try pl.allocDescriptorSet();
        pl.updateDescriptorSet(ds, &session.mat_buf, &acts_buf, &out_buf);
        persistent_dset = ds;
    }
    for (0..iters) |_| {
        const cmd = try ctx.beginBatch();
        const p_mv = ctx.profileBegin(cmd, name);
        const ds = if (persistent_dset) |dset| blk: {
            pl.recordDescriptor(cmd, dset, @intCast(mat.rows), @intCast(mat.cols));
            break :blk dset;
        } else try pl.record(cmd, &session.mat_buf, &acts_buf, &out_buf, @intCast(mat.rows), @intCast(mat.cols));
        ctx.profileEnd(cmd, p_mv);
        try ctx.submitBatch(cmd);
        if (persistent_dset == null) {
            var ds_mut = ds;
            pl.freeDescriptorSet(&ds_mut);
        }
    }
    const t1 = clk.now(io);
    const profile_after = ctx.profileStats(name);
    const ns = t0.durationTo(t1).nanoseconds;
    const avg_us = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(iters)) / 1000.0;
    const gpu_count_delta = profile_after.count - profile_before.count;
    const gpu_ns_delta = profile_after.total_ns - profile_before.total_ns;
    const gpu_us = if (gpu_count_delta == 0)
        0.0
    else
        @as(f64, @floatFromInt(gpu_ns_delta)) / @as(f64, @floatFromInt(gpu_count_delta)) / 1000.0;
    const cpu_us = if (gpu_count_delta == 0) 0.0 else @max(0.0, avg_us - gpu_us);
    const total_bytes = @as(f64, @floatFromInt(mat.data.len)) * @as(f64, @floatFromInt(iters));
    const gbps = total_bytes / (@as(f64, @floatFromInt(ns)) / 1_000_000_000.0) / 1_000_000_000.0;
    try out.print("{s:32} {s:7} {d:9} {d:9} {d:10.2} {d:10.2} {d:10.2} {d:10.2}\n", .{
        name, mat.type_.label(), mat.rows, mat.cols, avg_us, gpu_us, cpu_us, gbps,
    });
}

fn cmdTokenize(out: *std.Io.Writer, path: []const u8, text: []const u8, io: std.Io, gpa: std.mem.Allocator) !void {
    var reader = try gguf_reader.GgufReader.open(path, io, gpa);
    defer reader.deinit();

    var vocab = try vocab_mod.fromGguf(&reader, gpa);
    defer vocab.deinit();

    try out.print("tokenizer: {s}  vocab: {}  merges: {}\n", .{
        vocab.model, vocab.tokens.len, vocab.merge_rank.count(),
    });

    const ids = try bpe.encode(text, &vocab, gpa);
    defer gpa.free(ids);
    const decoded = try bpe.decode(ids, &vocab, gpa);
    defer gpa.free(decoded);

    try out.print("input:   \"{s}\"\n", .{text});
    try out.print("ids:     {any}\n", .{ids});
    try out.print("decoded: \"{s}\"\n", .{decoded});

    try out.print("\ntokens:\n", .{});
    for (ids) |id| {
        if (id < vocab.tokens.len) try out.print("  {:6}  {s}\n", .{ id, vocab.tokens[id] });
    }
}

const GenerateOptions = struct {
    max_tokens: u32 = 256,
    temperature: f32 = 0.8,
    top_p: f32 = 0.9,
    top_k: u32 = 40,
    seed: u64 = 42,
    threads: u32 = 0, // 0 = auto
    chat: bool = false,
    gpu: bool = false,
    // Override the auto-detected EOT token. null = use vocab.eot_token_id.
    stop_token: ?[]const u8 = null,
    // Restrict GPU to layers [start..=end] inclusive; null = all layers.
    gpu_layer_range: ?[2]usize = null,
};

fn cmdGenerate(
    out: *std.Io.Writer,
    path: []const u8,
    prompt: []const u8,
    opts: GenerateOptions,
    io: std.Io,
    gpa: std.mem.Allocator,
) !void {
    std.debug.print("loading {s}...\n", .{path});
    const clk = std.Io.Clock.real;
    const t_load_start = clk.now(io);

    var reader = try gguf_reader.GgufReader.open(path, io, gpa);
    defer reader.deinit();

    var vocab = try vocab_mod.fromGguf(&reader, gpa);
    defer vocab.deinit();

    // Resolve stop token: explicit --stop-token overrides auto-detected EOT.
    const stop_id: ?u32 = blk: {
        if (opts.stop_token) |s| {
            if (vocab.token_to_id.get(s)) |id| break :blk id;
            std.debug.print("warning: --stop-token '{s}' not found in vocab, ignoring\n", .{s});
            break :blk null;
        }
        break :blk vocab.eot_token_id;
    };
    if (stop_id) |id| std.debug.print("  stop token: {} '{s}'\n", .{ id, if (id < vocab.tokens.len) vocab.tokens[id] else "?" });

    const rendered_prompt = if (opts.chat) blk: {
        const messages = [_]chat_tmpl.Message{.{ .role = .user, .content = prompt }};
        break :blk try chat_tmpl.apply(gpa, &vocab, &messages, true);
    } else null;
    defer {
        if (rendered_prompt) |p| gpa.free(p);
    }

    const prompt_text = rendered_prompt orelse prompt;
    const prompt_ids = try bpe.encode(prompt_text, &vocab, gpa);
    defer gpa.free(prompt_ids);

    const n_threads: usize = if (opts.threads > 0) opts.threads else std.Thread.getCpuCount() catch 1;
    const pool = try tp.ThreadPool.init(gpa, n_threads, io);
    defer pool.deinit(gpa);

    var prng = std.Random.DefaultPrng.init(opts.seed);
    const rng = prng.random();
    const params = sample_mod.SampleParams{
        .temperature = opts.temperature,
        .top_p = opts.top_p,
        .top_k = opts.top_k,
    };

    try out.print("{s}", .{prompt_text});
    try out.flush();

    if (g4_loader.isGemma4(&reader)) {
        const g4cfg = try g4_loader.configFromGguf(&reader, gpa);
        std.debug.print(
            "  [Gemma4] layers={} heads={} d_model={} experts={}/{} vocab={}\n",
            .{ g4cfg.n_layers, g4cfg.n_heads, g4cfg.d_model, g4cfg.n_experts, g4cfg.n_experts_used, g4cfg.vocab_size },
        );
        var weights = try g4_loader.loadWeights(&reader, g4cfg, gpa);
        defer weights.deinit();

        // Optional GPU offload: upload attention + dense-FFN matrices to VRAM.
        var gpu_weights: ?g4_gpu.GpuWeights = null;
        if (opts.gpu) {
            // Prefer killing llmtoy over system services/Claude if OOM fires.
            setOomAdj(500);
            std.debug.print("uploading weights to GPU VRAM...\n", .{});
            const t_gpu0 = clk.now(io);
            gpu_weights = g4_gpu.GpuWeights.init(&weights, g4cfg, gpa) catch |e| blk: {
                std.debug.print("GPU init failed ({}), using CPU\n", .{e});
                break :blk null;
            };
            if (gpu_weights != null) {
                const t_gpu1 = clk.now(io);
                const ms = @divTrunc(t_gpu0.durationTo(t_gpu1).nanoseconds, std.time.ns_per_ms);
                std.debug.print("GPU upload done in {} ms\n", .{ms});
                printGemma4GpuRuntimeOptions();
            }
        }
        defer if (gpu_weights) |*gw| gw.deinit();

        const max_seq: usize = @min(g4cfg.max_seq_len, 4096);
        if (gpu_weights) |*gw| gw.initKvVram(g4cfg, max_seq) catch |e| {
            std.debug.print("KV VRAM alloc failed ({s}), continuing without GPU KV cache\n", .{@errorName(e)});
        };
        const gpu_ptr: ?*g4_gpu.GpuWeights = if (gpu_weights != null) &gpu_weights.? else null;
        var kv = try g4_kv.Gemma4KvCache.init(g4cfg, max_seq, gpa);
        defer kv.deinit();
        const t_prefill_start = clk.now(io);
        printSetupTiming(t_load_start, t_prefill_start);
        std.debug.print("prefilling {} tokens (threads={})...\n", .{ prompt_ids.len, n_threads });
        var last_logits: []f32 = &.{};
        var last_greedy_token: u32 = 0;
        var has_last_greedy_token = false;
        const use_gpu_greedy = opts.temperature == 0.0 and gpu_ptr != null;
        for (prompt_ids, 0..) |tok, pos| {
            if (last_logits.len > 0) gpa.free(last_logits);
            const need_logits = pos + 1 == prompt_ids.len;
            const greedy_out: ?*u32 = if (need_logits and use_gpu_greedy) &last_greedy_token else null;
            last_logits = try g4_fwd.forwardOne(tok, pos, &kv, &weights, g4cfg, pool, gpa, gpu_ptr, null, opts.gpu_layer_range, need_logits, greedy_out);
            has_last_greedy_token = greedy_out != null and last_logits.len == 0;
        }
        const t_prefill = clk.now(io);
        printTokenTiming("prefill", prompt_ids.len, t_prefill_start, t_prefill);
        var n_gen: u32 = 0;
        var pos: usize = prompt_ids.len;
        const t_gen_start = clk.now(io);
        while (n_gen < opts.max_tokens) : (n_gen += 1) {
            const next_tok = if (has_last_greedy_token)
                last_greedy_token
            else
                try sample_mod.sample(last_logits, params, rng, gpa);
            if (last_logits.len > 0) gpa.free(last_logits);
            has_last_greedy_token = false;
            if (next_tok == vocab.eos_token_id) {
                last_logits = &.{};
                break;
            }
            if (stop_id != null and next_tok == stop_id.?) {
                last_logits = &.{};
                break;
            }
            var tok_buf: [64]u8 = undefined;
            const tok_len = bpe.decodeOne(next_tok, &vocab, &tok_buf);
            try out.writeAll(tok_buf[0..tok_len]);
            try out.flush();
            const greedy_out: ?*u32 = if (use_gpu_greedy) &last_greedy_token else null;
            last_logits = try g4_fwd.forwardOne(next_tok, pos, &kv, &weights, g4cfg, pool, gpa, gpu_ptr, null, opts.gpu_layer_range, true, greedy_out);
            has_last_greedy_token = greedy_out != null and last_logits.len == 0;
            pos += 1;
        }
        if (last_logits.len > 0) gpa.free(last_logits);
        const t_end = clk.now(io);
        printTokenTiming("generation", n_gen, t_gen_start, t_end);
    } else {
        const cfg = try loader.configFromGguf(&reader, gpa);
        std.debug.print(
            "  layers={} heads={}/{} d_model={} d_ffn={} vocab={}\n",
            .{ cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.d_model, cfg.d_ffn, cfg.vocab_size },
        );
        var weights = try loader.loadWeights(&reader, cfg, gpa);
        defer weights.deinit();
        var kv = try kv_mod.KvCache.init(cfg, gpa);
        defer kv.deinit();
        const t_prefill_start = clk.now(io);
        printSetupTiming(t_load_start, t_prefill_start);
        std.debug.print("prefilling {} tokens (threads={})...\n", .{ prompt_ids.len, n_threads });
        var last_logits: []f32 = undefined;
        for (prompt_ids, 0..) |tok, pos| {
            if (pos > 0) gpa.free(last_logits);
            last_logits = try fwd.forwardOneModel(tok, pos, &kv, &weights, cfg, pool, gpa);
        }
        const t_prefill = clk.now(io);
        printTokenTiming("prefill", prompt_ids.len, t_prefill_start, t_prefill);
        var n_gen: u32 = 0;
        var pos: usize = prompt_ids.len;
        const t_gen_start = clk.now(io);
        while (n_gen < opts.max_tokens) : (n_gen += 1) {
            const next_tok = try sample_mod.sample(last_logits, params, rng, gpa);
            gpa.free(last_logits);
            if (next_tok == vocab.eos_token_id) {
                last_logits = &.{};
                break;
            }
            if (stop_id != null and next_tok == stop_id.?) {
                last_logits = &.{};
                break;
            }
            var tok_buf: [64]u8 = undefined;
            const tok_len = bpe.decodeOne(next_tok, &vocab, &tok_buf);
            try out.writeAll(tok_buf[0..tok_len]);
            try out.flush();
            last_logits = try fwd.forwardOneModel(next_tok, pos, &kv, &weights, cfg, pool, gpa);
            pos += 1;
        }
        if (last_logits.len > 0) gpa.free(last_logits);
        const t_end = clk.now(io);
        printTokenTiming("generation", n_gen, t_gen_start, t_end);
    }
    try out.print("\n", .{});
    try out.flush();
}

// Write `adj` to /proc/self/oom_score_adj so llmtoy is the preferred OOM
// victim rather than system services or the Claude session. Silently ignores
// failures (non-root processes may not be allowed to raise the adj).
fn setOomAdj(adj: i32) void {
    const fd = std.posix.openat(std.posix.AT.FDCWD, "/proc/self/oom_score_adj", .{ .ACCMODE = .WRONLY }, 0) catch return;
    defer _ = std.os.linux.close(fd);
    var buf: [12]u8 = undefined;
    const s = std.fmt.bufPrint(&buf, "{}\n", .{adj}) catch return;
    _ = std.os.linux.write(fd, s.ptr, s.len);
}

fn printSetupTiming(start: anytype, end: anytype) void {
    const ms = @divTrunc(start.durationTo(end).nanoseconds, std.time.ns_per_ms);
    std.debug.print("  setup: {} ms\n", .{ms});
}

fn envFlagEnabled(name: [:0]const u8, default: bool) bool {
    if (std.c.getenv(name)) |raw|
        return !std.mem.eql(u8, std.mem.span(raw), "0");
    return default;
}

fn printGemma4GpuRuntimeOptions() void {
    std.debug.print("  Gemma4 GPU options: attn_cpu_kv_shadow={} attention_async={} expert_reuse_cmd={} expert_gpu_router={}\n", .{
        envFlagEnabled("LLMTOY_ATTN_CPU_KV_SHADOW", false),
        envFlagEnabled("LLMTOY_ATTENTION_ASYNC", true),
        envFlagEnabled("LLMTOY_EXPERT_REUSE_CMD", true),
        envFlagEnabled("LLMTOY_EXPERT_GPU_ROUTER", false),
    });
}

fn printTokenTiming(label: []const u8, tokens: anytype, start: anytype, end: anytype) void {
    const ns = start.durationTo(end).nanoseconds;
    const ms = @divTrunc(ns, std.time.ns_per_ms);
    const seconds = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(std.time.ns_per_s));
    const tps = if (seconds > 0.0) @as(f64, @floatFromInt(tokens)) / seconds else 0.0;
    std.debug.print("  {s}: {} tokens in {} ms ({d:.2} tok/s)\n", .{ label, tokens, ms, tps });
}

// Parse "L0:L1" into a [2]usize layer range.
fn parseLayerRange(s: []const u8) !?[2]usize {
    const colon = std.mem.indexOfScalar(u8, s, ':') orelse return error.BadLayerRange;
    const lo = try std.fmt.parseInt(usize, s[0..colon], 10);
    const hi = try std.fmt.parseInt(usize, s[colon + 1 ..], 10);
    if (lo > hi) return error.BadLayerRange;
    return .{ lo, hi };
}

// ── compare command ───────────────────────────────────────────────────────────

const CompareOptions = struct {
    threads: u32 = 0,
    chat: bool = false,
    gpu_layer_range: ?[2]usize = null,
};

/// Run one forward pass on CPU and one on GPU for the first prompt token,
/// then print per-layer residual divergence and final top-5 logit comparison.
fn cmdCompare(
    out: *std.Io.Writer,
    path: []const u8,
    prompt: []const u8,
    opts: CompareOptions,
    io: std.Io,
    gpa: std.mem.Allocator,
) !void {
    std.debug.print("loading {s}...\n", .{path});

    var reader = try gguf_reader.GgufReader.open(path, io, gpa);
    defer reader.deinit();

    if (!g4_loader.isGemma4(&reader)) {
        std.debug.print("compare requires a Gemma4 model (GPU path is Gemma4-only)\n", .{});
        return error.NotGemma4;
    }

    var vocab = try vocab_mod.fromGguf(&reader, gpa);
    defer vocab.deinit();

    const rendered_prompt = if (opts.chat) blk: {
        const messages = [_]chat_tmpl.Message{.{ .role = .user, .content = prompt }};
        break :blk try chat_tmpl.apply(gpa, &vocab, &messages, true);
    } else null;
    defer if (rendered_prompt) |p| gpa.free(p);
    const prompt_text = rendered_prompt orelse prompt;

    const prompt_ids = try bpe.encode(prompt_text, &vocab, gpa);
    defer gpa.free(prompt_ids);
    if (prompt_ids.len == 0) return error.EmptyPrompt;

    const g4cfg = try g4_loader.configFromGguf(&reader, gpa);
    std.debug.print("  [Gemma4] layers={} d_model={} experts={}/{}\n", .{ g4cfg.n_layers, g4cfg.d_model, g4cfg.n_experts, g4cfg.n_experts_used });

    var weights = try g4_loader.loadWeights(&reader, g4cfg, gpa);
    defer weights.deinit();

    const n_threads: usize = if (opts.threads > 0) opts.threads else std.Thread.getCpuCount() catch 1;
    const pool = try tp.ThreadPool.init(gpa, n_threads, io);
    defer pool.deinit(gpa);

    setOomAdj(500);
    std.debug.print("uploading weights to GPU VRAM...\n", .{});
    var gpu_weights = g4_gpu.GpuWeights.init(&weights, g4cfg, gpa) catch |e| {
        std.debug.print("GPU init failed ({}), cannot compare without GPU\n", .{e});
        return e;
    };
    defer gpu_weights.deinit();
    const gpu_ptr = &gpu_weights;
    printGemma4GpuRuntimeOptions();

    const d = g4cfg.d_model;
    const nl = g4cfg.n_layers;

    // Allocate flat backing storage for taps; slice into per-layer views.
    const cpu_flat = try gpa.alloc(f32, nl * d);
    defer gpa.free(cpu_flat);
    const gpu_flat = try gpa.alloc(f32, nl * d);
    defer gpa.free(gpu_flat);
    const cpu_taps = try gpa.alloc([]f32, nl);
    defer gpa.free(cpu_taps);
    const gpu_taps = try gpa.alloc([]f32, nl);
    defer gpa.free(gpu_taps);
    for (0..nl) |l| {
        cpu_taps[l] = cpu_flat[l * d ..][0..d];
        gpu_taps[l] = gpu_flat[l * d ..][0..d];
    }

    const cmp_token = prompt_ids[0];
    const max_seq: usize = @min(g4cfg.max_seq_len, 4096);
    gpu_weights.initKvVram(g4cfg, max_seq) catch |e| {
        std.debug.print("KV VRAM alloc failed ({s}), continuing without GPU KV cache\n", .{@errorName(e)});
    };

    std.debug.print("running CPU forward pass (token={})...\n", .{cmp_token});
    var kv_cpu = try g4_kv.Gemma4KvCache.init(g4cfg, max_seq, gpa);
    defer kv_cpu.deinit();
    const cpu_logits = try g4_fwd.forwardOne(cmp_token, 0, &kv_cpu, &weights, g4cfg, pool, gpa, null, cpu_taps, null, true, null);
    defer gpa.free(cpu_logits);

    std.debug.print("running GPU forward pass...\n", .{});
    var kv_gpu = try g4_kv.Gemma4KvCache.init(g4cfg, max_seq, gpa);
    defer kv_gpu.deinit();
    const gpu_logits = try g4_fwd.forwardOne(cmp_token, 0, &kv_gpu, &weights, g4cfg, pool, gpa, gpu_ptr, gpu_taps, opts.gpu_layer_range, true, null);
    defer gpa.free(gpu_logits);

    // ── per-layer residual comparison ─────────────────────────────────────────

    try out.print("per-layer residual comparison  (token={}, pos=0", .{cmp_token});
    if (opts.gpu_layer_range) |r| try out.print(", gpu_layers={}:{}", .{ r[0], r[1] });
    try out.print("):\n", .{});

    var first_fail: ?usize = null;
    for (0..nl) |l| {
        const cx = cpu_taps[l];
        const gx = gpu_taps[l];

        var max_abs: f32 = 0.0;
        var max_ref: f32 = 0.0;
        var cpu_am: usize = 0;
        var gpu_am: usize = 0;
        var cpu_am_v: f32 = -std.math.inf(f32);
        var gpu_am_v: f32 = -std.math.inf(f32);

        for (cx, gx, 0..) |c, g, i| {
            const d_ = @abs(c - g);
            if (d_ > max_abs) max_abs = d_;
            if (@abs(c) > max_ref) max_ref = @abs(c);
            if (c > cpu_am_v) {
                cpu_am_v = c;
                cpu_am = i;
            }
            if (g > gpu_am_v) {
                gpu_am_v = g;
                gpu_am = i;
            }
        }

        const rel = max_abs / (max_ref + 1e-6);
        const ok = (cpu_am == gpu_am);
        if (!ok and first_fail == null) first_fail = l;

        try out.print("layer {:>3}  max|D|={d:.5}  rel={d:.3}%  argmax={s}\n", .{ l, max_abs, rel * 100.0, if (ok) "ok" else "FAIL" });
    }

    if (first_fail) |l| {
        try out.print("\nfirst argmax mismatch at layer {}\n", .{l});
    } else {
        try out.print("\nall layer argmaxes match\n", .{});
    }

    // ── top-5 final logit comparison ──────────────────────────────────────────

    const V = struct { id: usize, val: f32 };
    const top_n: usize = @min(5, cpu_logits.len);
    var cpu_top = [_]V{.{ .id = 0, .val = -std.math.inf(f32) }} ** 5;
    var gpu_top = [_]V{.{ .id = 0, .val = -std.math.inf(f32) }} ** 5;

    for (0..top_n) |k| {
        var best_id: usize = 0;
        var best_val: f32 = -std.math.inf(f32);
        for (cpu_logits, 0..) |v, i| {
            var dup = false;
            for (cpu_top[0..k]) |p| {
                if (p.id == i) {
                    dup = true;
                    break;
                }
            }
            if (!dup and v > best_val) {
                best_val = v;
                best_id = i;
            }
        }
        cpu_top[k] = .{ .id = best_id, .val = best_val };
    }

    for (0..top_n) |k| {
        var best_id: usize = 0;
        var best_val: f32 = -std.math.inf(f32);
        for (gpu_logits, 0..) |v, i| {
            var dup = false;
            for (gpu_top[0..k]) |p| {
                if (p.id == i) {
                    dup = true;
                    break;
                }
            }
            if (!dup and v > best_val) {
                best_val = v;
                best_id = i;
            }
        }
        gpu_top[k] = .{ .id = best_id, .val = best_val };
    }

    try out.print("\ntop-{} CPU:", .{top_n});
    for (cpu_top[0..top_n]) |t| try out.print("  {}({d:.3})", .{ t.id, t.val });
    try out.print("\ntop-{} GPU:", .{top_n});
    for (gpu_top[0..top_n]) |t| try out.print("  {}({d:.3})", .{ t.id, t.val });
    try out.print("\n", .{});

    if (cpu_top[0].id == gpu_top[0].id) {
        try out.print("final argmax: {} (match)\n", .{cpu_top[0].id});
    } else {
        try out.print("final argmax: CPU={} GPU={} (MISMATCH)\n", .{ cpu_top[0].id, gpu_top[0].id });
    }
    try out.flush();
}
