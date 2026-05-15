const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{
        .default_target = .{ .cpu_model = .native },
    });
    const optimize = b.option(std.builtin.OptimizeMode, "optimize",
        "Optimization mode (default: ReleaseFast)") orelse .ReleaseFast;

    // Compile GLSL compute shaders to SPIR-V via glslc (must be on PATH).
    // The WriteFiles step puts matvec_f32.spv alongside shaders.zig so that
    // @embedFile("matvec_f32.spv") in the generated source resolves correctly.
    const wf = b.addWriteFiles();
    const matvec_spv = compileShader(b, "src/gpu/shaders/matvec_f32.glsl");
    _ = wf.addCopyFile(matvec_spv, "matvec_f32.spv");
    const matvec_q8_0_spv = compileShader(b, "src/gpu/shaders/matvec_q8_0.glsl");
    _ = wf.addCopyFile(matvec_q8_0_spv, "matvec_q8_0.spv");
    const matvec_q3_k_spv = compileShader(b, "src/gpu/shaders/matvec_q3_k.glsl");
    _ = wf.addCopyFile(matvec_q3_k_spv, "matvec_q3_k.spv");
    const matvec_q4_k_spv = compileShader(b, "src/gpu/shaders/matvec_q4_k.glsl");
    _ = wf.addCopyFile(matvec_q4_k_spv, "matvec_q4_k.spv");
    const matvec_q5_1_spv = compileShader(b, "src/gpu/shaders/matvec_q5_1.glsl");
    _ = wf.addCopyFile(matvec_q5_1_spv, "matvec_q5_1.spv");
    const matvec_q5_0_spv = compileShader(b, "src/gpu/shaders/matvec_q5_0.glsl");
    _ = wf.addCopyFile(matvec_q5_0_spv, "matvec_q5_0.spv");
    const matvec_fused_gu_q3k_spv = compileShader(b, "src/gpu/shaders/matvec_fused_gu_q3k.glsl");
    _ = wf.addCopyFile(matvec_fused_gu_q3k_spv, "matvec_fused_gu_q3k.spv");
    const expert_accum_spv = compileShader(b, "src/gpu/shaders/expert_accum.glsl");
    _ = wf.addCopyFile(expert_accum_spv, "expert_accum.spv");
    const quantize_q8_1_spv = compileShader(b, "src/gpu/shaders/quantize_q8_1.glsl");
    _ = wf.addCopyFile(quantize_q8_1_spv, "quantize_q8_1.spv");
    const matvec_q4_k_q8_1_spv = compileShader(b, "src/gpu/shaders/matvec_q4_k_q8_1.glsl");
    _ = wf.addCopyFile(matvec_q4_k_q8_1_spv, "matvec_q4_k_q8_1.spv");
    const matvec_q3_k_q8_1_spv = compileShader(b, "src/gpu/shaders/matvec_q3_k_q8_1.glsl");
    _ = wf.addCopyFile(matvec_q3_k_q8_1_spv, "matvec_q3_k_q8_1.spv");
    const matvec_fused_gu_q3k_q8_1_spv = compileShader(b, "src/gpu/shaders/matvec_fused_gu_q3k_q8_1.glsl");
    _ = wf.addCopyFile(matvec_fused_gu_q3k_q8_1_spv, "matvec_fused_gu_q3k_q8_1.spv");
    const matvec_q5_0_q8_1_spv = compileShader(b, "src/gpu/shaders/matvec_q5_0_q8_1.glsl");
    _ = wf.addCopyFile(matvec_q5_0_q8_1_spv, "matvec_q5_0_q8_1.spv");
    const matvec_q5_1_q8_1_spv = compileShader(b, "src/gpu/shaders/matvec_q5_1_q8_1.glsl");
    _ = wf.addCopyFile(matvec_q5_1_q8_1_spv, "matvec_q5_1_q8_1.spv");
    const rmsnorm_spv = compileShader(b, "src/gpu/shaders/rmsnorm.glsl");
    _ = wf.addCopyFile(rmsnorm_spv, "rmsnorm.spv");
    const elem_add_spv = compileShader(b, "src/gpu/shaders/elem_add.glsl");
    _ = wf.addCopyFile(elem_add_spv, "elem_add.spv");
    const elem_scale_spv = compileShader(b, "src/gpu/shaders/elem_scale.glsl");
    _ = wf.addCopyFile(elem_scale_spv, "elem_scale.spv");
    const rope_table_spv = compileShader(b, "src/gpu/shaders/rope_neox_table.glsl");
    _ = wf.addCopyFile(rope_table_spv, "rope_neox_table.spv");
    const rope_theta_spv = compileShader(b, "src/gpu/shaders/rope_neox_theta.glsl");
    _ = wf.addCopyFile(rope_theta_spv, "rope_neox_theta.spv");
    // align(4): VkShaderModuleCreateInfo.pCode requires 4-byte aligned SPIR-V data.
    // Declaring the embedded file with align(4) makes &matvec_f32 satisfy that
    // requirement without a runtime allocation or copy.
    const shaders_src = wf.add("shaders.zig",
        \\pub const matvec_f32          align(4) = @embedFile("matvec_f32.spv").*;
        \\pub const matvec_q8_0         align(4) = @embedFile("matvec_q8_0.spv").*;
        \\pub const matvec_q3_k         align(4) = @embedFile("matvec_q3_k.spv").*;
        \\pub const matvec_q4_k         align(4) = @embedFile("matvec_q4_k.spv").*;
        \\pub const matvec_q5_1         align(4) = @embedFile("matvec_q5_1.spv").*;
        \\pub const matvec_q5_0         align(4) = @embedFile("matvec_q5_0.spv").*;
        \\pub const matvec_fused_gu_q3k align(4) = @embedFile("matvec_fused_gu_q3k.spv").*;
        \\pub const expert_accum        align(4) = @embedFile("expert_accum.spv").*;
        \\pub const quantize_q8_1       align(4) = @embedFile("quantize_q8_1.spv").*;
        \\pub const matvec_q4_k_q8_1    align(4) = @embedFile("matvec_q4_k_q8_1.spv").*;
        \\pub const matvec_q3_k_q8_1    align(4) = @embedFile("matvec_q3_k_q8_1.spv").*;
        \\pub const matvec_fused_gu_q3k_q8_1 align(4) = @embedFile("matvec_fused_gu_q3k_q8_1.spv").*;
        \\pub const matvec_q5_0_q8_1    align(4) = @embedFile("matvec_q5_0_q8_1.spv").*;
        \\pub const matvec_q5_1_q8_1    align(4) = @embedFile("matvec_q5_1_q8_1.spv").*;
        \\pub const rmsnorm             align(4) = @embedFile("rmsnorm.spv").*;
        \\pub const elem_add            align(4) = @embedFile("elem_add.spv").*;
        \\pub const elem_scale          align(4) = @embedFile("elem_scale.spv").*;
        \\pub const rope_neox_table     align(4) = @embedFile("rope_neox_table.spv").*;
        \\pub const rope_neox_theta     align(4) = @embedFile("rope_neox_theta.spv").*;
    );
    const shaders_mod = b.createModule(.{ .root_source_file = shaders_src });

    const exe = b.addExecutable(.{
        .name = "llmtoy",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    addGpuSupport(b, exe, shaders_mod);
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    const run_step = b.step("run", "Run llmtoy");
    run_step.dependOn(&run_cmd.step);

    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    addGpuSupport(b, tests, shaders_mod);
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run all tests");
    test_step.dependOn(&run_tests.step);
}

fn compileShader(b: *std.Build, src: []const u8) std.Build.LazyPath {
    // vulkan1.3 lets shaders use GL_EXT_integer_dot_product,
    // GL_EXT_shader_explicit_arithmetic_types_int8 / int16 / float16,
    // and the subgroup-arithmetic / clustered ops needed by the Q8_1 path.
    // Compatibility note: existing shaders still compile under 1.3.
    const run = b.addSystemCommand(&.{
        "glslc",
        "--target-env=vulkan1.3",
        "-fshader-stage=compute",
    });
    run.addFileArg(b.path(src));
    run.addArg("-o");
    const name = std.fs.path.stem(std.fs.path.basename(src));
    return run.addOutputFileArg(b.fmt("{s}.spv", .{name}));
}

fn addGpuSupport(b: *std.Build, artifact: *std.Build.Step.Compile, shaders_mod: *std.Build.Module) void {
    const m = artifact.root_module;
    m.addImport("gpu_shaders", shaders_mod);
    m.linkSystemLibrary("vulkan", .{});
    m.link_libc = true;
    // LIBRARY_PATH set by the nix shell hook; helps the linker find libvulkan.so
    if (b.graph.environ_map.get("LIBRARY_PATH")) |lib_path| {
        var it = std.mem.splitScalar(u8, lib_path, ':');
        while (it.next()) |p| {
            if (p.len > 0) m.addLibraryPath(.{ .cwd_relative = p });
        }
    }
}
