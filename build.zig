const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{
        .default_target = .{ .cpu_model = .native },
    });
    const optimize = b.option(std.builtin.OptimizeMode, "optimize", "Optimization mode (default: ReleaseFast)") orelse .ReleaseFast;

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
    const quantize_q8_1_batched_spv = compileShader(b, "src/gpu/shaders/quantize_q8_1_batched.glsl");
    _ = wf.addCopyFile(quantize_q8_1_batched_spv, "quantize_q8_1_batched.spv");
    const matvec_q4_k_q8_1_spv = compileShader(b, "src/gpu/shaders/matvec_q4_k_q8_1.glsl");
    _ = wf.addCopyFile(matvec_q4_k_q8_1_spv, "matvec_q4_k_q8_1.spv");
    const matvec_q4_k_q8_1_mmvq_spv = compileShader(b, "src/gpu/shaders/matvec_q4_k_q8_1_mmvq.glsl");
    _ = wf.addCopyFile(matvec_q4_k_q8_1_mmvq_spv, "matvec_q4_k_q8_1_mmvq.spv");
    const matvec_q4_k_q8_1_r4_spv = compileShader(b, "src/gpu/shaders/matvec_q4_k_q8_1_r4.glsl");
    _ = wf.addCopyFile(matvec_q4_k_q8_1_r4_spv, "matvec_q4_k_q8_1_r4.spv");
    const matvec_q3_k_q8_1_spv = compileShader(b, "src/gpu/shaders/matvec_q3_k_q8_1.glsl");
    _ = wf.addCopyFile(matvec_q3_k_q8_1_spv, "matvec_q3_k_q8_1.spv");
    const matvec_q3_k_q8_1_mmvq_spv = compileShader(b, "src/gpu/shaders/matvec_q3_k_q8_1_mmvq.glsl");
    _ = wf.addCopyFile(matvec_q3_k_q8_1_mmvq_spv, "matvec_q3_k_q8_1_mmvq.spv");
    const matvec_fused_gu_q3k_q8_1_spv = compileShader(b, "src/gpu/shaders/matvec_fused_gu_q3k_q8_1.glsl");
    _ = wf.addCopyFile(matvec_fused_gu_q3k_q8_1_spv, "matvec_fused_gu_q3k_q8_1.spv");
    const expert_gate_up_id_q3_k_q8_1_spv = compileShader(b, "src/gpu/shaders/expert_gate_up_id_q3_k_q8_1.glsl");
    _ = wf.addCopyFile(expert_gate_up_id_q3_k_q8_1_spv, "expert_gate_up_id_q3_k_q8_1.spv");
    const expert_gate_up_id_q3_k_q8_1_r2_spv = compileShader(b, "src/gpu/shaders/expert_gate_up_id_q3_k_q8_1_r2.glsl");
    _ = wf.addCopyFile(expert_gate_up_id_q3_k_q8_1_r2_spv, "expert_gate_up_id_q3_k_q8_1_r2.spv");
    const expert_gate_up_id_q3_k_q8_1_r4_spv = compileShader(b, "src/gpu/shaders/expert_gate_up_id_q3_k_q8_1_r4.glsl");
    _ = wf.addCopyFile(expert_gate_up_id_q3_k_q8_1_r4_spv, "expert_gate_up_id_q3_k_q8_1_r4.spv");
    const matvec_q5_0_q8_1_spv = compileShader(b, "src/gpu/shaders/matvec_q5_0_q8_1.glsl");
    _ = wf.addCopyFile(matvec_q5_0_q8_1_spv, "matvec_q5_0_q8_1.spv");
    const matvec_q5_0_q8_1_mmvq_spv = compileShader(b, "src/gpu/shaders/matvec_q5_0_q8_1_mmvq.glsl");
    _ = wf.addCopyFile(matvec_q5_0_q8_1_mmvq_spv, "matvec_q5_0_q8_1_mmvq.spv");
    const matvec_q5_1_q8_1_spv = compileShader(b, "src/gpu/shaders/matvec_q5_1_q8_1.glsl");
    _ = wf.addCopyFile(matvec_q5_1_q8_1_spv, "matvec_q5_1_q8_1.spv");
    const matvec_q5_1_q8_1_mmvq_spv = compileShader(b, "src/gpu/shaders/matvec_q5_1_q8_1_mmvq.glsl");
    _ = wf.addCopyFile(matvec_q5_1_q8_1_mmvq_spv, "matvec_q5_1_q8_1_mmvq.spv");
    const matvec_q6_k_q8_1_spv = compileShader(b, "src/gpu/shaders/matvec_q6_k_q8_1.glsl");
    _ = wf.addCopyFile(matvec_q6_k_q8_1_spv, "matvec_q6_k_q8_1.spv");
    const matvec_q6_k_q8_1_fast_spv = compileShader(b, "src/gpu/shaders/matvec_q6_k_q8_1_fast.glsl");
    _ = wf.addCopyFile(matvec_q6_k_q8_1_fast_spv, "matvec_q6_k_q8_1_fast.spv");
    const matvec_q6_k_q8_1_mmvq_spv = compileShader(b, "src/gpu/shaders/matvec_q6_k_q8_1_mmvq.glsl");
    _ = wf.addCopyFile(matvec_q6_k_q8_1_mmvq_spv, "matvec_q6_k_q8_1_mmvq.spv");
    const matvec_q5_k_q8_1_spv = compileShader(b, "src/gpu/shaders/matvec_q5_k_q8_1.glsl");
    _ = wf.addCopyFile(matvec_q5_k_q8_1_spv, "matvec_q5_k_q8_1.spv");
    const matvec_q5_k_q8_1_mmvq_spv = compileShader(b, "src/gpu/shaders/matvec_q5_k_q8_1_mmvq.glsl");
    _ = wf.addCopyFile(matvec_q5_k_q8_1_mmvq_spv, "matvec_q5_k_q8_1_mmvq.spv");
    const matvec_iq4_nl_q8_1_spv = compileShader(b, "src/gpu/shaders/matvec_iq4_nl_q8_1.glsl");
    _ = wf.addCopyFile(matvec_iq4_nl_q8_1_spv, "matvec_iq4_nl_q8_1.spv");
    const expert_down_id_q5_0_q8_1_spv = compileShader(b, "src/gpu/shaders/expert_down_id_q5_0_q8_1.glsl");
    _ = wf.addCopyFile(expert_down_id_q5_0_q8_1_spv, "expert_down_id_q5_0_q8_1.spv");
    const expert_down_id_q5_1_q8_1_spv = compileShader(b, "src/gpu/shaders/expert_down_id_q5_1_q8_1.glsl");
    _ = wf.addCopyFile(expert_down_id_q5_1_q8_1_spv, "expert_down_id_q5_1_q8_1.spv");
    const expert_down_id_iq4_nl_q8_1_spv = compileShader(b, "src/gpu/shaders/expert_down_id_iq4_nl_q8_1.glsl");
    _ = wf.addCopyFile(expert_down_id_iq4_nl_q8_1_spv, "expert_down_id_iq4_nl_q8_1.spv");
    const expert_down_id_iq4_nl_q8_1_r2_spv = compileShader(b, "src/gpu/shaders/expert_down_id_iq4_nl_q8_1_r2.glsl");
    _ = wf.addCopyFile(expert_down_id_iq4_nl_q8_1_r2_spv, "expert_down_id_iq4_nl_q8_1_r2.spv");
    const expert_down_id_iq4_nl_q8_1_b16_spv = compileShader(b, "src/gpu/shaders/expert_down_id_iq4_nl_q8_1_b16.glsl");
    _ = wf.addCopyFile(expert_down_id_iq4_nl_q8_1_b16_spv, "expert_down_id_iq4_nl_q8_1_b16.spv");
    const expert_down_id_iq4_nl_q8_1_iacc_spv = compileShader(b, "src/gpu/shaders/expert_down_id_iq4_nl_q8_1_iacc.glsl");
    _ = wf.addCopyFile(expert_down_id_iq4_nl_q8_1_iacc_spv, "expert_down_id_iq4_nl_q8_1_iacc.spv");
    const expert_down_sum_id_iq4_nl_q8_1_spv = compileShader(b, "src/gpu/shaders/expert_down_sum_id_iq4_nl_q8_1.glsl");
    _ = wf.addCopyFile(expert_down_sum_id_iq4_nl_q8_1_spv, "expert_down_sum_id_iq4_nl_q8_1.spv");
    const rmsnorm_spv = compileShader(b, "src/gpu/shaders/rmsnorm.glsl");
    _ = wf.addCopyFile(rmsnorm_spv, "rmsnorm.spv");
    const rmsnorm_128_spv = compileShader(b, "src/gpu/shaders/rmsnorm_128.glsl");
    _ = wf.addCopyFile(rmsnorm_128_spv, "rmsnorm_128.spv");
    const add_rmsnorm_spv = compileShader(b, "src/gpu/shaders/add_rmsnorm.glsl");
    _ = wf.addCopyFile(add_rmsnorm_spv, "add_rmsnorm.spv");
    const add_rmsnorm_128_spv = compileShader(b, "src/gpu/shaders/add_rmsnorm_128.glsl");
    _ = wf.addCopyFile(add_rmsnorm_128_spv, "add_rmsnorm_128.spv");
    const rmsnorm_perhead_spv = compileShader(b, "src/gpu/shaders/rmsnorm_perhead.glsl");
    _ = wf.addCopyFile(rmsnorm_perhead_spv, "rmsnorm_perhead.spv");
    const rmsnorm_perhead_rope_theta_spv = compileShader(b, "src/gpu/shaders/rmsnorm_perhead_rope_theta.glsl");
    _ = wf.addCopyFile(rmsnorm_perhead_rope_theta_spv, "rmsnorm_perhead_rope_theta.spv");
    const rmsnorm_perhead_rope_table_spv = compileShader(b, "src/gpu/shaders/rmsnorm_perhead_rope_table.glsl");
    _ = wf.addCopyFile(rmsnorm_perhead_rope_table_spv, "rmsnorm_perhead_rope_table.spv");
    const elem_add_spv = compileShader(b, "src/gpu/shaders/elem_add.glsl");
    _ = wf.addCopyFile(elem_add_spv, "elem_add.spv");
    const elem_scale_spv = compileShader(b, "src/gpu/shaders/elem_scale.glsl");
    _ = wf.addCopyFile(elem_scale_spv, "elem_scale.spv");
    const elem_add_scale_spv = compileShader(b, "src/gpu/shaders/elem_add_scale.glsl");
    _ = wf.addCopyFile(elem_add_scale_spv, "elem_add_scale.spv");
    const logit_softcap_spv = compileShader(b, "src/gpu/shaders/logit_softcap.glsl");
    _ = wf.addCopyFile(logit_softcap_spv, "logit_softcap.spv");
    const argmax_spv = compileShader(b, "src/gpu/shaders/argmax.glsl");
    _ = wf.addCopyFile(argmax_spv, "argmax.spv");
    const gelu_mul_spv = compileShader(b, "src/gpu/shaders/gelu_mul.glsl");
    _ = wf.addCopyFile(gelu_mul_spv, "gelu_mul.spv");
    const rope_table_spv = compileShader(b, "src/gpu/shaders/rope_neox_table.glsl");
    _ = wf.addCopyFile(rope_table_spv, "rope_neox_table.spv");
    const rope_theta_spv = compileShader(b, "src/gpu/shaders/rope_neox_theta.glsl");
    _ = wf.addCopyFile(rope_theta_spv, "rope_neox_theta.spv");
    const attn_qk_spv = compileShader(b, "src/gpu/shaders/attn_qk_softmax.glsl");
    _ = wf.addCopyFile(attn_qk_spv, "attn_qk_softmax.spv");
    const attn_av_spv = compileShader(b, "src/gpu/shaders/attn_av.glsl");
    _ = wf.addCopyFile(attn_av_spv, "attn_av.spv");
    const attn_fused_small_spv = compileShader(b, "src/gpu/shaders/attn_fused_small.glsl");
    _ = wf.addCopyFile(attn_fused_small_spv, "attn_fused_small.spv");
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
        \\pub const quantize_q8_1_batched align(4) = @embedFile("quantize_q8_1_batched.spv").*;
        \\pub const matvec_q4_k_q8_1    align(4) = @embedFile("matvec_q4_k_q8_1.spv").*;
        \\pub const matvec_q4_k_q8_1_mmvq align(4) = @embedFile("matvec_q4_k_q8_1_mmvq.spv").*;
        \\pub const matvec_q4_k_q8_1_r4 align(4) = @embedFile("matvec_q4_k_q8_1_r4.spv").*;
        \\pub const matvec_q3_k_q8_1    align(4) = @embedFile("matvec_q3_k_q8_1.spv").*;
        \\pub const matvec_q3_k_q8_1_mmvq align(4) = @embedFile("matvec_q3_k_q8_1_mmvq.spv").*;
        \\pub const matvec_fused_gu_q3k_q8_1 align(4) = @embedFile("matvec_fused_gu_q3k_q8_1.spv").*;
        \\pub const expert_gate_up_id_q3_k_q8_1 align(4) = @embedFile("expert_gate_up_id_q3_k_q8_1.spv").*;
        \\pub const expert_gate_up_id_q3_k_q8_1_r2 align(4) = @embedFile("expert_gate_up_id_q3_k_q8_1_r2.spv").*;
        \\pub const expert_gate_up_id_q3_k_q8_1_r4 align(4) = @embedFile("expert_gate_up_id_q3_k_q8_1_r4.spv").*;
        \\pub const matvec_q5_0_q8_1    align(4) = @embedFile("matvec_q5_0_q8_1.spv").*;
        \\pub const matvec_q5_0_q8_1_mmvq align(4) = @embedFile("matvec_q5_0_q8_1_mmvq.spv").*;
        \\pub const matvec_q5_1_q8_1    align(4) = @embedFile("matvec_q5_1_q8_1.spv").*;
        \\pub const matvec_q5_1_q8_1_mmvq align(4) = @embedFile("matvec_q5_1_q8_1_mmvq.spv").*;
        \\pub const matvec_q6_k_q8_1    align(4) = @embedFile("matvec_q6_k_q8_1.spv").*;
        \\pub const matvec_q6_k_q8_1_fast align(4) = @embedFile("matvec_q6_k_q8_1_fast.spv").*;
        \\pub const matvec_q6_k_q8_1_mmvq align(4) = @embedFile("matvec_q6_k_q8_1_mmvq.spv").*;
        \\pub const matvec_q5_k_q8_1    align(4) = @embedFile("matvec_q5_k_q8_1.spv").*;
        \\pub const matvec_q5_k_q8_1_mmvq align(4) = @embedFile("matvec_q5_k_q8_1_mmvq.spv").*;
        \\pub const matvec_iq4_nl_q8_1  align(4) = @embedFile("matvec_iq4_nl_q8_1.spv").*;
        \\pub const expert_down_id_q5_0_q8_1 align(4) = @embedFile("expert_down_id_q5_0_q8_1.spv").*;
        \\pub const expert_down_id_q5_1_q8_1 align(4) = @embedFile("expert_down_id_q5_1_q8_1.spv").*;
        \\pub const expert_down_id_iq4_nl_q8_1 align(4) = @embedFile("expert_down_id_iq4_nl_q8_1.spv").*;
        \\pub const expert_down_id_iq4_nl_q8_1_r2 align(4) = @embedFile("expert_down_id_iq4_nl_q8_1_r2.spv").*;
        \\pub const expert_down_id_iq4_nl_q8_1_b16 align(4) = @embedFile("expert_down_id_iq4_nl_q8_1_b16.spv").*;
        \\pub const expert_down_id_iq4_nl_q8_1_iacc align(4) = @embedFile("expert_down_id_iq4_nl_q8_1_iacc.spv").*;
        \\pub const expert_down_sum_id_iq4_nl_q8_1 align(4) = @embedFile("expert_down_sum_id_iq4_nl_q8_1.spv").*;
        \\pub const rmsnorm             align(4) = @embedFile("rmsnorm.spv").*;
        \\pub const rmsnorm_128         align(4) = @embedFile("rmsnorm_128.spv").*;
        \\pub const add_rmsnorm         align(4) = @embedFile("add_rmsnorm.spv").*;
        \\pub const add_rmsnorm_128     align(4) = @embedFile("add_rmsnorm_128.spv").*;
        \\pub const rmsnorm_perhead     align(4) = @embedFile("rmsnorm_perhead.spv").*;
        \\pub const rmsnorm_perhead_rope_theta align(4) = @embedFile("rmsnorm_perhead_rope_theta.spv").*;
        \\pub const rmsnorm_perhead_rope_table align(4) = @embedFile("rmsnorm_perhead_rope_table.spv").*;
        \\pub const elem_add            align(4) = @embedFile("elem_add.spv").*;
        \\pub const elem_scale          align(4) = @embedFile("elem_scale.spv").*;
        \\pub const elem_add_scale      align(4) = @embedFile("elem_add_scale.spv").*;
        \\pub const logit_softcap       align(4) = @embedFile("logit_softcap.spv").*;
        \\pub const argmax              align(4) = @embedFile("argmax.spv").*;
        \\pub const gelu_mul            align(4) = @embedFile("gelu_mul.spv").*;
        \\pub const rope_neox_table     align(4) = @embedFile("rope_neox_table.spv").*;
        \\pub const rope_neox_theta     align(4) = @embedFile("rope_neox_theta.spv").*;
        \\pub const attn_qk_softmax     align(4) = @embedFile("attn_qk_softmax.spv").*;
        \\pub const attn_av             align(4) = @embedFile("attn_av.spv").*;
        \\pub const attn_fused_small    align(4) = @embedFile("attn_fused_small.spv").*;
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
