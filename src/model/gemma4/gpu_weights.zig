/// GPU-resident weight sessions for Gemma4.
///
/// Phase 1: attention + dense FFN — one layer's ≤7 matrices per vkQueueSubmit.
/// Phase 2: MoE experts — all 128 experts per layer uploaded in batches of 16
///          (~47 MiB staging peak per batch). Expert sessions stored in flat
///          arrays [n_layers × n_experts]; null entries fall back to CPU.
///
/// lm_head is uploaded separately and falls back to CPU on alloc failure.
///
/// Matrices with no GPU shader (Q5_K, IQ4_NL, …) get null sessions and fall
/// back to the CPU path transparently.
const std = @import("std");
const math = @import("../../ops/math.zig");
const vk_mod = @import("../../gpu/vk.zig");
const vk = vk_mod.vk;
const GgmlType = @import("../../gguf/types.zig").GgmlType;
const GpuCtx = @import("../../gpu/context.zig").GpuContext;
const GpuBuffer = @import("../../gpu/buffer.zig").GpuBuffer;
const mv_mod = @import("../../gpu/matvec.zig");
const MatvecPipeline = mv_mod.MatvecPipeline;
const MatvecSession = mv_mod.MatvecSession;
const FusedGateUpPipeline = mv_mod.FusedGateUpPipeline;
const ExpertGateUpIdPipeline = mv_mod.ExpertGateUpIdPipeline;
const ExpertDownIdPipeline = mv_mod.ExpertDownIdPipeline;
const AccumPipeline = mv_mod.AccumPipeline;
const QuantizeQ8_1Pipeline = mv_mod.QuantizeQ8_1Pipeline;
const QuantizeQ8_1BatchedPipeline = mv_mod.QuantizeQ8_1BatchedPipeline;
const RmsnormPipeline = mv_mod.RmsnormPipeline;
const RmsnormPerHeadPipeline = mv_mod.RmsnormPerHeadPipeline;
const ElemAddPipeline = mv_mod.ElemAddPipeline;
const ElemScalePipeline = mv_mod.ElemScalePipeline;
const GeluMulPipeline = mv_mod.GeluMulPipeline;
const RopeNeoxTablePipeline = mv_mod.RopeNeoxTablePipeline;
const RopeNeoxThetaPipeline = mv_mod.RopeNeoxThetaPipeline;
const AttnQkSoftmaxPipeline = mv_mod.AttnQkSoftmaxPipeline;
const AttnAvPipeline = mv_mod.AttnAvPipeline;
const wt_ = @import("weights.zig");
const Gemma4Weights = wt_.Gemma4Weights;
const Gemma4LayerWeights = wt_.Gemma4LayerWeights;
const RawMatrix = wt_.RawMatrix;
const Gemma4Config = @import("config.zig").Gemma4Config;

pub const GpuLayerWeights = struct {
    wq: ?MatvecSession,
    wk: ?MatvecSession,
    wv: ?MatvecSession,
    wo: ?MatvecSession,
    w_gate: ?MatvecSession,
    w_up: ?MatvecSession,
    w_down: ?MatvecSession,

    // 7j: each per-layer RMSNorm weight pre-uploaded to a device-local f32
    // buffer.  Lets the GPU rmsnorm shader bind it directly; sized to the
    // norm's element count (d_model for body norms, hd for q_norm/k_norm).
    attn_norm_buf: ?GpuBuffer,
    post_attention_norm_buf: ?GpuBuffer,
    q_norm_buf: ?GpuBuffer,
    k_norm_buf: ?GpuBuffer,
    ffn_norm_buf: ?GpuBuffer,
    pre_ffw_norm_2_buf: ?GpuBuffer,
    post_ffw_norm_1_buf: ?GpuBuffer,
    post_ffw_norm_2_buf: ?GpuBuffer,
    post_ffw_norm_buf: ?GpuBuffer,

    pub fn deinitAll(self: *GpuLayerWeights) void {
        if (self.wq) |*s| s.deinit();
        if (self.wk) |*s| s.deinit();
        if (self.wv) |*s| s.deinit();
        if (self.wo) |*s| s.deinit();
        if (self.w_gate) |*s| s.deinit();
        if (self.w_up) |*s| s.deinit();
        if (self.w_down) |*s| s.deinit();
        if (self.attn_norm_buf) |*b| b.deinit();
        if (self.post_attention_norm_buf) |*b| b.deinit();
        if (self.q_norm_buf) |*b| b.deinit();
        if (self.k_norm_buf) |*b| b.deinit();
        if (self.ffn_norm_buf) |*b| b.deinit();
        if (self.pre_ffw_norm_2_buf) |*b| b.deinit();
        if (self.post_ffw_norm_1_buf) |*b| b.deinit();
        if (self.post_ffw_norm_2_buf) |*b| b.deinit();
        if (self.post_ffw_norm_buf) |*b| b.deinit();
    }
};

pub const GpuWeights = struct {
    ctx: GpuCtx,
    pl_f32: MatvecPipeline,
    pl_q8_0: MatvecPipeline,
    pl_q3_k: MatvecPipeline,
    pl_q4_k: MatvecPipeline,
    pl_q3_k_q8_1: MatvecPipeline,
    pl_q4_k_q8_1: MatvecPipeline,
    pl_q5_0_q8_1: MatvecPipeline,
    pl_q5_1_q8_1: MatvecPipeline,
    pl_q6_k_q8_1: MatvecPipeline,
    pl_q5_k_q8_1: MatvecPipeline,
    pl_iq4_nl_q8_1: MatvecPipeline,
    pl_quantize_q8_1: QuantizeQ8_1Pipeline,
    pl_quantize_q8_1_batched: QuantizeQ8_1BatchedPipeline,
    pl_q5_1: MatvecPipeline,
    pl_q5_0: MatvecPipeline,
    pl_fused_gu: FusedGateUpPipeline,
    pl_fused_gu_q8_1: FusedGateUpPipeline,
    pl_expert_gate_up_id_q3_k: ExpertGateUpIdPipeline,
    pl_expert_down_id_q5_1: ExpertDownIdPipeline,
    pl_expert_down_id_iq4_nl: ExpertDownIdPipeline,
    pl_accum: AccumPipeline,
    // 7j primitives — per-token f32 ops on VRAM residual stream.
    pl_rmsnorm: RmsnormPipeline,
    pl_rmsnorm_perhead: RmsnormPerHeadPipeline,
    pl_elem_add: ElemAddPipeline,
    pl_elem_scale: ElemScalePipeline,
    pl_gelu_mul: GeluMulPipeline,
    pl_rope_table: RopeNeoxTablePipeline,
    pl_rope_theta: RopeNeoxThetaPipeline,
    pl_attn_qk: AttnQkSoftmaxPipeline,
    pl_attn_av: AttnAvPipeline,
    layers: []GpuLayerWeights,
    lm_head: ?MatvecSession,
    // Shared host-coherent I/O buffers sized to the largest matrix across all
    // sessions. Eliminates 420+ individual per-session HOST_COHERENT allocations.
    shared_vec: ?GpuBuffer,
    shared_out: ?GpuBuffer,
    // Device-local Q8_1-quantized activation, sized to q8_1OutBytes(max_cols).
    // Quantize-shader writes; mul_mat_vec_q*_q8_1 shaders read.  Lives in VRAM
    // so neither path crosses PCIe per token.
    shared_acts_q8_1: ?GpuBuffer,
    // 7j VRAM-resident residual stream + intermediate. Both sized to
    // max_cols (≥ d_model). x_vram holds the running residual; xb_vram holds
    // norm outputs and other transient f32 vectors.
    x_vram: ?GpuBuffer,
    xb_vram: ?GpuBuffer,
    // Dense FFN VRAM staging — buffers consumed within a single command
    // submit by the fused gate→up→gelu*up→w_down→post_ffw_norm_1 chain.
    // Sized to d_ffn (gate/up/gelu_mul output) and d_model (w_down output).
    gate_vram: ?GpuBuffer,
    up_vram: ?GpuBuffer,
    ffn_vram: ?GpuBuffer,
    // Host-coherent download buffer for the final post_ffw_norm_1 result of
    // the fused dense FFN chain.  d_model floats.
    dense_ffn_out_buf: ?GpuBuffer,
    // Phase 7j step 2 — wo absorbed into the dense FFN submit.
    //   attn_in_buf: HOST_COHERENT upload buffer for attn_concat (caller
    //                writes after CPU sdpAttn finishes).  Sized to max_nq.
    //   attn_vram:   wo output (rows = d_model), modified in place by
    //                post_attention_norm, then read by elem_add.
    attn_in_buf: ?GpuBuffer,
    attn_vram: ?GpuBuffer,
    // Persistent host-coherent staging buffer for x_vram uploads/downloads —
    // sized to max_cols. Avoids re-allocating a staging buffer per upload.
    stage_buf: ?GpuBuffer,
    // Global norm weights uploaded to VRAM (out_norm) and optional precomputed
    // RoPE inverse frequencies (Gemma4 global layers).
    out_norm_buf: ?GpuBuffer,
    rope_freqs_buf: ?GpuBuffer,
    // Flat [n_layers × n_experts] expert session arrays; null = CPU fallback.
    expert_gate: ?[]?MatvecSession,
    expert_up: ?[]?MatvecSession,
    expert_down: ?[]?MatvecSession,
    // Optional flat [n_experts, 2, d_expert, d_model] gate/up tensor per layer.
    // Env-gated for now: the ID shader is the intended final shape, but still
    // needs to prove a production wall-time win before becoming default.
    expert_gate_up_flat: ?[]?MatvecSession,
    // One flat down tensor per layer, layout [n_experts, d_model, d_expert].
    // Used by expert_down_id_* shaders to route selected experts in one dispatch.
    expert_down_flat: ?[]?MatvecSession,
    n_experts: usize,
    n_experts_used: usize,
    // I/O buffers for batched expert dispatch:
    // expert_in_buf:      HOST_COHERENT, persistently mapped — CPU writes moe_in.
    // expert_mid_bufs:    device-local VRAM — fused shader writes, down shader reads.
    // expert_all_out_buf: device-local VRAM — flat [n_experts_used × d_model] f32;
    //                     down shader writes sub-ranges, accum shader reads the whole.
    // expert_scales_buf:  HOST_COHERENT, persistently mapped — CPU writes router scales.
    // moe_gpu_buf:        device-local VRAM — accum shader writes the MoE result.
    // moe_stage_buf:      HOST_COHERENT staging buffer for legacy/debug CPU readback.
    expert_in_buf: ?GpuBuffer,
    expert_mid_bufs: ?[]GpuBuffer,
    // Per-expert Q8_1-quantized mid buffers — populated by a quantize dispatch
    // between phase 1 and phase 2 when down has a Q8_1 pipeline. Sized to
    // q8_1OutBytes(d_expert).
    expert_mid_q8_1_bufs: ?[]GpuBuffer,
    expert_mid_q8_1_flat_buf: ?GpuBuffer,
    expert_all_out_buf: ?GpuBuffer,
    expert_scales_buf: ?GpuBuffer,
    expert_accum_scales_buf: ?GpuBuffer,
    expert_ids_buf: ?GpuBuffer,
    moe_gpu_buf: ?GpuBuffer,
    moe_stage_buf: ?GpuBuffer,
    expert_in_slice: ?[]f32,
    expert_scales_slice: ?[]f32,
    expert_accum_scales_slice: ?[]f32,
    expert_ids_slice: ?[]u32,
    moe_stage_slice: ?[]f32,
    expert_quant_input_dset: ?vk.VkDescriptorSet,
    expert_quant_mid_batched_dset: ?vk.VkDescriptorSet,
    expert_accum_dset: ?vk.VkDescriptorSet,
    expert_gate_up_id_dsets: ?[]?vk.VkDescriptorSet,
    expert_down_id_dsets: ?[]?vk.VkDescriptorSet,
    expert_down_id_dset_is_iq4: ?[]bool,
    expert_reuse_cmds: ?[]?vk.VkCommandBuffer,
    // Per-projection output buffers for batched dispatch.
    // QKV: all three read the same input → one upload, three parallel dispatches.
    // gate+up: both read the same FFN-norm input → same pattern.
    q_out_buf: ?GpuBuffer,
    k_out_buf: ?GpuBuffer,
    v_out_buf: ?GpuBuffer,
    gate_out_buf: ?GpuBuffer,
    up_out_buf: ?GpuBuffer,
    // 7l.1 — per-layer KV cache in VRAM. Mirrors Gemma4KvCache layout:
    //   k_vram[l] / v_vram[l]: device-local, stride = cap[l] * nkv(l) floats.
    //   kv_cap[l]: per-layer capacity (positions stored). SWA layers cap to
    //              min(max_seq, sliding_window); global layers cap to max_seq.
    //   kv_stage: persistent host-coherent staging buffer used to write one
    //             slot's K or V into k_vram/v_vram via vkCmdCopyBuffer.
    //             Sized to max(nkv(l)) * sizeof(f32).
    // Allocated lazily via initKvVram(cfg, max_seq) once max_seq is known
    // (GpuWeights.init runs before the KV cache is created).
    k_vram: ?[]GpuBuffer,
    v_vram: ?[]GpuBuffer,
    kv_cap: ?[]usize,
    kv_stage: ?GpuBuffer,
    // 7l.2/3 — fused-attention scratch.
    //   scores_vram: device-local; n_heads * max(cap[l]) floats. Written by
    //               attn_qk_softmax shader, read by attn_av shader.
    //   attn_max_win: max(cap[l]) across layers (for scores stride).
    // Allocated alongside k_vram/v_vram in initKvVram.
    scores_vram: ?GpuBuffer,
    attn_max_win: usize,
    allocator: std.mem.Allocator,

    pub fn init(g4w: *const Gemma4Weights, g4cfg: Gemma4Config, allocator: std.mem.Allocator) !GpuWeights {
        const avail_mb = availableMemoryMB();
        std.debug.print("  available system RAM: {} MiB\n", .{avail_mb});
        if (avail_mb < 500) return error.InsufficientMemory;

        std.debug.print("  init: creating Vulkan context (VRAM={} MiB GTT={} MiB)\n", .{ vramUsedMB(), gttUsedMB() });
        var ctx = try GpuCtx.init();
        errdefer ctx.deinit();
        std.debug.print("  init: VkDevice ready (VRAM={} MiB GTT={} MiB sys={} MiB)\n", .{ vramUsedMB(), gttUsedMB(), availableMemoryMB() });

        var pl_f32 = try MatvecPipeline.initF32(&ctx);
        errdefer pl_f32.deinit();
        std.debug.print("  init: pl_f32  ok (VRAM={} MiB GTT={} MiB sys={} MiB)\n", .{ vramUsedMB(), gttUsedMB(), availableMemoryMB() });
        var pl_q8_0 = try MatvecPipeline.initQ8_0(&ctx);
        errdefer pl_q8_0.deinit();
        std.debug.print("  init: pl_q8_0 ok (VRAM={} MiB GTT={} MiB sys={} MiB)\n", .{ vramUsedMB(), gttUsedMB(), availableMemoryMB() });
        var pl_q3_k = try MatvecPipeline.initQ3K(&ctx);
        errdefer pl_q3_k.deinit();
        std.debug.print("  init: pl_q3_k ok (VRAM={} MiB GTT={} MiB sys={} MiB)\n", .{ vramUsedMB(), gttUsedMB(), availableMemoryMB() });
        var pl_q4_k = try MatvecPipeline.initQ4K(&ctx);
        errdefer pl_q4_k.deinit();
        std.debug.print("  init: pl_q4_k ok (VRAM={} MiB GTT={} MiB sys={} MiB)\n", .{ vramUsedMB(), gttUsedMB(), availableMemoryMB() });
        var pl_q3_k_q8_1 = try MatvecPipeline.initQ3KQ8_1(&ctx);
        errdefer pl_q3_k_q8_1.deinit();
        std.debug.print("  init: pl_q3_k_q8_1 ok\n", .{});
        var pl_q4_k_q8_1 = try MatvecPipeline.initQ4KQ8_1(&ctx);
        errdefer pl_q4_k_q8_1.deinit();
        std.debug.print("  init: pl_q4_k_q8_1 ok\n", .{});
        var pl_q5_0_q8_1 = try MatvecPipeline.initQ5_0Q8_1(&ctx);
        errdefer pl_q5_0_q8_1.deinit();
        std.debug.print("  init: pl_q5_0_q8_1 ok\n", .{});
        var pl_q5_1_q8_1 = try MatvecPipeline.initQ5_1Q8_1(&ctx);
        errdefer pl_q5_1_q8_1.deinit();
        std.debug.print("  init: pl_q5_1_q8_1 ok\n", .{});
        var pl_q6_k_q8_1 = try MatvecPipeline.initQ6KQ8_1(&ctx);
        errdefer pl_q6_k_q8_1.deinit();
        std.debug.print("  init: pl_q6_k_q8_1 ok\n", .{});
        var pl_q5_k_q8_1 = try MatvecPipeline.initQ5KQ8_1(&ctx);
        errdefer pl_q5_k_q8_1.deinit();
        std.debug.print("  init: pl_q5_k_q8_1 ok\n", .{});
        var pl_iq4_nl_q8_1 = try MatvecPipeline.initIQ4NLQ8_1(&ctx);
        errdefer pl_iq4_nl_q8_1.deinit();
        std.debug.print("  init: pl_iq4_nl_q8_1 ok\n", .{});
        var pl_quantize_q8_1 = try QuantizeQ8_1Pipeline.init(&ctx);
        errdefer pl_quantize_q8_1.deinit();
        std.debug.print("  init: pl_quantize_q8_1 ok\n", .{});
        var pl_quantize_q8_1_batched = try QuantizeQ8_1BatchedPipeline.init(&ctx);
        errdefer pl_quantize_q8_1_batched.deinit();
        std.debug.print("  init: pl_quantize_q8_1_batched ok\n", .{});
        var pl_q5_1 = try MatvecPipeline.initQ5_1(&ctx);
        errdefer pl_q5_1.deinit();
        std.debug.print("  init: pl_q5_1 ok (VRAM={} MiB GTT={} MiB sys={} MiB)\n", .{ vramUsedMB(), gttUsedMB(), availableMemoryMB() });
        var pl_q5_0 = try MatvecPipeline.initQ5_0(&ctx);
        errdefer pl_q5_0.deinit();
        std.debug.print("  init: pl_q5_0 ok (VRAM={} MiB GTT={} MiB sys={} MiB)\n", .{ vramUsedMB(), gttUsedMB(), availableMemoryMB() });
        var pl_fused_gu = try FusedGateUpPipeline.init(&ctx);
        errdefer pl_fused_gu.deinit();
        std.debug.print("  init: pl_fused_gu ok\n", .{});
        var pl_fused_gu_q8_1 = try FusedGateUpPipeline.initQ8_1(&ctx);
        errdefer pl_fused_gu_q8_1.deinit();
        std.debug.print("  init: pl_fused_gu_q8_1 ok\n", .{});
        var pl_expert_gate_up_id_q3_k = try ExpertGateUpIdPipeline.initQ3KQ8_1(&ctx);
        errdefer pl_expert_gate_up_id_q3_k.deinit();
        std.debug.print("  init: pl_expert_gate_up_id_q3_k ok\n", .{});
        var pl_expert_down_id_q5_1 = try ExpertDownIdPipeline.initQ5_1Q8_1(&ctx);
        errdefer pl_expert_down_id_q5_1.deinit();
        std.debug.print("  init: pl_expert_down_id_q5_1 ok\n", .{});
        var pl_expert_down_id_iq4_nl = try ExpertDownIdPipeline.initIQ4NLQ8_1(&ctx);
        errdefer pl_expert_down_id_iq4_nl.deinit();
        std.debug.print("  init: pl_expert_down_id_iq4_nl ok\n", .{});
        var pl_accum = try AccumPipeline.init(&ctx);
        errdefer pl_accum.deinit();
        std.debug.print("  init: pl_accum ok\n", .{});
        var pl_rmsnorm = try RmsnormPipeline.init(&ctx);
        errdefer pl_rmsnorm.deinit();
        var pl_rmsnorm_perhead = try RmsnormPerHeadPipeline.init(&ctx);
        errdefer pl_rmsnorm_perhead.deinit();
        var pl_elem_add = try ElemAddPipeline.init(&ctx);
        errdefer pl_elem_add.deinit();
        var pl_elem_scale = try ElemScalePipeline.init(&ctx);
        errdefer pl_elem_scale.deinit();
        var pl_gelu_mul = try GeluMulPipeline.init(&ctx);
        errdefer pl_gelu_mul.deinit();
        var pl_rope_table = try RopeNeoxTablePipeline.init(&ctx);
        errdefer pl_rope_table.deinit();
        var pl_rope_theta = try RopeNeoxThetaPipeline.init(&ctx);
        errdefer pl_rope_theta.deinit();
        var pl_attn_qk = try AttnQkSoftmaxPipeline.init(&ctx);
        errdefer pl_attn_qk.deinit();
        var pl_attn_av = try AttnAvPipeline.init(&ctx);
        errdefer pl_attn_av.deinit();
        std.debug.print("  init: pl_rmsnorm + elem + rope ok\n", .{});

        const layers = try allocator.alloc(GpuLayerWeights, g4cfg.n_layers);
        errdefer allocator.free(layers);
        for (layers) |*l| l.* = .{
            .wq = null,
            .wk = null,
            .wv = null,
            .wo = null,
            .w_gate = null,
            .w_up = null,
            .w_down = null,
            .attn_norm_buf = null,
            .post_attention_norm_buf = null,
            .q_norm_buf = null,
            .k_norm_buf = null,
            .ffn_norm_buf = null,
            .pre_ffw_norm_2_buf = null,
            .post_ffw_norm_1_buf = null,
            .post_ffw_norm_2_buf = null,
            .post_ffw_norm_buf = null,
        };

        var gw = GpuWeights{
            .ctx = ctx,
            .pl_f32 = pl_f32,
            .pl_q8_0 = pl_q8_0,
            .pl_q3_k = pl_q3_k,
            .pl_q4_k = pl_q4_k,
            .pl_q3_k_q8_1 = pl_q3_k_q8_1,
            .pl_q4_k_q8_1 = pl_q4_k_q8_1,
            .pl_q5_0_q8_1 = pl_q5_0_q8_1,
            .pl_q5_1_q8_1 = pl_q5_1_q8_1,
            .pl_q6_k_q8_1 = pl_q6_k_q8_1,
            .pl_q5_k_q8_1 = pl_q5_k_q8_1,
            .pl_iq4_nl_q8_1 = pl_iq4_nl_q8_1,
            .pl_quantize_q8_1 = pl_quantize_q8_1,
            .pl_quantize_q8_1_batched = pl_quantize_q8_1_batched,
            .pl_q5_1 = pl_q5_1,
            .pl_q5_0 = pl_q5_0,
            .pl_fused_gu = pl_fused_gu,
            .pl_fused_gu_q8_1 = pl_fused_gu_q8_1,
            .pl_expert_gate_up_id_q3_k = pl_expert_gate_up_id_q3_k,
            .pl_expert_down_id_q5_1 = pl_expert_down_id_q5_1,
            .pl_expert_down_id_iq4_nl = pl_expert_down_id_iq4_nl,
            .pl_accum = pl_accum,
            .pl_rmsnorm = pl_rmsnorm,
            .pl_rmsnorm_perhead = pl_rmsnorm_perhead,
            .pl_elem_add = pl_elem_add,
            .pl_elem_scale = pl_elem_scale,
            .pl_gelu_mul = pl_gelu_mul,
            .pl_rope_table = pl_rope_table,
            .pl_rope_theta = pl_rope_theta,
            .pl_attn_qk = pl_attn_qk,
            .pl_attn_av = pl_attn_av,
            .layers = layers,
            .lm_head = null,
            .shared_vec = null,
            .shared_out = null,
            .shared_acts_q8_1 = null,
            .x_vram = null,
            .xb_vram = null,
            .stage_buf = null,
            .gate_vram = null,
            .up_vram = null,
            .ffn_vram = null,
            .dense_ffn_out_buf = null,
            .attn_in_buf = null,
            .attn_vram = null,
            .out_norm_buf = null,
            .rope_freqs_buf = null,
            .expert_gate = null,
            .expert_up = null,
            .expert_down = null,
            .expert_gate_up_flat = null,
            .expert_down_flat = null,
            .n_experts = g4cfg.n_experts,
            .n_experts_used = g4cfg.n_experts_used,
            .expert_in_buf = null,
            .expert_mid_bufs = null,
            .expert_mid_q8_1_bufs = null,
            .expert_mid_q8_1_flat_buf = null,
            .expert_all_out_buf = null,
            .expert_scales_buf = null,
            .expert_accum_scales_buf = null,
            .expert_ids_buf = null,
            .moe_gpu_buf = null,
            .moe_stage_buf = null,
            .expert_in_slice = null,
            .expert_scales_slice = null,
            .expert_accum_scales_slice = null,
            .expert_ids_slice = null,
            .moe_stage_slice = null,
            .expert_quant_input_dset = null,
            .expert_quant_mid_batched_dset = null,
            .expert_accum_dset = null,
            .expert_gate_up_id_dsets = null,
            .expert_down_id_dsets = null,
            .expert_down_id_dset_is_iq4 = null,
            .expert_reuse_cmds = null,
            .q_out_buf = null,
            .k_out_buf = null,
            .v_out_buf = null,
            .gate_out_buf = null,
            .up_out_buf = null,
            .k_vram = null,
            .v_vram = null,
            .kv_cap = null,
            .kv_stage = null,
            .scores_vram = null,
            .attn_max_win = 0,
            .allocator = allocator,
        };
        errdefer gw.deinit();

        // One vkQueueSubmit per layer: peak staging ~22 MiB instead of ~1 GiB.
        // Preemptive abort: if sys RAM drops below 8 GiB, return error so the
        // caller falls back to CPU instead of letting the OOM killer fire.
        for (0..g4cfg.n_layers) |l| {
            const mem_before = availableMemoryMB();
            if (mem_before < 8 * 1024) {
                std.debug.print("  GPU upload aborted at layer {}: only {} MiB available\n", .{ l, mem_before });
                return error.InsufficientMemory;
            }
            try uploadLayerBatch(&ctx, &gw.layers[l], &g4w.layers[l]);
            // Upload per-layer norm weights to VRAM for GPU rmsnorm dispatches.
            // Each is tiny (≤ 12 KiB for d_model=2816) so we don't bother batching.
            const glw = &gw.layers[l];
            const lwx = &g4w.layers[l];
            glw.attn_norm_buf = try uploadF32(&ctx, lwx.attn_norm);
            glw.post_attention_norm_buf = try uploadF32(&ctx, lwx.post_attention_norm);
            glw.q_norm_buf = try uploadF32(&ctx, lwx.q_norm);
            glw.k_norm_buf = try uploadF32(&ctx, lwx.k_norm);
            glw.ffn_norm_buf = try uploadF32(&ctx, lwx.ffn_norm);
            glw.pre_ffw_norm_2_buf = try uploadF32(&ctx, lwx.pre_ffw_norm_2);
            glw.post_ffw_norm_1_buf = try uploadF32(&ctx, lwx.post_ffw_norm_1);
            glw.post_ffw_norm_2_buf = try uploadF32(&ctx, lwx.post_ffw_norm_2);
            glw.post_ffw_norm_buf = try uploadF32(&ctx, lwx.post_ffw_norm);
            const vram_used = vramUsedMB();
            const gtt_used = gttUsedMB();
            const mem_avail = availableMemoryMB();
            std.debug.print("  layer {:2}: VRAM={} MiB GTT={} MiB sys_avail={} MiB\n", .{ l, vram_used, gtt_used, mem_avail });
        }

        // lm_head staging can be 400+ MiB; fall back to CPU on alloc failure.
        // Global norms + RoPE freqs uploaded once for the whole model.
        gw.out_norm_buf = try uploadF32(&ctx, g4w.out_norm);
        gw.rope_freqs_buf = try uploadF32(&ctx, g4w.rope_freqs);

        gw.lm_head = uploadSingleBatch(&ctx, g4w.lm_head) catch |e| blk: {
            std.debug.print("  lm_head GPU upload failed ({s}), using CPU\n", .{@errorName(e)});
            break :blk null;
        };

        // Create ONE shared vec_buf + out_buf sized to the largest matrix.
        // Replaces 420+ per-session HOST_COHERENT allocations with 2.
        var max_rows: u32 = 1;
        var max_cols: u32 = 1;
        for (gw.layers) |l| {
            for ([_]?MatvecSession{ l.wq, l.wk, l.wv, l.wo, l.w_gate, l.w_up, l.w_down }) |ms| {
                if (ms) |s| {
                    if (s.rows > max_rows) max_rows = s.rows;
                    if (s.cols > max_cols) max_cols = s.cols;
                }
            }
        }
        if (gw.lm_head) |s| {
            if (s.rows > max_rows) max_rows = s.rows;
            if (s.cols > max_cols) max_cols = s.cols;
        }
        gw.shared_vec = try GpuBuffer.initHostCoherent(&gw.ctx, max_cols * @sizeOf(f32), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        gw.shared_out = try GpuBuffer.initHostCoherent(&gw.ctx, max_rows * @sizeOf(f32), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        // Q8_1 activation buffer lives in VRAM; quantize-shader writes, matvec reads.
        const q8_1_bytes = mv_mod.q8_1OutBytes(max_cols);
        gw.shared_acts_q8_1 = try GpuBuffer.initDeviceLocal(&gw.ctx, q8_1_bytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        std.debug.print("  shared I/O bufs: vec={} KiB out={} KiB q8_1_acts={} KiB\n", .{ max_cols * @sizeOf(f32) / 1024, max_rows * @sizeOf(f32) / 1024, q8_1_bytes / 1024 });

        // 7j VRAM residual + scratch + staging. Sized to the maximum f32
        // vector we'll handle (d_model OR n_heads*head_dim_global, whichever
        // is larger — max_cols covers both since wo's cols dimension equals
        // n_heads*head_dim_global).
        gw.x_vram = try GpuBuffer.initDeviceLocal(&gw.ctx, max_cols * @sizeOf(f32), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        gw.xb_vram = try GpuBuffer.initDeviceLocal(&gw.ctx, max_cols * @sizeOf(f32), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        gw.stage_buf = try GpuBuffer.initStaging(&gw.ctx, max_cols * @sizeOf(f32));
        std.debug.print("  7j VRAM bufs: x={} KiB xb={} KiB stage={} KiB\n", .{ max_cols * @sizeOf(f32) / 1024, max_cols * @sizeOf(f32) / 1024, max_cols * @sizeOf(f32) / 1024 });

        // Dense FFN fused-chain buffers.  gate/up_vram hold the gate+up matvec
        // outputs (d_ffn floats); the fused gelu_mul rewrites gate_vram in place
        // as gelu(gate)*up before w_down reads it.  ffn_vram holds the w_down
        // output (d_model floats) before the in-place post_ffw_norm_1 rmsnorm
        // copies into dense_ffn_out_buf for download.
        const dffn_bytes = g4cfg.d_ffn * @sizeOf(f32);
        gw.gate_vram = try GpuBuffer.initDeviceLocal(&gw.ctx, dffn_bytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        gw.up_vram = try GpuBuffer.initDeviceLocal(&gw.ctx, dffn_bytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        gw.ffn_vram = try GpuBuffer.initDeviceLocal(&gw.ctx, g4cfg.d_model * @sizeOf(f32), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        gw.dense_ffn_out_buf = try GpuBuffer.initHostCoherent(&gw.ctx, g4cfg.d_model * @sizeOf(f32), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        std.debug.print("  dense FFN VRAM: gate={} KiB up={} KiB ffn={} KiB out={} KiB\n", .{ dffn_bytes / 1024, dffn_bytes / 1024, g4cfg.d_model * @sizeOf(f32) / 1024, g4cfg.d_model * @sizeOf(f32) / 1024 });

        // wo input/output buffers for the combined attn-residual + dense FFN
        // submit.  attn_in_buf is HOST_COHERENT because the caller uploads
        // attn_concat right before the submit; attn_vram lives in VRAM so the
        // post_attention_norm + elem_add chain runs without PCIe traffic.
        var max_nq: u32 = 0;
        for (gw.layers) |l| if (l.wo) |s| if (s.cols > max_nq) {
            max_nq = s.cols;
        };
        if (max_nq == 0) max_nq = @intCast(g4cfg.d_model);
        gw.attn_in_buf = try GpuBuffer.initHostCoherent(&gw.ctx, max_nq * @sizeOf(f32), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        gw.attn_vram = try GpuBuffer.initDeviceLocal(&gw.ctx, g4cfg.d_model * @sizeOf(f32), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        std.debug.print("  wo fusion bufs: attn_in={} KiB (host) attn_vram={} KiB (VRAM)\n", .{ max_nq * @sizeOf(f32) / 1024, g4cfg.d_model * @sizeOf(f32) / 1024 });

        // Per-projection output buffers for batched QKV and gate+up dispatch.
        // Sized per-projection so all three (or two) can be in-flight simultaneously.
        var max_q_rows: u32 = 1;
        var max_k_rows: u32 = 1;
        var max_v_rows: u32 = 1;
        var max_gate_rows: u32 = 1;
        var max_up_rows: u32 = 1;
        for (gw.layers) |l| {
            if (l.wq) |s| if (s.rows > max_q_rows) {
                max_q_rows = s.rows;
            };
            if (l.wk) |s| if (s.rows > max_k_rows) {
                max_k_rows = s.rows;
            };
            if (l.wv) |s| {
                if (s.rows > max_v_rows) max_v_rows = s.rows;
            } else if (l.wk) |s| {
                if (s.rows > max_v_rows) max_v_rows = s.rows;
            }
            if (l.w_gate) |s| if (s.rows > max_gate_rows) {
                max_gate_rows = s.rows;
            };
            if (l.w_up) |s| if (s.rows > max_up_rows) {
                max_up_rows = s.rows;
            };
        }
        // K and V outputs are vkCmdCopyBuffer sources during the 7l.1 GPU
        // norms+RoPE+KV-append submit. Shared-V global layers also copy K into
        // V before V normalization, so these buffers need both transfer bits.
        const qkv_out_usage: vk.VkBufferUsageFlags = @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        gw.q_out_buf = try GpuBuffer.initHostCoherent(&gw.ctx, max_q_rows * @sizeOf(f32), qkv_out_usage);
        gw.k_out_buf = try GpuBuffer.initHostCoherent(&gw.ctx, max_k_rows * @sizeOf(f32), qkv_out_usage);
        gw.v_out_buf = try GpuBuffer.initHostCoherent(&gw.ctx, max_v_rows * @sizeOf(f32), qkv_out_usage);
        gw.gate_out_buf = try GpuBuffer.initHostCoherent(&gw.ctx, max_gate_rows * @sizeOf(f32), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        gw.up_out_buf = try GpuBuffer.initHostCoherent(&gw.ctx, max_up_rows * @sizeOf(f32), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        std.debug.print("  batch output bufs: q={} KiB k={} KiB v={} KiB gate={} KiB up={} KiB\n", .{
            max_q_rows * @sizeOf(f32) / 1024,  max_k_rows * @sizeOf(f32) / 1024,
            max_v_rows * @sizeOf(f32) / 1024,  max_gate_rows * @sizeOf(f32) / 1024,
            max_up_rows * @sizeOf(f32) / 1024,
        });

        // Upload all MoE experts to VRAM (128 experts × 30 layers ≈ 9.6 GiB).
        // Batched by 16 experts per command buffer (~47 MiB staging peak per batch).
        const n_total = g4cfg.n_layers * g4cfg.n_experts;
        gw.expert_gate = try allocator.alloc(?MatvecSession, n_total);
        @memset(gw.expert_gate.?, null);
        gw.expert_up = try allocator.alloc(?MatvecSession, n_total);
        @memset(gw.expert_up.?, null);
        gw.expert_down = try allocator.alloc(?MatvecSession, n_total);
        @memset(gw.expert_down.?, null);
        const expert_gu_env = std.c.getenv("LLMTOY_EXPERT_GU_ID");
        const enable_gate_up_id = expert_gu_env == null or !std.mem.eql(u8, std.mem.span(expert_gu_env.?), "0");
        gw.expert_gate_up_flat = try allocator.alloc(?MatvecSession, g4cfg.n_layers);
        @memset(gw.expert_gate_up_flat.?, null);
        gw.expert_down_flat = try allocator.alloc(?MatvecSession, g4cfg.n_layers);
        @memset(gw.expert_down_flat.?, null);
        gw.expert_gate_up_id_dsets = try allocator.alloc(?vk.VkDescriptorSet, g4cfg.n_layers);
        @memset(gw.expert_gate_up_id_dsets.?, null);
        gw.expert_down_id_dsets = try allocator.alloc(?vk.VkDescriptorSet, g4cfg.n_layers);
        @memset(gw.expert_down_id_dsets.?, null);
        gw.expert_down_id_dset_is_iq4 = try allocator.alloc(bool, g4cfg.n_layers);
        @memset(gw.expert_down_id_dset_is_iq4.?, false);
        if (std.c.getenv("LLMTOY_EXPERT_REUSE_CMD") != null) {
            gw.expert_reuse_cmds = try allocator.alloc(?vk.VkCommandBuffer, g4cfg.n_layers);
            @memset(gw.expert_reuse_cmds.?, null);
        }
        std.debug.print("  uploading {} experts × {} layers to GPU...\n", .{ g4cfg.n_experts, g4cfg.n_layers });
        for (0..g4cfg.n_layers) |l| {
            if (enable_gate_up_id and isExpertGateUpIdSupported(g4w.layers[l].gate_up_exps.type_)) {
                gw.expert_gate_up_flat.?[l] = uploadSingleBatch(&gw.ctx, g4w.layers[l].gate_up_exps) catch |e| blk: {
                    std.debug.print("  layer {}: flat expert-gate-up upload failed ({s}), using per-expert gate/up\n", .{ l, @errorName(e) });
                    break :blk null;
                };
            }
            if (isExpertDownIdSupported(g4w.layers[l].down_exps.type_)) {
                gw.expert_down_flat.?[l] = uploadSingleBatch(&gw.ctx, g4w.layers[l].down_exps) catch |e| blk: {
                    std.debug.print("  layer {}: flat expert-down upload failed ({s}), using per-expert down\n", .{ l, @errorName(e) });
                    break :blk null;
                };
            }
            const off = l * g4cfg.n_experts;
            uploadExpertsBatch(
                &gw.ctx,
                gw.expert_gate.?[off..][0..g4cfg.n_experts],
                gw.expert_up.?[off..][0..g4cfg.n_experts],
                gw.expert_down.?[off..][0..g4cfg.n_experts],
                &g4w.layers[l],
                g4cfg.d_model,
                g4cfg.d_expert,
                gw.expert_gate_up_flat.?[l] != null,
                gw.expert_down_flat.?[l] != null,
            ) catch |e| std.debug.print("  layer {}: expert upload failed ({s})\n", .{ l, @errorName(e) });
        }
        std.debug.print("  experts done: VRAM={} MiB GTT={} MiB\n", .{ vramUsedMB(), gttUsedMB() });

        // Expert I/O buffers for fused batched dispatch:
        // - expert_in_buf:      HOST_COHERENT, persistently mapped (CPU writes moe_in)
        // - expert_mid_bufs:    device-local VRAM (fused writes, down reads — GPU only)
        // - expert_all_out_buf: device-local VRAM, flat n×d_model; down writes sub-ranges,
        //                       accum reads the whole
        // - expert_scales_buf:  HOST_COHERENT, persistently mapped (CPU writes router scales)
        // - moe_gpu_buf:        device-local VRAM (accum writes, GPU combine reads)
        // - moe_stage_buf:      HOST_COHERENT staging for legacy/debug CPU readback
        const nu = g4cfg.n_experts_used;
        gw.expert_in_buf = try GpuBuffer.initHostCoherent(&gw.ctx, g4cfg.d_model * @sizeOf(f32), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        gw.expert_in_slice = try gw.expert_in_buf.?.mapSlice(f32, g4cfg.d_model);

        gw.expert_mid_bufs = try allocator.alloc(GpuBuffer, nu);
        for (0..nu) |k| {
            gw.expert_mid_bufs.?[k] = try GpuBuffer.initDeviceLocal(&gw.ctx, g4cfg.d_expert * @sizeOf(f32), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        }
        // Per-expert Q8_1 mid buffers (sized for the d_expert column count).
        // 8 experts × q8_1OutBytes(704) = 8 × 864 = 6912 bytes total — trivial.
        const mid_q8_1_bytes = mv_mod.q8_1OutBytes(@intCast(g4cfg.d_expert));
        gw.expert_mid_q8_1_bufs = try allocator.alloc(GpuBuffer, nu);
        for (0..nu) |k| {
            gw.expert_mid_q8_1_bufs.?[k] = try GpuBuffer.initDeviceLocal(&gw.ctx, mid_q8_1_bytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        }
        gw.expert_mid_q8_1_flat_buf = try GpuBuffer.initDeviceLocal(&gw.ctx, nu * mid_q8_1_bytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        gw.expert_all_out_buf = try GpuBuffer.initDeviceLocal(&gw.ctx, nu * g4cfg.d_model * @sizeOf(f32), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        gw.expert_scales_buf = try GpuBuffer.initHostCoherent(&gw.ctx, nu * @sizeOf(f32), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        gw.expert_scales_slice = try gw.expert_scales_buf.?.mapSlice(f32, nu);
        gw.expert_accum_scales_buf = try GpuBuffer.initHostCoherent(&gw.ctx, nu * @sizeOf(f32), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        gw.expert_accum_scales_slice = try gw.expert_accum_scales_buf.?.mapSlice(f32, nu);
        gw.expert_ids_buf = try GpuBuffer.initHostCoherent(&gw.ctx, nu * @sizeOf(u32), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        gw.expert_ids_slice = try gw.expert_ids_buf.?.mapSlice(u32, nu);
        gw.moe_gpu_buf = try GpuBuffer.initDeviceLocal(&gw.ctx, g4cfg.d_model * @sizeOf(f32), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        gw.moe_stage_buf = try GpuBuffer.initStaging(&gw.ctx, g4cfg.d_model * @sizeOf(f32));
        gw.moe_stage_slice = try gw.moe_stage_buf.?.mapSlice(f32, g4cfg.d_model);

        std.debug.print("  expert I/O bufs: in={} KiB, {} × mid={} KiB (VRAM), all_out={} KiB (VRAM), moe_out={} KiB (VRAM)\n", .{
            g4cfg.d_model * @sizeOf(f32) / 1024,
            nu,
            g4cfg.d_expert * @sizeOf(f32) / 1024,
            nu * g4cfg.d_model * @sizeOf(f32) / 1024,
            g4cfg.d_model * @sizeOf(f32) / 1024,
        });

        return gw;
    }

    // 7l.1 — allocate per-layer device-local K/V cache buffers.
    // Sizing mirrors Gemma4KvCache: SWA layers cap to min(max_seq, sliding_window);
    // global layers cap to max_seq. The staging buffer is sized to the largest
    // per-position slot across layers (max nkv(l) * sizeof(f32)).
    //
    // This is the foundation step for Phase 7l (attention on GPU). After
    // initKvVram returns, the buffers exist but are not yet referenced by any
    // forward pass — 7l.1c wires them in.
    pub fn initKvVram(self: *GpuWeights, cfg: Gemma4Config, max_seq: usize) !void {
        if (self.k_vram != null) return error.AlreadyInitialized;
        const n_layers = cfg.n_layers;

        var k = try self.allocator.alloc(GpuBuffer, n_layers);
        errdefer self.allocator.free(k);
        var ki: usize = 0;
        errdefer for (k[0..ki]) |*b| b.deinit();

        var v = try self.allocator.alloc(GpuBuffer, n_layers);
        errdefer self.allocator.free(v);
        var vi: usize = 0;
        errdefer for (v[0..vi]) |*b| b.deinit();

        const cap = try self.allocator.alloc(usize, n_layers);
        errdefer self.allocator.free(cap);

        var max_nkv: usize = 0;
        var total_bytes: usize = 0;
        for (0..n_layers) |l| {
            const seq_cap = if (cfg.is_swa[l]) @min(max_seq, cfg.sliding_window) else max_seq;
            const nkv = cfg.nkv(l);
            const bytes = seq_cap * nkv * @sizeOf(f32);
            cap[l] = seq_cap;
            k[l] = try GpuBuffer.initDeviceLocal(&self.ctx, bytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            ki = l + 1;
            v[l] = try GpuBuffer.initDeviceLocal(&self.ctx, bytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            vi = l + 1;
            if (nkv > max_nkv) max_nkv = nkv;
            total_bytes += 2 * bytes;
        }

        const stage = try GpuBuffer.initStaging(&self.ctx, max_nkv * @sizeOf(f32));
        errdefer {
            var s = stage;
            s.deinit();
        }

        // 7l.2/3 — scores buffer for the fused-attention chain. Sized to
        // n_heads × max(cap[l]) floats so the largest layer fits.
        var max_cap: usize = 0;
        for (cap) |c| {
            if (c > max_cap) max_cap = c;
        }
        const scores_bytes = cfg.n_heads * max_cap * @sizeOf(f32);
        const scores = try GpuBuffer.initDeviceLocal(&self.ctx, scores_bytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        self.k_vram = k;
        self.v_vram = v;
        self.kv_cap = cap;
        self.kv_stage = stage;
        self.scores_vram = scores;
        self.attn_max_win = max_cap;

        std.debug.print("  KV VRAM: total={} MiB, stage={} KiB, scores={} KiB\n", .{ total_bytes / (1024 * 1024), (max_nkv * @sizeOf(f32)) / 1024, scores_bytes / 1024 });
    }

    // Upload one position's K or V row into the per-layer VRAM cache at slot.
    // Used during 7l.1 bring-up when the CPU still computes K/V; replaced in
    // 7l.1c by an in-submit `vkCmdCopyBuffer` from the matvec output buffer.
    pub fn uploadKvSlotK(self: *const GpuWeights, layer: usize, slot: usize, k: []const f32) !void {
        const k_vram = self.k_vram orelse return error.NotOnGpu;
        const stage = &(self.kv_stage orelse return error.NotOnGpu);
        const bytes = k.len * @sizeOf(f32);
        try stage.upload(std.mem.sliceAsBytes(k));
        try self.ctx.copyBufferRegion(stage.handle, k_vram[layer].handle, 0, slot * bytes, bytes);
    }
    pub fn uploadKvSlotV(self: *const GpuWeights, layer: usize, slot: usize, v: []const f32) !void {
        const v_vram = self.v_vram orelse return error.NotOnGpu;
        const stage = &(self.kv_stage orelse return error.NotOnGpu);
        const bytes = v.len * @sizeOf(f32);
        try stage.upload(std.mem.sliceAsBytes(v));
        try self.ctx.copyBufferRegion(stage.handle, v_vram[layer].handle, 0, slot * bytes, bytes);
    }

    // Download one position's K or V row from the per-layer VRAM cache. For
    // testing / verification — production paths read the cache directly from
    // attention shaders.
    pub fn downloadKvSlotK(self: *const GpuWeights, layer: usize, slot: usize, k: []f32) !void {
        const k_vram = self.k_vram orelse return error.NotOnGpu;
        const stage = &(self.kv_stage orelse return error.NotOnGpu);
        const bytes = k.len * @sizeOf(f32);
        try self.ctx.copyBufferRegion(k_vram[layer].handle, stage.handle, slot * bytes, 0, bytes);
        try stage.download(std.mem.sliceAsBytes(k));
    }
    pub fn downloadKvSlotV(self: *const GpuWeights, layer: usize, slot: usize, v: []f32) !void {
        const v_vram = self.v_vram orelse return error.NotOnGpu;
        const stage = &(self.kv_stage orelse return error.NotOnGpu);
        const bytes = v.len * @sizeOf(f32);
        try self.ctx.copyBufferRegion(v_vram[layer].handle, stage.handle, slot * bytes, 0, bytes);
        try stage.download(std.mem.sliceAsBytes(v));
    }

    pub fn deinit(self: *GpuWeights) void {
        if (self.scores_vram) |*b| b.deinit();
        if (self.kv_stage) |*b| b.deinit();
        if (self.kv_cap) |c| self.allocator.free(c);
        if (self.v_vram) |bs| {
            for (bs) |*b| b.deinit();
            self.allocator.free(bs);
        }
        if (self.k_vram) |bs| {
            for (bs) |*b| b.deinit();
            self.allocator.free(bs);
        }
        if (self.expert_reuse_cmds) |cmds| {
            for (cmds) |cmd| if (cmd) |c| self.ctx.freeReusableCommandBuffer(c);
            self.allocator.free(cmds);
        }
        if (self.expert_down_id_dsets) |sets| {
            for (sets, 0..) |*ds, i| if (ds.*) |set| {
                var tmp = set;
                const is_iq4 = if (self.expert_down_id_dset_is_iq4) |flags| flags[i] else false;
                const pool = if (is_iq4) self.pl_expert_down_id_iq4_nl.desc_pool else self.pl_expert_down_id_q5_1.desc_pool;
                _ = vk.vkFreeDescriptorSets(self.ctx.device, pool, 1, &tmp);
            };
            self.allocator.free(sets);
        }
        if (self.expert_down_id_dset_is_iq4) |flags| self.allocator.free(flags);
        if (self.expert_gate_up_id_dsets) |sets| {
            for (sets) |*ds| if (ds.*) |set| {
                var tmp = set;
                _ = vk.vkFreeDescriptorSets(self.ctx.device, self.pl_expert_gate_up_id_q3_k.desc_pool, 1, &tmp);
            };
            self.allocator.free(sets);
        }
        if (self.expert_accum_dset) |set| {
            var tmp = set;
            _ = vk.vkFreeDescriptorSets(self.ctx.device, self.pl_accum.desc_pool, 1, &tmp);
        }
        if (self.expert_quant_mid_batched_dset) |set| {
            var tmp = set;
            _ = vk.vkFreeDescriptorSets(self.ctx.device, self.pl_quantize_q8_1_batched.desc_pool, 1, &tmp);
        }
        if (self.expert_quant_input_dset) |set| {
            var tmp = set;
            _ = vk.vkFreeDescriptorSets(self.ctx.device, self.pl_quantize_q8_1.desc_pool, 1, &tmp);
        }
        if (self.moe_stage_buf) |*b| {
            b.unmap();
            b.deinit();
        }
        if (self.moe_gpu_buf) |*b| b.deinit();
        if (self.expert_ids_buf) |*b| {
            b.unmap();
            b.deinit();
        }
        if (self.expert_accum_scales_buf) |*b| {
            b.unmap();
            b.deinit();
        }
        if (self.expert_scales_buf) |*b| {
            b.unmap();
            b.deinit();
        }
        if (self.expert_all_out_buf) |*b| b.deinit();
        if (self.expert_mid_q8_1_flat_buf) |*b| b.deinit();
        if (self.expert_mid_q8_1_bufs) |bs| {
            for (bs) |*b| b.deinit();
            self.allocator.free(bs);
        }
        if (self.expert_mid_bufs) |bs| {
            for (bs) |*b| b.deinit();
            self.allocator.free(bs);
        }
        if (self.expert_in_buf) |*b| {
            b.unmap();
            b.deinit();
        }
        if (self.expert_down) |ed| {
            for (ed) |*ms| if (ms.*) |*s| s.deinit();
            self.allocator.free(ed);
        }
        if (self.expert_down_flat) |ed| {
            for (ed) |*ms| if (ms.*) |*s| s.deinit();
            self.allocator.free(ed);
        }
        if (self.expert_gate_up_flat) |eguf| {
            for (eguf) |*ms| if (ms.*) |*s| s.deinit();
            self.allocator.free(eguf);
        }
        if (self.expert_up) |eu| {
            for (eu) |*ms| if (ms.*) |*s| s.deinit();
            self.allocator.free(eu);
        }
        if (self.expert_gate) |eg| {
            for (eg) |*ms| if (ms.*) |*s| s.deinit();
            self.allocator.free(eg);
        }
        if (self.up_out_buf) |*b| b.deinit();
        if (self.gate_out_buf) |*b| b.deinit();
        if (self.v_out_buf) |*b| b.deinit();
        if (self.k_out_buf) |*b| b.deinit();
        if (self.q_out_buf) |*b| b.deinit();
        if (self.rope_freqs_buf) |*b| b.deinit();
        if (self.out_norm_buf) |*b| b.deinit();
        if (self.attn_vram) |*b| b.deinit();
        if (self.attn_in_buf) |*b| b.deinit();
        if (self.dense_ffn_out_buf) |*b| b.deinit();
        if (self.ffn_vram) |*b| b.deinit();
        if (self.up_vram) |*b| b.deinit();
        if (self.gate_vram) |*b| b.deinit();
        if (self.stage_buf) |*b| b.deinit();
        if (self.xb_vram) |*b| b.deinit();
        if (self.x_vram) |*b| b.deinit();
        if (self.shared_acts_q8_1) |*b| b.deinit();
        if (self.shared_out) |*b| b.deinit();
        if (self.shared_vec) |*b| b.deinit();
        if (self.lm_head) |*s| s.deinit();
        for (self.layers) |*l| l.deinitAll();
        self.allocator.free(self.layers);
        self.pl_accum.deinit();
        self.pl_expert_down_id_iq4_nl.deinit();
        self.pl_expert_down_id_q5_1.deinit();
        self.pl_expert_gate_up_id_q3_k.deinit();
        self.pl_fused_gu_q8_1.deinit();
        self.pl_fused_gu.deinit();
        self.pl_q5_0.deinit();
        self.pl_q5_1.deinit();
        self.pl_attn_av.deinit();
        self.pl_attn_qk.deinit();
        self.pl_rope_theta.deinit();
        self.pl_rope_table.deinit();
        self.pl_gelu_mul.deinit();
        self.pl_elem_scale.deinit();
        self.pl_elem_add.deinit();
        self.pl_rmsnorm_perhead.deinit();
        self.pl_rmsnorm.deinit();
        self.pl_quantize_q8_1_batched.deinit();
        self.pl_quantize_q8_1.deinit();
        self.pl_q5_1_q8_1.deinit();
        self.pl_q5_0_q8_1.deinit();
        self.pl_q6_k_q8_1.deinit();
        self.pl_q5_k_q8_1.deinit();
        self.pl_iq4_nl_q8_1.deinit();
        self.pl_q4_k_q8_1.deinit();
        self.pl_q3_k_q8_1.deinit();
        self.pl_q4_k.deinit();
        self.pl_q3_k.deinit();
        self.pl_q8_0.deinit();
        self.pl_f32.deinit();
        self.ctx.deinit();
    }

    // Dispatch wq, wk, (optionally wv) in a single command buffer.
    // All three read the same xb → upload once, 2–3 parallel dispatches, 1 submit.
    // Pipelines are passed by the caller so gpu_weights.zig doesn't need raw types.
    // Returns error.NotOnGpu if any needed session is missing.
    pub fn runLayerQKV(
        self: *const GpuWeights,
        layer: usize,
        wq_pl: *const MatvecPipeline,
        wk_pl: *const MatvecPipeline,
        wv_pl: ?*const MatvecPipeline,
        xb: []const f32,
        q_out: []f32,
        k_out: []f32,
        v_out: []f32,
    ) !void {
        const lw = &self.layers[layer];
        const wq = lw.wq orelse return error.NotOnGpu;
        const wk = lw.wk orelse return error.NotOnGpu;
        const q_buf = &(self.q_out_buf orelse return error.NotOnGpu);
        const k_buf = &(self.k_out_buf orelse return error.NotOnGpu);

        try self.shared_vec.?.upload(std.mem.sliceAsBytes(xb));

        const cmd = try self.ctx.beginBatch();
        var q_dset: vk.VkDescriptorSet = null;
        var k_dset: vk.VkDescriptorSet = null;
        var v_dset: ?vk.VkDescriptorSet = null;

        q_dset = try wq.recordMv(cmd, wq_pl, &self.shared_vec.?, q_buf);
        k_dset = try wk.recordMv(cmd, wk_pl, &self.shared_vec.?, k_buf);
        if (wv_pl) |vpl| {
            const wv = lw.wv orelse return error.NotOnGpu;
            const v_buf = &(self.v_out_buf orelse return error.NotOnGpu);
            v_dset = try wv.recordMv(cmd, vpl, &self.shared_vec.?, v_buf);
        }

        try self.ctx.submitBatch(cmd);

        _ = vk.vkFreeDescriptorSets(self.ctx.device, wq_pl.desc_pool, 1, &q_dset);
        _ = vk.vkFreeDescriptorSets(self.ctx.device, wk_pl.desc_pool, 1, &k_dset);
        if (v_dset) |*ds| _ = vk.vkFreeDescriptorSets(self.ctx.device, wv_pl.?.desc_pool, 1, ds);

        try q_buf.download(std.mem.sliceAsBytes(q_out));
        try k_buf.download(std.mem.sliceAsBytes(k_out));
        if (v_dset != null) try self.v_out_buf.?.download(std.mem.sliceAsBytes(v_out));
    }

    // Q8_1-activation matvec: one f32 → Q8_1 quantize pass, then one matvec.
    // The two dispatches share a command buffer with a compute-compute barrier
    // between them so the matvec sees the freshly-written Q8_1 acts.
    // Caller supplies the matvec pipeline via q8_1PipelineFor(t).
    pub fn runQ8_1Mv(
        self: *const GpuWeights,
        pl: *const MatvecPipeline,
        sess: *const MatvecSession,
        xb: []const f32,
        out: []f32,
    ) !void {
        std.debug.assert(sess.cols % 32 == 0);
        const vec_buf = &self.shared_vec.?;
        const out_buf = &self.shared_out.?;
        const acts_buf = &self.shared_acts_q8_1.?;

        try vec_buf.upload(std.mem.sliceAsBytes(xb));

        const cmd = try self.ctx.beginBatch();
        var quant_label_buf: [48]u8 = undefined;
        const quant_label = std.fmt.bufPrint(&quant_label_buf, "quantize_q8_1.cols{}", .{sess.cols}) catch "quantize_q8_1";
        const p_quant = self.ctx.profileBegin(cmd, quant_label);
        const q_dset = try self.pl_quantize_q8_1.record(cmd, vec_buf, acts_buf, sess.cols);
        self.ctx.profileEnd(cmd, p_quant);
        GpuCtx.recordShaderBarrier(cmd);
        var mv_label_buf: [72]u8 = undefined;
        const mv_label = std.fmt.bufPrint(&mv_label_buf, "matvec_q8_1.single.{}x{}", .{ sess.rows, sess.cols }) catch "matvec_q8_1.single";
        const p_mv = self.ctx.profileBegin(cmd, mv_label);
        const mv_dset = try pl.record(cmd, &sess.mat_buf, acts_buf, out_buf, sess.rows, sess.cols);
        self.ctx.profileEnd(cmd, p_mv);
        try self.ctx.submitBatch(cmd);

        _ = vk.vkFreeDescriptorSets(self.ctx.device, self.pl_quantize_q8_1.desc_pool, 1, &q_dset);
        _ = vk.vkFreeDescriptorSets(self.ctx.device, pl.desc_pool, 1, &mv_dset);

        try out_buf.download(std.mem.sliceAsBytes(out));
    }

    // QKV variant of runQ8_1Mv: one upload, one quantize, then 2–3 matvecs all
    // reading the same Q8_1 acts buffer. Saves 2 PCIe uploads + 2 quantize
    // dispatches vs three independent runQ8_1Mv calls.
    //
    // Caller must have already verified that wq, wk (and wv if not shared) are
    // all Q4_K and have GPU sessions. `v_out` is null for shared-V layers
    // (caller will @memcpy k into v).
    pub fn runLayerQKVQ8_1(
        self: *const GpuWeights,
        layer: usize,
        wq_pl: *const MatvecPipeline,
        wk_pl: *const MatvecPipeline,
        wv_pl: ?*const MatvecPipeline, // null when wv is shared (callers @memcpy k → v)
        xb: []const f32,
        q_out: []f32,
        k_out: []f32,
        v_out: ?[]f32,
    ) !void {
        const lw = &self.layers[layer];
        const wq = lw.wq orelse return error.NotOnGpu;
        const wk = lw.wk orelse return error.NotOnGpu;
        std.debug.assert(wq.cols % 256 == 0);
        std.debug.assert(wq.cols == wk.cols);

        const vec_buf = &self.shared_vec.?;
        const acts_buf = &self.shared_acts_q8_1.?;
        const q_buf = &(self.q_out_buf orelse return error.NotOnGpu);
        const k_buf = &(self.k_out_buf orelse return error.NotOnGpu);

        try vec_buf.upload(std.mem.sliceAsBytes(xb));

        const cmd = try self.ctx.beginBatch();
        const quant_dset = try self.pl_quantize_q8_1.record(cmd, vec_buf, acts_buf, wq.cols);
        GpuCtx.recordShaderBarrier(cmd);

        const q_mv_dset = try wq_pl.record(cmd, &wq.mat_buf, acts_buf, q_buf, wq.rows, wq.cols);
        const k_mv_dset = try wk_pl.record(cmd, &wk.mat_buf, acts_buf, k_buf, wk.rows, wk.cols);

        var v_mv_dset: ?vk.VkDescriptorSet = null;
        if (v_out != null) {
            const wv = lw.wv orelse return error.NotOnGpu;
            const v_buf = &(self.v_out_buf orelse return error.NotOnGpu);
            const v_pl = wv_pl orelse return error.NoQ8_1Pipeline;
            std.debug.assert(wv.cols == wq.cols);
            v_mv_dset = try v_pl.record(cmd, &wv.mat_buf, acts_buf, v_buf, wv.rows, wv.cols);
        }

        try self.ctx.submitBatch(cmd);

        const dev = self.ctx.device;
        _ = vk.vkFreeDescriptorSets(dev, self.pl_quantize_q8_1.desc_pool, 1, &quant_dset);
        _ = vk.vkFreeDescriptorSets(dev, wq_pl.desc_pool, 1, &q_mv_dset);
        _ = vk.vkFreeDescriptorSets(dev, wk_pl.desc_pool, 1, &k_mv_dset);
        if (v_mv_dset) |*ds| _ = vk.vkFreeDescriptorSets(dev, wv_pl.?.desc_pool, 1, ds);

        try q_buf.download(std.mem.sliceAsBytes(q_out));
        try k_buf.download(std.mem.sliceAsBytes(k_out));
        if (v_out) |v| try self.v_out_buf.?.download(std.mem.sliceAsBytes(v));
    }

    // 7j-integrated attention path: same as runLayerQKVQ8_1 but with attn_norm
    // folded into the same submit. Caller passes the unnormalized residual `x`;
    // the GPU runs rmsnorm(x, attn_norm) → xb_vram, then quantize → shared_acts_q8_1,
    // then 2–3 QKV matvecs. One submit replaces the CPU rmsnorm + the existing
    // QKV submit. Saves one xb upload and 30 × ~10 µs of CPU work per token.
    pub fn runLayerAttnQ8_1(
        self: *const GpuWeights,
        layer: usize,
        eps: f32,
        wq_pl: *const MatvecPipeline,
        wk_pl: *const MatvecPipeline,
        wv_pl: ?*const MatvecPipeline, // null when wv is shared (caller @memcpy k → v)
        x: []const f32, // unnormalized residual
        q_out: []f32,
        k_out: []f32,
        v_out: ?[]f32,
    ) !void {
        const lw = &self.layers[layer];
        const wq = lw.wq orelse return error.NotOnGpu;
        const wk = lw.wk orelse return error.NotOnGpu;
        const attn_norm_buf = &(lw.attn_norm_buf orelse return error.NotOnGpu);
        std.debug.assert(wq.cols % 256 == 0); // rmsnorm shader requires n % 256 == 0
        std.debug.assert(wq.cols == wk.cols);
        std.debug.assert(wq.cols == x.len);

        const vec_buf = &self.shared_vec.?; // x lives here (HOST_COHERENT)
        const xb_buf = &(self.xb_vram orelse return error.NotOnGpu); // rmsnorm output (VRAM)
        const acts_buf = &self.shared_acts_q8_1.?; // quantize output (VRAM)
        const q_buf = &(self.q_out_buf orelse return error.NotOnGpu);
        const k_buf = &(self.k_out_buf orelse return error.NotOnGpu);

        try vec_buf.upload(std.mem.sliceAsBytes(x));

        const cmd = try self.ctx.beginBatch();
        const norm_dset = try self.recordLayerRmsnorm(cmd, layer, vec_buf, attn_norm_buf, xb_buf, @intCast(x.len), eps, false);
        GpuCtx.recordShaderBarrier(cmd);

        const quant_dset = try self.pl_quantize_q8_1.record(cmd, xb_buf, acts_buf, wq.cols);
        GpuCtx.recordShaderBarrier(cmd);

        const q_mv_dset = try wq_pl.record(cmd, &wq.mat_buf, acts_buf, q_buf, wq.rows, wq.cols);
        const k_mv_dset = try wk_pl.record(cmd, &wk.mat_buf, acts_buf, k_buf, wk.rows, wk.cols);

        var v_mv_dset: ?vk.VkDescriptorSet = null;
        if (v_out != null) {
            const wv = lw.wv orelse return error.NotOnGpu;
            const v_buf = &(self.v_out_buf orelse return error.NotOnGpu);
            const v_pl = wv_pl orelse return error.NoQ8_1Pipeline;
            std.debug.assert(wv.cols == wq.cols);
            v_mv_dset = try v_pl.record(cmd, &wv.mat_buf, acts_buf, v_buf, wv.rows, wv.cols);
        }

        try self.ctx.submitBatch(cmd);

        const dev = self.ctx.device;
        _ = vk.vkFreeDescriptorSets(dev, self.pl_rmsnorm.desc_pool, 1, &norm_dset);
        _ = vk.vkFreeDescriptorSets(dev, self.pl_quantize_q8_1.desc_pool, 1, &quant_dset);
        _ = vk.vkFreeDescriptorSets(dev, wq_pl.desc_pool, 1, &q_mv_dset);
        _ = vk.vkFreeDescriptorSets(dev, wk_pl.desc_pool, 1, &k_mv_dset);
        if (v_mv_dset) |*ds| _ = vk.vkFreeDescriptorSets(dev, wv_pl.?.desc_pool, 1, ds);

        try q_buf.download(std.mem.sliceAsBytes(q_out));
        try k_buf.download(std.mem.sliceAsBytes(k_out));
        if (v_out) |v| try self.v_out_buf.?.download(std.mem.sliceAsBytes(v));
    }

    // 7l.1c — full attention front-end on the GPU in one submit.
    //
    // Same chain as runLayerAttnQ8_1 plus:
    //   - per-head Q rmsnorm  (q_norm, Gemma 1+w convention) in-place on q_out_buf
    //   - per-head K rmsnorm  (k_norm, Gemma 1+w convention) in-place on k_out_buf
    //   - per-head V rmsnormRaw                              in-place on v_out_buf
    //   - RoPE on Q (table for global, theta for SWA)
    //   - RoPE on K
    //   - vkCmdCopyBuffer K and V into the per-layer VRAM KV cache at slot offset
    //
    // The caller still receives Q, K, V on the CPU (q_out/k_out/v_out) because
    // 7l.2/3 (GPU Q·K^T+softmax / attn·V) aren't here yet — the CPU sdpAttn
    // still drives attention compute. Once those land, the CPU downloads
    // disappear and Q stays in VRAM too.
    //
    // If wv_pl is null, V is shared from the raw K projection. This covers
    // Gemma4 global layers without falling back to CPU attention.
    pub fn runLayerAttnQ8_1KvVram(
        self: *const GpuWeights,
        layer: usize,
        eps: f32,
        wq_pl: *const MatvecPipeline,
        wk_pl: *const MatvecPipeline,
        wv_pl: ?*const MatvecPipeline,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        pos: u32,
        is_swa: bool,
        rope_theta_swa: f32,
        kv_slot: u32,
        x: []const f32,
        q_out: []f32,
        k_out: []f32,
        v_out: []f32,
    ) !void {
        const lw = &self.layers[layer];
        const wq = lw.wq orelse return error.NotOnGpu;
        const wk = lw.wk orelse return error.NotOnGpu;
        const attn_norm_buf = &(lw.attn_norm_buf orelse return error.NotOnGpu);
        const q_norm_buf = &(lw.q_norm_buf orelse return error.NotOnGpu);
        const k_norm_buf = &(lw.k_norm_buf orelse return error.NotOnGpu);
        std.debug.assert(wq.cols % 256 == 0);
        std.debug.assert(wq.cols == wk.cols);
        std.debug.assert(wq.cols == x.len);
        std.debug.assert(head_dim % 256 == 0);

        const vec_buf = &self.shared_vec.?;
        const xb_buf = &(self.xb_vram orelse return error.NotOnGpu);
        const acts_buf = &self.shared_acts_q8_1.?;
        const q_buf = &(self.q_out_buf orelse return error.NotOnGpu);
        const k_buf = &(self.k_out_buf orelse return error.NotOnGpu);
        const v_buf = &(self.v_out_buf orelse return error.NotOnGpu);
        const k_cache = &((self.k_vram orelse return error.NotOnGpu)[layer]);
        const v_cache = &((self.v_vram orelse return error.NotOnGpu)[layer]);
        const rope_freqs_buf = &(self.rope_freqs_buf orelse return error.NotOnGpu);

        const nkv = @as(u32, @intCast(n_kv_heads * head_dim));
        const slot_bytes: vk.VkDeviceSize = @as(vk.VkDeviceSize, nkv) * @sizeOf(f32);
        const slot_offset: vk.VkDeviceSize = @as(vk.VkDeviceSize, kv_slot) * slot_bytes;

        try vec_buf.upload(std.mem.sliceAsBytes(x));

        const cmd = try self.ctx.beginBatch();

        // ── 1. attn_norm(x) → xb_vram
        const p_norm = self.ctx.profileBegin(cmd, "attn_front.rmsnorm");
        const norm_dset = try self.recordLayerRmsnorm(cmd, layer, vec_buf, attn_norm_buf, xb_buf, @intCast(x.len), eps, false);
        self.ctx.profileEnd(cmd, p_norm);
        GpuCtx.recordShaderBarrier(cmd);

        // ── 2. quantize(xb_vram) → acts
        const p_quant = self.ctx.profileBegin(cmd, "attn_front.quantize_q8_1");
        const quant_dset = try self.pl_quantize_q8_1.record(cmd, xb_buf, acts_buf, wq.cols);
        self.ctx.profileEnd(cmd, p_quant);
        GpuCtx.recordShaderBarrier(cmd);

        // ── 3. QKV matvecs (parallel; share acts read)
        const p_q = self.ctx.profileBegin(cmd, "attn_front.wq");
        const q_mv_dset = try wq_pl.record(cmd, &wq.mat_buf, acts_buf, q_buf, wq.rows, wq.cols);
        self.ctx.profileEnd(cmd, p_q);
        const p_k = self.ctx.profileBegin(cmd, "attn_front.wk");
        const k_mv_dset = try wk_pl.record(cmd, &wk.mat_buf, acts_buf, k_buf, wk.rows, wk.cols);
        self.ctx.profileEnd(cmd, p_k);
        var v_mv_dset: ?vk.VkDescriptorSet = null;
        if (wv_pl) |vpl| {
            const wv = lw.wv orelse return error.NotOnGpu;
            std.debug.assert(wq.cols == wv.cols);
            const p_v = self.ctx.profileBegin(cmd, "attn_front.wv");
            v_mv_dset = try vpl.record(cmd, &wv.mat_buf, acts_buf, v_buf, wv.rows, wv.cols);
            self.ctx.profileEnd(cmd, p_v);
            GpuCtx.recordShaderBarrier(cmd);
        } else {
            GpuCtx.recordShaderBarrier(cmd);
            GpuCtx.recordShaderToTransferBarrier(cmd);
            const p_v_copy = self.ctx.profileBegin(cmd, "attn_front.v_from_k_copy");
            GpuCtx.recordCopy(cmd, k_buf.handle, v_buf.handle, slot_bytes);
            self.ctx.profileEnd(cmd, p_v_copy);
            GpuCtx.recordTransferToShaderBarrier(cmd);
        }

        // ── 4. Per-head normalization in-place
        //   Q: rmsnorm with q_norm,    Gemma (1+w) convention
        //   K: rmsnorm with k_norm,    Gemma (1+w) convention
        //   V: rmsnormRaw (no weight). v_buf is bound to the W slot too —
        //      the shader doesn't read it when use_weight==0, so any valid
        //      buffer works.
        const p_qn = self.ctx.profileBegin(cmd, "attn_front.q_norm");
        const qn_dset = try self.pl_rmsnorm_perhead.record(cmd, q_buf, q_norm_buf, q_buf, n_heads, head_dim, eps, true, true);
        self.ctx.profileEnd(cmd, p_qn);
        const p_kn = self.ctx.profileBegin(cmd, "attn_front.k_norm");
        const kn_dset = try self.pl_rmsnorm_perhead.record(cmd, k_buf, k_norm_buf, k_buf, n_kv_heads, head_dim, eps, true, true);
        self.ctx.profileEnd(cmd, p_kn);
        const p_vn = self.ctx.profileBegin(cmd, "attn_front.v_norm");
        const vn_dset = try self.pl_rmsnorm_perhead.record(cmd, v_buf, v_buf, v_buf, n_kv_heads, head_dim, eps, false, false);
        self.ctx.profileEnd(cmd, p_vn);
        GpuCtx.recordShaderBarrier(cmd);

        // ── 5. RoPE on Q and K (V isn't rotated)
        var qr_dset: vk.VkDescriptorSet = null;
        var kr_dset: vk.VkDescriptorSet = null;
        var rope_pool: vk.VkDescriptorPool = undefined;
        if (is_swa) {
            const p_rope_q = self.ctx.profileBegin(cmd, "attn_front.rope_q_theta");
            qr_dset = try self.pl_rope_theta.record(cmd, q_buf, pos, head_dim, rope_theta_swa, n_heads);
            self.ctx.profileEnd(cmd, p_rope_q);
            const p_rope_k = self.ctx.profileBegin(cmd, "attn_front.rope_k_theta");
            kr_dset = try self.pl_rope_theta.record(cmd, k_buf, pos, head_dim, rope_theta_swa, n_kv_heads);
            self.ctx.profileEnd(cmd, p_rope_k);
            rope_pool = self.pl_rope_theta.desc_pool;
        } else {
            const p_rope_q = self.ctx.profileBegin(cmd, "attn_front.rope_q_table");
            qr_dset = try self.pl_rope_table.record(cmd, q_buf, rope_freqs_buf, pos, head_dim, n_heads);
            self.ctx.profileEnd(cmd, p_rope_q);
            const p_rope_k = self.ctx.profileBegin(cmd, "attn_front.rope_k_table");
            kr_dset = try self.pl_rope_table.record(cmd, k_buf, rope_freqs_buf, pos, head_dim, n_kv_heads);
            self.ctx.profileEnd(cmd, p_rope_k);
            rope_pool = self.pl_rope_table.desc_pool;
        }
        GpuCtx.recordShaderToTransferBarrier(cmd);

        // ── 6. Append K, V to the per-layer VRAM cache at this slot
        const p_k_copy = self.ctx.profileBegin(cmd, "attn_front.k_cache_copy");
        GpuCtx.recordCopyRegion(cmd, k_buf.handle, k_cache.handle, 0, slot_offset, slot_bytes);
        self.ctx.profileEnd(cmd, p_k_copy);
        const p_v_copy = self.ctx.profileBegin(cmd, "attn_front.v_cache_copy");
        GpuCtx.recordCopyRegion(cmd, v_buf.handle, v_cache.handle, 0, slot_offset, slot_bytes);
        self.ctx.profileEnd(cmd, p_v_copy);

        try self.ctx.submitBatch(cmd);

        const dev = self.ctx.device;
        _ = vk.vkFreeDescriptorSets(dev, self.pl_rmsnorm.desc_pool, 1, &norm_dset);
        _ = vk.vkFreeDescriptorSets(dev, self.pl_quantize_q8_1.desc_pool, 1, &quant_dset);
        _ = vk.vkFreeDescriptorSets(dev, wq_pl.desc_pool, 1, &q_mv_dset);
        _ = vk.vkFreeDescriptorSets(dev, wk_pl.desc_pool, 1, &k_mv_dset);
        if (v_mv_dset) |*ds| _ = vk.vkFreeDescriptorSets(dev, wv_pl.?.desc_pool, 1, ds);
        _ = vk.vkFreeDescriptorSets(dev, self.pl_rmsnorm_perhead.desc_pool, 1, &qn_dset);
        _ = vk.vkFreeDescriptorSets(dev, self.pl_rmsnorm_perhead.desc_pool, 1, &kn_dset);
        _ = vk.vkFreeDescriptorSets(dev, self.pl_rmsnorm_perhead.desc_pool, 1, &vn_dset);
        _ = vk.vkFreeDescriptorSets(dev, rope_pool, 1, &qr_dset);
        _ = vk.vkFreeDescriptorSets(dev, rope_pool, 1, &kr_dset);

        try q_buf.download(std.mem.sliceAsBytes(q_out));
        try k_buf.download(std.mem.sliceAsBytes(k_out));
        try v_buf.download(std.mem.sliceAsBytes(v_out));
    }

    // 7l.2/3 — GPU attention compute. Replaces the per-head CPU sdpAttn loop.
    //
    // Reads:
    //   - Q from q_out_buf (host-coherent — GPU still reads it; has the rope'd
    //     norm'd Q for this token, written by runLayerAttnQ8_1KvVram).
    //   - K from k_vram[layer] (per-layer VRAM cache, populated by 7l.1c).
    //   - V from v_vram[layer] (per-layer VRAM cache).
    //
    // Writes:
    //   - attn_in_buf (host-coherent, sized max_nq). After this submit, the
    //     wo input is already on the GPU side; runLayerAttnResidualDenseFfnQ8_1
    //     should skip its own upload (caller passes skip_attn_upload=true).
    //
    // For verification / non-full-fused wo paths, the caller can also receive
    // a CPU copy of attn_concat via `attn_out`. Pass null to skip the download.
    pub fn runLayerAttention(
        self: *const GpuWeights,
        layer: usize,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        pos: u32,
        win_len: u32,
        cap: u32,
        scale: f32,
        attn_out: ?[]f32,
    ) !void {
        const q_buf = &(self.q_out_buf orelse return error.NotOnGpu);
        const k_cache = &((self.k_vram orelse return error.NotOnGpu)[layer]);
        const v_cache = &((self.v_vram orelse return error.NotOnGpu)[layer]);
        const scores_buf = &(self.scores_vram orelse return error.NotOnGpu);
        const attn_buf = &(self.attn_in_buf orelse return error.NotOnGpu);

        const seq = pos + 1;
        const n_q_per_kv = n_heads / n_kv_heads;

        const cmd = try self.ctx.beginBatch();
        const p_qk = self.ctx.profileBegin(cmd, "attention.qk_softmax");
        const qk_dset = try self.pl_attn_qk.record(cmd, q_buf, k_cache, scores_buf, n_heads, seq, win_len, head_dim, n_kv_heads, n_q_per_kv, cap, scale);
        self.ctx.profileEnd(cmd, p_qk);
        GpuCtx.recordShaderBarrier(cmd);
        const p_av = self.ctx.profileBegin(cmd, "attention.av");
        const av_dset = try self.pl_attn_av.record(cmd, scores_buf, v_cache, attn_buf, n_heads, seq, win_len, head_dim, n_kv_heads, n_q_per_kv, cap);
        self.ctx.profileEnd(cmd, p_av);
        try self.ctx.submitBatch(cmd);

        const dev = self.ctx.device;
        _ = vk.vkFreeDescriptorSets(dev, self.pl_attn_qk.desc_pool, 1, &qk_dset);
        _ = vk.vkFreeDescriptorSets(dev, self.pl_attn_av.desc_pool, 1, &av_dset);

        if (attn_out) |o| {
            try attn_buf.download(std.mem.sliceAsBytes(o));
        }
    }

    // Pipeline for the Q8_1-activation integer-dot path. Returns null when
    // the weight type has no Q8_1 shader yet (Q5_K, Q5_0, Q5_1, Q8_0, IQ4_NL,
    // F32 — these would either fall back to the f32-activation path or run
    // on CPU).
    pub fn q8_1PipelineFor(self: *const GpuWeights, t: GgmlType) ?*const MatvecPipeline {
        return switch (t) {
            .q3_k => &self.pl_q3_k_q8_1,
            .q4_k => &self.pl_q4_k_q8_1,
            .q5_0 => &self.pl_q5_0_q8_1,
            .q5_1 => &self.pl_q5_1_q8_1,
            .q6_k => &self.pl_q6_k_q8_1,
            .q5_k => &self.pl_q5_k_q8_1,
            .iq4_nl => &self.pl_iq4_nl_q8_1,
            else => null,
        };
    }

    fn expertDownIdPipelineFor(self: *const GpuWeights, t: GgmlType) ?*const ExpertDownIdPipeline {
        return switch (t) {
            .q5_1 => &self.pl_expert_down_id_q5_1,
            .iq4_nl => &self.pl_expert_down_id_iq4_nl,
            else => null,
        };
    }

    // Q8_1 dense FFN gate+up: one upload, one quantize, two parallel matvec
    // dispatches that share shared_acts_q8_1, two downloads. Mirrors
    // runLayerQKVQ8_1 but with 2 matmuls instead of 3.
    pub fn runLayerGateUpQ8_1(
        self: *const GpuWeights,
        layer: usize,
        gate_pl: *const MatvecPipeline,
        up_pl: *const MatvecPipeline,
        xb: []const f32,
        gate_out: []f32,
        up_out: []f32,
    ) !void {
        const lw = &self.layers[layer];
        const w_gate = lw.w_gate orelse return error.NotOnGpu;
        const w_up = lw.w_up orelse return error.NotOnGpu;
        std.debug.assert(w_gate.cols % 32 == 0);
        std.debug.assert(w_gate.cols == w_up.cols);

        const vec_buf = &self.shared_vec.?;
        const acts_buf = &self.shared_acts_q8_1.?;
        const gate_buf = &(self.gate_out_buf orelse return error.NotOnGpu);
        const up_buf = &(self.up_out_buf orelse return error.NotOnGpu);

        try vec_buf.upload(std.mem.sliceAsBytes(xb));

        const cmd = try self.ctx.beginBatch();
        const quant_dset = try self.pl_quantize_q8_1.record(cmd, vec_buf, acts_buf, w_gate.cols);
        GpuCtx.recordShaderBarrier(cmd);

        const gate_dset = try gate_pl.record(cmd, &w_gate.mat_buf, acts_buf, gate_buf, w_gate.rows, w_gate.cols);
        const up_dset = try up_pl.record(cmd, &w_up.mat_buf, acts_buf, up_buf, w_up.rows, w_up.cols);

        try self.ctx.submitBatch(cmd);

        const dev = self.ctx.device;
        _ = vk.vkFreeDescriptorSets(dev, self.pl_quantize_q8_1.desc_pool, 1, &quant_dset);
        _ = vk.vkFreeDescriptorSets(dev, gate_pl.desc_pool, 1, &gate_dset);
        _ = vk.vkFreeDescriptorSets(dev, up_pl.desc_pool, 1, &up_dset);

        try gate_buf.download(std.mem.sliceAsBytes(gate_out));
        try up_buf.download(std.mem.sliceAsBytes(up_out));
    }

    // 7j-integrated dense FFN gate+up path. Caller passes the unnormalized
    // residual `x`; GPU runs rmsnorm(x, ffn_norm) → xb_vram, quantize →
    // shared_acts_q8_1, then gate + up matvecs. Replaces CPU rmsnorm + the
    // existing gate+up submit with one combined submit.
    pub fn runLayerFfnGateUpQ8_1(
        self: *const GpuWeights,
        layer: usize,
        eps: f32,
        gate_pl: *const MatvecPipeline,
        up_pl: *const MatvecPipeline,
        x: []const f32, // unnormalized residual
        gate_out: []f32,
        up_out: []f32,
    ) !void {
        const lw = &self.layers[layer];
        const w_gate = lw.w_gate orelse return error.NotOnGpu;
        const w_up = lw.w_up orelse return error.NotOnGpu;
        const ffn_norm_buf = &(lw.ffn_norm_buf orelse return error.NotOnGpu);
        std.debug.assert(w_gate.cols % 256 == 0);
        std.debug.assert(w_gate.cols == w_up.cols);
        std.debug.assert(w_gate.cols == x.len);

        const vec_buf = &self.shared_vec.?;
        const xb_buf = &(self.xb_vram orelse return error.NotOnGpu);
        const acts_buf = &self.shared_acts_q8_1.?;
        const gate_buf = &(self.gate_out_buf orelse return error.NotOnGpu);
        const up_buf = &(self.up_out_buf orelse return error.NotOnGpu);

        try vec_buf.upload(std.mem.sliceAsBytes(x));

        const cmd = try self.ctx.beginBatch();
        const norm_dset = try self.recordLayerRmsnorm(cmd, layer, vec_buf, ffn_norm_buf, xb_buf, @intCast(x.len), eps, false);
        GpuCtx.recordShaderBarrier(cmd);

        const quant_dset = try self.pl_quantize_q8_1.record(cmd, xb_buf, acts_buf, w_gate.cols);
        GpuCtx.recordShaderBarrier(cmd);

        const gate_dset = try gate_pl.record(cmd, &w_gate.mat_buf, acts_buf, gate_buf, w_gate.rows, w_gate.cols);
        const up_dset = try up_pl.record(cmd, &w_up.mat_buf, acts_buf, up_buf, w_up.rows, w_up.cols);

        try self.ctx.submitBatch(cmd);

        const dev = self.ctx.device;
        _ = vk.vkFreeDescriptorSets(dev, self.pl_rmsnorm.desc_pool, 1, &norm_dset);
        _ = vk.vkFreeDescriptorSets(dev, self.pl_quantize_q8_1.desc_pool, 1, &quant_dset);
        _ = vk.vkFreeDescriptorSets(dev, gate_pl.desc_pool, 1, &gate_dset);
        _ = vk.vkFreeDescriptorSets(dev, up_pl.desc_pool, 1, &up_dset);

        try gate_buf.download(std.mem.sliceAsBytes(gate_out));
        try up_buf.download(std.mem.sliceAsBytes(up_out));
    }

    // 7j integrated dense FFN — the BIG submit-count saver for Gemma4.
    // One command-buffer submission does:
    //   1. rmsnorm(x, ffn_norm)              → xb_vram        (GPU)
    //   2. quantize_q8_1(xb_vram)             → shared_acts_q8_1
    //   3. gate matvec (Q8_1)                 → gate_vram
    //   4. up   matvec (Q8_1)                 → up_vram
    //   5. gelu_mul: gate_vram = gelu(g) * u  (in-place)
    //   6. w_down matvec (f32-acts pl_q3_k)   → ffn_vram
    //   7. rmsnorm(ffn_vram, post_ffw_norm_1) → dense_ffn_out_buf
    // Then a single download of dense_ffn_out_buf into ffn_out.
    //
    // Replaces 2 prior submits (gate+up, then w_down) with 1, and eliminates
    // 2 PCIe round-trips per layer (gate/up download + gelu_buf upload).
    //
    // Requirements:
    //   - w_gate/w_up have Q8_1 pipelines (caller passes gate_pl/up_pl).
    //   - w_down has a regular (f32-acts) pipeline (caller passes down_pl).
    //   - x.len = d_model is a multiple of 256 (rmsnorm), of 32 (quantize),
    //     and of 256 (Q8_1 matvec for Q3_K/Q4_K weights).
    //   - d_ffn (gate/up output) need not be 256-aligned — the gelu_mul +
    //     w_down (f32-acts) dispatches handle any width.
    pub fn runLayerDenseFfnQ8_1(
        self: *const GpuWeights,
        layer: usize,
        eps: f32,
        gate_pl: *const MatvecPipeline, // Q8_1 pipeline for w_gate
        up_pl: *const MatvecPipeline, // Q8_1 pipeline for w_up
        down_pl: *const MatvecPipeline, // f32-acts pipeline for w_down
        x: []const f32, // unnormalized residual (d_model)
        ffn_out: []f32, // post_ffw_norm_1 result (d_model)
    ) !void {
        const lw = &self.layers[layer];
        const w_gate = lw.w_gate orelse return error.NotOnGpu;
        const w_up = lw.w_up orelse return error.NotOnGpu;
        const w_down = lw.w_down orelse return error.NotOnGpu;
        const ffn_norm_buf = &(lw.ffn_norm_buf orelse return error.NotOnGpu);
        const post_ffw_norm_1_buf = &(lw.post_ffw_norm_1_buf orelse return error.NotOnGpu);
        std.debug.assert(w_gate.cols % 256 == 0);
        std.debug.assert(w_gate.cols == w_up.cols);
        std.debug.assert(w_gate.cols == x.len);
        std.debug.assert(w_gate.rows == w_up.rows);
        std.debug.assert(w_down.cols == w_gate.rows); // d_ffn
        std.debug.assert(w_down.rows == x.len); // d_model
        std.debug.assert(w_down.rows == ffn_out.len);

        const vec_buf = &self.shared_vec.?; // host-coherent x
        const xb_buf = &(self.xb_vram orelse return error.NotOnGpu);
        const acts_buf = &self.shared_acts_q8_1.?;
        const gate_buf = &(self.gate_vram orelse return error.NotOnGpu);
        const up_buf = &(self.up_vram orelse return error.NotOnGpu);
        const ffn_buf = &(self.ffn_vram orelse return error.NotOnGpu);
        const out_buf = &(self.dense_ffn_out_buf orelse return error.NotOnGpu);

        try vec_buf.upload(std.mem.sliceAsBytes(x));

        const cmd = try self.ctx.beginBatch();
        const norm_dset = try self.recordLayerRmsnorm(cmd, layer, vec_buf, ffn_norm_buf, xb_buf, @intCast(x.len), eps, false);
        GpuCtx.recordShaderBarrier(cmd);

        const quant_dset = try self.pl_quantize_q8_1.record(cmd, xb_buf, acts_buf, w_gate.cols);
        GpuCtx.recordShaderBarrier(cmd);

        const gate_dset = try gate_pl.record(cmd, &w_gate.mat_buf, acts_buf, gate_buf, w_gate.rows, w_gate.cols);
        const up_dset = try up_pl.record(cmd, &w_up.mat_buf, acts_buf, up_buf, w_up.rows, w_up.cols);
        GpuCtx.recordShaderBarrier(cmd);

        const gelu_dset = try self.pl_gelu_mul.record(cmd, gate_buf, up_buf, w_gate.rows);
        GpuCtx.recordShaderBarrier(cmd);

        // w_down: rows = d_model, cols = d_ffn (not 256-aligned for d_ffn=2112,
        // so we deliberately run this on the f32-acts shader, not Q8_1).
        const down_dset = try down_pl.record(cmd, &w_down.mat_buf, gate_buf, ffn_buf, w_down.rows, w_down.cols);
        GpuCtx.recordShaderBarrier(cmd);

        const norm2_dset = try self.recordLayerRmsnorm(cmd, layer, ffn_buf, post_ffw_norm_1_buf, out_buf, @intCast(w_down.rows), eps, false);

        try self.ctx.submitBatch(cmd);

        const dev = self.ctx.device;
        _ = vk.vkFreeDescriptorSets(dev, self.pl_rmsnorm.desc_pool, 1, &norm_dset);
        _ = vk.vkFreeDescriptorSets(dev, self.pl_quantize_q8_1.desc_pool, 1, &quant_dset);
        _ = vk.vkFreeDescriptorSets(dev, gate_pl.desc_pool, 1, &gate_dset);
        _ = vk.vkFreeDescriptorSets(dev, up_pl.desc_pool, 1, &up_dset);
        _ = vk.vkFreeDescriptorSets(dev, self.pl_gelu_mul.desc_pool, 1, &gelu_dset);
        _ = vk.vkFreeDescriptorSets(dev, down_pl.desc_pool, 1, &down_dset);
        _ = vk.vkFreeDescriptorSets(dev, self.pl_rmsnorm.desc_pool, 1, &norm2_dset);

        try out_buf.download(std.mem.sliceAsBytes(ffn_out));
    }

    // 7j integrated wo + post-attention residual + dense FFN — collapses
    // the prior wo submit and the dense FFN submit into ONE GPU submit:
    //   1. quantize(attn_concat)              → shared_acts_q8_1   (acts for wo)
    //   2. wo matvec (Q8_1)                    → attn_vram           (rows=d_model)
    //   3. rmsnorm(attn_vram, post_attn_norm) → attn_vram (in place)
    //   4. elem_add(x_buf, attn_vram)          → x_buf (post-attn residual)
    //   5. rmsnorm(x_buf, ffn_norm)            → xb_vram
    //   6. quantize(xb_vram)                   → shared_acts_q8_1   (acts for FFN)
    //   7. gate matvec  (Q8_1)                 → gate_vram
    //   8. up   matvec  (Q8_1)                 → up_vram
    //   9. gelu_mul: gate_vram = gelu(g)*u    (in place)
    //  10. w_down matvec (f32-acts pl_q3_k)    → ffn_vram
    //  11. rmsnorm(ffn_vram, post_ffw_norm_1)  → dense_ffn_out_buf
    // After submit, host-coherent x_buf holds the post-attention residual
    // (x + attn_buf after norm) and dense_ffn_out_buf holds the dense FFN
    // output (post_ffw_norm_1 applied).  Caller downloads both.
    //
    // Saves one submit (the standalone wo submit) compared to the
    // runLayerDenseFfnQ8_1 entry point, and removes one CPU rmsnorm
    // (post_attention_norm) plus one CPU residual loop.  Per-layer GPU
    // submit count drops from 4 to 3 on the Q8_1 path.
    //
    // Requirements:
    //   - wo, w_gate, w_up have Q8_1 pipelines; w_down has a regular
    //     (f32-acts) pipeline.
    //   - Uploads x and attn_concat to shared_vec / attn_in_buf internally;
    //     caller just passes the slices.
    //   - On return, ffn_out holds post_ffw_norm_1(...) and x (input slice)
    //     has been overwritten with the post-attention residual.
    pub fn runLayerAttnResidualDenseFfnQ8_1(
        self: *const GpuWeights,
        layer: usize,
        eps: f32,
        wo_pl: *const MatvecPipeline, // Q8_1 pipeline for wo
        gate_pl: *const MatvecPipeline, // Q8_1 pipeline for w_gate
        up_pl: *const MatvecPipeline, // Q8_1 pipeline for w_up
        down_pl: *const MatvecPipeline, // f32-acts pipeline for w_down
        x: []f32, // in: unnormalized residual; out: x + post_attn_norm(wo(attn_concat))
        attn_concat: []const f32, // CPU sdpAttn output, length = nq
        ffn_out: []f32, // post_ffw_norm_1 result (d_model)
        skip_attn_upload: bool, // 7l.2/3: GPU attention already wrote attn_in_buf
    ) !void {
        const lw = &self.layers[layer];
        const wo = lw.wo orelse return error.NotOnGpu;
        const w_gate = lw.w_gate orelse return error.NotOnGpu;
        const w_up = lw.w_up orelse return error.NotOnGpu;
        const w_down = lw.w_down orelse return error.NotOnGpu;
        const post_attn_buf = &(lw.post_attention_norm_buf orelse return error.NotOnGpu);
        const ffn_norm_buf = &(lw.ffn_norm_buf orelse return error.NotOnGpu);
        const post_ffw_norm_1_buf = &(lw.post_ffw_norm_1_buf orelse return error.NotOnGpu);
        std.debug.assert(wo.cols == attn_concat.len);
        std.debug.assert(wo.cols % 32 == 0); // quantize block size
        std.debug.assert(wo.rows == x.len); // d_model
        std.debug.assert(w_gate.cols == x.len);
        std.debug.assert(w_gate.cols % 256 == 0); // rmsnorm needs %256
        std.debug.assert(w_gate.rows == w_up.rows);
        std.debug.assert(w_down.cols == w_gate.rows); // d_ffn
        std.debug.assert(w_down.rows == x.len);
        std.debug.assert(w_down.rows == ffn_out.len);

        const x_buf = &self.shared_vec.?; // host-coherent x
        const attn_in_buf = &(self.attn_in_buf orelse return error.NotOnGpu);
        const attn_vram = &(self.attn_vram orelse return error.NotOnGpu);
        const xb_buf = &(self.xb_vram orelse return error.NotOnGpu);
        const acts_buf = &self.shared_acts_q8_1.?;
        const gate_buf = &(self.gate_vram orelse return error.NotOnGpu);
        const up_buf = &(self.up_vram orelse return error.NotOnGpu);
        const ffn_buf = &(self.ffn_vram orelse return error.NotOnGpu);
        const out_buf = &(self.dense_ffn_out_buf orelse return error.NotOnGpu);

        try x_buf.upload(std.mem.sliceAsBytes(x));
        if (!skip_attn_upload) {
            try attn_in_buf.upload(std.mem.sliceAsBytes(attn_concat));
        }

        const cmd = try self.ctx.beginBatch();

        // wo: quantize attn_concat, then Q8_1 matvec into attn_vram.
        const p_wo_quant = self.ctx.profileBegin(cmd, "post_attn.wo_quantize");
        const wo_quant_dset = try self.pl_quantize_q8_1.record(cmd, attn_in_buf, acts_buf, @intCast(attn_concat.len));
        self.ctx.profileEnd(cmd, p_wo_quant);
        GpuCtx.recordShaderBarrier(cmd);
        const p_wo = self.ctx.profileBegin(cmd, "post_attn.wo");
        const wo_dset = try wo_pl.record(cmd, &wo.mat_buf, acts_buf, attn_vram, wo.rows, wo.cols);
        self.ctx.profileEnd(cmd, p_wo);
        GpuCtx.recordShaderBarrier(cmd);

        // post_attention_norm in place on attn_vram.
        const p_post_attn = self.ctx.profileBegin(cmd, "post_attn.rmsnorm");
        const post_attn_dset = try self.recordLayerRmsnorm(cmd, layer, attn_vram, post_attn_buf, attn_vram, @intCast(wo.rows), eps, false);
        self.ctx.profileEnd(cmd, p_post_attn);
        GpuCtx.recordShaderBarrier(cmd);

        // Residual: x_buf += attn_vram.  x_buf is HOST_COHERENT shared_vec
        // pre-loaded by the caller; after this dispatch it carries the new x.
        const p_add = self.ctx.profileBegin(cmd, "post_attn.residual_add");
        const add_dset = try self.pl_elem_add.record(cmd, x_buf, attn_vram, @intCast(wo.rows));
        self.ctx.profileEnd(cmd, p_add);
        GpuCtx.recordShaderBarrier(cmd);

        // ffn_norm(x_buf) → xb_vram.
        const p_ffn_norm = self.ctx.profileBegin(cmd, "dense_ffn.rmsnorm");
        const ffn_norm_dset = try self.recordLayerRmsnorm(cmd, layer, x_buf, ffn_norm_buf, xb_buf, @intCast(wo.rows), eps, false);
        self.ctx.profileEnd(cmd, p_ffn_norm);
        GpuCtx.recordShaderBarrier(cmd);

        // Re-quantize xb for the FFN Q8_1 matvecs.
        const p_ffn_quant = self.ctx.profileBegin(cmd, "dense_ffn.quantize_q8_1");
        const ffn_quant_dset = try self.pl_quantize_q8_1.record(cmd, xb_buf, acts_buf, w_gate.cols);
        self.ctx.profileEnd(cmd, p_ffn_quant);
        GpuCtx.recordShaderBarrier(cmd);

        const p_gate = self.ctx.profileBegin(cmd, "dense_ffn.gate");
        const gate_dset = try gate_pl.record(cmd, &w_gate.mat_buf, acts_buf, gate_buf, w_gate.rows, w_gate.cols);
        self.ctx.profileEnd(cmd, p_gate);
        const p_up = self.ctx.profileBegin(cmd, "dense_ffn.up");
        const up_dset = try up_pl.record(cmd, &w_up.mat_buf, acts_buf, up_buf, w_up.rows, w_up.cols);
        self.ctx.profileEnd(cmd, p_up);
        GpuCtx.recordShaderBarrier(cmd);

        const p_gelu = self.ctx.profileBegin(cmd, "dense_ffn.gelu_mul");
        const gelu_dset = try self.pl_gelu_mul.record(cmd, gate_buf, up_buf, w_gate.rows);
        self.ctx.profileEnd(cmd, p_gelu);
        GpuCtx.recordShaderBarrier(cmd);

        const p_down = self.ctx.profileBeginFmt(cmd, "dense_ffn.down.L{d:0>2}.{d}x{d}", .{ layer, w_down.rows, w_down.cols });
        const down_dset = try down_pl.record(cmd, &w_down.mat_buf, gate_buf, ffn_buf, w_down.rows, w_down.cols);
        self.ctx.profileEnd(cmd, p_down);
        GpuCtx.recordShaderBarrier(cmd);

        const p_post_ffw = self.ctx.profileBegin(cmd, "dense_ffn.post_norm");
        const post_ffw_dset = try self.recordLayerRmsnorm(cmd, layer, ffn_buf, post_ffw_norm_1_buf, out_buf, @intCast(w_down.rows), eps, false);
        self.ctx.profileEnd(cmd, p_post_ffw);

        try self.ctx.submitBatch(cmd);

        const dev = self.ctx.device;
        _ = vk.vkFreeDescriptorSets(dev, self.pl_quantize_q8_1.desc_pool, 1, &wo_quant_dset);
        _ = vk.vkFreeDescriptorSets(dev, wo_pl.desc_pool, 1, &wo_dset);
        _ = vk.vkFreeDescriptorSets(dev, self.pl_rmsnorm.desc_pool, 1, &post_attn_dset);
        _ = vk.vkFreeDescriptorSets(dev, self.pl_elem_add.desc_pool, 1, &add_dset);
        _ = vk.vkFreeDescriptorSets(dev, self.pl_rmsnorm.desc_pool, 1, &ffn_norm_dset);
        _ = vk.vkFreeDescriptorSets(dev, self.pl_quantize_q8_1.desc_pool, 1, &ffn_quant_dset);
        _ = vk.vkFreeDescriptorSets(dev, gate_pl.desc_pool, 1, &gate_dset);
        _ = vk.vkFreeDescriptorSets(dev, up_pl.desc_pool, 1, &up_dset);
        _ = vk.vkFreeDescriptorSets(dev, self.pl_gelu_mul.desc_pool, 1, &gelu_dset);
        _ = vk.vkFreeDescriptorSets(dev, down_pl.desc_pool, 1, &down_dset);
        _ = vk.vkFreeDescriptorSets(dev, self.pl_rmsnorm.desc_pool, 1, &post_ffw_dset);

        try out_buf.download(std.mem.sliceAsBytes(ffn_out));
        try x_buf.download(std.mem.sliceAsBytes(x));
    }

    // Dispatch w_gate and w_up in one command buffer (both read the same FFN-norm xb).
    pub fn runLayerGateUp(
        self: *const GpuWeights,
        layer: usize,
        gate_pl: *const MatvecPipeline,
        up_pl: *const MatvecPipeline,
        xb: []const f32,
        gate_out: []f32,
        up_out: []f32,
    ) !void {
        const lw = &self.layers[layer];
        const w_gate = lw.w_gate orelse return error.NotOnGpu;
        const w_up = lw.w_up orelse return error.NotOnGpu;
        const gate_buf = &(self.gate_out_buf orelse return error.NotOnGpu);
        const up_buf = &(self.up_out_buf orelse return error.NotOnGpu);

        try self.shared_vec.?.upload(std.mem.sliceAsBytes(xb));

        const cmd = try self.ctx.beginBatch();
        var gate_dset: vk.VkDescriptorSet = null;
        var up_dset: vk.VkDescriptorSet = null;

        gate_dset = try w_gate.recordMv(cmd, gate_pl, &self.shared_vec.?, gate_buf);
        up_dset = try w_up.recordMv(cmd, up_pl, &self.shared_vec.?, up_buf);

        try self.ctx.submitBatch(cmd);

        _ = vk.vkFreeDescriptorSets(self.ctx.device, gate_pl.desc_pool, 1, &gate_dset);
        _ = vk.vkFreeDescriptorSets(self.ctx.device, up_pl.desc_pool, 1, &up_dset);

        try gate_buf.download(std.mem.sliceAsBytes(gate_out));
        try up_buf.download(std.mem.sliceAsBytes(up_out));
    }

    pub fn expertGate(self: *const GpuWeights, l: usize, e: usize) ?MatvecSession {
        const eg = self.expert_gate orelse return null;
        return eg[l * self.n_experts + e];
    }
    pub fn expertUp(self: *const GpuWeights, l: usize, e: usize) ?MatvecSession {
        const eu = self.expert_up orelse return null;
        return eu[l * self.n_experts + e];
    }
    pub fn expertDown(self: *const GpuWeights, l: usize, e: usize) ?MatvecSession {
        const ed = self.expert_down orelse return null;
        return ed[l * self.n_experts + e];
    }

    // Upload an f32 vector to x_vram via the persistent staging buffer.
    // Synchronous: writes staging → records copy → submits → waits. Used
    // once after embed lookup to seed the residual stream; subsequent
    // per-layer ops update x_vram in place via shader dispatches.
    pub fn uploadX(self: *const GpuWeights, src: []const f32) !void {
        const stage = &(self.stage_buf orelse return error.NotOnGpu);
        const x = &(self.x_vram orelse return error.NotOnGpu);
        try stage.upload(std.mem.sliceAsBytes(src));
        try self.ctx.copyBuffer(stage.handle, x.handle, src.len * @sizeOf(f32));
    }

    // Download x_vram → out (f32 slice). Submits a copy + waits.
    pub fn downloadX(self: *const GpuWeights, out: []f32) !void {
        const stage = &(self.stage_buf orelse return error.NotOnGpu);
        const x = &(self.x_vram orelse return error.NotOnGpu);
        try self.ctx.copyBuffer(x.handle, stage.handle, out.len * @sizeOf(f32));
        try stage.download(std.mem.sliceAsBytes(out));
    }

    fn recordLayerRmsnorm(
        self: *const GpuWeights,
        cmd: vk.VkCommandBuffer,
        layer: usize,
        x_buf: *const GpuBuffer,
        w_buf: *const GpuBuffer,
        y_buf: *const GpuBuffer,
        n: u32,
        eps: f32,
        weight_offset: bool,
    ) !vk.VkDescriptorSet {
        // Layer 19 is numerically sensitive on the target Gemma4 APEX model:
        // fast parallel reduction keeps layer argmaxes but can swap the final
        // top-2 logits when the MoE tail stays fully GPU-resident.
        if (layer == 19) {
            return self.pl_rmsnorm.recordPrecise(cmd, x_buf, w_buf, y_buf, n, eps, weight_offset);
        }
        return self.pl_rmsnorm.record(cmd, x_buf, w_buf, y_buf, n, eps, weight_offset);
    }

    // Pipeline for the f32-activation path.
    pub fn pipelineFor(self: *const GpuWeights, t: GgmlType) *const MatvecPipeline {
        return switch (t) {
            .f32 => &self.pl_f32,
            .q8_0 => &self.pl_q8_0,
            .q3_k => &self.pl_q3_k,
            .q4_k => &self.pl_q4_k,
            .q5_1 => &self.pl_q5_1,
            .q5_0 => &self.pl_q5_0,
            else => unreachable,
        };
    }

    // Run all active experts for one layer in ONE GPU submission:
    //   Phase 1: n fused gate-gelu-up dispatches, write device-local mid_bufs
    //   Phase 2: n down dispatches, write sub-ranges of device-local expert_all_out_buf
    //   Phase 3: 1 accum dispatch, reads all_out + scales, writes device-local moe_gpu_buf
    // CPU writes moe_in and scales. Legacy/debug callers can request a staging
    // readback; the opt-in VRAM-tail forward path consumes moe_gpu_buf directly.
    // Returns error.ExpertNotOnGpu if any session is missing; caller falls back to CPU.
    pub fn runExpertBatch(
        self: *GpuWeights,
        layer: usize,
        top_idx: []const usize,
        gate_up_type: GgmlType,
        down_type: GgmlType,
        down_exps_scale: []const f32,
        moe_in: []const f32,
        router_out: []const f32,
        moe_buf: []f32,
        skip_readback: bool,
    ) !void {
        const n = top_idx.len;
        const eg_sessions = self.expert_gate orelse return error.ExpertNotOnGpu;
        const eu_sessions = self.expert_up orelse return error.ExpertNotOnGpu;
        const ed_sessions = self.expert_down orelse return error.ExpertNotOnGpu;
        const mid_bufs = self.expert_mid_bufs orelse return error.ExpertNotOnGpu;
        const all_out_buf = &(self.expert_all_out_buf orelse return error.ExpertNotOnGpu);
        const in_buf = &(self.expert_in_buf orelse return error.ExpertNotOnGpu);
        const scales_buf = &(self.expert_scales_buf orelse return error.ExpertNotOnGpu);
        const accum_scales_buf = &(self.expert_accum_scales_buf orelse return error.ExpertNotOnGpu);
        const ids_buf = &(self.expert_ids_buf orelse return error.ExpertNotOnGpu);
        const moe_out_buf = &(self.moe_gpu_buf orelse return error.ExpertNotOnGpu);
        const moe_stage_buf = &(self.moe_stage_buf orelse return error.ExpertNotOnGpu);
        const in_slice = self.expert_in_slice orelse return error.ExpertNotOnGpu;
        const scales_slice = self.expert_scales_slice orelse return error.ExpertNotOnGpu;
        const accum_scales_slice = self.expert_accum_scales_slice orelse return error.ExpertNotOnGpu;
        const ids_slice = self.expert_ids_slice orelse return error.ExpertNotOnGpu;
        const moe_stage_slice = self.moe_stage_slice orelse return error.ExpertNotOnGpu;
        const acts_q8_1 = &(self.shared_acts_q8_1 orelse return error.ExpertNotOnGpu);
        const mid_q8_1_flat_buf = &(self.expert_mid_q8_1_flat_buf orelse return error.ExpertNotOnGpu);

        const pl_dn_q8_1 = self.q8_1PipelineFor(down_type);
        const use_q8_1_dn = pl_dn_q8_1 != null;
        const pl_dn_id = self.expertDownIdPipelineFor(down_type);
        const down_flat = if (self.expert_down_flat) |dfs| dfs[layer] else null;
        const use_id_dn = use_q8_1_dn and pl_dn_id != null and down_flat != null;
        const gate_up_flat = if (self.expert_gate_up_flat) |gufs| gufs[layer] else null;
        const use_id_gu = gate_up_type == .q3_k and use_q8_1_dn and gate_up_flat != null;
        const reuse_id_dsets = std.c.getenv("LLMTOY_EXPERT_REUSE_DSETS") != null;
        const reuse_quant_input_dset = reuse_id_dsets and use_id_gu;
        const reuse_gate_up_id_dset = reuse_id_dsets and use_id_gu;
        const reuse_quant_mid_batched_dset = reuse_id_dsets and use_id_gu;
        const reuse_down_id_dset = reuse_id_dsets and use_id_dn;
        const reuse_accum_dset = reuse_id_dsets and (use_id_gu or use_id_dn);
        const reuse_cmd = std.c.getenv("LLMTOY_EXPERT_REUSE_CMD") != null and use_id_gu and use_id_dn;

        for (top_idx) |eidx| {
            if (!use_id_gu) {
                if (eg_sessions[layer * self.n_experts + eidx] == null or
                    eu_sessions[layer * self.n_experts + eidx] == null)
                    return error.ExpertNotOnGpu;
            }
            if (!use_id_dn and ed_sessions[layer * self.n_experts + eidx] == null)
                return error.ExpertNotOnGpu;
        }

        // Optional Q8_1-acts pipeline for down. When non-null, each expert's
        // f32 mid_buf gets quantized to its own Q8_1 mid_q8_1_buf between
        // phases, and down reads that instead of the raw f32 mid_buf.
        const pl_dn_f32 = if (use_q8_1_dn) null else self.pipelineFor(down_type);
        const mid_q8_1_bufs = self.expert_mid_q8_1_bufs;

        // Use the Q8_1 fused gate+up shader when gate/up are Q3_K.  Both fused
        // shaders share the same binding layout; we just swap pipelines.
        const use_q8_1_gu = gate_up_type == .q3_k;
        const pl_gu = if (use_q8_1_gu) &self.pl_fused_gu_q8_1 else &self.pl_fused_gu;
        // For the Q8_1 path the gate+up dispatches read `acts_q8_1` instead of
        // `in_buf`. moe_in.len gives us the column count for the quantize call.
        const gu_in_buf = if (use_q8_1_gu) acts_q8_1 else in_buf;

        // Write inputs and per-expert scales to HOST_COHERENT buffers.
        @memcpy(in_slice, moe_in);
        for (0..n) |k| {
            const eidx = top_idx[k];
            scales_slice[k] = down_exps_scale[eidx] * router_out[eidx];
            accum_scales_slice[k] = if (use_id_dn) 1.0 else scales_slice[k];
            ids_slice[k] = @intCast(eidx);
        }
        var reused_cmd = false;
        const cmd = blk: {
            if (reuse_cmd) {
                const cmds = self.expert_reuse_cmds orelse return error.ExpertNotOnGpu;
                if (cmds[layer] == null)
                    cmds[layer] = try self.ctx.allocReusableCommandBuffer();
                reused_cmd = true;
                break :blk try self.ctx.beginReusableBatch(cmds[layer].?);
            }
            break :blk try self.ctx.beginBatch();
        };
        var fused_dsets: [16]vk.VkDescriptorSet = undefined;
        var quant_dn_dsets: [16]vk.VkDescriptorSet = undefined;
        var down_dsets: [16]vk.VkDescriptorSet = undefined;
        var quant_dset: ?vk.VkDescriptorSet = null;
        var gate_up_id_dset: ?vk.VkDescriptorSet = null;
        var quant_mid_id_dset: ?vk.VkDescriptorSet = null;

        // Optional Phase 0: f32 moe_in → Q8_1 in shared_acts_q8_1 once.
        if (use_q8_1_gu) {
            const p_quant = self.ctx.profileBegin(cmd, "moe.quantize_input");
            if (reuse_quant_input_dset) {
                if (self.expert_quant_input_dset == null)
                    self.expert_quant_input_dset = try self.pl_quantize_q8_1.allocSet(in_buf, acts_q8_1);
                self.pl_quantize_q8_1.recordWithSet(cmd, self.expert_quant_input_dset.?, @intCast(moe_in.len));
            } else {
                quant_dset = try self.pl_quantize_q8_1.record(cmd, in_buf, acts_q8_1, @intCast(moe_in.len));
            }
            self.ctx.profileEnd(cmd, p_quant);
            GpuCtx.recordShaderBarrier(cmd);
        }

        // Phase 1: fused gate-gelu-up. The opt-in expert-ID path writes a
        // flat [slot, d_expert] f32 mid into expert_all_out_buf; otherwise the
        // current path writes one mid_buf per selected expert.
        if (use_id_gu) {
            const p_gu = self.ctx.profileBegin(cmd, "moe.fused_gate_up");
            if (reuse_gate_up_id_dset) {
                const sets = self.expert_gate_up_id_dsets orelse return error.ExpertNotOnGpu;
                if (sets[layer] == null)
                    sets[layer] = try self.pl_expert_gate_up_id_q3_k.allocSet(&gate_up_flat.?.mat_buf, gu_in_buf, ids_buf, all_out_buf);
                self.pl_expert_gate_up_id_q3_k.recordWithSet(cmd, sets[layer].?, @intCast(gate_up_flat.?.rows / (2 * self.n_experts)), @intCast(gate_up_flat.?.cols), @intCast(n));
            } else {
                gate_up_id_dset = try self.pl_expert_gate_up_id_q3_k.record(cmd, &gate_up_flat.?.mat_buf, gu_in_buf, ids_buf, all_out_buf, @intCast(gate_up_flat.?.rows / (2 * self.n_experts)), @intCast(gate_up_flat.?.cols), @intCast(n));
            }
            self.ctx.profileEnd(cmd, p_gu);
        } else {
            for (0..n) |k| {
                const sg = eg_sessions[layer * self.n_experts + top_idx[k]].?;
                const su = eu_sessions[layer * self.n_experts + top_idx[k]].?;
                const p_gu = self.ctx.profileBegin(cmd, "moe.fused_gate_up");
                fused_dsets[k] = try pl_gu.record(cmd, &sg.mat_buf, &su.mat_buf, gu_in_buf, &mid_bufs[k], sg.rows, sg.cols);
                self.ctx.profileEnd(cmd, p_gu);
            }
        }

        // Barrier: fused writes mid_bufs; either quantize or down reads them.
        GpuCtx.recordShaderBarrier(cmd);

        // Phase 1.5 (Q8_1 down only): quantize each expert's f32 mid_buf into
        // its own Q8_1 mid_q8_1_buf. One dispatch per expert; cheap relative
        // to the down matmul that follows.
        if (use_q8_1_dn) {
            const d_expert: u32 = if (use_id_dn)
                down_flat.?.cols
            else
                ed_sessions[layer * self.n_experts + top_idx[0]].?.cols;
            const mid_q8_1_bytes = mv_mod.q8_1OutBytes(d_expert);
            if (use_id_gu and use_id_dn) {
                const p_quant_dn = self.ctx.profileBegin(cmd, "moe.quantize_mid");
                if (reuse_quant_mid_batched_dset) {
                    if (self.expert_quant_mid_batched_dset == null)
                        self.expert_quant_mid_batched_dset = try self.pl_quantize_q8_1_batched.allocSet(all_out_buf, mid_q8_1_flat_buf);
                    self.pl_quantize_q8_1_batched.recordWithSet(cmd, self.expert_quant_mid_batched_dset.?, d_expert, @intCast(n));
                } else {
                    quant_mid_id_dset = try self.pl_quantize_q8_1_batched.record(cmd, all_out_buf, mid_q8_1_flat_buf, d_expert, @intCast(n));
                }
                self.ctx.profileEnd(cmd, p_quant_dn);
            } else if (use_id_gu) {
                for (0..n) |k| {
                    const p_quant_dn = self.ctx.profileBegin(cmd, "moe.quantize_mid");
                    quant_dn_dsets[k] = try self.pl_quantize_q8_1.recordRangeToOffset(cmd, all_out_buf, k * d_expert * @sizeOf(f32), d_expert * @sizeOf(f32), &mid_q8_1_bufs.?[k], 0, mid_q8_1_bytes, d_expert);
                    self.ctx.profileEnd(cmd, p_quant_dn);
                }
            } else {
                for (0..n) |k| {
                    const p_quant_dn = self.ctx.profileBegin(cmd, "moe.quantize_mid");
                    quant_dn_dsets[k] = if (use_id_dn)
                        try self.pl_quantize_q8_1.recordToOffset(cmd, &mid_bufs[k], mid_q8_1_flat_buf, k * mid_q8_1_bytes, mid_q8_1_bytes, d_expert)
                    else
                        try self.pl_quantize_q8_1.record(cmd, &mid_bufs[k], &mid_q8_1_bufs.?[k], d_expert);
                    self.ctx.profileEnd(cmd, p_quant_dn);
                }
            }
            GpuCtx.recordShaderBarrier(cmd);
        }

        // Phase 2: down matmul; each expert writes into its sub-range of expert_all_out_buf.
        const d_model: u64 = blk: {
            if (use_id_dn) break :blk down_flat.?.rows / self.n_experts;
            const sd0 = ed_sessions[layer * self.n_experts + top_idx[0]].?;
            break :blk sd0.rows;
        };
        var down_id_dset: ?vk.VkDescriptorSet = null;
        if (use_id_dn) {
            const p_down = self.ctx.profileBegin(cmd, "moe.down");
            if (reuse_down_id_dset) {
                const sets = self.expert_down_id_dsets orelse return error.ExpertNotOnGpu;
                if (sets[layer] == null) {
                    sets[layer] = try pl_dn_id.?.allocSet(&down_flat.?.mat_buf, mid_q8_1_flat_buf, ids_buf, scales_buf, all_out_buf);
                    if (self.expert_down_id_dset_is_iq4) |flags| flags[layer] = down_type == .iq4_nl;
                }
                pl_dn_id.?.recordWithSet(cmd, sets[layer].?, @intCast(d_model), @intCast(down_flat.?.cols), @intCast(n));
            } else {
                down_id_dset = try pl_dn_id.?.record(cmd, &down_flat.?.mat_buf, mid_q8_1_flat_buf, ids_buf, scales_buf, all_out_buf, @intCast(d_model), @intCast(down_flat.?.cols), @intCast(n));
            }
            self.ctx.profileEnd(cmd, p_down);
        } else {
            for (0..n) |k| {
                const sd = ed_sessions[layer * self.n_experts + top_idx[k]].?;
                const out_off = k * d_model * @sizeOf(f32);
                const out_sz = d_model * @sizeOf(f32);
                const dn_in_buf: *const GpuBuffer = if (use_q8_1_dn)
                    &mid_q8_1_bufs.?[k]
                else
                    &mid_bufs[k];
                const pl_dn = if (use_q8_1_dn) pl_dn_q8_1.? else pl_dn_f32.?;
                const p_down = self.ctx.profileBegin(cmd, "moe.down");
                down_dsets[k] = try pl_dn.recordToRange(cmd, &sd.mat_buf, dn_in_buf, all_out_buf.handle, out_off, out_sz, sd.rows, sd.cols);
                self.ctx.profileEnd(cmd, p_down);
            }
        }

        // Barrier: down writes all_out, accum reads all_out
        GpuCtx.recordShaderBarrier(cmd);

        // Phase 3: weighted accumulation on GPU
        const p_accum = self.ctx.profileBegin(cmd, "moe.accum");
        var accum_dset: ?vk.VkDescriptorSet = null;
        if (reuse_accum_dset) {
            if (self.expert_accum_dset == null)
                self.expert_accum_dset = try self.pl_accum.allocSet(all_out_buf, accum_scales_buf, moe_out_buf);
            self.pl_accum.recordWithSet(cmd, self.expert_accum_dset.?, @intCast(d_model), @intCast(n));
        } else {
            accum_dset = try self.pl_accum.record(cmd, all_out_buf, accum_scales_buf, moe_out_buf, @intCast(d_model), @intCast(n));
        }
        self.ctx.profileEnd(cmd, p_accum);

        if (reused_cmd) {
            try self.ctx.submitReusableBatch(cmd);
        } else {
            try self.ctx.submitBatch(cmd);
        }

        const pl_dn_used = if (use_q8_1_dn) pl_dn_q8_1.? else pl_dn_f32.?;
        if (reuse_gate_up_id_dset and use_id_gu) {
            // Persistent descriptor set is owned by GpuWeights.
        } else if (gate_up_id_dset) |*ds| {
            _ = vk.vkFreeDescriptorSets(self.ctx.device, self.pl_expert_gate_up_id_q3_k.desc_pool, 1, ds);
        } else {
            for (fused_dsets[0..n]) |*ds|
                _ = vk.vkFreeDescriptorSets(self.ctx.device, pl_gu.desc_pool, 1, ds);
        }
        if (reuse_down_id_dset and use_id_dn) {
            // Persistent descriptor set is owned by GpuWeights.
        } else if (use_id_dn) {
            var ds = down_id_dset.?;
            _ = vk.vkFreeDescriptorSets(self.ctx.device, pl_dn_id.?.desc_pool, 1, &ds);
        } else {
            for (down_dsets[0..n]) |*ds|
                _ = vk.vkFreeDescriptorSets(self.ctx.device, pl_dn_used.desc_pool, 1, ds);
        }
        if (use_q8_1_dn) {
            if (reuse_quant_mid_batched_dset and use_id_gu and use_id_dn) {
                // Persistent descriptor set is owned by GpuWeights.
            } else if (quant_mid_id_dset) |*ds| {
                _ = vk.vkFreeDescriptorSets(self.ctx.device, self.pl_quantize_q8_1_batched.desc_pool, 1, ds);
            } else {
                for (quant_dn_dsets[0..n]) |*ds|
                    _ = vk.vkFreeDescriptorSets(self.ctx.device, self.pl_quantize_q8_1.desc_pool, 1, ds);
            }
        }
        if (quant_dset) |*ds|
            _ = vk.vkFreeDescriptorSets(self.ctx.device, self.pl_quantize_q8_1.desc_pool, 1, ds);
        if (accum_dset) |set| {
            var accum_ds = set;
            _ = vk.vkFreeDescriptorSets(self.ctx.device, self.pl_accum.desc_pool, 1, &accum_ds);
        }

        // Add GPU-accumulated expert result into moe_buf only for legacy/debug
        // callers. The production forward path keeps this vector in VRAM and
        // consumes it with runLayerMoeResidualOnGpu.
        if (!skip_readback) {
            try self.ctx.copyBuffer(moe_out_buf.handle, moe_stage_buf.handle, d_model * @sizeOf(f32));
            for (moe_buf, moe_stage_slice) |*m, v| m.* += v;
        }
    }

    // Finish Gemma's dense-FFN + MoE block on GPU after runExpertBatch has
    // produced moe_gpu_buf in VRAM. The final residual is downloaded because
    // the current layer orchestration still uses CPU router/compare state.
    pub fn runLayerMoeResidualOnGpu(
        self: *const GpuWeights,
        layer: usize,
        eps: f32,
        x: []f32,
        dense_ffn: []const f32,
        layer_output_scale: f32,
    ) !void {
        const lw = &self.layers[layer];
        const post_ffw_norm_2_buf = &(lw.post_ffw_norm_2_buf orelse return error.NotOnGpu);
        const post_ffw_norm_buf = &(lw.post_ffw_norm_buf orelse return error.NotOnGpu);
        const x_buf = &self.shared_vec.?;
        const dense_buf = &(self.dense_ffn_out_buf orelse return error.NotOnGpu);
        const moe_buf = &(self.moe_gpu_buf orelse return error.NotOnGpu);
        const moe_norm_buf = &(self.ffn_vram orelse return error.NotOnGpu);
        const combined_norm_buf = &(self.xb_vram orelse return error.NotOnGpu);

        std.debug.assert(x.len == dense_ffn.len);
        std.debug.assert(x.len % 256 == 0);

        try x_buf.upload(std.mem.sliceAsBytes(x));
        try dense_buf.upload(std.mem.sliceAsBytes(dense_ffn));

        const cmd = try self.ctx.beginBatch();

        const p_moe_norm = self.ctx.profileBegin(cmd, "moe.post_norm");
        const moe_norm_dset = try self.recordLayerRmsnorm(cmd, layer, moe_buf, post_ffw_norm_2_buf, moe_norm_buf, @intCast(x.len), eps, false);
        self.ctx.profileEnd(cmd, p_moe_norm);
        GpuCtx.recordShaderBarrier(cmd);

        const p_combine = self.ctx.profileBegin(cmd, "ffn_moe.combine");
        const combine_dset = try self.pl_elem_add.record(cmd, dense_buf, moe_norm_buf, @intCast(x.len));
        self.ctx.profileEnd(cmd, p_combine);
        GpuCtx.recordShaderBarrier(cmd);

        const p_post_ffw = self.ctx.profileBegin(cmd, "ffn_moe.post_norm");
        const post_ffw_dset = try self.recordLayerRmsnorm(cmd, layer, dense_buf, post_ffw_norm_buf, combined_norm_buf, @intCast(x.len), eps, false);
        self.ctx.profileEnd(cmd, p_post_ffw);
        GpuCtx.recordShaderBarrier(cmd);

        const p_residual = self.ctx.profileBegin(cmd, "ffn_moe.residual_add");
        const residual_dset = try self.pl_elem_add.record(cmd, x_buf, combined_norm_buf, @intCast(x.len));
        self.ctx.profileEnd(cmd, p_residual);

        var scale_dset: ?vk.VkDescriptorSet = null;
        if (layer_output_scale != 1.0) {
            GpuCtx.recordShaderBarrier(cmd);
            const p_scale = self.ctx.profileBegin(cmd, "ffn_moe.layer_scale");
            scale_dset = try self.pl_elem_scale.record(cmd, x_buf, @intCast(x.len), layer_output_scale);
            self.ctx.profileEnd(cmd, p_scale);
        }

        try self.ctx.submitBatch(cmd);

        const dev = self.ctx.device;
        _ = vk.vkFreeDescriptorSets(dev, self.pl_rmsnorm.desc_pool, 1, &moe_norm_dset);
        _ = vk.vkFreeDescriptorSets(dev, self.pl_elem_add.desc_pool, 1, &combine_dset);
        _ = vk.vkFreeDescriptorSets(dev, self.pl_rmsnorm.desc_pool, 1, &post_ffw_dset);
        _ = vk.vkFreeDescriptorSets(dev, self.pl_elem_add.desc_pool, 1, &residual_dset);
        if (scale_dset) |*ds|
            _ = vk.vkFreeDescriptorSets(dev, self.pl_elem_scale.desc_pool, 1, ds);

        try x_buf.download(std.mem.sliceAsBytes(x));
    }
};

// --- module-level helpers ---

fn isGpuSupported(t: GgmlType) bool {
    return switch (t) {
        .f32, .q8_0, .q3_k, .q4_k, .q5_1, .q5_0, .q6_k, .q5_k, .iq4_nl => true,
        else => false,
    };
}

fn isExpertDownIdSupported(t: GgmlType) bool {
    return switch (t) {
        .q5_1, .iq4_nl => true,
        else => false,
    };
}

fn isExpertGateUpIdSupported(t: GgmlType) bool {
    return switch (t) {
        .q3_k => true,
        else => false,
    };
}

// Schedule one matrix upload into an open command buffer.
// Commits the staging buffer to sbufs[nsb.*] and increments nsb.
// Returns null for unsupported quant types (→ CPU fallback).
fn schedUpload(
    ctx: *const GpuCtx,
    mat: RawMatrix,
    cmd: vk.VkCommandBuffer,
    sbufs: []GpuBuffer,
    nsb: *usize,
) !?MatvecSession {
    if (!isGpuSupported(mat.type_)) return null;

    var tmp: ?GpuBuffer = null;
    errdefer if (tmp) |*s| s.deinit();

    tmp = try GpuBuffer.initStaging(ctx, mat.data.len);
    try tmp.?.upload(mat.data);

    const sess = try MatvecSession.allocEmpty(ctx, mat.data.len, @intCast(mat.rows), @intCast(mat.cols));

    GpuCtx.recordCopy(cmd, tmp.?.handle, sess.mat_buf.handle, mat.data.len);

    sbufs[nsb.*] = tmp.?;
    tmp = null; // committed → cancel errdefer
    nsb.* += 1;

    return sess;
}

// Upload an f32 slice to a fresh device-local buffer. One submit per call —
// only used at init time for the small (≤ 12 KiB) norm weights.
fn uploadF32(ctx: *const GpuCtx, src: []const f32) !GpuBuffer {
    const size = src.len * @sizeOf(f32);
    var staging = try GpuBuffer.initStaging(ctx, size);
    defer staging.deinit();
    try staging.upload(std.mem.sliceAsBytes(src));

    var dst = try GpuBuffer.initDeviceLocal(ctx, size, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    errdefer dst.deinit();
    try ctx.copyBuffer(staging.handle, dst.handle, size);
    return dst;
}

// Upload one layer's ≤7 matrices in a single command buffer submission.
fn uploadLayerBatch(ctx: *const GpuCtx, glayer: *GpuLayerWeights, lw: *const Gemma4LayerWeights) !void {
    var stagings: [7]GpuBuffer = undefined;
    var n_stagings: usize = 0;
    errdefer for (0..n_stagings) |i| stagings[i].deinit();

    const cmd = try ctx.beginBatchCopy();
    glayer.wq = try schedUpload(ctx, lw.wq, cmd, &stagings, &n_stagings);
    glayer.wk = try schedUpload(ctx, lw.wk, cmd, &stagings, &n_stagings);
    if (lw.wv) |wv|
        glayer.wv = try schedUpload(ctx, wv, cmd, &stagings, &n_stagings);
    glayer.wo = try schedUpload(ctx, lw.wo, cmd, &stagings, &n_stagings);
    glayer.w_gate = try schedUpload(ctx, lw.w_gate, cmd, &stagings, &n_stagings);
    glayer.w_up = try schedUpload(ctx, lw.w_up, cmd, &stagings, &n_stagings);
    glayer.w_down = try schedUpload(ctx, lw.w_down, cmd, &stagings, &n_stagings);

    try ctx.submitBatchCopy(cmd);
    for (0..n_stagings) |i| stagings[i].deinit();
}

// Upload all experts for one layer in batches of EXPERT_BATCH per command buffer.
// Each batch creates at most EXPERT_BATCH×3 staging buffers (~47 MiB at batch=16).
const EXPERT_BATCH: usize = 16;

fn uploadExpertsBatch(
    ctx: *const GpuCtx,
    gate_out: []?MatvecSession,
    up_out: []?MatvecSession,
    down_out: []?MatvecSession,
    lw: *const Gemma4LayerWeights,
    d_model: usize,
    d_expert: usize,
    skip_gate_up: bool,
    skip_down: bool,
) !void {
    const gu_row = math.rowBytes(lw.gate_up_exps.type_, d_model);
    const gu_each = d_expert * gu_row; // bytes for one expert's gate or up
    const dn_row = math.rowBytes(lw.down_exps.type_, d_expert);
    const dn_each = d_model * dn_row; // bytes for one expert's down

    var e: usize = 0;
    while (e < gate_out.len) : (e += EXPERT_BATCH) {
        const end = @min(e + EXPERT_BATCH, gate_out.len);
        var stagings: [EXPERT_BATCH * 3]GpuBuffer = undefined;
        var n_stagings: usize = 0;
        errdefer for (0..n_stagings) |i| stagings[i].deinit();

        const cmd = try ctx.beginBatchCopy();
        for (e..end) |eidx| {
            if (!skip_gate_up) {
                gate_out[eidx] = try schedUpload(ctx, .{ .data = lw.gate_up_exps.data[eidx * 2 * gu_each ..][0..gu_each], .type_ = lw.gate_up_exps.type_, .rows = d_expert, .cols = d_model }, cmd, &stagings, &n_stagings);
                up_out[eidx] = try schedUpload(ctx, .{ .data = lw.gate_up_exps.data[eidx * 2 * gu_each + gu_each ..][0..gu_each], .type_ = lw.gate_up_exps.type_, .rows = d_expert, .cols = d_model }, cmd, &stagings, &n_stagings);
            }
            if (!skip_down) {
                down_out[eidx] = try schedUpload(ctx, .{ .data = lw.down_exps.data[eidx * dn_each ..][0..dn_each], .type_ = lw.down_exps.type_, .rows = d_model, .cols = d_expert }, cmd, &stagings, &n_stagings);
            }
        }
        try ctx.submitBatchCopy(cmd);
        for (0..n_stagings) |i| stagings[i].deinit();
    }
}

// Upload a single matrix in its own command buffer submission.
fn uploadSingleBatch(ctx: *const GpuCtx, mat: RawMatrix) !?MatvecSession {
    var stagings: [1]GpuBuffer = undefined;
    var n_stagings: usize = 0;
    errdefer for (0..n_stagings) |i| stagings[i].deinit();

    const cmd = try ctx.beginBatchCopy();
    const sess = try schedUpload(ctx, mat, cmd, &stagings, &n_stagings);
    try ctx.submitBatchCopy(cmd);
    for (0..n_stagings) |i| stagings[i].deinit();
    return sess;
}

fn readSysU64(path: []const u8) u64 {
    const fd = std.posix.openat(std.posix.AT.FDCWD, path, .{}, 0) catch return 0;
    defer _ = std.os.linux.close(fd);
    var buf: [32]u8 = undefined;
    const n = std.posix.read(fd, &buf) catch return 0;
    const s = std.mem.trim(u8, buf[0..n], " \t\n");
    return std.fmt.parseInt(u64, s, 10) catch 0;
}

fn vramUsedMB() u64 {
    return readSysU64("/sys/class/drm/card1/device/mem_info_vram_used") / (1024 * 1024);
}

fn gttUsedMB() u64 {
    return readSysU64("/sys/class/drm/card1/device/mem_info_gtt_used") / (1024 * 1024);
}

// Read MemAvailable from /proc/meminfo; returns maxInt on any failure.
fn availableMemoryMB() u64 {
    const fd = std.posix.openat(std.posix.AT.FDCWD, "/proc/meminfo", .{}, 0) catch return std.math.maxInt(u64);
    defer _ = std.os.linux.close(fd);
    var buf: [4096]u8 = undefined;
    const n = std.posix.read(fd, &buf) catch return std.math.maxInt(u64);
    const text = buf[0..n];
    const prefix = "MemAvailable:";
    const pos = std.mem.indexOf(u8, text, prefix) orelse return std.math.maxInt(u64);
    const after = std.mem.trim(u8, text[pos + prefix.len ..][0..@min(32, text.len - pos - prefix.len)], " \t\n");
    const end = std.mem.indexOfAny(u8, after, " \t\n") orelse after.len;
    const kb = std.fmt.parseInt(u64, after[0..end], 10) catch return std.math.maxInt(u64);
    return kb / 1024;
}
