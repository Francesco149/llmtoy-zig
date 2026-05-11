/// GGUF → Gemma4Config + Gemma4Weights loader.
///
/// Reads gemma4-specific metadata (expert counts, sliding-window pattern,
/// per-layer KV head counts) and maps each GGUF tensor to the right slot.

const std    = @import("std");
const cfg_   = @import("config.zig");
const wt_    = @import("weights.zig");
const gguf   = @import("../../gguf/reader.zig");
const types  = @import("../../gguf/types.zig");
const dq     = @import("../../quant/dequant.zig");
const math   = @import("../../ops/math.zig");

pub const Gemma4Config  = cfg_.Gemma4Config;
pub const Gemma4Weights = wt_.Gemma4Weights;

pub const LoadError = error{
    MissingMetadata,
    MissingTensor,
    UnsupportedQuantType,
    OutOfMemory,
    TooManyLayers,
};

/// Detect whether a GGUF file is Gemma4 architecture.
pub fn isGemma4(reader: *const gguf.GgufReader) bool {
    const arch = reader.metaString("general.architecture") orelse return false;
    return std.mem.eql(u8, arch, "gemma4");
}

pub fn configFromGguf(reader: *const gguf.GgufReader, _: std.mem.Allocator) !Gemma4Config {
    const n_layers: usize = @intCast(reader.metaU64("gemma4.block_count") orelse
        return LoadError.MissingMetadata);
    if (n_layers > cfg_.MAX_LAYERS) return LoadError.TooManyLayers;

    // Read per-layer sliding-window pattern (bool array: true = local/SWA, false = global).
    var is_swa = [_]bool{false} ** cfg_.MAX_LAYERS;
    if (reader.metaArray("gemma4.attention.sliding_window_pattern")) |arr| {
        for (0..@min(n_layers, arr.values.len)) |l| {
            is_swa[l] = arr.values[l].bool_;
        }
    }

    // Read per-layer KV head counts (int32 array).
    var n_kv_heads = [_]usize{8} ** cfg_.MAX_LAYERS;
    if (reader.metaArray("gemma4.attention.head_count_kv")) |arr| {
        for (0..@min(n_layers, arr.values.len)) |l| {
            n_kv_heads[l] = @intCast(@max(1, arr.values[l].int32));
        }
    }

    const vocab_size: usize = blk: {
        if (reader.metaU64("gemma4.vocab_size")) |v| break :blk @intCast(v);
        if (reader.metaU64("tokenizer.ggml.vocab_size")) |v| break :blk @intCast(v);
        if (reader.metaArray("tokenizer.ggml.tokens")) |a| break :blk a.values.len;
        return LoadError.MissingMetadata;
    };

    const eps_raw = reader.metaF32("gemma4.attention.layer_norm_rms_epsilon") orelse 1e-6;
    const eps     = if (eps_raw < 1e-10) 1e-6 else eps_raw;

    return Gemma4Config{
        .vocab_size      = vocab_size,
        .d_model         = @intCast(reader.metaU64("gemma4.embedding_length") orelse
            return LoadError.MissingMetadata),
        .n_layers        = n_layers,
        .n_heads         = @intCast(reader.metaU64("gemma4.attention.head_count") orelse
            return LoadError.MissingMetadata),
        .d_ffn           = @intCast(reader.metaU64("gemma4.feed_forward_length") orelse
            return LoadError.MissingMetadata),
        .d_expert        = @intCast(reader.metaU64("gemma4.expert_feed_forward_length") orelse
            return LoadError.MissingMetadata),
        .n_experts       = @intCast(reader.metaU64("gemma4.expert_count") orelse
            return LoadError.MissingMetadata),
        .n_experts_used  = @intCast(reader.metaU64("gemma4.expert_used_count") orelse 8),
        .head_dim_swa    = @intCast(reader.metaU64("gemma4.attention.key_length_swa") orelse 256),
        .head_dim_global = @intCast(reader.metaU64("gemma4.attention.key_length")     orelse 512),
        .max_seq_len     = @intCast(reader.metaU64("gemma4.context_length") orelse 4096),
        .sliding_window  = @intCast(reader.metaU64("gemma4.attention.sliding_window") orelse 1024),
        .rope_theta_swa  = reader.metaF32("gemma4.rope.freq_base_swa")  orelse 10_000.0,
        .rope_theta_global = reader.metaF32("gemma4.rope.freq_base")    orelse 1_000_000.0,
        .eps             = eps,
        .logit_softcap   = reader.metaF32("gemma4.final_logit_softcapping") orelse 30.0,
        .is_swa          = is_swa,
        .n_kv_heads      = n_kv_heads,
    };
}

pub fn loadWeights(
    reader: *const gguf.GgufReader,
    cfg: Gemma4Config,
    backing: std.mem.Allocator,
) !Gemma4Weights {
    var tensor_map = std.StringHashMap(types.TensorInfo).init(backing);
    defer tensor_map.deinit();
    for (reader.data.tensors) |t| try tensor_map.put(t.name, t);

    var arena = std.heap.ArenaAllocator.init(backing);
    errdefer arena.deinit();
    const aa = arena.allocator();

    const d = cfg.d_model;

    const out_norm = try loadNorm(reader, &tensor_map, "output_norm.weight", d, aa);
    const token_emb = try rawMatrix(reader, &tensor_map, "token_embd.weight",
        cfg.vocab_size, d);
    const lm_head = rawMatrix(reader, &tensor_map, "output.weight",
        cfg.vocab_size, d) catch token_emb;

    // Global-attention RoPE frequencies: shape [head_dim_global/2] stored as F32.
    const rope_freqs = try loadRopeFreqs(reader, &tensor_map, cfg.head_dim_global, aa);

    const layers = try aa.alloc(wt_.Gemma4LayerWeights, cfg.n_layers);
    var name_buf: [128]u8 = undefined;

    for (layers, 0..) |*lw, l| {
        const hd  = cfg.headDim(l);
        const nq  = cfg.nq(l);
        const nkv = cfg.nkv(l);

        lw.attn_norm          = try loadNorm(reader, &tensor_map,
            try ln(&name_buf, l, "attn_norm.weight"),           d,  aa);
        lw.post_attention_norm = try loadNorm(reader, &tensor_map,
            try ln(&name_buf, l, "post_attention_norm.weight"), d,  aa);
        lw.q_norm             = try loadNorm(reader, &tensor_map,
            try ln(&name_buf, l, "attn_q_norm.weight"),         hd, aa);
        lw.k_norm             = try loadNorm(reader, &tensor_map,
            try ln(&name_buf, l, "attn_k_norm.weight"),         hd, aa);
        lw.ffn_norm           = try loadNorm(reader, &tensor_map,
            try ln(&name_buf, l, "ffn_norm.weight"),            d,  aa);
        lw.pre_ffw_norm_2     = try loadNorm(reader, &tensor_map,
            try ln(&name_buf, l, "pre_ffw_norm_2.weight"),      d,  aa);
        lw.post_ffw_norm_1    = try loadNorm(reader, &tensor_map,
            try ln(&name_buf, l, "post_ffw_norm_1.weight"),     d,  aa);
        lw.post_ffw_norm_2    = try loadNorm(reader, &tensor_map,
            try ln(&name_buf, l, "post_ffw_norm_2.weight"),     d,  aa);
        lw.post_ffw_norm      = try loadNorm(reader, &tensor_map,
            try ln(&name_buf, l, "post_ffw_norm.weight"),       d,  aa);

        lw.layer_output_scale = try loadScalar(reader, &tensor_map,
            try ln(&name_buf, l, "layer_output_scale.weight"));

        lw.wq = try rawMatrix(reader, &tensor_map, try ln(&name_buf, l, "attn_q.weight"),      nq,  d);
        lw.wk = try rawMatrix(reader, &tensor_map, try ln(&name_buf, l, "attn_k.weight"),      nkv, d);
        lw.wv = rawMatrix(reader, &tensor_map, try ln(&name_buf, l, "attn_v.weight"), nkv, d) catch null;
        lw.wo = try rawMatrix(reader, &tensor_map, try ln(&name_buf, l, "attn_output.weight"), d,   nq);

        lw.w_gate = try rawMatrix(reader, &tensor_map, try ln(&name_buf, l, "ffn_gate.weight"), cfg.d_ffn, d);
        lw.w_up   = try rawMatrix(reader, &tensor_map, try ln(&name_buf, l, "ffn_up.weight"),   cfg.d_ffn, d);
        lw.w_down = try rawMatrix(reader, &tensor_map, try ln(&name_buf, l, "ffn_down.weight"), d, cfg.d_ffn);

        lw.router_w     = try rawMatrix(reader, &tensor_map,
            try ln(&name_buf, l, "ffn_gate_inp.weight"), cfg.n_experts, d);
        lw.router_scale = try loadNorm(reader, &tensor_map,
            try ln(&name_buf, l, "ffn_gate_inp.scale"), d, aa);
        lw.gate_up_exps = try rawMatrix3D(reader, &tensor_map,
            try ln(&name_buf, l, "ffn_gate_up_exps.weight"),
            cfg.n_experts, 2 * cfg.d_expert, d);
        lw.down_exps    = try rawMatrix3D(reader, &tensor_map,
            try ln(&name_buf, l, "ffn_down_exps.weight"),
            cfg.n_experts, d, cfg.d_expert);
        lw.down_exps_scale = try loadNorm(reader, &tensor_map,
            try ln(&name_buf, l, "ffn_down_exps.scale"), cfg.n_experts, aa);
    }

    return Gemma4Weights{
        .token_emb  = token_emb,
        .layers     = layers,
        .out_norm   = out_norm,
        .lm_head    = lm_head,
        .rope_freqs = rope_freqs,
        .arena      = arena,
    };
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn ln(buf: []u8, l: usize, suffix: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "blk.{d}.{s}", .{ l, suffix });
}

fn loadNorm(
    reader: *const gguf.GgufReader,
    map: *const std.StringHashMap(types.TensorInfo),
    name: []const u8,
    n: usize,
    allocator: std.mem.Allocator,
) ![]f32 {
    const info = map.get(name) orelse return LoadError.MissingTensor;
    const raw  = reader.tensorBytes(info);
    const out  = try allocator.alloc(f32, n);
    switch (info.type_) {
        .f32  => dq.dequantF32(raw, out),
        .f16  => dq.dequantF16(raw, out),
        .q8_0 => dq.dequantQ8_0(raw, out),
        else  => return LoadError.UnsupportedQuantType,
    }
    return out;
}

fn loadScalar(
    reader: *const gguf.GgufReader,
    map: *const std.StringHashMap(types.TensorInfo),
    name: []const u8,
) !f32 {
    const info = map.get(name) orelse return 1.0;
    const raw  = reader.tensorBytes(info);
    var buf: [1]f32 = undefined;
    switch (info.type_) {
        .f32 => dq.dequantF32(raw, &buf),
        .f16 => dq.dequantF16(raw, &buf),
        else => return 1.0,
    }
    return buf[0];
}

fn loadRopeFreqs(
    reader: *const gguf.GgufReader,
    map: *const std.StringHashMap(types.TensorInfo),
    head_dim_global: usize,
    allocator: std.mem.Allocator,
) ![]f32 {
    const n = head_dim_global / 2;
    const out = try allocator.alloc(f32, n);
    const info = map.get("rope_freqs.weight") orelse {
        // Fallback: compute from default theta
        for (0..n) |i| {
            out[i] = 1.0 / std.math.pow(
                f32, 1_000_000.0,
                @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(head_dim_global)),
            );
        }
        return out;
    };
    const raw = reader.tensorBytes(info);
    // rope_freqs.weight may have head_dim_global elements (one per dim, stored in pairs)
    // or head_dim_global/2 elements (one per pair).
    const elem_count = raw.len / 4; // assuming F32
    if (elem_count == n) {
        dq.dequantF32(raw, out);
    } else if (elem_count == head_dim_global) {
        // One value per dimension; take every other (they come in identical pairs).
        const tmp: []f32 = try allocator.alloc(f32, head_dim_global);
        defer allocator.free(tmp);
        dq.dequantF32(raw, tmp);
        for (0..n) |i| out[i] = tmp[2 * i];
    } else {
        // Unexpected size: fall back to computed
        for (0..n) |i| {
            out[i] = 1.0 / std.math.pow(
                f32, 1_000_000.0,
                @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(head_dim_global)),
            );
        }
    }
    return out;
}

fn rawMatrix(
    reader: *const gguf.GgufReader,
    map: *const std.StringHashMap(types.TensorInfo),
    name: []const u8,
    rows: usize,
    cols: usize,
) !wt_.RawMatrix {
    const info = map.get(name) orelse return LoadError.MissingTensor;
    return wt_.RawMatrix{
        .data  = reader.tensorBytes(info),
        .type_ = info.type_,
        .rows  = rows,
        .cols  = cols,
    };
}

/// Load a 3-D expert weight tensor as a flat RawMatrix.
/// GGUF stores [cols, rows_per_expert, n_experts]; we expose rows = n_experts * rows_per_expert.
fn rawMatrix3D(
    reader: *const gguf.GgufReader,
    map: *const std.StringHashMap(types.TensorInfo),
    name: []const u8,
    n_experts: usize,
    rows_per_expert: usize,
    cols: usize,
) !wt_.RawMatrix {
    const info = map.get(name) orelse return LoadError.MissingTensor;
    return wt_.RawMatrix{
        .data  = reader.tensorBytes(info),
        .type_ = info.type_,
        .rows  = n_experts * rows_per_expert,
        .cols  = cols,
    };
}
