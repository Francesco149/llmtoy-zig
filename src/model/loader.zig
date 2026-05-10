/// GGUF → Config + ModelWeights loader.
///
/// Reads architecture metadata to build a Config, then maps each GGUF tensor
/// to the corresponding slot in ModelWeights. Norm weights (small) are
/// dequantized to f32; matrix weights keep their original quantized bytes.
///
/// Tensor naming follows the standard GGUF LLaMA convention used by llama.cpp:
///
///   token_embd.weight          — token embeddings
///   output_norm.weight         — final RMSNorm
///   output.weight              — LM head
///   blk.{l}.attn_norm.weight   — attention RMSNorm layer l
///   blk.{l}.attn_q.weight      — Q projection
///   blk.{l}.attn_k.weight      — K projection
///   blk.{l}.attn_v.weight      — V projection
///   blk.{l}.attn_output.weight — output projection
///   blk.{l}.ffn_norm.weight    — FFN RMSNorm
///   blk.{l}.ffn_gate.weight    — SwiGLU gate
///   blk.{l}.ffn_up.weight      — SwiGLU up
///   blk.{l}.ffn_down.weight    — SwiGLU down

const std    = @import("std");
const Config = @import("config.zig").Config;
const mw     = @import("model_weights.zig");
const gguf   = @import("../gguf/reader.zig");
const types  = @import("../gguf/types.zig");
const dq     = @import("../quant/dequant.zig");

pub const LoadError = error{
    MissingMetadata,
    MissingTensor,
    UnsupportedQuantType,
    OutOfMemory,
};

/// Build a Config from GGUF metadata.
pub fn configFromGguf(reader: *const gguf.GgufReader, allocator: std.mem.Allocator) !Config {
    const arch = reader.metaString("general.architecture") orelse return LoadError.MissingMetadata;

    const ctx_key      = try std.fmt.allocPrint(allocator, "{s}.context_length",                  .{arch});
    defer allocator.free(ctx_key);
    const emb_key      = try std.fmt.allocPrint(allocator, "{s}.embedding_length",                .{arch});
    defer allocator.free(emb_key);
    const blk_key      = try std.fmt.allocPrint(allocator, "{s}.block_count",                     .{arch});
    defer allocator.free(blk_key);
    const head_key     = try std.fmt.allocPrint(allocator, "{s}.attention.head_count",            .{arch});
    defer allocator.free(head_key);
    const kv_head_key  = try std.fmt.allocPrint(allocator, "{s}.attention.head_count_kv",         .{arch});
    defer allocator.free(kv_head_key);
    const ffn_key      = try std.fmt.allocPrint(allocator, "{s}.feed_forward_length",             .{arch});
    defer allocator.free(ffn_key);
    const eps_key      = try std.fmt.allocPrint(allocator, "{s}.attention.layer_norm_rms_epsilon", .{arch});
    defer allocator.free(eps_key);
    const rope_key     = try std.fmt.allocPrint(allocator, "{s}.rope.freq_base",                  .{arch});
    defer allocator.free(rope_key);

    const vocab_size = blk: {
        // Try {arch}.vocab_size first, fall back to tokenizer.ggml.vocab_size.
        const arch_vk = try std.fmt.allocPrint(allocator, "{s}.vocab_size", .{arch});
        defer allocator.free(arch_vk);
        if (reader.metaU64(arch_vk)) |v| break :blk v;
        if (reader.metaU64("tokenizer.ggml.vocab_size")) |v| break :blk v;
        // Last resort: count the vocab array.
        if (reader.metaArray("tokenizer.ggml.tokens")) |arr| break :blk @as(u64, arr.values.len);
        return LoadError.MissingMetadata;
    };

    return Config{
        .vocab_size   = @intCast(vocab_size),
        .d_model      = @intCast(reader.metaU64(emb_key)     orelse return LoadError.MissingMetadata),
        .n_layers     = @intCast(reader.metaU64(blk_key)     orelse return LoadError.MissingMetadata),
        .n_heads      = @intCast(reader.metaU64(head_key)    orelse return LoadError.MissingMetadata),
        .n_kv_heads   = @intCast(reader.metaU64(kv_head_key) orelse return LoadError.MissingMetadata),
        .d_ffn        = @intCast(reader.metaU64(ffn_key)     orelse return LoadError.MissingMetadata),
        .max_seq_len  = @intCast(reader.metaU64(ctx_key)     orelse 4096),
        .rope_theta   = reader.metaF32(rope_key) orelse 500_000.0,
        .eps          = reader.metaF32(eps_key)  orelse 1e-5,
    };
}

/// Load weights from a GGUF file into a ModelWeights.
///
/// The `reader` mmap must remain alive for the lifetime of the returned
/// ModelWeights (RawMatrix slices point into it directly).
pub fn loadWeights(
    reader: *const gguf.GgufReader,
    cfg: Config,
    backing: std.mem.Allocator,
) !mw.ModelWeights {
    // Build a name → TensorInfo lookup.
    var tensor_map = std.StringHashMap(types.TensorInfo).init(backing);
    defer tensor_map.deinit();
    for (reader.data.tensors) |t| {
        try tensor_map.put(t.name, t);
    }

    var arena = std.heap.ArenaAllocator.init(backing);
    errdefer arena.deinit();
    const aa = arena.allocator();

    const d   = cfg.d_model;
    const hd  = cfg.headDim();
    const nq  = cfg.n_heads    * hd;
    const nkv = cfg.n_kv_heads * hd;

    // Norm vectors: dequantized to f32 at load time.
    const out_norm = try loadNorm(reader, &tensor_map, "output_norm.weight", d, aa);
    const token_emb = rawMatrix(reader, &tensor_map, "token_embd.weight",
        cfg.vocab_size, d) catch return LoadError.MissingTensor;

    // LM head: some models tie it to the embedding; others have a separate tensor.
    const lm_head = rawMatrix(reader, &tensor_map, "output.weight",
        cfg.vocab_size, d) catch token_emb;

    const layers = try aa.alloc(mw.LayerWeights, cfg.n_layers);
    for (layers, 0..) |*lw, l| {
        var name_buf: [128]u8 = undefined;

        lw.attn_norm = try loadNorm(reader, &tensor_map,
            try layerName(&name_buf, l, "attn_norm.weight"), d, aa);
        lw.ffn_norm  = try loadNorm(reader, &tensor_map,
            try layerName(&name_buf, l, "ffn_norm.weight"),  d, aa);

        lw.wq     = try rawMatrix(reader, &tensor_map, try layerName(&name_buf, l, "attn_q.weight"),      nq,           d);
        lw.wk     = try rawMatrix(reader, &tensor_map, try layerName(&name_buf, l, "attn_k.weight"),      nkv,          d);
        lw.wv     = try rawMatrix(reader, &tensor_map, try layerName(&name_buf, l, "attn_v.weight"),      nkv,          d);
        lw.wo     = try rawMatrix(reader, &tensor_map, try layerName(&name_buf, l, "attn_output.weight"), d,            nq);
        lw.w_gate = try rawMatrix(reader, &tensor_map, try layerName(&name_buf, l, "ffn_gate.weight"),    cfg.d_ffn,    d);
        lw.w_up   = try rawMatrix(reader, &tensor_map, try layerName(&name_buf, l, "ffn_up.weight"),      cfg.d_ffn,    d);
        lw.w_down = try rawMatrix(reader, &tensor_map, try layerName(&name_buf, l, "ffn_down.weight"),    d,            cfg.d_ffn);
    }

    return mw.ModelWeights{
        .token_emb = token_emb,
        .layers    = layers,
        .out_norm  = out_norm,
        .lm_head   = lm_head,
        .arena     = arena,
    };
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn layerName(buf: []u8, l: usize, suffix: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "blk.{d}.{s}", .{ l, suffix });
}

fn loadNorm(
    reader: *const gguf.GgufReader,
    map: *const std.StringHashMap(types.TensorInfo),
    name: []const u8,
    d: usize,
    allocator: std.mem.Allocator,
) ![]f32 {
    const info = map.get(name) orelse return LoadError.MissingTensor;
    const raw  = reader.tensorBytes(info);
    const out  = try allocator.alloc(f32, d);
    switch (info.type_) {
        .f32  => dq.dequantF32(raw, out),
        .f16  => dq.dequantF16(raw, out),
        .q8_0 => dq.dequantQ8_0(raw, out),
        else  => return LoadError.UnsupportedQuantType,
    }
    return out;
}

fn rawMatrix(
    reader: *const gguf.GgufReader,
    map: *const std.StringHashMap(types.TensorInfo),
    name: []const u8,
    rows: usize,
    cols: usize,
) !mw.RawMatrix {
    const info = map.get(name) orelse return LoadError.MissingTensor;
    return mw.RawMatrix{
        .data  = reader.tensorBytes(info),
        .type_ = info.type_,
        .rows  = rows,
        .cols  = cols,
    };
}
