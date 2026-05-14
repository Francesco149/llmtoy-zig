const std         = @import("std");
const gguf_reader = @import("gguf/reader.zig");
const vocab_mod   = @import("tokenizer/vocab.zig");
const bpe         = @import("tokenizer/bpe.zig");
const chat_tmpl   = @import("tokenizer/chat_template.zig");
const loader      = @import("model/loader.zig");
const fwd         = @import("model/forward.zig");
const kv_mod      = @import("model/kv_cache.zig");
const sample_mod  = @import("model/sample.zig");
const tp          = @import("ops/thread_pool.zig");
const g4_loader   = @import("model/gemma4/loader.zig");
const g4_fwd      = @import("model/gemma4/forward.zig");
const g4_kv       = @import("model/gemma4/kv_cache.zig");
const gpu_ctx     = @import("gpu/context.zig");
const gpu_matvec  = @import("gpu/matvec.zig");

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
    _ = @import("gpu/matvec.zig");
}

pub fn main(init: std.process.Init) !void {
    const io    = init.io;
    const gpa   = init.gpa;
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
        try cmdGpuInfo(out);
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
    } else if (std.mem.eql(u8, args[1], "generate")) {
        if (args.len < 4) {
            std.debug.print(
                "usage: llmtoy generate <model.gguf> <prompt> [--chat] [--max-tokens N] [--temperature T] [--top-p P] [--top-k K] [--seed S] [--threads N]\n",
                .{},
            );
            return error.MissingArg;
        }
        const model_path = args[2];
        const prompt     = args[3];

        // Parse optional flags.
        var max_tokens:  u32   = 256;
        var temperature: f32   = 0.8;
        var top_p:       f32   = 0.9;
        var top_k:       u32   = 40;
        var seed:        u64   = 42;
        var threads:     u32   = 0; // 0 = auto (getCpuCount)
        var chat:        bool  = false;
        var i: usize = 4;
        while (i < args.len) {
            const flag = args[i];
            if (std.mem.eql(u8, flag, "--chat")) {
                chat = true;
                i += 1;
                continue;
            }
            if (i + 1 >= args.len) break;
            const val  = args[i + 1];
            if (std.mem.eql(u8, flag, "--max-tokens"))  max_tokens  = try std.fmt.parseInt(u32, val, 10);
            if (std.mem.eql(u8, flag, "--temperature")) temperature = try std.fmt.parseFloat(f32, val);
            if (std.mem.eql(u8, flag, "--top-p"))       top_p       = try std.fmt.parseFloat(f32, val);
            if (std.mem.eql(u8, flag, "--top-k"))       top_k       = try std.fmt.parseInt(u32, val, 10);
            if (std.mem.eql(u8, flag, "--seed"))        seed        = try std.fmt.parseInt(u64, val, 10);
            if (std.mem.eql(u8, flag, "--threads"))     threads     = try std.fmt.parseInt(u32, val, 10);
            i += 2;
        }

        try cmdGenerate(out, model_path, prompt, .{
            .max_tokens  = max_tokens,
            .temperature = temperature,
            .top_p       = top_p,
            .top_k       = top_k,
            .seed        = seed,
            .threads     = threads,
            .chat        = chat,
        }, io, gpa);
    } else {
        try usagePrint(out);
    }
}

fn usagePrint(out: *std.Io.Writer) !void {
    try out.writeAll(
        \\llmtoy-zig  —  educational LLM inference
        \\
        \\  llmtoy gpu-info                        list Vulkan device and run a matvec smoke test
        \\  llmtoy info <model.gguf>               print model metadata and tensor summary
        \\  llmtoy tokenize <model.gguf> <text>    BPE-encode text, print IDs and decoded tokens
        \\  llmtoy generate <model.gguf> <prompt> [--chat] [--max-tokens N] [--temperature T] [--top-p P] [--top-k K] [--seed S] [--threads N]
        \\
    );
}

fn cmdGpuInfo(out: *std.Io.Writer) !void {
    var ctx = gpu_ctx.GpuContext.init() catch |e| {
        try out.print("gpu init failed: {}\n", .{e});
        return;
    };
    defer ctx.deinit();

    const name = ctx.deviceName();
    try out.print("GPU device: {s}\n", .{std.mem.sliceTo(&name, 0)});

    // Smoke test: 4×4 identity * [1,2,3,4] = [1,2,3,4]
    var pipeline = gpu_matvec.MatvecPipeline.init(&ctx) catch |e| {
        try out.print("pipeline init failed: {}\n", .{e});
        return;
    };
    defer pipeline.deinit();

    const mat = [16]f32{ 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };
    const vec = [4]f32{ 1, 2, 3, 4 };
    var result = [4]f32{ 0, 0, 0, 0 };
    try gpu_matvec.matvecF32(&ctx, &pipeline, &mat, &vec, &result, 4, 4);

    try out.print("matvec smoke test: [{d:.0}, {d:.0}, {d:.0}, {d:.0}] (expect [1, 2, 3, 4])\n",
        .{ result[0], result[1], result[2], result[3] });
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

        const ctx_key  = try std.fmt.allocPrint(gpa, "{s}.context_length",       .{arch});
        defer gpa.free(ctx_key);
        const emb_key  = try std.fmt.allocPrint(gpa, "{s}.embedding_length",     .{arch});
        defer gpa.free(emb_key);
        const blk_key  = try std.fmt.allocPrint(gpa, "{s}.block_count",          .{arch});
        defer gpa.free(blk_key);
        const head_key = try std.fmt.allocPrint(gpa, "{s}.attention.head_count", .{arch});
        defer gpa.free(head_key);

        if (reader.metaU32(ctx_key))  |v| try out.print("ctx_len:   {}\n",  .{v});
        if (reader.metaU32(emb_key))  |v| try out.print("emb_dim:   {}\n",  .{v});
        if (reader.metaU32(blk_key))  |v| try out.print("n_layers:  {}\n",  .{v});
        if (reader.metaU32(head_key)) |v| try out.print("n_heads:   {}\n",  .{v});
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
        const key   = @intFromEnum(tensor.type_);
        const entry = try counts.getOrPutValue(key, 0);
        entry.value_ptr.* += 1;
    }
    var it = counts.iterator();
    while (it.next()) |entry| {
        const gtype: @import("gguf/types.zig").GgmlType = @enumFromInt(entry.key_ptr.*);
        try out.print("  {s}: {}\n", .{ gtype.label(), entry.value_ptr.* });
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

fn cmdTokenize(out: *std.Io.Writer, path: []const u8, text: []const u8, io: std.Io, gpa: std.mem.Allocator) !void {
    var reader = try gguf_reader.GgufReader.open(path, io, gpa);
    defer reader.deinit();

    var vocab = try vocab_mod.fromGguf(&reader, gpa);
    defer vocab.deinit();

    try out.print("tokenizer: {s}  vocab: {}  merges: {}\n", .{
        vocab.model, vocab.tokens.len, vocab.merge_rank.count(),
    });

    const ids     = try bpe.encode(text, &vocab, gpa);
    defer gpa.free(ids);
    const decoded = try bpe.decode(ids, &vocab, gpa);
    defer gpa.free(decoded);

    try out.print("input:   \"{s}\"\n", .{text});
    try out.print("ids:     {any}\n",   .{ids});
    try out.print("decoded: \"{s}\"\n", .{decoded});

    try out.print("\ntokens:\n", .{});
    for (ids) |id| {
        if (id < vocab.tokens.len) try out.print("  {:6}  {s}\n", .{ id, vocab.tokens[id] });
    }
}

const GenerateOptions = struct {
    max_tokens:  u32  = 256,
    temperature: f32  = 0.8,
    top_p:       f32  = 0.9,
    top_k:       u32  = 40,
    seed:        u64  = 42,
    threads:     u32  = 0, // 0 = auto
    chat:        bool = false,
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
        .top_p       = opts.top_p,
        .top_k       = opts.top_k,
    };

    try out.print("{s}", .{prompt_text});
    try out.flush();

    if (g4_loader.isGemma4(&reader)) {
        const g4cfg = try g4_loader.configFromGguf(&reader, gpa);
        std.debug.print(
            "  [Gemma4] layers={} heads={} d_model={} experts={}/{} vocab={}\n",
            .{ g4cfg.n_layers, g4cfg.n_heads, g4cfg.d_model,
               g4cfg.n_experts, g4cfg.n_experts_used, g4cfg.vocab_size },
        );
        var weights = try g4_loader.loadWeights(&reader, g4cfg, gpa);
        defer weights.deinit();
        const max_seq: usize = @min(g4cfg.max_seq_len, 4096);
        var kv = try g4_kv.Gemma4KvCache.init(g4cfg, max_seq, gpa);
        defer kv.deinit();
        const t_prefill_start = clk.now(io);
        printSetupTiming(t_load_start, t_prefill_start);
        std.debug.print("prefilling {} tokens (threads={})...\n", .{ prompt_ids.len, n_threads });
        var last_logits: []f32 = undefined;
        for (prompt_ids, 0..) |tok, pos| {
            if (pos > 0) gpa.free(last_logits);
            last_logits = try g4_fwd.forwardOne(tok, pos, &kv, &weights, g4cfg, pool, gpa);
        }
        const t_prefill = clk.now(io);
        printTokenTiming("prefill", prompt_ids.len, t_prefill_start, t_prefill);
        var n_gen: u32 = 0;
        var pos: usize = prompt_ids.len;
        const t_gen_start = clk.now(io);
        while (n_gen < opts.max_tokens) : (n_gen += 1) {
            const next_tok = try sample_mod.sample(last_logits, params, rng, gpa);
            gpa.free(last_logits);
            if (next_tok == vocab.eos_token_id) break;
            var tok_buf: [64]u8 = undefined;
            const tok_len = bpe.decodeOne(next_tok, &vocab, &tok_buf);
            try out.writeAll(tok_buf[0..tok_len]);
            try out.flush();
            last_logits = try g4_fwd.forwardOne(next_tok, pos, &kv, &weights, g4cfg, pool, gpa);
            pos += 1;
        }
        gpa.free(last_logits);
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
            if (next_tok == vocab.eos_token_id) break;
            var tok_buf: [64]u8 = undefined;
            const tok_len = bpe.decodeOne(next_tok, &vocab, &tok_buf);
            try out.writeAll(tok_buf[0..tok_len]);
            try out.flush();
            last_logits = try fwd.forwardOneModel(next_tok, pos, &kv, &weights, cfg, pool, gpa);
            pos += 1;
        }
        gpa.free(last_logits);
        const t_end = clk.now(io);
        printTokenTiming("generation", n_gen, t_gen_start, t_end);
    }
    try out.print("\n", .{});
    try out.flush();
}

fn printSetupTiming(start: anytype, end: anytype) void {
    const ms = @divTrunc(start.durationTo(end).nanoseconds, std.time.ns_per_ms);
    std.debug.print("  setup: {} ms\n", .{ms});
}

fn printTokenTiming(label: []const u8, tokens: anytype, start: anytype, end: anytype) void {
    const ns = start.durationTo(end).nanoseconds;
    const ms = @divTrunc(ns, std.time.ns_per_ms);
    const seconds = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(std.time.ns_per_s));
    const tps = if (seconds > 0.0) @as(f64, @floatFromInt(tokens)) / seconds else 0.0;
    std.debug.print("  {s}: {} tokens in {} ms ({d:.2} tok/s)\n", .{ label, tokens, ms, tps });
}
