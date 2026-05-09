const std = @import("std");
const gguf_reader = @import("gguf/reader.zig");
const vocab_mod = @import("tokenizer/vocab.zig");
const bpe = @import("tokenizer/bpe.zig");

// Pull tests from sub-modules into the test binary.
comptime {
    _ = @import("gguf/reader.zig");
    _ = @import("tokenizer/bpe.zig");
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = init.gpa;
    const arena = init.arena.allocator();

    const args = try init.minimal.args.toSlice(arena);

    var out_buf: [8192]u8 = undefined;
    var out_fw = std.Io.File.stdout().writer(io, &out_buf);
    const out = &out_fw.interface;
    defer out.flush() catch {};

    if (args.len < 2) {
        try usagePrint(out);
        return;
    }

    if (std.mem.eql(u8, args[1], "info")) {
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
    } else {
        try usagePrint(out);
    }
}

fn usagePrint(out: *std.Io.Writer) !void {
    try out.writeAll(
        \\llmtoy-zig  —  educational LLM inference
        \\
        \\  llmtoy info <model.gguf>              print model metadata and tensor summary
        \\  llmtoy tokenize <model.gguf> <text>   BPE-encode text, print IDs and decoded tokens
        \\
    );
}

fn cmdInfo(out: *std.Io.Writer, path: []const u8, io: std.Io, gpa: std.mem.Allocator) !void {
    var reader = try gguf_reader.GgufReader.open(path, io, gpa);
    defer reader.deinit();

    const size_gib = @as(f64, @floatFromInt(reader.mmap.len)) / (1024 * 1024 * 1024);

    try out.print("file:      {s}\n", .{path});
    try out.print("version:   GGUF v{}\n", .{reader.data.version});
    try out.print("size:      {d:.2} GiB\n", .{size_gib});
    try out.print("tensors:   {}\n", .{reader.data.tensors.len});
    try out.print("metadata:  {} entries\n", .{reader.data.metadata.count()});

    // Key architecture metadata.
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

    // Quant type distribution across tensors.
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

    // First few tensors as a sanity check.
    try out.print("\nfirst 8 tensors:\n", .{});
    for (reader.data.tensors[0..@min(8, reader.data.tensors.len)]) |tensor| {
        try out.print("  [{s}] {s}  dims={any}  offset={}\n", .{
            tensor.type_.label(), tensor.name, tensor.dims, tensor.offset,
        });
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

    const ids = try bpe.encode(text, &vocab, gpa);
    defer gpa.free(ids);

    const decoded = try bpe.decode(ids, &vocab, gpa);
    defer gpa.free(decoded);

    try out.print("input:   \"{s}\"\n", .{text});
    try out.print("ids:     {any}\n", .{ids});
    try out.print("decoded: \"{s}\"\n", .{decoded});

    try out.print("\ntokens:\n", .{});
    for (ids) |id| {
        if (id < vocab.tokens.len) {
            try out.print("  {:6}  {s}\n", .{ id, vocab.tokens[id] });
        }
    }
}
