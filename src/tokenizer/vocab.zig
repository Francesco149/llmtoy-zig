const std = @import("std");
const t = @import("../gguf/types.zig");
const reader_mod = @import("../gguf/reader.zig");

pub const Vocab = struct {
    model: []const u8,
    tokens: []const []const u8,
    token_to_id: std.StringHashMap(u32),
    merge_rank: std.StringHashMap(u32),
    bos_token_id: u32,
    eos_token_id: u32,
    add_bos: bool,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *Vocab) void {
        self.allocator.free(self.tokens);
        self.token_to_id.deinit();
        self.merge_rank.deinit();
    }
};

/// Build a Vocab from an open GgufReader.
/// Token/merge strings are borrowed from the reader's arena — keep the reader alive.
/// The tokens slice, token_to_id and merge_rank hashmaps are owned by `allocator`.
pub fn fromGguf(reader: *const reader_mod.GgufReader, allocator: std.mem.Allocator) !Vocab {
    const data = &reader.data;

    const model = data.metaString("tokenizer.ggml.model") orelse return error.MissingTokenizerModel;

    const tokens_arr = data.metaArray("tokenizer.ggml.tokens") orelse return error.MissingTokens;
    const tokens = try allocator.alloc([]const u8, tokens_arr.values.len);
    errdefer allocator.free(tokens);
    for (tokens_arr.values, 0..) |v, i| {
        tokens[i] = if (v == .string) v.string else return error.InvalidTokenEntry;
    }

    var token_to_id = std.StringHashMap(u32).init(allocator);
    errdefer token_to_id.deinit();
    try token_to_id.ensureTotalCapacity(@intCast(tokens.len));
    for (tokens, 0..) |tok, i| {
        try token_to_id.put(tok, @intCast(i));
    }

    var merge_rank = std.StringHashMap(u32).init(allocator);
    errdefer merge_rank.deinit();
    if (data.metaArray("tokenizer.ggml.merges")) |merges_arr| {
        try merge_rank.ensureTotalCapacity(@intCast(merges_arr.values.len));
        for (merges_arr.values, 0..) |v, i| {
            const merge_str = if (v == .string) v.string else return error.InvalidMergeEntry;
            try merge_rank.put(merge_str, @intCast(i));
        }
    }

    const bos_id = data.metaU32("tokenizer.ggml.bos_token_id") orelse 1;
    const eos_id = data.metaU32("tokenizer.ggml.eos_token_id") orelse 2;
    const add_bos = data.metaBool("tokenizer.ggml.add_bos_token") orelse false;

    return .{
        .model = model,
        .tokens = tokens,
        .token_to_id = token_to_id,
        .merge_rank = merge_rank,
        .bos_token_id = bos_id,
        .eos_token_id = eos_id,
        .add_bos = add_bos,
        .allocator = allocator,
    };
}
