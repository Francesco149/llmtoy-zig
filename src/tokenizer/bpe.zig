const std = @import("std");
const Vocab = @import("vocab.zig").Vocab;

// GPT-2 bytes_to_unicode: maps each byte (0-255) to a unicode codepoint.
//
// "Base" bytes that map to themselves:
//   33-126  (printable ASCII, '!'..'~')
//   161-172 (Latin-1 supplement: '¡'..'¬')
//   174-255 (Latin-1 supplement: '®'..'ÿ')
//
// The remaining 68 bytes (0-32, 127, 128-160, 173) map to codepoints 256-323
// in iteration order, so e.g. space (0x20 = 32) → 0x120 = Ġ.
const BYTE_TO_CP: [256]u21 = blk: {
    var table: [256]u21 = undefined;
    var n: u21 = 256;
    for (0..256) |b| {
        const in_base = (b >= 33 and b <= 126) or (b >= 161 and b <= 172) or (b >= 174 and b <= 255);
        if (in_base) {
            table[b] = @intCast(b);
        } else {
            table[b] = n;
            n += 1;
        }
    }
    break :blk table;
};

// Reverse: unicode codepoint → byte.  Valid for codepoints produced by BYTE_TO_CP.
const CP_TO_BYTE: [324]u8 = blk: {
    var table: [324]u8 = [_]u8{0} ** 324;
    for (BYTE_TO_CP, 0..) |cp, b| {
        table[cp] = @intCast(b);
    }
    break :blk table;
};

// True iff `cp` is a codepoint produced by BYTE_TO_CP (one of the 256 unique outputs).
fn cpIsByteEncoded(cp: u21) bool {
    return (cp >= 33 and cp <= 126) or
        (cp >= 161 and cp <= 172) or
        (cp >= 174 and cp <= 255) or
        (cp >= 256 and cp <= 323);
}

/// Encode `text` into token IDs using byte-level BPE.
///
/// Pre-tokenization: simple whitespace split, with a leading space prepended
/// to every word after the first (matching GPT-2 convention: " world" not "world").
/// This is a simplification — a production tokenizer would use a Unicode-aware
/// regex (e.g. the tiktoken pattern for Qwen3).
///
/// Caller owns the returned slice.
pub fn encode(text: []const u8, vocab: *const Vocab, allocator: std.mem.Allocator) ![]u32 {
    var ids: std.ArrayList(u32) = .{ .items = &.{}, .capacity = 0 };
    errdefer ids.deinit(allocator);

    if (vocab.add_bos) {
        try ids.append(allocator, vocab.bos_token_id);
    }

    // Walk through text, splitting on spaces.
    var first_piece = true;
    var i: usize = 0;
    while (i < text.len) {
        // skip leading spaces
        while (i < text.len and text[i] == ' ') i += 1;
        if (i >= text.len) break;
        // find end of word
        const start = i;
        while (i < text.len and text[i] != ' ') i += 1;
        const word = text[start..i];

        try encodePiece(word, !first_piece, vocab, &ids, allocator);
        first_piece = false;
    }

    return ids.toOwnedSlice(allocator);
}

fn encodePiece(
    word: []const u8,
    with_space: bool,
    vocab: *const Vocab,
    ids: *std.ArrayList(u32),
    allocator: std.mem.Allocator,
) !void {
    // Use an arena for all BPE intermediate allocations.
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    // Build initial symbol list: one utf8-encoded codepoint per byte.
    var symbols: std.ArrayList([]const u8) = .{ .items = &.{}, .capacity = 0 };

    if (with_space) {
        try symbols.append(aa, try cpToUtf8(aa, BYTE_TO_CP[' ']));
    }
    for (word) |byte| {
        try symbols.append(aa, try cpToUtf8(aa, BYTE_TO_CP[byte]));
    }

    // BPE merge loop: repeatedly find the lowest-ranked adjacent pair and merge.
    while (symbols.items.len > 1) {
        var best_rank: u32 = std.math.maxInt(u32);
        var best_pos: usize = 0;
        var found = false;

        for (0..symbols.items.len - 1) |pos| {
            const pair = try std.fmt.allocPrint(aa, "{s} {s}", .{ symbols.items[pos], symbols.items[pos + 1] });
            if (vocab.merge_rank.get(pair)) |rank| {
                if (rank < best_rank) {
                    best_rank = rank;
                    best_pos = pos;
                    found = true;
                }
            }
        }

        if (!found) break;

        // Merge the pair at best_pos.
        const merged = try std.fmt.allocPrint(aa, "{s}{s}", .{
            symbols.items[best_pos],
            symbols.items[best_pos + 1],
        });
        symbols.items[best_pos] = merged;
        _ = symbols.orderedRemove(best_pos + 1);
    }

    // Look up token IDs.
    for (symbols.items) |sym| {
        if (vocab.token_to_id.get(sym)) |id| {
            try ids.append(allocator, id);
        }
        // Unknown symbols are silently skipped — rare in practice with byte-level BPE.
    }
}

fn cpToUtf8(allocator: std.mem.Allocator, cp: u21) ![]u8 {
    var buf: [4]u8 = undefined;
    const len = try std.unicode.utf8Encode(cp, &buf);
    return allocator.dupe(u8, buf[0..len]);
}

/// Decode token IDs back to bytes.
/// Caller owns the returned slice.
pub fn decode(token_ids: []const u32, vocab: *const Vocab, allocator: std.mem.Allocator) ![]u8 {
    var result: std.ArrayList(u8) = .{ .items = &.{}, .capacity = 0 };
    errdefer result.deinit(allocator);

    for (token_ids) |id| {
        if (id >= vocab.tokens.len) continue;
        const tok = vocab.tokens[id];

        var pos: usize = 0;
        while (pos < tok.len) {
            const seq_len = std.unicode.utf8ByteSequenceLength(tok[pos]) catch break;
            if (pos + seq_len > tok.len) break;
            const cp = std.unicode.utf8Decode(tok[pos..][0..seq_len]) catch break;
            pos += seq_len;

            if (cpIsByteEncoded(cp)) {
                try result.append(allocator, CP_TO_BYTE[cp]);
            } else {
                // Special token or literal unicode — emit raw UTF-8 bytes.
                var buf: [4]u8 = undefined;
                const written = std.unicode.utf8Encode(cp, &buf) catch continue;
                try result.appendSlice(allocator, buf[0..written]);
            }
        }
    }

    return result.toOwnedSlice(allocator);
}

// ── tests ─────────────────────────────────────────────────────────────────────

test "BYTE_TO_CP: space maps to Ġ (U+0120)" {
    try std.testing.expectEqual(@as(u21, 0x120), BYTE_TO_CP[' ']);
}

test "BYTE_TO_CP: printable ASCII maps to itself" {
    try std.testing.expectEqual(@as(u21, 'A'), BYTE_TO_CP['A']);
    try std.testing.expectEqual(@as(u21, 'z'), BYTE_TO_CP['z']);
    try std.testing.expectEqual(@as(u21, '!'), BYTE_TO_CP['!']);
}

test "BYTE_TO_CP: all 256 outputs are distinct" {
    var seen = std.AutoHashMap(u21, void).init(std.testing.allocator);
    defer seen.deinit();
    for (BYTE_TO_CP) |cp| {
        try std.testing.expect(!seen.contains(cp));
        try seen.put(cp, {});
    }
}

test "CP_TO_BYTE: round-trips all bytes" {
    for (BYTE_TO_CP, 0..) |cp, b| {
        try std.testing.expectEqual(@as(u8, @intCast(b)), CP_TO_BYTE[cp]);
    }
}
