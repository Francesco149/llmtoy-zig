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
    if (std.mem.eql(u8, vocab.model, "gemma4")) {
        return encodeGemma4(text, vocab, allocator);
    }

    var ids: std.ArrayList(u32) = .empty;
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

/// Gemma4 uses SentencePiece-style BPE: spaces are normalized to U+2581 and
/// merges run over raw UTF-8, not GPT-2 byte-encoded text.
fn encodeGemma4(text: []const u8, vocab: *const Vocab, allocator: std.mem.Allocator) ![]u32 {
    var ids: std.ArrayList(u32) = .empty;
    errdefer ids.deinit(allocator);

    if (vocab.add_bos) {
        try ids.append(allocator, vocab.bos_token_id);
    }

    var i: usize = 0;
    while (i < text.len) {
        if (try appendSpecialIfPresent(text, &i, vocab, &ids, allocator)) continue;

        const start = i;
        if (text[i] == '\n') {
            while (i < text.len and text[i] == '\n') : (i += 1) {}
        } else {
            while (i < text.len and text[i] != '\n') {
                if (specialTokenLenAt(text[i..], vocab) != null) break;
                i += std.unicode.utf8ByteSequenceLength(text[i]) catch 1;
            }
        }
        if (i > start) {
            try encodeGemma4Piece(text[start..i], vocab, &ids, allocator);
        }
    }

    return ids.toOwnedSlice(allocator);
}

fn encodeGemma4Piece(
    piece: []const u8,
    vocab: *const Vocab,
    ids: *std.ArrayList(u32),
    allocator: std.mem.Allocator,
) !void {
    if (vocab.token_to_id.get(piece)) |id| {
        try ids.append(allocator, id);
        return;
    }

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    var normalized: std.ArrayList(u8) = .empty;
    for (piece) |byte| {
        if (byte == ' ') {
            try normalized.appendSlice(aa, "▁");
        } else {
            try normalized.append(aa, byte);
        }
    }

    if (vocab.token_to_id.get(normalized.items)) |id| {
        try ids.append(allocator, id);
        return;
    }

    var symbols: std.ArrayList([]const u8) = .empty;
    var pos: usize = 0;
    while (pos < normalized.items.len) {
        const len = std.unicode.utf8ByteSequenceLength(normalized.items[pos]) catch 1;
        try symbols.append(aa, normalized.items[pos..][0..@min(len, normalized.items.len - pos)]);
        pos += @min(len, normalized.items.len - pos);
    }

    try runBpe(&symbols, vocab, aa);

    for (symbols.items) |sym| {
        if (vocab.token_to_id.get(sym)) |id| {
            try ids.append(allocator, id);
        }
    }
}

fn appendSpecialIfPresent(
    text: []const u8,
    index: *usize,
    vocab: *const Vocab,
    ids: *std.ArrayList(u32),
    allocator: std.mem.Allocator,
) !bool {
    if (specialTokenLenAt(text[index.*..], vocab)) |match| {
        try ids.append(allocator, match.id);
        index.* += match.len;
        return true;
    }
    return false;
}

const SpecialMatch = struct { len: usize, id: u32 };

fn specialTokenLenAt(text: []const u8, vocab: *const Vocab) ?SpecialMatch {
    if (text.len == 0 or text[0] != '<') return null;
    var best: ?SpecialMatch = null;
    for (vocab.tokens, 0..) |tok, id| {
        if (tok.len <= 2 or tok[0] != '<' or tok[tok.len - 1] != '>') continue;
        if (!std.mem.startsWith(u8, text, tok)) continue;
        if (best == null or tok.len > best.?.len) {
            best = .{ .len = tok.len, .id = @intCast(id) };
        }
    }
    return best;
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
    var symbols: std.ArrayList([]const u8) = .empty;

    if (with_space) {
        try symbols.append(aa, try cpToUtf8(aa, BYTE_TO_CP[' ']));
    }
    for (word) |byte| {
        try symbols.append(aa, try cpToUtf8(aa, BYTE_TO_CP[byte]));
    }

    // BPE merge loop: repeatedly find the lowest-ranked adjacent pair and merge.
    while (symbols.items.len > 1) {
        if (!try mergeBestPair(&symbols, vocab, aa)) break;
    }

    // Look up token IDs.
    for (symbols.items) |sym| {
        if (vocab.token_to_id.get(sym)) |id| {
            try ids.append(allocator, id);
        }
        // Unknown symbols are silently skipped — rare in practice with byte-level BPE.
    }
}

fn runBpe(symbols: *std.ArrayList([]const u8), vocab: *const Vocab, allocator: std.mem.Allocator) !void {
    while (symbols.items.len > 1) {
        if (!try mergeBestPair(symbols, vocab, allocator)) break;
    }
}

fn mergeBestPair(symbols: *std.ArrayList([]const u8), vocab: *const Vocab, allocator: std.mem.Allocator) !bool {
    var best_rank: u32 = std.math.maxInt(u32);
    var best_pos: usize = 0;
    var found = false;

    for (0..symbols.items.len - 1) |pos| {
        const pair = try std.fmt.allocPrint(allocator, "{s} {s}", .{ symbols.items[pos], symbols.items[pos + 1] });
        if (vocab.merge_rank.get(pair)) |rank| {
            if (rank < best_rank) {
                best_rank = rank;
                best_pos = pos;
                found = true;
            }
        }
    }

    if (!found) return false;

    const merged = try std.fmt.allocPrint(allocator, "{s}{s}", .{
        symbols.items[best_pos],
        symbols.items[best_pos + 1],
    });
    symbols.items[best_pos] = merged;
    _ = symbols.orderedRemove(best_pos + 1);
    return true;
}

fn cpToUtf8(allocator: std.mem.Allocator, cp: u21) ![]u8 {
    var buf: [4]u8 = undefined;
    const len = try std.unicode.utf8Encode(cp, &buf);
    return allocator.dupe(u8, buf[0..len]);
}

/// Decode token IDs back to bytes.
/// Caller owns the returned slice.
/// Decode a single token to raw UTF-8 bytes into `buf`.
/// Returns the number of bytes written (0 if token is out of range or empty).
/// `buf` should be at least 32 bytes; a token is rarely longer than that.
pub fn decodeOne(id: u32, vocab: *const Vocab, buf: []u8) usize {
    if (id >= vocab.tokens.len) return 0;
    const tok = vocab.tokens[id];
    if (std.mem.eql(u8, vocab.model, "gemma4")) {
        return decodeGemma4Token(tok, buf);
    }
    var out: usize = 0;
    var pos: usize = 0;
    while (pos < tok.len) {
        const seq_len = std.unicode.utf8ByteSequenceLength(tok[pos]) catch break;
        if (pos + seq_len > tok.len) break;
        const cp = std.unicode.utf8Decode(tok[pos..][0..seq_len]) catch break;
        pos += seq_len;
        if (cpIsByteEncoded(cp)) {
            if (out < buf.len) { buf[out] = CP_TO_BYTE[cp]; out += 1; }
        } else {
            var tmp: [4]u8 = undefined;
            const n = std.unicode.utf8Encode(cp, &tmp) catch continue;
            for (tmp[0..n]) |b| { if (out < buf.len) { buf[out] = b; out += 1; } }
        }
    }
    return out;
}

fn decodeGemma4Token(tok: []const u8, buf: []u8) usize {
    var out: usize = 0;
    var pos: usize = 0;
    while (pos < tok.len) {
        const seq_len = std.unicode.utf8ByteSequenceLength(tok[pos]) catch break;
        if (pos + seq_len > tok.len) break;
        const cp = std.unicode.utf8Decode(tok[pos..][0..seq_len]) catch break;
        pos += seq_len;
        if (cp == 0x2581) {
            if (out < buf.len) { buf[out] = ' '; out += 1; }
        } else {
            for (tok[pos - seq_len .. pos]) |b| {
                if (out < buf.len) { buf[out] = b; out += 1; }
            }
        }
    }
    return out;
}

pub fn decode(token_ids: []const u32, vocab: *const Vocab, allocator: std.mem.Allocator) ![]u8 {
    var result: std.ArrayList(u8) = .empty;
    errdefer result.deinit(allocator);

    for (token_ids) |id| {
        if (id >= vocab.tokens.len) continue;
        if (std.mem.eql(u8, vocab.model, "gemma4")) {
            var tok_buf: [256]u8 = undefined;
            const n = decodeGemma4Token(vocab.tokens[id], &tok_buf);
            try result.appendSlice(allocator, tok_buf[0..n]);
            continue;
        }
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

test "Gemma4 tokenizer normalizes spaces before BPE" {
    const allocator = std.testing.allocator;
    const token_lits = [_][]const u8{ "<eos>", "<bos>", "The", "▁capital", "▁of", "▁France", "▁is" };
    const tokens = try allocator.alloc([]const u8, token_lits.len);
    @memcpy(tokens, &token_lits);

    var token_to_id = std.StringHashMap(u32).init(allocator);
    try token_to_id.put("<bos>", 1);
    try token_to_id.put("The", 2);
    try token_to_id.put("▁capital", 3);
    try token_to_id.put("▁of", 4);
    try token_to_id.put("▁France", 5);
    try token_to_id.put("▁is", 6);

    const merge_rank = std.StringHashMap(u32).init(allocator);
    var vocab = Vocab{
        .model = "gemma4",
        .tokens = tokens,
        .token_to_id = token_to_id,
        .merge_rank = merge_rank,
        .bos_token_id = 1,
        .eos_token_id = 0,
        .add_bos = true,
        .allocator = allocator,
    };
    defer vocab.deinit();

    const ids = try encode(" capital", &vocab, allocator);
    defer allocator.free(ids);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 1, 3 }, ids);

    const decoded = try decode(ids, &vocab, allocator);
    defer allocator.free(decoded);
    try std.testing.expectEqualStrings("<bos> capital", decoded);
}

test "Gemma4 tokenizer keeps angle-bracket control tokens intact" {
    const allocator = std.testing.allocator;
    const token_lits = [_][]const u8{ "<eos>", "<bos>", "<|turn>", "<turn|>", "user", "\n" };
    const tokens = try allocator.alloc([]const u8, token_lits.len);
    @memcpy(tokens, &token_lits);

    var token_to_id = std.StringHashMap(u32).init(allocator);
    try token_to_id.put("<bos>", 1);
    try token_to_id.put("<|turn>", 2);
    try token_to_id.put("<turn|>", 3);
    try token_to_id.put("user", 4);
    try token_to_id.put("\n", 5);

    const merge_rank = std.StringHashMap(u32).init(allocator);
    var vocab = Vocab{
        .model = "gemma4",
        .tokens = tokens,
        .token_to_id = token_to_id,
        .merge_rank = merge_rank,
        .bos_token_id = 1,
        .eos_token_id = 0,
        .add_bos = true,
        .allocator = allocator,
    };
    defer vocab.deinit();

    const ids = try encode("<|turn>user\n<turn|>", &vocab, allocator);
    defer allocator.free(ids);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 1, 2, 4, 5, 3 }, ids);
}
