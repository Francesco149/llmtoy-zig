const std = @import("std");
const Vocab = @import("vocab.zig").Vocab;

pub const Role = enum {
    system,
    user,
    assistant,

    pub fn name(self: Role, vocab_model: []const u8) []const u8 {
        _ = vocab_model;
        return switch (self) {
            .system => "system",
            .user => "user",
            .assistant => "model",
        };
    }
};

pub const Message = struct {
    role: Role,
    content: []const u8,
};

pub const TemplateError = error{
    UnsupportedChatTemplate,
};

/// Apply the minimal chat template needed by current smoke tests.
///
/// This is intentionally not a Jinja interpreter. It gives the CLI a structured
/// place to put model-family templates, and can later grow into metadata-driven
/// template support.
pub fn apply(
    allocator: std.mem.Allocator,
    vocab: *const Vocab,
    messages: []const Message,
    add_generation_prompt: bool,
) ![]u8 {
    if (std.mem.eql(u8, vocab.model, "gemma4")) {
        return applyGemma4(allocator, messages, add_generation_prompt);
    }
    return TemplateError.UnsupportedChatTemplate;
}

fn applyGemma4(
    allocator: std.mem.Allocator,
    messages: []const Message,
    add_generation_prompt: bool,
) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);

    for (messages) |msg| {
        try appendFmt(&out, allocator, "<|turn>{s}\n{s}<turn|>\n", .{
            msg.role.name("gemma4"),
            std.mem.trim(u8, msg.content, " \t\r\n"),
        });
    }

    if (add_generation_prompt) {
        try out.appendSlice(allocator, "<|turn>model\n<|channel>thought\n<channel|>");
    }

    return out.toOwnedSlice(allocator);
}

fn applyChatML(
    allocator: std.mem.Allocator,
    messages: []const Message,
    add_generation_prompt: bool,
) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);

    for (messages) |msg| {
        const role = switch (msg.role) {
            .system => "system",
            .user => "user",
            .assistant => "assistant",
        };
        try appendFmt(&out, allocator, "<|im_start|>{s}\n{s}<|im_end|>\n", .{
            role,
            std.mem.trim(u8, msg.content, " \t\r\n"),
        });
    }

    if (add_generation_prompt) {
        try out.appendSlice(allocator, "<|im_start|>assistant\n");
    }

    return out.toOwnedSlice(allocator);
}

fn appendFmt(out: *std.ArrayList(u8), allocator: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !void {
    const rendered = try std.fmt.allocPrint(allocator, fmt, args);
    defer allocator.free(rendered);
    try out.appendSlice(allocator, rendered);
}

test "Gemma4 chat template renders turn markers" {
    const token_lits = [_][]const u8{ "<bos>" };
    const tokens = try std.testing.allocator.alloc([]const u8, token_lits.len);
    @memcpy(tokens, &token_lits);

    var vocab = Vocab{
        .model = "gemma4",
        .tokens = tokens,
        .token_to_id = std.StringHashMap(u32).init(std.testing.allocator),
        .merge_rank = std.StringHashMap(u32).init(std.testing.allocator),
        .bos_token_id = 2,
        .eos_token_id = 1,
        .add_bos = true,
        .allocator = std.testing.allocator,
    };
    defer vocab.deinit();

    const messages = [_]Message{.{ .role = .user, .content = "  What is this?\n" }};
    const rendered = try apply(std.testing.allocator, &vocab, &messages, true);
    defer std.testing.allocator.free(rendered);
    try std.testing.expectEqualStrings("<|turn>user\nWhat is this?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>", rendered);
}

test "ChatML template renders assistant generation prompt" {
    const token_lits = [_][]const u8{ "<bos>" };
    const tokens = try std.testing.allocator.alloc([]const u8, token_lits.len);
    @memcpy(tokens, &token_lits);

    var vocab = Vocab{
        .model = "gpt2",
        .tokens = tokens,
        .token_to_id = std.StringHashMap(u32).init(std.testing.allocator),
        .merge_rank = std.StringHashMap(u32).init(std.testing.allocator),
        .bos_token_id = 1,
        .eos_token_id = 2,
        .add_bos = false,
        .allocator = std.testing.allocator,
    };
    defer vocab.deinit();

    const messages = [_]Message{.{ .role = .user, .content = "hello" }};
    const rendered = try applyChatML(std.testing.allocator, &messages, true);
    defer std.testing.allocator.free(rendered);
    try std.testing.expectEqualStrings("<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n", rendered);
}
