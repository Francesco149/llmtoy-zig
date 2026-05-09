const std = @import("std");

pub fn main() !void {
    const stdout = std.io.getStdOut().writer();
    try stdout.print("llmtoy-zig: educational LLM inference\n", .{});
}

test "sanity" {
    try std.testing.expect(true);
}
