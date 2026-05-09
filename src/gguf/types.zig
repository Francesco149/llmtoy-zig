const std = @import("std");

pub const MAGIC: u32 = 0x46554747; // "GGUF" little-endian
pub const ALIGNMENT: usize = 32;

pub const MetaValueType = enum(u32) {
    uint8 = 0,
    int8 = 1,
    uint16 = 2,
    int16 = 3,
    uint32 = 4,
    int32 = 5,
    float32 = 6,
    bool_ = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    int64 = 11,
    float64 = 12,
};

// Non-exhaustive: llama.cpp keeps adding types.
pub const GgmlType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
    iq2_xxs = 16,
    iq2_xs = 17,
    iq3_xxs = 18,
    iq1_s = 19,
    iq4_nl = 20,
    iq3_s = 21,
    iq2_s = 22,
    iq4_xs = 23,
    i8 = 24,
    i16 = 25,
    i32 = 26,
    i64 = 27,
    f64 = 28,
    iq1_m = 29,
    bf16 = 30,
    _,

    pub fn label(self: GgmlType) []const u8 {
        return switch (self) {
            .f32 => "F32",
            .f16 => "F16",
            .q4_0 => "Q4_0",
            .q4_1 => "Q4_1",
            .q5_0 => "Q5_0",
            .q5_1 => "Q5_1",
            .q8_0 => "Q8_0",
            .q8_1 => "Q8_1",
            .q2_k => "Q2_K",
            .q3_k => "Q3_K",
            .q4_k => "Q4_K",
            .q5_k => "Q5_K",
            .q6_k => "Q6_K",
            .q8_k => "Q8_K",
            .bf16 => "BF16",
            .iq1_s => "IQ1_S",
            .iq1_m => "IQ1_M",
            .iq2_xxs => "IQ2_XXS",
            .iq2_xs => "IQ2_XS",
            .iq2_s => "IQ2_S",
            .iq3_xxs => "IQ3_XXS",
            .iq3_s => "IQ3_S",
            .iq4_nl => "IQ4_NL",
            .iq4_xs => "IQ4_XS",
            else => "unknown",
        };
    }
};

pub const Array = struct {
    elem_type: MetaValueType,
    values: []MetaValue,
};

pub const MetaValue = union(MetaValueType) {
    uint8: u8,
    int8: i8,
    uint16: u16,
    int16: i16,
    uint32: u32,
    int32: i32,
    float32: f32,
    bool_: bool,
    string: []const u8,
    array: Array,
    uint64: u64,
    int64: i64,
    float64: f64,

    pub fn format(self: MetaValue, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        switch (self) {
            .uint8 => |v| try writer.print("{}", .{v}),
            .int8 => |v| try writer.print("{}", .{v}),
            .uint16 => |v| try writer.print("{}", .{v}),
            .int16 => |v| try writer.print("{}", .{v}),
            .uint32 => |v| try writer.print("{}", .{v}),
            .int32 => |v| try writer.print("{}", .{v}),
            .float32 => |v| try writer.print("{d:.4}", .{v}),
            .bool_ => |v| try writer.print("{}", .{v}),
            .string => |v| try writer.print("{s}", .{v}),
            .array => |v| try writer.print("[{} x {s}]", .{ v.values.len, @tagName(v.elem_type) }),
            .uint64 => |v| try writer.print("{}", .{v}),
            .int64 => |v| try writer.print("{}", .{v}),
            .float64 => |v| try writer.print("{d:.4}", .{v}),
        }
    }
};

pub const TensorInfo = struct {
    name: []const u8,
    dims: []u64,
    type_: GgmlType,
    offset: u64,

    pub fn n_elements(self: TensorInfo) u64 {
        var n: u64 = 1;
        for (self.dims) |d| n *= d;
        return n;
    }
};
