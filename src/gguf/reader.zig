const std = @import("std");
const t = @import("types.zig");

/// Parsed GGUF header, metadata and tensor descriptors.
/// Backed by an internal arena; call deinit when done.
/// Does not hold the raw file bytes.
pub const GgufData = struct {
    arena: std.heap.ArenaAllocator,
    version: u32,
    metadata: std.StringHashMap(t.MetaValue),
    tensors: []t.TensorInfo,
    /// Byte offset in the source data where tensor weights begin.
    data_offset: usize,

    pub fn deinit(self: *GgufData) void {
        self.arena.deinit();
    }

    pub fn metaString(self: *const GgufData, key: []const u8) ?[]const u8 {
        const v = self.metadata.get(key) orelse return null;
        return if (v == .string) v.string else null;
    }

    pub fn metaU32(self: *const GgufData, key: []const u8) ?u32 {
        const v = self.metadata.get(key) orelse return null;
        return if (v == .uint32) v.uint32 else null;
    }

    pub fn metaArray(self: *const GgufData, key: []const u8) ?t.Array {
        const v = self.metadata.get(key) orelse return null;
        return if (v == .array) v.array else null;
    }

    pub fn metaBool(self: *const GgufData, key: []const u8) ?bool {
        const v = self.metadata.get(key) orelse return null;
        return if (v == .bool_) v.bool_ else null;
    }
};

/// Parse GGUF data from raw bytes.
/// The returned GgufData is fully self-contained (strings are duped into the arena).
pub fn parseBytes(data: []const u8, backing: std.mem.Allocator) !GgufData {
    var arena = std.heap.ArenaAllocator.init(backing);
    errdefer arena.deinit();
    const aa = arena.allocator();

    var pos: usize = 0;

    const magic = readU32(data, &pos);
    if (magic != t.MAGIC) return error.InvalidMagic;

    const version = readU32(data, &pos);
    if (version < 1 or version > 3) return error.UnsupportedVersion;

    // v1 used u32 for counts; v2+ use u64.
    const n_tensors: u64 = if (version == 1) readU32(data, &pos) else readU64(data, &pos);
    const n_kv: u64 = if (version == 1) readU32(data, &pos) else readU64(data, &pos);

    var metadata = std.StringHashMap(t.MetaValue).init(aa);
    for (0..n_kv) |_| {
        const key = try readString(data, &pos, aa);
        const vtype: t.MetaValueType = @enumFromInt(readU32(data, &pos));
        const value = try readMetaValue(data, &pos, vtype, aa);
        try metadata.put(key, value);
    }

    const tensors = try aa.alloc(t.TensorInfo, n_tensors);
    for (tensors) |*info| {
        info.name = try readString(data, &pos, aa);
        const n_dims = readU32(data, &pos);
        info.dims = try aa.alloc(u64, n_dims);
        for (info.dims) |*d| d.* = readU64(data, &pos);
        info.type_ = @enumFromInt(readU32(data, &pos));
        info.offset = readU64(data, &pos);
    }

    const data_offset = std.mem.alignForward(usize, pos, t.ALIGNMENT);

    return .{
        .arena = arena,
        .version = version,
        .metadata = metadata,
        .tensors = tensors,
        .data_offset = data_offset,
    };
}

// ── memory-mapped reader ───────────────────────────────────────────────────────

/// Opens a GGUF file, memory-maps it, and parses the header.
/// Call deinit to release the mmap and parsed data.
pub const GgufReader = struct {
    data: GgufData,
    mmap: []align(std.heap.page_size_min) const u8,
    file: std.Io.File,
    io: std.Io,

    pub fn open(path: []const u8, io: std.Io, backing: std.mem.Allocator) !GgufReader {
        const file = if (std.fs.path.isAbsolute(path))
            try std.Io.Dir.openFileAbsolute(io, path, .{})
        else
            try std.Io.Dir.cwd().openFile(io, path, .{});
        errdefer std.Io.File.close(file, io);

        const size = (try std.Io.File.stat(file, io)).size;
        if (size == 0) return error.EmptyFile;

        const mmap = try std.posix.mmap(
            null,
            size,
            .{ .READ = true },
            .{ .TYPE = .PRIVATE },
            file.handle,
            0,
        );
        errdefer std.posix.munmap(mmap);

        const data = try parseBytes(mmap, backing);
        return .{ .data = data, .mmap = mmap, .file = file, .io = io };
    }

    pub fn deinit(self: *GgufReader) void {
        self.data.deinit();
        std.posix.munmap(self.mmap);
        std.Io.File.close(self.file, self.io);
    }

    /// Raw bytes for a tensor's weight data.
    pub fn tensorBytes(self: *const GgufReader, info: t.TensorInfo) []const u8 {
        return self.mmap[self.data.data_offset + info.offset ..];
    }

    pub fn metaString(self: *const GgufReader, key: []const u8) ?[]const u8 {
        return self.data.metaString(key);
    }

    pub fn metaU32(self: *const GgufReader, key: []const u8) ?u32 {
        return self.data.metaU32(key);
    }

    pub fn metaArray(self: *const GgufReader, key: []const u8) ?t.Array {
        return self.data.metaArray(key);
    }

    pub fn metaBool(self: *const GgufReader, key: []const u8) ?bool {
        return self.data.metaBool(key);
    }
};

// ── low-level byte readers ─────────────────────────────────────────────────────

fn readU8(data: []const u8, pos: *usize) u8 {
    defer pos.* += 1;
    return data[pos.*];
}

fn readU16(data: []const u8, pos: *usize) u16 {
    defer pos.* += 2;
    return std.mem.readInt(u16, data[pos.*..][0..2], .little);
}

fn readU32(data: []const u8, pos: *usize) u32 {
    defer pos.* += 4;
    return std.mem.readInt(u32, data[pos.*..][0..4], .little);
}

fn readU64(data: []const u8, pos: *usize) u64 {
    defer pos.* += 8;
    return std.mem.readInt(u64, data[pos.*..][0..8], .little);
}

fn readString(data: []const u8, pos: *usize, allocator: std.mem.Allocator) ![]const u8 {
    const len = readU64(data, pos);
    const s = try allocator.dupe(u8, data[pos.*..][0..len]);
    pos.* += len;
    return s;
}

fn readMetaValue(data: []const u8, pos: *usize, vtype: t.MetaValueType, allocator: std.mem.Allocator) !t.MetaValue {
    return switch (vtype) {
        .uint8 => .{ .uint8 = readU8(data, pos) },
        .int8 => .{ .int8 = @bitCast(readU8(data, pos)) },
        .uint16 => .{ .uint16 = readU16(data, pos) },
        .int16 => .{ .int16 = @bitCast(readU16(data, pos)) },
        .uint32 => .{ .uint32 = readU32(data, pos) },
        .int32 => .{ .int32 = @bitCast(readU32(data, pos)) },
        .float32 => .{ .float32 = @bitCast(readU32(data, pos)) },
        .bool_ => .{ .bool_ = readU8(data, pos) != 0 },
        .string => .{ .string = try readString(data, pos, allocator) },
        .array => {
            const elem_type: t.MetaValueType = @enumFromInt(readU32(data, pos));
            const count = readU64(data, pos);
            const values = try allocator.alloc(t.MetaValue, count);
            for (values) |*v| v.* = try readMetaValue(data, pos, elem_type, allocator);
            return .{ .array = .{ .elem_type = elem_type, .values = values } };
        },
        .uint64 => .{ .uint64 = readU64(data, pos) },
        .int64 => .{ .int64 = @bitCast(readU64(data, pos)) },
        .float64 => .{ .float64 = @bitCast(readU64(data, pos)) },
    };
}

// ── helpers for building test data ────────────────────────────────────────────

fn putU32(buf: []u8, pos: *usize, v: u32) void {
    std.mem.writeInt(u32, buf[pos.*..][0..4], v, .little);
    pos.* += 4;
}

fn putU64(buf: []u8, pos: *usize, v: u64) void {
    std.mem.writeInt(u64, buf[pos.*..][0..8], v, .little);
    pos.* += 8;
}

fn putStr(buf: []u8, pos: *usize, s: []const u8) void {
    putU64(buf, pos, s.len);
    @memcpy(buf[pos.*..][0..s.len], s);
    pos.* += s.len;
}

// ── tests ─────────────────────────────────────────────────────────────────────

test "parseBytes: synthetic GGUF v3" {
    var buf: [512]u8 = undefined;
    var pos: usize = 0;

    // header
    putU32(&buf, &pos, t.MAGIC);
    putU32(&buf, &pos, 3); // version
    putU64(&buf, &pos, 1); // n_tensors
    putU64(&buf, &pos, 2); // n_kv

    // kv[0]: general.architecture = "test_arch"
    putStr(&buf, &pos, "general.architecture");
    putU32(&buf, &pos, @intFromEnum(t.MetaValueType.string));
    putStr(&buf, &pos, "test_arch");

    // kv[1]: test.context_length = 256 (uint32)
    putStr(&buf, &pos, "test.context_length");
    putU32(&buf, &pos, @intFromEnum(t.MetaValueType.uint32));
    putU32(&buf, &pos, 256);

    // tensor info: "weight", 2 dims [4, 4], f32, offset=0
    putStr(&buf, &pos, "weight");
    putU32(&buf, &pos, 2); // n_dims
    putU64(&buf, &pos, 4);
    putU64(&buf, &pos, 4);
    putU32(&buf, &pos, @intFromEnum(t.GgmlType.f32));
    putU64(&buf, &pos, 0); // offset

    // pad to alignment + 64 bytes of tensor data
    const header_end = pos;
    const aligned = std.mem.alignForward(usize, header_end, t.ALIGNMENT);
    @memset(buf[header_end..aligned], 0);
    pos = aligned;
    @memset(buf[pos .. pos + 64], 0);
    pos += 64;

    var gguf = try parseBytes(buf[0..pos], std.testing.allocator);
    defer gguf.deinit();

    try std.testing.expectEqual(@as(u32, 3), gguf.version);
    try std.testing.expectEqual(@as(usize, 2), gguf.metadata.count());
    try std.testing.expectEqual(@as(usize, 1), gguf.tensors.len);

    try std.testing.expectEqualStrings("test_arch", gguf.metaString("general.architecture").?);
    try std.testing.expectEqual(@as(u32, 256), gguf.metaU32("test.context_length").?);

    const tensor = gguf.tensors[0];
    try std.testing.expectEqualStrings("weight", tensor.name);
    try std.testing.expectEqual(@as(u64, 16), tensor.n_elements());
    try std.testing.expectEqual(t.GgmlType.f32, tensor.type_);
}
