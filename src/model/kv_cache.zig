const std = @import("std");
const Config = @import("config.zig").Config;

/// Per-layer key-value cache for incremental (one-token-at-a-time) decoding.
///
/// Layout: k[l][pos * n_kv_heads * head_dim + kv_h * head_dim ..]
///
/// Each call to forwardOne writes new K/V at position `pos` and reads all
/// cached positions 0..pos when computing attention. This turns the O(T²)
/// full-recompute approach from Phase 3 into O(T) per new token.
pub const KvCache = struct {
    k: [][]f32,
    v: [][]f32,
    n_layers:    usize,
    n_kv_heads:  usize,
    head_dim:    usize,
    max_seq_len: usize,
    allocator:   std.mem.Allocator,

    pub fn init(cfg: Config, allocator: std.mem.Allocator) !KvCache {
        const hd     = cfg.headDim();
        const stride = cfg.max_seq_len * cfg.n_kv_heads * hd;

        const k = try allocator.alloc([]f32, cfg.n_layers);
        errdefer allocator.free(k);
        for (k, 0..) |*s, i| {
            errdefer for (k[0..i]) |prev| allocator.free(prev);
            s.* = try allocator.alloc(f32, stride);
        }

        const v = try allocator.alloc([]f32, cfg.n_layers);
        errdefer {
            for (k) |s| allocator.free(s);
            allocator.free(k);
            allocator.free(v);
        }
        for (v, 0..) |*s, i| {
            errdefer for (v[0..i]) |prev| allocator.free(prev);
            s.* = try allocator.alloc(f32, stride);
        }

        return .{
            .k           = k,
            .v           = v,
            .n_layers    = cfg.n_layers,
            .n_kv_heads  = cfg.n_kv_heads,
            .head_dim    = hd,
            .max_seq_len = cfg.max_seq_len,
            .allocator   = allocator,
        };
    }

    pub fn deinit(self: *KvCache) void {
        for (self.k) |s| self.allocator.free(s);
        self.allocator.free(self.k);
        for (self.v) |s| self.allocator.free(s);
        self.allocator.free(self.v);
    }
};
