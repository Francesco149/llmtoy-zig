/// Per-layer KV cache for Gemma4.
///
/// SWA and global layers have different head dimensions and KV head counts,
/// so each layer gets its own allocation.  SWA layers store up to
/// min(max_seq_len, sliding_window) positions; global layers store everything.
///
/// Layout for layer l: k[l][pos * nkv_l + kv_h * head_dim_l + d]
/// where nkv_l = n_kv_heads[l] * head_dim[l].

const std     = @import("std");
const Gemma4Config = @import("config.zig").Gemma4Config;

pub const Gemma4KvCache = struct {
    k: [][]f32,         // [n_layers][] — each layer has its own stride
    v: [][]f32,
    /// Effective capacity per layer (positions stored).
    cap: []usize,
    max_seq_len: usize,
    allocator: std.mem.Allocator,

    pub fn init(cfg: Gemma4Config, max_seq: usize, allocator: std.mem.Allocator) !Gemma4KvCache {
        const k = try allocator.alloc([]f32, cfg.n_layers);
        errdefer allocator.free(k);
        var ki: usize = 0;
        errdefer for (k[0..ki]) |s| allocator.free(s);

        const v = try allocator.alloc([]f32, cfg.n_layers);
        errdefer allocator.free(v);
        var vi: usize = 0;
        errdefer for (v[0..vi]) |s| allocator.free(s);

        const cap = try allocator.alloc(usize, cfg.n_layers);
        errdefer allocator.free(cap);

        for (0..cfg.n_layers) |l| {
            const seq_cap = if (cfg.is_swa[l]) @min(max_seq, cfg.sliding_window) else max_seq;
            const stride  = seq_cap * cfg.nkv(l);
            cap[l] = seq_cap;
            k[l] = try allocator.alloc(f32, stride);
            ki = l + 1;
            v[l] = try allocator.alloc(f32, stride);
            vi = l + 1;
        }

        return .{
            .k           = k,
            .v           = v,
            .cap         = cap,
            .max_seq_len = max_seq,
            .allocator   = allocator,
        };
    }

    pub fn deinit(self: *Gemma4KvCache) void {
        for (self.k) |s| self.allocator.free(s);
        self.allocator.free(self.k);
        for (self.v) |s| self.allocator.free(s);
        self.allocator.free(self.v);
        self.allocator.free(self.cap);
    }
};
