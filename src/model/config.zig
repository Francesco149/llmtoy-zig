pub const Config = struct {
    vocab_size: usize,
    d_model: usize,
    n_layers: usize,
    n_heads: usize,      // Q attention heads
    n_kv_heads: usize,   // KV heads; n_kv_heads ≤ n_heads (GQA when <, MHA when equal)
    d_ffn: usize,
    max_seq_len: usize = 4096,
    rope_theta: f32 = 500_000.0, // Qwen3/Gemma4 use 1M; 500k is a safe default
    eps: f32 = 1e-5,

    pub fn headDim(self: Config) usize {
        return self.d_model / self.n_heads;
    }

    /// How many Q heads share one KV head.
    pub fn kvGroupSize(self: Config) usize {
        return self.n_heads / self.n_kv_heads;
    }
};
