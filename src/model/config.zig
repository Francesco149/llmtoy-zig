pub const Config = struct {
    vocab_size: usize,
    d_model: usize,
    n_layers: usize,
    n_heads: usize,
    d_ffn: usize,
    eps: f32 = 1e-5,

    pub fn headDim(self: Config) usize {
        return self.d_model / self.n_heads;
    }
};
