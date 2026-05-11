/// Gemma4 architecture configuration.
///
/// Gemma4 26B alternates between local sliding-window attention (SWA) and
/// global full-context attention layers.  Each layer type has different head
/// dimensions and RoPE settings.  All layers have dual FFN (dense SwiGLU +
/// sparse MoE) sharing the same residual stream.
pub const MAX_LAYERS = 64;

pub const Gemma4Config = struct {
    vocab_size: usize,
    d_model: usize,
    n_layers: usize,
    n_heads: usize,
    d_ffn: usize,              // dense FFN hidden size (2112)
    d_expert: usize,           // MoE expert hidden size (704)
    n_experts: usize,          // total experts (128)
    n_experts_used: usize,     // top-k per token (8)
    head_dim_swa: usize,       // head dim for SWA layers (256)
    head_dim_global: usize,    // head dim for global layers (512)
    max_seq_len: usize,
    sliding_window: usize,     // local attention window size (1024)
    rope_theta_swa: f32,       // RoPE theta for SWA layers (10000)
    rope_theta_global: f32,    // RoPE theta for global layers (1e6, fallback)
    eps: f32,
    logit_softcap: f32,        // final logit soft-capping value (30.0)

    // Per-layer flags (indexed by layer index; valid up to n_layers).
    is_swa:    [MAX_LAYERS]bool,  // true = sliding-window attention
    n_kv_heads:[MAX_LAYERS]usize, // KV head count per layer

    pub fn headDim(self: Gemma4Config, l: usize) usize {
        return if (self.is_swa[l]) self.head_dim_swa else self.head_dim_global;
    }

    pub fn nq(self: Gemma4Config, l: usize) usize {
        return self.n_heads * self.headDim(l);
    }

    pub fn nkv(self: Gemma4Config, l: usize) usize {
        return self.n_kv_heads[l] * self.headDim(l);
    }
};
