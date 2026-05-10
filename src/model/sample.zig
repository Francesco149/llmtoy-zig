/// Greedy decoding: return the index of the highest logit.
pub fn greedy(logits: []const f32) u32 {
    var best: u32 = 0;
    for (logits[1..], 1..) |v, i| {
        if (v > logits[best]) best = @intCast(i);
    }
    return best;
}
