# Profiling

Optimization work should start from a profile, not from a guess. The current
Gemma4 helper profiles one model run at a time and defaults to the Phase 6 APEX
I Mini target.

## Quick Counter Run

```sh
scripts/profile_gemma4.sh stat
```

This builds ReleaseFast and runs:

```text
perf stat -d -d -d -- ./zig-out/bin/llmtoy generate ... --threads 12
```

Use this for high-level counters: cycles, instructions, cache misses, branch
misses, and elapsed time. It is the first check after an optimization that
claims to reduce memory traffic or instruction count.

## Sampling Profile

```sh
TOKENS=16 scripts/profile_gemma4.sh record
```

This writes:

```text
profiles/gemma4-<timestamp>.data
profiles/gemma4-<timestamp>.report.txt
```

The text report is the first artifact to read. Look for symbols under
`quant.dequant`, `ops.math`, `model.gemma4.forward`, and `ops.thread_pool`.
Those names tell us whether time is going into quantized dot products, attention,
thread-pool coordination, activation functions, or allocation/setup.

`profiles/` is gitignored because `perf record` files are machine-local and can
be large.

## Smoke-Test Result

On the Ryzen 3600 host, a very short counter run completed successfully:

```sh
TOKENS=2 scripts/profile_gemma4.sh stat
```

The useful first-pass counters were:

```text
task-clock:              124401.70 ms, 6.2 CPUs utilized
instructions:            1.10e12
cycles:                  4.78e11
instructions per cycle:  2.3
L1d load miss rate:      3.2%
branch miss rate:        1.8%
```

This confirms the wrapper works. Use longer `record` runs before drawing
optimization conclusions; two generated tokens are only a smoke test.

## Environment Overrides

```sh
MODEL=/path/to/model.gguf \
PROMPT="What is the capital of France?" \
TOKENS=8 \
THREADS=12 \
scripts/profile_gemma4.sh record
```

Only run one large model/profile at a time. CPU Gemma4 runs are memory-heavy, and
parallel runs make profiles hard to interpret.

## Reading Results

Use `perf stat` when asking "did this reduce work?" Use `perf record` when
asking "where is the time going?"

Examples:

- high cache misses in quantized matvecs point at memory-layout or dequant work
- high time in `ops.thread_pool` points at too many small parallel jobs
- high time in `std.math.tanh` or activation code points at GELU/logit softcap
- high time in attention grows with context length and should be separated from
  short-prompt decode measurements
