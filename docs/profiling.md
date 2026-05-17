# Profiling

Optimization work should start from a profile, not from a guess. The current
Gemma4 helper profiles one model run at a time and defaults to the Phase 6 APEX
I Mini target.

## Quiet-System Preflight

Before recording numbers, check for noisy host processes:

```sh
scripts/check_benchmark_noise.sh
```

Do not record benchmark/profile results while unrelated high-CPU work is active.
In particular, kill or wait for stray `llama-cli`, `llmtoy generate`,
`profile_gemma4.sh`, or `perf` processes from earlier runs. Idle `llama-server`
processes are less damaging to CPU timings, but they can still matter if memory
pressure or swapping shows up.

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

## Vulkan Timestamp Profile

GPU dispatch timing is env-gated:

```sh
systemd-run --user --scope -p MemoryMax=40G --quiet -- \
  nix develop --command env LLMTOY_GPU_PROFILE=1 ./zig-out/bin/llmtoy generate "$MODEL" "$PROMPT" \
  --chat --gpu --temperature 0 --max-tokens 8
```

The profiler prints an aggregate table at shutdown with count, total ms, avg
us, min/max us, and percentage by dispatch label. Use this before shader work;
wall-clock tok/s alone is too coarse to distinguish matvec throughput from
attention, MoE, or submit overhead.

## Matvec Microbench

Use this before comparing shader variants:

```sh
nix develop --command ./zig-out/bin/llmtoy bench-matvec "$MODEL" --iters 64
```

To focus on one real tensor shape:

```sh
nix develop --command ./zig-out/bin/llmtoy bench-matvec "$MODEL" \
  --iters 256 --target L0.attn_q
```

The current harness measures one Q8_1-activation matvec dispatch at a time on
representative Gemma4 tensors. It quantizes the activation once before timing,
then includes the current descriptor, command-buffer, submit, wait, and cleanup
overhead in each measured iteration. Use it to choose between matvec kernels;
use full `generate --gpu` only after the microbench shows a real win.

Current experimental targets include bench-only MMVQ variants such as
`*.mmvq.b64.r1`, `*.mmvq.b64.r2`, and `*.mmvq.b64.r4`. These are comparison
targets, not production routing.

With `LLMTOY_GPU_PROFILE=1`, the table also prints per-dispatch GPU timestamp
time and the remaining CPU/submit overhead:

```sh
nix develop --command env LLMTOY_GPU_PROFILE=1 ./zig-out/bin/llmtoy bench-matvec "$MODEL" \
  --iters 128 --target L0.attn_q
```

Use `--reuse-descriptor` to measure the ceiling from reusing a descriptor set
for one stable matvec binding. This is a bench probe; production routing still
uses the normal path unless explicitly changed.

## llama.cpp Vulkan Reference

Use the Vulkan package, not the generic package:

```sh
nix shell nixpkgs#llama-cpp-vulkan -c llama-cli --list-devices
```

Expected on the target host:

```text
Vulkan0: AMD Radeon RX 7800 XT (RADV NAVI32)
```

Enable llama.cpp's Vulkan timestamp logger with:

```sh
GGML_VK_PERF_LOGGER=1 GGML_VK_PERF_LOGGER_FREQUENCY=1 \
VK_ICD_FILENAMES=/run/opengl-driver/share/vulkan/icd.d/radeon_icd.x86_64.json \
nix shell nixpkgs#llama-cpp-vulkan -c llama-cli ...
```

Cross-check production flags in `../nix-lab/hosts/lame/llama.nix` before
recording parity numbers.

## MoE GPU Baseline

Use the production expert path microbench when comparing against llama.cpp's
`MUL_MAT_ID_VEC` / `MUL_MAT_ID_MUL` Vulkan trace:

```sh
nix develop --command env LLMTOY_GPU_PROFILE=1 ./zig-out/bin/llmtoy \
  bench-moe /opt/ai-lab/models/mudler/gemma-4-26B-A4B-it-APEX-GGUF/gemma-4-26B-A4B-APEX-I-Mini.gguf \
  --iters 64 --layer 0
```

Current baseline on the RX 7800 XT:

- layer 0 Q3_K/Q5_1 top-8 MoE: 614.08 us wall, 84.97 us GPU phases
- layer 10 Q3_K/IQ4_NL top-8 MoE: 629.70 us wall, 96.81 us GPU phases

The GPU phase timing is useful for shader comparisons; the wall timing exposes
the current descriptor/command-recording overhead that llama.cpp avoids with its
expert-id pipeline shape.

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
parallel runs make profiles hard to interpret. Re-run the quiet-system preflight
before each number you intend to publish in the phase notes.

## Reading Results

Use `perf stat` when asking "did this reduce work?" Use `perf record` when
asking "where is the time going?"

Examples:

- high cache misses in quantized matvecs point at memory-layout or dequant work
- high time in `ops.thread_pool` points at too many small parallel jobs
- high time in `std.math.tanh` or activation code points at GELU/logit softcap
- high time in attention grows with context length and should be separated from
  short-prompt decode measurements
