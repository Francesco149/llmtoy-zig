# Phase 6 Addendum: Gemma4 CPU Optimizations

This addendum tracks optimization work for the current concrete target:

```text
CPU:   Ryzen 5 3600, 6 cores / 12 threads, Zen 2, AVX2 + F16C
RAM:   64 GB
Model: gemma-4-26B-A4B-APEX-I-Mini.gguf
Run:   32 generated tokens, 12 threads, temperature 0.1, top-k 40, top-p 0.9
```

The rule for this phase is simple: make one meaningful change at a time, run the
Gemma4 llama.cpp comparison, and write down what changed. A faster number is not
enough; the goal is to understand why the change helped or why it did not.

## Baseline

Command:

```sh
nix develop --command python3 scripts/regression_compare.py \
  --model /opt/ai-lab/models/mudler/gemma-4-26B-A4B-it-APEX-GGUF/gemma-4-26B-A4B-APEX-I-Mini.gguf \
  --prompt "Briefly explain the full forward pass of a MoE model" \
  --chat \
  --max-tokens 32 \
  --threads 12 \
  --temperature 0.1 \
  --top-k 40 \
  --top-p 0.9 \
  --expect-substring MoE \
  --timeout-s 240
```

Initial measured result after the NeoX RoPE correctness fix:

```text
llmtoy:   33.573 s
llama.cpp 24.265 s
```

The current CLI reports setup, prefill, and generation separately. Older notes
below list total harness wall time because they were captured before that split.
Use the split timings for future comparisons, especially `generation tok/s`.

Profiling workflow lives in [profiling.md](/opt/ai-lab/llmtoy-zig/docs/profiling.md).
Run `scripts/profile_gemma4.sh stat` for counters and
`scripts/profile_gemma4.sh record` for sampled hot symbols.

## Optimization 1: Hoist Per-Layer Temporary Buffers

Gemma4's forward pass used to allocate dense FFN buffers, MoE input buffers,
router buffers, selected-expert buffers, and expert scratch inside the layer loop.
That meant every generated token performed allocator work for every layer.

The first cleanup moved those buffers to the top of `forwardOne` and reused them
for all layers:

```zig
const gate_buf  = try allocator.alloc(f32, cfg.d_ffn);
const up_buf    = try allocator.alloc(f32, cfg.d_ffn);
const eg        = try allocator.alloc(f32, cfg.d_expert);
const eu        = try allocator.alloc(f32, cfg.d_expert);
const ed        = try allocator.alloc(f32, d);
```

Result:

```text
llmtoy:   33.921 s
llama.cpp 24.343 s
```

This did not improve wall time. That is a useful negative result: allocator churn
was untidy, but the hot path is dominated by quantized matrix-vector products,
especially the dense FFN and the top-8 expert gate/up/down matvecs.

## Optimization 2: Fused IQ4_NL Dot Product

APEX I Mini uses `IQ4_NL` for many large matrices. The generic path was:

```text
compressed row bytes -> dequantize full row to f32 scratch -> f32 dot(vec)
```

For a row with 2816 columns, that writes 2816 floats to scratch just so the next
loop can immediately read them back. The fused path is:

```text
compressed row bytes -> table lookup + scale + multiply with vec -> sum
```

`IQ4_NL` stores 32 weights per block:

```text
[ f16 scale ][ 16 bytes of packed nibbles ]
              lo nibble -> weight 0..15
              hi nibble -> weight 16..31
```

Each 4-bit value indexes a small non-linear table:

```zig
acc += IQ4_NL_TABLE[q & 0xF] * vec[base + j];
acc += IQ4_NL_TABLE[q >> 4]  * vec[base + j + 16];
```

Then the block sum is multiplied by the f16 scale. This is like reading a
dictionary-compressed sentence without expanding the whole sentence first: each
code points to a real value, and we use it immediately.

Result:

```text
llmtoy:   33.451 s
llama.cpp 24.007 s
```

This is only a small improvement in the current scalar implementation, but it is
directionally correct and removes the f32 row materialization for `IQ4_NL`.

## Disassembly Notes

The ReleaseFast disassembly for `quant.dequant.dotIQ4NL` shows what the compiler
made from the scalar fused loop:

```text
vcvtph2ps       convert the f16 block scale to f32
movzbl/shr/and  unpack each packed nibble
vmovss          load one table value
vmulss          multiply table value by vec element
vaddss          accumulate scalar products
```

The important part is what is absent on the `IQ4_NL` fast path: there is no call
to `dequantRow`, no f32 scratch-row write, and no separate `dotf32` pass.

It is still scalar per nibble. A future version can try to batch the 16 low
nibbles and 16 high nibbles into vector lanes, but AVX2 has no cheap f32 gather
for this exact 16-entry lookup table. The likely next wins are:

- reduce thread-pool wakeups around small expert matvecs
- specialize Gemma4/A4B row types so `mat_type` branches disappear in hot loops
- fuse gate and up expert projections for the packed `gate_up_exps` tensor
- add a benchmark mode that excludes model load entirely for stable decode-only
  measurements

## Profile 1: Where The Time Actually Goes

After adding the profiling wrapper, a sampled run used:

```sh
TOKENS=8 scripts/profile_gemma4.sh record
```

Run shape:

```text
prefill:    25 tokens in 17762 ms (1.41 tok/s)
generation:  8 tokens in  5749 ms (1.39 tok/s)
samples:    14488
```

Top symbols from `perf report --stdio --no-children`:

```text
43.31% ops.math.dequantRow
        37.90% dequantQ3K
26.53% quant.dequant.dequantQ6K
13.79% quant.dequant.dequantQ4K
 5.32% ops.math.RowJob.poolRun
 5.11% quant.dequant.dotQ5_0
 1.99% quant.dequant.dotIQ4NL
 0.39% model.gemma4.forward.forwardOne
```

The main lesson is that the Gemma4 path is not currently bottlenecked on the
high-level transformer structure. Attention, RoPE, softmax, RMSNorm, GELU, and
MoE routing barely show up compared with quantized row decoding.

For this APEX I Mini quant, the expensive loop is:

```text
packed K-quant row -> dequantize row to f32 scratch -> vector dot with activation
```

Q3_K is the largest single target because Gemma4 uses it for the dense FFN
gate/up matrices and the packed expert gate/up tensor. Those are called
repeatedly for every token.

### Negative Experiment: Scalar Q3_K Fused Dot

A simple fused Q3_K dot was tested:

```text
packed Q3_K row -> decode each value -> multiply activation immediately
```

Correctness held for the fixed-seed smoke output, but performance got worse:

```text
baseline short run:       generation 1.41 tok/s
scalar fused Q3_K:        generation 1.17 tok/s
grouped scalar Q3_K:      generation 1.33 tok/s
```

The reason is that the old path, while wasteful, still feeds a fully dequantized
row into `dotf32`, which uses 8-wide SIMD with multiple accumulators. The scalar
fused loop saved scratch traffic but gave up too much SIMD throughput.

So the lowest hanging fruit is not "fuse Q3_K naively"; it is one of:

- implement a real AVX2 Q3_K/Q4_K/Q6_K vector-dot path
- quantize the activation vector to Q8 blocks and port the llama.cpp
  `q*_K × q8_K` dot strategy
- specialize the K-quant matvec path enough that the compiler can see fixed
  quant types and fewer branches inside hot row loops

The profile makes Q3_K the first target, but the implementation needs to keep
SIMD in the inner dot loop.
