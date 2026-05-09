# Phase 1 — GGUF Parsing

## What is GGUF?

GGUF (GPT-Generated Unified Format) is the file format used by llama.cpp and the broader quantized model ecosystem. It replaced the older GGML format and bundles all model data into a single self-describing file:

```
[ header ]
  magic:     u32  = 0x46554747 ("GGUF")
  version:   u32  (v1/v2/v3)
  n_tensors: u64  (u32 in v1)
  n_kv:      u64  (u32 in v1)

[ metadata key-value pairs ]  ×n_kv
  key:        u64-prefixed string
  value_type: u32  (MetaValueType enum)
  value:      (depends on type)

[ tensor info ]  ×n_tensors
  name:   u64-prefixed string
  n_dims: u32
  dims:   [n_dims]u64
  type:   u32  (GgmlType enum)
  offset: u64  (from start of data section)

[ padding to 32-byte alignment ]

[ tensor data ]  (raw bytes, accessed via offset + data_section_start)
```

All integers are **little-endian**. Strings are length-prefixed (u64) with no null terminator.

## What we built

`src/gguf/types.zig` — Type definitions:
- `MetaValueType` — 13 value types (uint8 through float64, string, array)
- `GgmlType` — tensor quantization types (F32, F16, Q4_K, Q8_0, IQ2_S, ...)
- `MetaValue` — tagged union over all metadata value types
- `TensorInfo` — name, dims, quantization type, byte offset

`src/gguf/reader.zig` — Parser + mmap'd reader:
- `parseBytes(data, allocator)` — pure parser, takes any `[]const u8`
- `GgufReader.open(path, io, allocator)` — opens and mmaps a GGUF file, calls `parseBytes`

### Design decisions

**Memory-mapping**: We mmap the file rather than reading it. For multi-GB model files this matters:
- Pages load on demand — only the header pages are touched at startup
- Zero-copy: tensor weights can be read directly from the mmap'd region later
- OS manages the cache; re-runs reuse the in-memory pages

**Arena allocator for parsed data**: All parsed strings and slices live in one arena. `deinit` frees everything in one call with zero fragmentation.

**`parseBytes` + `GgufReader` split**: The parser is fully testable without touching the filesystem. Tests build synthetic GGUF bytes in a stack buffer and call `parseBytes` directly.

**`std.StringHashMap` for metadata**: Metadata KV pairs don't need ordered iteration; the managed hash map has a simpler API than the unmanaged variant.

## Zig 0.16 notes

This phase revealed several breaking changes in Zig 0.16 vs older tutorials:
- `std.io` → `std.Io` with I/O context (`Io`) threaded through all operations
- `pub fn main()` → `pub fn main(init: std.process.Init)` — GPA, IO, arena provided
- `std.fs.*` deprecated → `std.Io.Dir.*` (takes `io` parameter)
- `std.ArrayList.init(a)` → `.empty` (allocator per-method, see `std.array_hash_map`)
- `std.posix.PROT.READ` → `.{ .READ = true }` (packed struct literal)
- mmap alignment: `std.mem.page_size` → `std.heap.page_size_min`

## Results on target models

```
Qwen3.6 35B A3B (APEX-I-Mini):
  GGUF v3 | 13.33 GiB | 733 tensors | ctx_len=262144 | emb_dim=2048 | 40 layers
  Quants: Q5_K(34) Q4_K(178) Q3_K(159) IQ2_S(60) Q6_K(1) F32(301)
  MoE: 256 experts visible in ffn_down_exps tensor

Gemma4 26B A4B (APEX-I-Mini):
  GGUF v3 | 12.09 GiB | 658 tensors | ctx_len=262144 | emb_dim=2816 | 30 layers
  Quants: Q4_K(65) Q3_K(138) Q5_0(48) IQ4_NL(10) Q5_K(2) Q5_1(2) Q6_K(1) F32(392)
```

## Zig comptime — two patterns we used

### `comptime _: []const u8` in `MetaValue.format`

`std.fmt` requires any type that wants to be printable to implement a `format` method with this exact signature:

```zig
pub fn format(self: T, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void
```

The `comptime fmt` parameter is the format specifier string — whatever you write between `{}` in a format string. `{s}` passes `"s"`, `{d}` passes `"d"`, `{}` passes `""`. It's `comptime` because Zig's format machinery is entirely compile-time: `std.fmt.format` unrolls the whole dispatch tree at compile time, so there's no runtime overhead and format errors are caught at compile time rather than at runtime.

The callee can branch on it: `if (comptime std.mem.eql(u8, fmt, "s")) { ... }`. In `MetaValue.format` we name it `_` because we intentionally ignore the specifier — `MetaValue` always prints the same way regardless of whether you write `{s}` or `{}`. The `comptime` on the parameter is part of the interface contract, not optional.

### `comptime { _ = @import(...); }` in `main.zig`

In Zig, `test` blocks only end up in the test binary if the file that contains them is transitively reachable from the test root (`main.zig`). Every module-level `@import` is already followed by the compiler, so the existing `const gguf_reader = @import("gguf/reader.zig");` would include `reader.zig`'s tests on its own.

The explicit `comptime { _ = @import("gguf/reader.zig"); }` block is a second, named-nothing reference. It says: "include this file's declarations (including its `test` blocks) as a side effect of compilation, and I'm not keeping the result." The practical difference:

| form | creates a binding | intent communicated |
|------|------------------|---------------------|
| `const gguf_reader = @import(...)` | yes — `gguf_reader.T` is usable | "I need this module's API" |
| `comptime { _ = @import(...); }` | no | "include this for its tests, nothing else" |

Both are evaluated at compile time (all `@import` calls are inherently comptime). The comptime block is a guard: if the runtime import were removed later during a refactor, the tests would silently disappear without it. The explicit block makes the test-inclusion intent survive independently.

## Next

Phase 2: tokenization — load the BPE vocabulary from GGUF metadata and implement
`encode` / `decode` for both model families.
