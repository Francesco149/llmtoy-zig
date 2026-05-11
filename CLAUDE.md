# llmtoy-zig — Agent Instructions

Educational LLM inference engine built incrementally in Zig.

## What this is

Step-by-step inference implementation targeting:
- **Models**: Gemma 4 26B A4B (APEX-I-Mini) and Qwen3.6 35B A3B (APEX-I-Mini), both GGUF format
- **Hardware**: Ryzen 3600 / 5900x, 64 GB RAM, consumer GPU (Vulkan, on the 3600 machine)
- **OS**: Linux x86_64

Each phase is an independent educational unit — old paths stay runnable and tested forever.

## Project layout

```
src/              Zig source (keep files small and decoupled)
docs/roadmap.md   Phase-by-phase plan — check here for current phase
docs/phases/      Per-phase write-ups and notes
docs/benchmarks/  Benchmark results per optimization step
scripts/          Build wrappers, benchmark runners
```

Root stays clean — no extra files at the top level.

## Build & test

**Always use the Nix flake dev shell** — `zig` is not on PATH outside it.
Claude Code must prefix every zig invocation with `nix develop --command`:

```sh
nix develop --command zig build
nix develop --command zig build test
nix develop --command ./zig-out/bin/llmtoy info <model>
```

In a human terminal, `nix develop` (or direnv with `.envrc`) enters the shell once.

All CI-equivalent checks must pass before committing: `zig build test`.

## Conventions

- **Files**: single-purpose, small. Split early rather than growing large files.
- **Comments**: only write one when the WHY is non-obvious. No what/how narration.
- **Tests**: write regression tests for every path as we add it. Old paths get integration tests so we can verify they still work as we restructure.
- **Benchmarks**: document with hyperfine. Store results in `docs/benchmarks/` so we can track the progression.
- **Commits**: use the user's **global** git identity (never set local git config). Co-author every commit with Claude:
  ```
  Co-Authored-By: Claude <noreply@anthropic.com>
  ```
- **Phases**: each phase has its own doc in `docs/phases/phaseN-*.md` explaining the concepts and implementation choices.

## Philosophy

1. Simplest working implementation first, then optimize.
2. Each optimization is its own educational unit with measured before/after.
3. Modularity first — we swap backends in and out. CPU naive → CPU SIMD → GPU.
4. Fast compilation is a feature. Keep compile times visible and fast.
5. No premature abstractions. Three similar lines beats a bad abstraction.

## Current phase

**Phase 5 complete.** See `docs/roadmap.md` for the full plan.

Completed steps (Qwen2.5-0.5B Q4_K_M, Ryzen 3600):
- Step 1: raw Thread.spawn multi-threading → 1.54× (1.45 → 2.23 tok/s)
- Step 2: AVX2 SIMD dot (`@Vector(8, f32)`, 4 accumulators) + native target → 3.03× (4.40 tok/s 12t)
- Step 3: fused Q8_0 dequant+dot (`dotQ8_0`) → 14.06 tok/s on Q8_0
- Step 4: fused Q5_0 dequant+dot (`dotQ5_0`) → 3.32× (4.82 tok/s 1t; 12t regressed due to spawn)
- Step 5: persistent thread pool (`ThreadPool`, `std.Io.Mutex/Condition`) → **9.72× overall** (14.1 tok/s 12t)

Phase 6 candidates: MoE routing (Qwen3.6/Gemma4), BF16 weights, batched prefill.

## Zig 0.16 API patterns

We're on Zig **0.16.0**. Many tutorials and older code use pre-0.14 APIs. Key differences:

**Entry point** — main now receives I/O + allocators:
```zig
pub fn main(init: std.process.Init) !void {
    const io  = init.io;           // std.Io context
    const gpa = init.gpa;          // general-purpose allocator
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena); // []const [:0]const u8
}
```

**Stdout/stderr**:
```zig
var buf: [8192]u8 = undefined;
var fw = std.Io.File.stdout().writer(io, &buf);
const out = &fw.interface;         // *std.Io.Writer
try out.print("hello {s}\n", .{name});
try out.flush();                   // always flush before exit
// stderr: std.debug.print() still works, no io needed
```

**File I/O** (`std.fs` is deprecated → `std.Io.Dir`):
```zig
// open (absolute)
const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
// open (relative)
const f = try std.Io.Dir.cwd().openFile(io, path, .{});
const stat = try std.Io.File.stat(f, io);  // .size: u64
std.Io.File.close(f, io);
```

**mmap** — alignment changed, PROT is a packed struct literal:
```zig
const mmap = try std.posix.mmap(null, size, .{ .READ = true },
    .{ .TYPE = .PRIVATE }, file.handle, 0);
// type: []align(std.heap.page_size_min) u8
std.posix.munmap(mmap);
```

**Hash maps** — `std.StringArrayHashMap` gone; use `std.StringHashMap` (managed, unchanged API):
```zig
var m = std.StringHashMap(V).init(allocator);
defer m.deinit();
try m.put(key, value);
```

**ArrayList** — in 0.16, `std.ArrayList(T)` IS the unmanaged type (renamed); `std.ArrayListUnmanaged` is a deprecated alias for the same thing. The managed variant moved to `std.array_list.Managed(T)`. Use `.empty` to initialise.
```zig
// std.ArrayList = unmanaged; allocator at each call site:
var list: std.ArrayList(T) = .empty;
defer list.deinit(allocator);
try list.append(allocator, item);

// Managed (allocator stored inside the struct):
var list = std.array_list.Managed(T).init(allocator);
defer list.deinit();
try list.append(item);
```

**Writer in tests** — no `std.io.fixedBufferStream`; use `std.Io.Writer.fixed`:
```zig
var buf: [4096]u8 = undefined;
var w: std.Io.Writer = .fixed(&buf);
try w.writeAll("hello");
```

**Testing I/O** — `std.testing.io` available inside tests.

**Synchronisation** — `std.Thread.Mutex`/`std.Thread.Condition` do not exist in 0.16.
Use `std.Io.Mutex` and `std.Io.Condition`; every operation requires `io: std.Io`:
```zig
mutex: std.Io.Mutex = std.Io.Mutex.init,
cond:  std.Io.Condition = std.Io.Condition.init,

mutex.lockUncancelable(io);
defer mutex.unlock(io);
cond.waitUncancelable(io, &mutex);
cond.signal(io);
cond.broadcast(io);
```
Store `io: std.Io` in any struct that needs sync (passed from `init.io` in main).

**build.zig** — `root_source_file` moved into `root_module`:
```zig
b.addExecutable(.{
    .name = "foo",
    .root_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target, .optimize = optimize,
    }),
});
```

**Block expressions** — blocks produce values via `break :label value`, not "last expression":
```zig
const TABLE: [N]T = blk: {
    var t: [N]T = undefined;
    // ... fill t ...
    break :blk t;   // "the value of blk is t"
};
```
Module-level `const` initialisers are implicitly comptime. The label is required because bare
`break` only exits loops — the label disambiguates block vs. loop.

**build.zig.zon** — requires `fingerprint` field (run `zig build` once to get the suggested value).

## Model paths

```
/opt/ai-lab/models/mudler/gemma-4-26B-A4B-it-APEX-GGUF/gemma-4-26B-A4B-APEX-I-Mini.gguf
/opt/ai-lab/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-I-Mini.gguf
```
