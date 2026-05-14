# Phase 7 — GPU Compute via Vulkan

## Why Vulkan for compute?

Several GPU compute APIs exist: CUDA, ROCm/HIP, OpenCL, Vulkan, Metal. The goal
here is a single codebase that runs on consumer AMD hardware under Linux without
proprietary drivers or a vendor-specific SDK. Vulkan is the right choice:

- Works on AMD with the open-source RADV Mesa driver (no ROCm needed)
- Vulkan 1.1+ is ubiquitous on modern Linux desktop hardware
- Compute shaders are written in GLSL or HLSL, compiled to SPIR-V — portable
  intermediate representation
- The spec is public; any question has an authoritative answer

The target machine has an AMD Radeon RX 7800 XT (NAVI32) with the RADV driver
and Vulkan 1.4. That is much faster for matrix operations than the Ryzen 3600
CPU.

## Vulkan compute concepts

Vulkan was designed for graphics, but everything it exposes is also useful for
pure compute. The concepts from graphics all carry over:

```
CPU side                         GPU side
─────────────────────────────────────────────────────────
Instance                         Vulkan runtime
PhysicalDevice                   The actual GPU chip
Device                           Logical connection to the GPU
Queue                            Stream of work submitted to the GPU
CommandPool / CommandBuffer      Recorded sequence of GPU commands
Buffer + DeviceMemory            Raw memory the GPU can read/write
DescriptorSetLayout              "Shape" of what a shader expects (binding slots)
DescriptorSet                    Concrete binding of real buffers to binding slots
PipelineLayout                   Combination of descriptor layout + push constants
ShaderModule                     Compiled SPIR-V shader
Pipeline                         Compiled compute program (shader + layout)
```

For inference, the only GPU primitive we need is a **compute pipeline**: upload
data to GPU buffers, dispatch shader threads, read the result back.

## Initialization sequence

Every Vulkan program starts the same way:

```
vkCreateInstance
  ↓
vkEnumeratePhysicalDevices  → pick one with a compute queue
  ↓
vkCreateDevice              → logical connection + queue
  ↓
vkGetDeviceQueue
  ↓
vkCreateCommandPool
```

A queue is the pipe through which work flows to the GPU. For pure compute we
need a queue family that has `VK_QUEUE_COMPUTE_BIT`. On AMD RADV this is
typically family 0 (which also supports graphics).

All of this lives in `src/gpu/context.zig` as `GpuContext`.

## Memory model

Vulkan memory is explicit. The application allocates `VkDeviceMemory` and binds
it to `VkBuffer` objects. Memory has property flags:

```
VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT    CPU can map and read/write this memory
VK_MEMORY_PROPERTY_HOST_COHERENT_BIT   Writes are immediately visible (no flush)
VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT    On-chip GPU memory (fastest for compute)
```

Two memory regimes matter for LLM inference:

**Host-coherent** (`HOST_VISIBLE | HOST_COHERENT`): the CPU can map and write
directly. No explicit flush needed — the GPU sees the write immediately. Used
for per-token activations (the vector, the output) where we need a new value
every token anyway. Limited by PCIe bandwidth (≈ 64 GB/s for PCIe 4.0 x16).

**Device-local** (`DEVICE_LOCAL`): on-chip GPU VRAM. Cannot be mapped by the
CPU. Accessed at full GPU memory bandwidth (≈ 432 GB/s on RX 7800 XT). Used
for weight matrices that are uploaded once at model-load time and read thousands
of times — one read per token per layer.

To get data into a device-local buffer, we use a **staging buffer**: a
temporary host-coherent buffer, copy the data in, then call `vkCmdCopyBuffer`
to transfer GPU-side, then free the staging buffer.

`GpuBuffer` in `src/gpu/buffer.zig` has three constructors:

```zig
GpuBuffer.initHostCoherent(ctx, size, usage)  // for activations, outputs
GpuBuffer.initDeviceLocal(ctx, size, usage)   // for weight matrices
GpuBuffer.initStaging(ctx, size)              // temporary upload helper
```

`GpuContext.copyBuffer` does the device-side transfer via a one-shot command
buffer.

## Compute shaders and SPIR-V

GPU programs ("shaders") for Vulkan are compiled to SPIR-V binary. We write
them in GLSL (a C-like language for shaders) and compile with `glslc`:

```sh
glslc --target-env=vulkan1.1 -fshader-stage=compute shader.glsl -o shader.spv
```

`glslc` is part of the `shaderc` package. In the nix dev shell, it is on PATH.

The build system compiles shaders automatically as part of `zig build`. The
SPIR-V bytes are embedded in the binary via `@embedFile`, so no runtime file
loading is needed.

### The matvec_f32 shader

The first shader is the simplest useful primitive for LLM inference: multiply
a matrix by a vector.

```glsl
layout(local_size_x = 64) in;         // 64 threads per workgroup

layout(set=0, binding=0) readonly buffer MatBuf { float mat[]; };
layout(set=0, binding=1) readonly buffer VecBuf { float vec_in[]; };
layout(set=0, binding=2) writeonly buffer OutBuf { float vec_out[]; };

layout(push_constant) uniform PC { uint rows; uint cols; } pc;

void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= pc.rows) return;
    float acc = 0.0;
    for (uint c = 0; c < pc.cols; c++)
        acc = fma(mat[row * pc.cols + c], vec_in[c], acc);
    vec_out[row] = acc;
}
```

Each GPU thread computes one output row. With 64 threads per workgroup and one
workgroup per 64 rows, the dispatch is:

```
groups = ceil(rows / 64)
vkCmdDispatch(groups, 1, 1)
```

For a 2816-element output (the Gemma4 d_model), that is 44 workgroups × 64
threads = 2816 threads — one per element.

## Descriptor sets and push constants

A **descriptor set** binds real VkBuffers to the binding slots the shader
declares. The layout declares that binding 0 is a storage buffer; the
descriptor set says "binding 0 = this specific buffer."

**Push constants** are small values (≤ 128 bytes) written directly into the
command buffer. For matvec we push `rows` and `cols` as two u32 values. This
avoids creating a separate UBO buffer for a pair of integers.

## Pipeline creation order

```
vkCreateDescriptorSetLayout   (layout: 3 storage buffers)
vkCreatePipelineLayout        (descriptor layout + push constants)
vkCreateShaderModule          (load SPIR-V)
vkCreateComputePipelines      (shader + pipeline layout)
vkCreateDescriptorPool        (pool to allocate descriptor sets from)
```

The pipeline is created once and reused for every matvec call.

## Per-call dispatch

For each matvec:

```
vkAllocateDescriptorSets         (get a set from the pool)
vkUpdateDescriptorSets           (bind mat/vec/out buffers)
vkAllocateCommandBuffers         (one-shot command buffer)
  vkBeginCommandBuffer
  vkCmdBindPipeline
  vkCmdBindDescriptorSets
  vkCmdPushConstants             (rows + cols)
  vkCmdDispatch
  vkEndCommandBuffer
vkQueueSubmit
vkQueueWaitIdle                  (synchronize: CPU blocks until GPU is done)
vkFreeCommandBuffers
vkFreeDescriptorSets
```

For the first pass, `vkQueueWaitIdle` is the simplest synchronization. Every
matvec call is fully synchronous: GPU runs, CPU waits, CPU reads result.
Later, pipeline parallelism (submit → do other CPU work → wait) is an
optimization opportunity.

## Zig-specific notes

### C import strategy

Vulkan is a C API. Zig's `@cImport` translates C types into Zig types, but
each call creates its own opaque type namespace. If two files each call
`@cImport(@cInclude("vulkan/vulkan.h"))`, the `VkDevice` in file A is a
different opaque type than `VkDevice` in file B — they don't unify, and
passing one to a function expecting the other is a compile error.

The fix is a single shared import file (`src/gpu/vk.zig`):

```zig
pub const vk = @cImport(@cInclude("vulkan/vulkan.h"));
```

Every other GPU file does `const vk = @import("vk.zig").vk;`. Because they
all reference the same compiled module, the types unify.

### Handle initialization

On 64-bit Linux, Vulkan 1.4 defines ALL handle types (both dispatchable and
non-dispatchable) as struct pointers:

```c
// VK_USE_64_BIT_PTR_DEFINES = 1 on x86_64
#define VK_DEFINE_NON_DISPATCHABLE_HANDLE(object) typedef struct object##_T *object;
```

Zig translates these as `?*opaque_struct`. They must be initialized to `null`,
not `0`. The null pipeline cache, null fence, and null allocation callbacks
all follow this pattern:

```zig
var pipeline: vk.VkPipeline = null;
vk.vkCreateComputePipelines(dev, null, 1, &ci, null, &pipeline)
```

### SPIR-V alignment

`VkShaderModuleCreateInfo.pCode` must point to 4-byte aligned memory.
`@embedFile` returns a byte array with alignment 1 by default. The fix: declare
the embedded constant with an explicit `align(4)` annotation in `shaders.zig`:

```zig
pub const matvec_f32 align(4) = @embedFile("matvec_f32.spv").*;
```

The `.*` dereferences the file slice into an array value; `align(4)` tells Zig
to place it at a 4-byte-aligned address. Taking `&shaders.matvec_f32` then
satisfies `pCode`'s alignment requirement without any allocation or copy:

```zig
const spv = &shaders.matvec_f32;
const shader_ci = vk.VkShaderModuleCreateInfo{
    .codeSize = shaders.matvec_f32.len,
    .pCode = @ptrCast(spv),
};
```

### Build system integration

`build.zig` compiles each GLSL shader to SPIR-V as a `Step.Run`, captures the
output via `addOutputFileArg`, copies it alongside a generated `shaders.zig`
via `addWriteFiles`, and builds a Zig module from that file. The result is
imported as `"gpu_shaders"`.

The Vulkan library is found via the nix shell's `LIBRARY_PATH` env var (set by
the shell hook in `flake.nix`). Vulkan headers are found via `CPATH`.

## Weight-resident sessions

For inference, the same weight matrix is multiplied by a different vector every
token. Allocating and uploading a staging buffer per token would waste PCIe
bandwidth. `MatvecSession` pre-uploads the matrix at construction time and keeps
it resident in VRAM:

```zig
// at model load:
var session = try MatvecSession.init(ctx, weight_matrix, rows, cols);
defer session.deinit();

// per token:
try session.run(ctx, pipeline, activation_vec, output_vec);
```

Under the hood, `init` creates a staging buffer, uploads the matrix, calls
`copyBuffer` to transfer it to device-local VRAM, then frees the staging buffer.
The resulting `mat_buf` stays in VRAM for the lifetime of the session. `vec_buf`
and `out_buf` are host-coherent because they change every token regardless.

## What this achieves

After Phase 7, `llmtoy gpu-info` prints the GPU device name and verifies that
a 4×4 identity matvec produces the correct result, using a device-local weight
matrix uploaded via a staging buffer. The infrastructure is in place for
offloading the Gemma4 forward pass matmul operations to the GPU.

## Smoke test

```sh
nix develop --command ./zig-out/bin/llmtoy gpu-info
```

Expected output:
```
GPU device: AMD Radeon RX 7800 XT (RADV NAVI32)
matvec smoke test: [1, 2, 3, 4] (expect [1, 2, 3, 4])
```

The full test suite (`zig build test`) includes a GPU matvec correctness test
that skips gracefully if no Vulkan device is available.

## Next steps

- Add quantized matvec shaders (Q8×fp32, Q4_K×fp32, Q3_K×fp32) matching the
  GGUF quant types used by Gemma4 APEX I Mini
- Wire `MatvecSession` into the Gemma4 forward pass as an optional `--gpu` backend
- Benchmark GPU matvec throughput vs the AVX2 CPU path on Gemma4 generation speed
