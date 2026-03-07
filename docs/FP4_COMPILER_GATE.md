# FP4 Compiler Gate

Date verified: 2026-03-07

## Why this gate exists

`leanstack` previously considered a narrower FP4-first pivot:

- semantic base: `Qwen/Qwen3-8B`
- deployment artifact: `nvidia/Qwen3-8B-FP4`
- hardware target: `NVIDIA GB10 / sm_121`

That pivot only makes sense if the public `cuTile` stack can actually express and compile the decisive FP4 kernels.

As of 2026-03-07, this is no longer the active first-format target. The active mainline is `Qwen/Qwen3-1.7B-Base` BF16, and this document remains as a negative sub-gate for the deferred FP4 route.

## What is already true

### Official external evidence

- NVIDIA publishes a `Qwen3-8B-FP4` deployment artifact.
- NVIDIA CUTLASS / CuTe documentation for Blackwell describes narrow-precision and blockscaled kernel support in the broader stack.
- `tileiras` on the remote machine supports `--gpu-name=sm_121`.

That means Blackwell FP4 support is plausible in the broader NVIDIA compiler and kernel ecosystem.

### Remote toolchain facts

Remote inspection of `/home/pto/venv-cutile/lib/python3.12/site-packages/cuda/tile` shows:

- `cuda.tile` version `1.1.0`
- public dtype symbols for:
  - `float16`
  - `float32`
  - `float64`
  - `bfloat16`
  - `tfloat32`
  - `float8_e4m3fn`
  - `float8_e5m2`
- no public `fp4`, `float4`, or `nvfp4` dtype symbol in the exposed Python dtype registry
- bytecode type definitions expose `BF16`, `TF32`, and `F8*`, but no visible FP4 type token

This is the critical uncertainty:

- the backend target exists
- the public frontend FP4 surface is not yet proven

## What counts as success

Phase 1 only succeeds if at least one of these is demonstrated on the remote GB10:

1. a minimal FP4 or NVFP4 GEMM/linear kernel is authored in the public `cuTile` Python DSL, lowered through:
   - `cuTile DSL -> TileIR/tilebc -> tileiras -> cubin`
2. or, if public DSL coverage is the blocker, a tightly scoped PTX escape hatch is used for the hotspot while preserving:
   - explicit artifact capture
   - `sm_121` targeting
   - SASS inspection

In both cases, the kernel must run on the remote machine and produce a captured artifact trail.

## Current executable gate

This repo now exposes the gate directly through:

- `experiments/cutile/fp4_compiler_gate.py`
- `scripts/remote_fp4_gate.sh`

Current remote result on 2026-03-07:

- status: `blocked`
- blocker: `public cuda.tile frontend does not expose a complete FP4 authoring surface`
- backend target availability: `sm_121` is present in `tileiras`

That means the gate is no longer documentation-only. It is an executable check whose current answer is negative.

## What does not count as success

The following do not clear the gate:

- "Blackwell supports FP4 in principle"
- "CUTLASS supports Blackwell FP4 somewhere in the stack"
- "TensorRT-LLM or another runtime can run the FP4 checkpoint"
- "The model card exists"

Those facts matter, but they do not prove the public `leanstack` authoring path is viable.

## Current decision rule

- if the public `cuTile` path can express FP4 on `sm_121`, continue with the FP4 runtime
- if only a small hotspot needs PTX, continue but record the PTX wedge explicitly
- if the public path cannot emit any credible FP4 kernel, stop treating `Qwen3-8B-FP4` as the active first runtime target and either:
  - wait for toolchain coverage
  - choose a different first-format target
  - or declare the tooling gap as a negative research result
