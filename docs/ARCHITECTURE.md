# Architecture

## Goal

Build a narrow, auditable inference stack for `Qwen/Qwen3-32B` on Blackwell that is centered on three moving parts:

1. `compiler`: lowers model-level ops to cuTile and TileIR-backed kernels.
2. `runtime`: owns request batching, KV cache placement, and kernel dispatch.
3. `serve`: exposes a minimal front end only after the execution and benchmark story is stable.

## Non-goals

- Reproducing vLLM surface area in the first phase.
- Supporting every model family at once before `Qwen/Qwen3-32B` is stable.
- Hiding compiler behavior behind framework magic.
- Embedding `vLLM`, `SGLang`, `llama.cpp`, or another inference runtime inside the core serving path.

## Design rules

### 1. Compiler-first kernels

Every performance-critical path must have a traceable lowering chain:

`model op -> adapter selection -> cuTile kernel -> TileIR -> cubin -> SASS`

If a path cannot be inspected at this level, it does not belong in the core stack.

### 2. Runtime is a small state machine

The runtime should only do five things:

1. admission control
2. request scheduling
3. KV cache bookkeeping
4. kernel dispatch
5. token sampling

Everything else belongs in adapters, tooling, or offline compilation.

### 3. Model support is adapter-driven

Each model family gets an adapter that declares:

- tensor layout
- rotary embedding policy
- normalization and MLP fusion rules
- KV cache format
- attention kernel requirements

The runtime should not accumulate model-specific branches.

### 4. Frameworks are baselines, not dependencies

`vLLM`, `SGLang`, and `llama.cpp` should inform comparison tables, not the core execution path.

If a capability is needed in `leanstack`, it should be re-expressed in `leanstack` terms:

- explicit kernel requirement
- explicit runtime state
- explicit benchmark consequence

### 5. Qwen3-32B and Blackwell define the first contract

The initial adapter and runtime should be shaped around the first verified target:

- `Qwen/Qwen3-32B`
- dense transformer blocks
- GQA with 64 query heads and 8 KV heads
- BF16 weight path
- Blackwell-class memory and tensor-core behavior

This is a deliberate optimization target, not a generic abstraction accident.

### 6. Remote validation is mandatory

The DGX Spark machine is the truth source for kernel bring-up. Local development defines structure; remote runs decide whether the stack is real.

## Layer map

### Compiler

- kernel catalog
- tile templates
- graph partitioning
- offline compilation cache

### Runtime

- block manager
- prefill/decode scheduler
- execution graph
- sampler

### Serve

- HTTP or gRPC edge
- request translation
- metrics and tracing

## Replacement strategy

The repo intentionally starts with a narrow vertical slice:

1. known-good cuTile kernel bring-up
2. remote artifact capture
3. adapter contract for `Qwen/Qwen3-32B`
4. minimal runtime that can execute one prefill and one decode loop
5. benchmark comparison against framework baselines
6. API layer only after the execution path is stable

This keeps the stack understandable while still targeting a full model run.
