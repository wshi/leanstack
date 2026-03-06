# Architecture

## Goal

Replace a sprawling LLM serving stack with a narrow, auditable system that is centered on three moving parts:

1. `compiler`: lowers model-level ops to cuTile and TileIR-backed kernels.
2. `runtime`: owns request batching, KV cache placement, and kernel dispatch.
3. `serve`: exposes a minimal OpenAI-compatible front end once the runtime is stable.

## Non-goals

- Reproducing vLLM surface area in the first phase.
- Supporting every model family at once.
- Hiding compiler behavior behind framework magic.

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

### 4. Remote validation is mandatory

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
3. adapter contract for Qwen-family models
4. minimal runtime that can execute one prefill and one decode loop
5. API layer only after the execution path is stable

This keeps the stack understandable while still targeting a full model run.
