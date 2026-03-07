# Architecture

## Goal

Build a narrow, auditable inference stack for `Qwen/Qwen3-8B` BF16 on `GB10 / sm_121` that is centered on three moving parts and explicitly avoids paying broad compatibility costs up front:

1. `compiler`: lowers model-level ops to cuTile and TileIR-backed kernels.
2. `runtime`: owns request batching, KV cache placement, and kernel dispatch.
3. `serve`: exposes a minimal front end only after the execution and benchmark story is stable.

## Non-goals

- Reproducing vLLM surface area in the first phase.
- Supporting every model family at once before the first `Qwen3-8B BF16` contract is stable.
- Supporting every hardware backend behind one generic abstraction.
- Hiding compiler behavior behind framework magic.
- Embedding `vLLM`, `SGLang`, `llama.cpp`, or another inference runtime inside the core serving path.

## Design rules

### 1. Compiler-first kernels

Every performance-critical path must have a traceable lowering chain:

`model op -> adapter selection -> cuTile kernel -> TileIR -> cubin -> SASS`

If a path cannot be inspected at this level, it does not belong in the core stack.

The current remote compiler target should be treated concretely as `GB10 / sm_121`, not as an abstract Blackwell placeholder.

That also implies a strict fallback order:

- `cuTile/TileIR` is the default authoring path
- `PTX` is an escape hatch when the public cuTile frontend cannot yet express a hotspot, especially for future FP8 or FP4 work
- `SASS` is a verification artifact, not the main source language

### 2. Runtime is a small state machine

The runtime should only do five things:

1. admission control
2. request scheduling
3. KV cache bookkeeping
4. kernel dispatch
5. token sampling

Everything else belongs in adapters, tooling, or offline compilation.

### 2a. Compatibility tax is a first-class architectural concern

Whenever a new layer, interface, or fallback path is proposed, the default question is:

`Is this required for the Qwen3-8B BF16 + GB10 contract, or is it a compatibility tax?`

If it is only a compatibility tax, it should be deferred.

### 2b. Runtime uncertainty should be designed out

For the first contract, the stack should avoid discovering core execution decisions at runtime.

The preferred shape is:

- static model geometry
- static precision policy
- static kernel inventory
- static memory layout
- static dispatch order
- dynamic user request only

If runtime behavior still depends on framework heuristics such as `device_map="auto"` or opportunistic CPU offload, the stack is not yet in its target form.

### 3. Model support is adapter-driven

Each model family gets an adapter that declares:

- tensor layout
- precision policy
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

For the same reason, `transformers` should gradually move from "execution provider" to "correctness oracle":

- acceptable today for tokenizer, config, and semantic cross-checks
- not acceptable as the long-term owner of the model execution path

### 5. Semantic contract and checkpoint contract define the first path

The initial adapter and runtime should be shaped around the first verified target:

- semantic contract: `Qwen/Qwen3-8B`
- active deployment contract: the public BF16 checkpoint for `Qwen/Qwen3-8B`
- dense transformer blocks
- GQA with 32 query heads and 8 KV heads
- explicit BF16 linears
- Blackwell-class memory and tensor-core behavior on `sm_121`

This is a deliberate optimization target, not a generic abstraction accident.

That implies a more aggressive simplification rule:

- no runtime search for architecture-specific behavior
- no hardware-agnostic placement logic in the first path
- no hidden decision points beyond those induced by user input length and decode progress

### 5a. Semantic contract, checkpoint contract, and precision gates are separate

`leanstack` should not blur together:

- the semantic base model
- the deployment checkpoint
- the kernel authoring path
- the precision gate result

For the active target:

- `Qwen/Qwen3-8B` defines geometry and prompt semantics
- the public BF16 checkpoint defines the active runtime weight contract
- the precision gate defines whether FP8 or FP4 can become a later deployment contract

This separation matters because a public model card or a working vendor runtime does not prove the public `leanstack` compiler path is viable.

### 5b. Agent regeneration is part of the design

The stack should be small enough that an agent can reasonably:

- rewrite a kernel
- adjust an adapter
- change a scheduler rule
- regenerate benchmark glue

without having to understand a sprawling, compatibility-first codebase.

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
3. executable precision gate on `sm_121`
4. adapter contract for `Qwen3-8B` semantics and BF16 checkpoint layout
5. minimal runtime that can execute one prefill and one decode loop
6. benchmark comparison against framework baselines and their compatibility-heavy process shape
7. API layer only after the execution path is stable

This keeps the stack understandable while still targeting a full model run.
