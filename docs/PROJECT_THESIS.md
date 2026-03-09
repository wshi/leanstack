# Project Thesis

Date: 2026-03-09

## Thesis

`leanstack` is not trying to become another large inference framework.
It is also no longer trying to become a smaller generic inference framework.

The project thesis is narrower and harder:

- use `cuTile -> TileIR -> cubin -> SASS` as the core execution path
- target one concrete semantic contract first: `Qwen/Qwen3-1.7B-Base`
- target one concrete deployment contract first: the public `Qwen/Qwen3-1.7B-Base` BF16 checkpoint
- target one concrete hardware class first: Blackwell, currently the remote GB10 / DGX Spark machine
- target the actual remote compiler surface as `sm_121`, not a vague "future Blackwell" abstraction
- treat compatibility-driven software complexity as a tax, not a requirement
- build an appliance, not a generic runtime: `leanpack` for serving artifacts and `leanserve` for a static resident decode service
- measure whether an agent-built, hardware-near appliance can stay materially simpler than the current framework-heavy ecosystem while also showing a real performance advantage on that fixed contract
- treat "slightly faster than `vLLM`" as insufficient; the active aspiration is now a `30%+` throughput win on the primary official decode profile

## Central hypothesis

The dominant LLM serving stacks are expensive partly because they try to preserve broad compatibility:

- many model families
- many hardware targets
- many deployment modes
- many fallback paths

That compatibility is valuable, but it also creates a large software tax.

`leanstack` explores the opposite bet:

- narrow the target to a model-chip pair
- let an agent synthesize and regenerate the missing software
- spend tokens instead of carrying a permanently generalized runtime
- recover efficiency, inspectability, and customization from that narrower scope
- eliminate as much runtime uncertainty as possible so the user request becomes the dominant dynamic variable

From first principles, that means the project should optimize the terms that actually set single-request decode throughput:

- bytes moved per token
- decisive kernel quality
- KV read/write cost
- host intervention frequency

It should not expect repo size or runtime neatness, by themselves, to win a throughput comparison.

The project is therefore not only a performance effort. It is an economic and architectural experiment about when agent cost is lower than compatibility cost.

## Hard constraints

### 1. No borrowed runtime in the core path

The serving path must not depend on `vLLM`, `SGLang`, `llama.cpp`, `TensorRT-LLM`, or another inference runtime.

Those projects are comparison points and reference material only.

### 1a. Compatibility is opt-in, not default

The project does not begin by asking how to support the widest surface area.

It begins by asking how small the stack can become if:

- the hardware target is fixed
- the model target is fixed
- the agent can rewrite code quickly
- execution-time uncertainty is intentionally removed from everything except the user request

### 2. Qwen3-1.7B-Base BF16 is the first contract

The initial system is now shaped around one active deployment contract and two deferred precision investigations:

- active semantic and checkpoint contract: `Qwen/Qwen3-1.7B-Base` BF16
- deferred narrow-precision investigations: public FP8 and FP4 authoring on `sm_121`

The intended static contract is:

- 28 transformer layers
- hidden size 2048
- grouped-query attention with 16 query heads and 8 KV heads
- head dimension 128
- BF16 linears and activations on the first path
- fixed GB10 / `sm_121` target

The intended outcome is that this contract becomes static:

- fixed model geometry
- fixed packed weight layout
- fixed tensor layout
- fixed precision policy
- fixed page layout
- fixed kernel set
- fixed scheduler shape
- fixed hardware target
- fixed prompt-token buckets for the official comparison path

The user request should be the only first-class dynamic input.

The first gate is therefore not "run the whole model."

The first gate is:

- keep the executable precision gate green for BF16 on `sm_121`
- record FP8 and FP4 blockers explicitly instead of hand-waving them away

### 3. Blackwell is the first hardware contract

The project should exploit the fact that the target machine is Blackwell-class hardware instead of hiding behind a generic abstraction boundary.

That means:

- explicit kernel layouts
- explicit memory movement
- explicit compilation artifacts
- explicit performance accounting

That hardware contract also defines the preferred compiler policy:

- mainline: `cuTile -> TileIR -> cubin`
- diagnostic fallback: `PTX` only when the cuTile path misses a decisive hotspot and the project needs to understand why
- ground truth: inspect `SASS`, but do not make direct SASS authoring the default path

### 4. Agentic development is part of the point

The goal is not only to produce a fast stack.

The goal is to show that an agent-guided workflow can build and maintain a smaller, more hardware-native stack than the current accumulation of framework layers, helper daemons, adapters, compatibility shims, and opaque optimization passes.

### 5. Token budget is an explicit engineering resource

In this repo, token spend is treated as a real engineering input:

- prompt tokens
- response tokens
- iteration count

The intended outcome is not "zero software cost."

The intended outcome is that a bounded agent token budget can replace a large amount of compatibility-oriented software and manual framework plumbing.

## What success looks like

### Technical success

- the repo keeps an explicit, owned BF16 path running on the remote GB10 and the official hot path stays on `cuTile/TileIR`
- the repo defines an offline serving-artifact format and a resident appliance contract for the fixed Qwen/GB10 pair
- the repo explicitly records why FP8 and FP4 are still blocked on the public stack
- `Qwen3-1.7B-Base` BF16 runs end to end on the remote Blackwell machine through `leanpack + leanserve`
- the critical path is inspectable down to TileIR and SASS
- the stack exposes a small and auditable runtime surface
- the core path does not rely on framework-managed uncertainty such as automatic placement or CPU offload

### Comparative success

- `leanstack` is benchmarked against exact-format BF16 external baselines on the same machine and model profile
- `vLLM` and `SGLang` remain required comparison points when they can run the same BF16 checkpoint or a clearly labeled equivalent snapshot
- `llama.cpp` is tracked as a secondary deployment reference when the weight format is not apples-to-apples
- the result table includes `generated tokens/s`, latency, memory use, process shape, operational complexity, and software-stack size proxies
- the primary throughput target is no longer parity; the primary target is `>= 1.30x` warmed `vLLM` on the main exact-bucket decode profile

### Go / no-go success

- if `leanstack` cannot show a real performance or complexity advantage on the fixed `Qwen3-1.7B-Base BF16 + GB10` appliance contract, the repo should say so directly
- if the packed BF16 appliance can only slightly beat `vLLM`, the repo should not claim that the thesis is proven; the repo should then require a stronger asymmetry such as exact self-speculative decode
- if the public `cuTile` path cannot clear FP8 or FP4 later, that result is part of the research outcome rather than something to hide

### Research success

- the repo documents whether an agent-driven, hardware-near stack can simplify the modern LLM software path without surrendering practical performance
- if the answer is negative on a given workload, the repo states which missing kernels, scheduling rules, compiler gaps, or compatibility requirements caused the miss
