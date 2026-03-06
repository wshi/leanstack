# Project Thesis

Date: 2026-03-06

## Thesis

`leanstack` is not trying to become another large inference framework.

The project thesis is narrower and harder:

- use a `cuTile -> TileIR -> cubin -> SASS` execution path as the core of the stack
- target one concrete model first: `Qwen/Qwen3-32B`
- target one concrete hardware class first: Blackwell, currently the remote GB10 / DGX Spark machine
- measure whether an agent-built, hardware-near stack can stay materially simpler than the current framework-heavy ecosystem while remaining performance-competitive

## Hard constraints

### 1. No borrowed runtime in the core path

The serving path must not depend on `vLLM`, `SGLang`, `llama.cpp`, `TensorRT-LLM`, or another inference runtime.

Those projects are comparison points and reference material only.

### 2. Qwen3-32B is the first contract

The initial system is shaped around the public and remotely verified `Qwen/Qwen3-32B` contract:

- dense causal LM
- 64 transformer layers
- grouped-query attention with 64 query heads and 8 KV heads
- head dimension 128
- hidden size 5120
- intermediate size 25600
- BF16 checkpoint
- rotary position embedding with `rope_theta=1_000_000`

### 3. Blackwell is the first hardware contract

The project should exploit the fact that the target machine is Blackwell-class hardware instead of hiding behind a generic abstraction boundary.

That means:

- explicit kernel layouts
- explicit memory movement
- explicit compilation artifacts
- explicit performance accounting

### 4. Agentic development is part of the point

The goal is not only to produce a fast stack.

The goal is to show that an agent-guided workflow can build and maintain a smaller, more hardware-native stack than the current accumulation of framework layers, helper daemons, adapters, compatibility shims, and opaque optimization passes.

## What success looks like

### Technical success

- `Qwen/Qwen3-32B` runs end to end on the remote Blackwell machine through `leanstack`
- the critical path is inspectable down to TileIR and SASS
- the stack exposes a small and auditable runtime surface

### Comparative success

- `leanstack` is benchmarked against `vLLM` and `SGLang` on the same machine and model profile
- `llama.cpp` is tracked as a secondary deployment reference when the weight format is not apples-to-apples
- the result table includes `generated tokens/s`, latency, memory use, process shape, and operational complexity

### Research success

- the repo documents whether an agent-driven, hardware-near stack can simplify the modern LLM software path without surrendering practical performance
- if the answer is negative on a given workload, the repo states which missing kernels, scheduling rules, or compiler gaps caused the miss
