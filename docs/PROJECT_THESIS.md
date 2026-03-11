# Project Thesis

Date: 2026-03-11

## Unifying hypothesis

In an agentic software development era, a stable virtual instruction set (VIS) is not just a compiler artifact — it is the primary economic and architectural moat for AI hardware. An agent can synthesize, benchmark, and regenerate a hardware-specific inference stack cheaply enough to make narrow, VIS-native appliances superior to compatibility-heavy generic frameworks. This is first proven on GPU using Tile IR, then reproduced on custom silicon using our own VIS.

## Two-phase structure

`leanstack` is a two-phase research and engineering project.

**Phase 1 — GPU: Validate Tile IR as an Agentic-Era VIS.** Can a stable virtual instruction set (Tile IR) enable an agent to synthesize a more efficient and simpler inference stack than one built for broad compatibility? NVIDIA PTX proved that a stable ISA intermediate enables rapid hardware iteration without breaking software investments. Tile IR is PTX's successor for the tensor-core era. leanstack is designed to be the reference existence proof that Tile IR-native stacks can outperform compatibility-first stacks — not just on throughput, but on total software cost.

**Phase 2 — Custom Silicon: Reproduce the VIS Moat with Our Own Instruction Set.** Given everything learned in Phase 1, can the same agentic-synthesis methodology be applied to build a Tile-IR-equivalent virtual instruction set for our own chip, and demonstrate that it becomes the economic foundation of our hardware ecosystem?

The through-line is: leanstack is an experiment in VIS-centric agentic inference, first validated on Tile IR, then applied to our own VIS.

## Why VIS matters

The article "Compatibility and Fragmentation in the AI Era" argues that the AI accelerator landscape has returned to a 1950s-style proliferation of incompatible architectures. The lesson from IBM S/360, NVIDIA PTX, and now Tile IR is identical: the hardware vendor that owns a stable, well-abstracted virtual instruction set controls the software investment lock-in for their entire ecosystem.

The dominant LLM serving stacks are expensive partly because they try to preserve broad compatibility:

- many model families
- many hardware targets
- many deployment modes
- many fallback paths

That compatibility is valuable, but it also creates a large software tax. `leanstack` explores the opposite bet:

- narrow the target to a model-chip pair
- let an agent synthesize and regenerate the missing software via a stable VIS
- spend tokens instead of carrying a permanently generalized runtime
- recover efficiency, inspectability, and customization from that narrower scope
- eliminate as much runtime uncertainty as possible so the user request becomes the dominant dynamic variable

The project is therefore not only a performance effort. It is an economic and architectural experiment about when agent cost is lower than compatibility cost, and whether a stable VIS is the enabling abstraction that makes that trade-off work.

## Phase 1 thesis (current focus)

### Core question

Can a stable virtual instruction set (Tile IR) enable an agent to synthesize a more efficient and simpler inference stack than one built for broad compatibility?

### Immediate approach

- use `cuTile -> TileIR -> cubin -> SASS` as the core execution path
- target one concrete semantic contract first: `Qwen/Qwen3-1.7B-Base`
- target one concrete deployment contract first: the public `Qwen/Qwen3-1.7B-Base` BF16 checkpoint
- target one concrete hardware class first: Blackwell, currently the remote GB10 / DGX Spark machine
- target the actual remote compiler surface as `sm_121`, not a vague "future Blackwell" abstraction
- treat compatibility-driven software complexity as a tax, not a requirement
- build an appliance, not a generic runtime: `leanpack` for serving artifacts and `leanserve` for a static resident decode service
- measure whether an agent-built, hardware-near appliance can stay materially simpler than the current framework-heavy ecosystem while also showing a real performance advantage on that fixed contract
- the active aspiration is `30%+` throughput win on the primary official decode profile over warmed vLLM

### What Phase 1 must document

Beyond raw throughput, Phase 1 must produce evidence for or against the VIS thesis:

- which compatibility layers vLLM/SGLang carry that leanstack does not
- what agent token budget was spent to synthesize the stack vs. the compatibility software it replaced
- whether Tile IR provides cross-generation portability (when the next sm target appears)
- whether the model-chip contract schema is generalizable to new model families

## Phase 2 thesis (declared long-term destination)

### Core question

Given everything learned in Phase 1, can the same agentic-synthesis methodology be applied to build a Tile-IR-equivalent virtual instruction set for our own chip, and demonstrate that it becomes the economic foundation of our hardware ecosystem?

### Design principles (informed by Phase 1)

- define our own tile-based virtual instruction set as a stable specification, versioned independently of physical chip generations
- the VIS must be: tile-native (not thread-native), portable across our chip generations, expressive enough to cover attention, linear, and normalization kernels without falling back to scalar paths
- the agent-synthesis pipeline built in Phase 1 should be retargetable to our VIS as a backend
- leanpack format should be generalized to support our chip as a first-class target alongside sm_121
- the economic argument must be reproducible: measure agent token budget to bring up a new model on our chip vs. compatibility software cost for a generic runtime

Phase 2 is not yet in active execution. Its presence in this document is a declaration of intent, not a commitment to a schedule.

## Hard constraints

All Phase 1 hard constraints remain in force. Phase 2 inherits them in spirit and will add its own when active.

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

### 6. VIS stability is a design requirement, not an accident

The virtual instruction set — Tile IR in Phase 1, our own VIS in Phase 2 — must be treated as a stable contract boundary:

- kernel code targets the VIS, not the physical ISA directly
- when the physical target changes (new sm version, new chip generation), VIS-targeting code should recompile, not be rewritten
- VIS versioning must be explicit and tracked alongside leanpack artifact versioning

This constraint does not mean the VIS cannot evolve. It means evolution must be managed as a compatibility contract, not as an ad-hoc refactor.

## What success looks like

### Phase 1 technical success

- the repo keeps an explicit, owned BF16 path running on the remote GB10 and the official hot path stays on `cuTile/TileIR`
- the repo defines an offline serving-artifact format and a resident appliance contract for the fixed Qwen/GB10 pair
- the repo explicitly records why FP8 and FP4 are still blocked on the public stack
- `Qwen3-1.7B-Base` BF16 runs end to end on the remote Blackwell machine through `leanpack + leanserve`
- the critical path is inspectable down to TileIR and SASS
- the stack exposes a small and auditable runtime surface
- the core path does not rely on framework-managed uncertainty such as automatic placement or CPU offload

### Phase 1 comparative success

- `leanstack` is benchmarked against exact-format BF16 external baselines on the same machine and model profile
- `vLLM` and `SGLang` remain required comparison points when they can run the same BF16 checkpoint or a clearly labeled equivalent snapshot
- `llama.cpp` is tracked as a secondary deployment reference when the weight format is not apples-to-apples
- the result table includes `generated tokens/s`, latency, memory use, process shape, operational complexity, and software-stack size proxies
- the primary throughput target is `>= 1.30x` warmed `vLLM` on the main exact-bucket decode profile

### Phase 1 VIS thesis success

- the model-chip contract schema is documented as a formalizable artifact, not just a thesis paragraph
- the agent token budget spent to bring up the Qwen/GB10 pair is recorded and compared against the compatibility software it replaced
- when a second model family or sm target is attempted, the Tile IR abstraction boundary either holds (recompile, not rewrite) or the failure is documented precisely

### Phase 2 success (criteria to be refined when Phase 2 becomes active)

- our own VIS specification exists as a stable, versioned document
- the agent-synthesis pipeline can target our VIS as a backend
- a first model runs on our chip through our VIS with measured performance and recorded agent token budget
- the economic argument is quantified: agent cost to bring up a new model on our chip vs. compatibility software cost for a generic runtime on the same chip

### Go / no-go

- if `leanstack` cannot show a real performance or complexity advantage on the fixed `Qwen3-1.7B-Base BF16 + GB10` appliance contract, the repo should say so directly
- if the packed BF16 appliance can only slightly beat `vLLM`, the repo should not claim that the thesis is proven; the repo should then require a stronger asymmetry such as exact speculative decode
- if the public `cuTile` path cannot clear FP8 or FP4 later, that result is part of the research outcome rather than something to hide
- if Phase 2 VIS design proves intractable given Phase 1 learnings, the repo documents why

### Research success

- the repo documents whether an agent-driven, VIS-native stack can simplify the modern LLM software path without surrendering practical performance
- the repo documents whether the VIS abstraction boundary is the critical enabler of that simplification
- if the answer is negative on a given workload, the repo states which missing kernels, scheduling rules, compiler gaps, VIS limitations, or compatibility requirements caused the miss
