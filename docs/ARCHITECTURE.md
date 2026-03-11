# Architecture

Date: 2026-03-11

## Goal

Build a narrow, auditable inference stack for `Qwen/Qwen3-1.7B-Base` BF16 on `GB10 / sm_121` that is centered on three moving parts and explicitly avoids paying broad compatibility costs up front:

1. `compiler`: lowers model-level ops to cuTile and TileIR-backed kernels.
2. `runtime`: owns request batching, KV cache placement, and kernel dispatch.
3. `serve`: exposes a minimal front end only after the execution and benchmark story is stable.

## Non-goals

- Reproducing vLLM surface area in the first phase.
- Supporting every model family at once before the first `Qwen3-1.7B-Base` BF16 contract is stable.
- Supporting every hardware backend behind one generic abstraction.
- Hiding compiler behavior behind framework magic.
- Embedding `vLLM`, `SGLang`, `llama.cpp`, or another inference runtime inside the core serving path.

## Design rules

### 1. Compiler-first kernels

Every performance-critical path must have a traceable lowering chain:

`model op -> adapter selection -> cuTile kernel -> TileIR -> cubin -> SASS`

If a path cannot be inspected at this level, it does not belong in the core stack.

The current remote compiler target should be treated concretely as `GB10 / sm_121`, not as an abstract Blackwell placeholder.

That also implies a strict backend order:

- `cuTile/TileIR` is the official authoring path
- `PTX` is a diagnostic escape hatch when the project needs to understand a compiler miss
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

`Is this required for the Qwen3-1.7B-Base BF16 + GB10 contract, or is it a compatibility tax?`

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

- semantic contract: `Qwen/Qwen3-1.7B-Base`
- active deployment contract: the public BF16 checkpoint for `Qwen/Qwen3-1.7B-Base`
- dense transformer blocks
- GQA with 16 query heads and 8 KV heads
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

- `Qwen/Qwen3-1.7B-Base` defines geometry and prompt semantics
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
4. adapter contract for `Qwen3-1.7B-Base` semantics and BF16 checkpoint layout
5. minimal runtime that can execute one prefill and one decode loop
6. benchmark comparison against framework baselines and their compatibility-heavy process shape
7. API layer only after the execution path is stable

This keeps the stack understandable while still targeting a full model run.

## Long-term architecture principles

The following principles apply to both Phase 1 (GPU / Tile IR) and Phase 2 (custom silicon / our own VIS). They are intended to shape design decisions today so that Phase 2 does not require a rewrite of everything built in Phase 1.

### 1. The model-chip contract schema is a first-class artifact

The current `Qwen3-1.7B-Base BF16 + GB10/sm_121` contract is the first instance of a general pattern. That pattern should be formalized:

- **model geometry**: layers, hidden size, heads, GQA config, head dimension, vocabulary size, normalization type, activation type
- **chip capability**: sm version (or our VIS target version), tensor-core generation, memory bandwidth, compute peak per precision, memory capacity
- **derived contract**: kernel inventory, dispatch policy, precision policy, KV cache layout, prompt-bucket manifest, leanpack artifact structure

The schema is not just documentation. It is the input to the agent-synthesis pipeline: given a new model-chip pair, the agent should be able to read the contract schema and generate the kernel inventory, dispatch policy, and leanpack artifact. The schema should be machine-readable and versioned.

In Phase 1, the schema is implicit in `model_registry.py` and `manifest.json`. The path toward M2 requires making it explicit and formalizable.

### 2. Agent token budget is an explicit engineering metric

Alongside throughput, latency, and memory, the project tracks agent token budget as a first-class metric:

- **bring-up cost**: how many agent tokens (prompt + response + iteration) does it take to bring up a new model-chip pair from the contract schema to a running leanpack + leanserve appliance?
- **maintenance cost**: how many agent tokens does it take to update an existing appliance when the model or chip contract changes?
- **comparison baseline**: how much compatibility-oriented software (measured in lines of code, dependency count, configuration surface) does the agent token budget replace?

The economic thesis of leanstack is that agent cost < compatibility cost for narrow appliances. This metric is the primary evidence for or against that claim.

### 3. Leanpack as a potentially multi-target artifact

The current `leanpack/v0` format targets `sm_121` only. The format should be designed so that a future version can target multiple backends:

- `sm_121` (Blackwell, Tile IR)
- a future sm generation (Tile IR, recompiled)
- our own chip (our own VIS)

This does not mean leanpack must support all targets today. It means:

- the manifest schema should include an explicit `target` field that distinguishes backend-specific sections from backend-agnostic sections
- packed weight tensors (which are precision-specific but not ISA-specific) should be factored separately from compiled kernel artifacts (which are ISA-specific)
- the leanserve loader should be organized so that target-specific dispatch is a plugin, not a hardcoded assumption

The current `leanpack/v0` layout already separates weights from metadata. The next step is to make the separation between "model artifact" and "target artifact" explicit in the manifest schema.

### 4. VIS portability as a factoring requirement

Components that are VIS-agnostic should be factored to remain so. Components that are VIS-specific should be clearly marked.

**VIS-agnostic** (should remain portable across Tile IR and our future VIS):

- model-chip contract schema and its parser
- leanpack manifest schema and weight packing logic
- leanserve request admission, scheduling, and KV cache management
- benchmark harness and comparison protocol
- agent-synthesis pipeline orchestration (prompt construction, benchmark evaluation, promotion logic)

**VIS-specific** (expected to differ per target):

- kernel source code (cuTile source for Tile IR; our language for our VIS)
- compiled artifacts (cubin for sm_121; our binary format for our chip)
- kernel dispatch and launch mechanics
- memory layout optimizations that depend on physical hardware topology

The factoring boundary should be: everything above "launch this kernel with these arguments on these buffers" is VIS-agnostic. Everything at and below that line is VIS-specific.

This factoring is not a premature abstraction exercise. It is a constraint that prevents Phase 1 engineering from accidentally embedding sm_121 assumptions into layers that should be reusable in Phase 2.

### 5. Cross-generation portability as a testable property

Tile IR's value proposition is that it provides cross-generation portability: code targeting Tile IR should recompile without reauthoring when the physical sm target changes. This is the analog of what PTX proved across GPU generations.

leanstack should be the public evidence that Tile IR delivers the same promise for tensor-core workloads. To test this:

- leanpack format versioning should be aligned to Tile IR specification versioning
- when the next Blackwell-successor sm target appears, leanstack kernels should recompile without source changes
- if recompilation fails or produces regressions, the failure should be documented precisely: which Tile IR operations broke, which kernel patterns required reauthoring, and what the cost was

This property is not testable today (only one sm target exists). But the architecture should be organized so that it becomes testable as soon as a second target appears.
