# Roadmap

Date: 2026-03-11

This document captures the Phase 1 milestones and Phase 2 design principles as a living reference. Each milestone has a core question, success criteria, and the measurement that would falsify it.

## Phase 1 — GPU: Validate Tile IR as an Agentic-Era VIS

### M1 — Appliance Proof (active)

**Core question:** On a single fixed model-chip pair, can a Tile IR-native appliance outperform a compatibility-first framework by a decisive margin?

**Fixed contract:**

- Model: `Qwen/Qwen3-1.7B-Base` BF16
- Hardware: GB10 / sm_121
- Execution path: `cuTile -> TileIR -> cubin -> SASS`
- Build artifacts: `leanpack` + `leanserve`
- Comparison baseline: warmed vLLM on the same machine, same checkpoint, same prompt bucket

**Success criteria:**

- `>= 1.30x` warmed vLLM on `decode_64_256` for the **fixed-contract official path**
- leanpack + leanserve runs end to end from packed artifact on GB10
- critical path is inspectable down to TileIR and SASS
- official benchmark rows are explicitly labeled as `fixed-contract` or `exploratory`
- exact speculative decode preserves output-equivalence with the base model when used

**Falsification:** If, after packed weights + resident appliance + decisive cuTile decode kernels + speculative decode, the stack cannot exceed warmed vLLM by 30% on the fixed contract, then the compatibility tax is not the dominant bottleneck for this workload. The remaining bottleneck is kernel maturity or algorithmic efficiency.

**Current status (2026-03-12):**

- fixed-contract packed appliance path is near warmed-vLLM parity, but not yet a decisive `>= 1.30x` lead
- exploratory dual-model speculative runs can exceed `+30%` on selected prompts
- those exploratory results are useful for direction finding but are not yet sufficient as the primary VIS-on-DSA proof line

**Remaining M1 work:**

- Document which compatibility layers vLLM/SGLang carry that leanstack does not (the "compatibility tax inventory")
- Move decisive cuTile kernels from microbenchmarks into the appliance hot path (kv_proj, down_proj, logits are still on eager PyTorch)
- Evaluate the +30% result stability across diverse text types (creative/high-entropy text currently shows only +2.6%)
- Record agent token budget spent on the Qwen/GB10 bring-up
- Keep compare harness and UI locked to the fixed official contract for claim-bearing results

### M2 — Synthesis Generalization

**Core question:** Can the agent-synthesis pipeline bring up a new model-chip pair from a contract schema to a running appliance in bounded time and token budget?

**Deliverables:**

- Formalize a `model-chip contract` schema: given model geometry (layers, hidden size, heads, GQA config) + chip capability (sm version, tensor-core generation, memory bandwidth), generate the kernel inventory, dispatch policy, and leanpack artifact automatically
- Agent-driven kernel synthesis pipeline: for a new model-chip pair, agent synthesizes, benchmarks, and promotes cuTile kernels with bounded token budget
- A new model family (e.g., GLM or Llama architecture) on GB10 reaches a credible first bring-up

**Success criteria:**

- A new model family reaches first bring-up in < 1 agent-day (measured in wall-clock time and token budget)
- The contract schema is machine-readable and versioned
- The agent token budget spent vs. compatibility software eliminated is quantified — this is the core economic measurement

**Falsification:** If the agent token budget for a second model family is comparable to or larger than the engineering cost of adding the same model to an existing framework, then the economic thesis does not hold at the model-family level. The agentic approach only works if narrow synthesis is genuinely cheaper than broad compatibility for each new target.

**Measurement:**

- Agent tokens (prompt + response + iterations) to reach first correct output on the new model
- Agent tokens to reach performance parity with vLLM on the new model
- Lines of leanstack code added vs. lines of framework code that would be needed for equivalent support
- Number of cuTile kernels that transferred unchanged vs. required reauthoring

### M3 — Tile IR Cross-Generation Portability

**Core question:** Does the Tile IR abstraction boundary hold across GPU generations — i.e., can leanstack kernels recompile without reauthoring when the sm target changes?

**Deliverables:**

- leanpack format versioned and aligned to Tile IR specification versioning
- When the next Blackwell-successor sm target appears, leanstack kernels recompile without source changes
- Demonstration that the Tile IR abstraction boundary holds across at least two sm generations

**Success criteria:**

- Kernels authored for sm_121 compile for the next sm target without source modifications
- Performance on the new target is within a reasonable range (no catastrophic regression) after recompilation
- The leanpack artifact can be regenerated for the new target by changing only the target field in the contract schema

**Falsification:** If more than a small fraction of cuTile kernel sources require modification for the new sm target, then Tile IR does not deliver the cross-generation portability that PTX delivered for thread-level code. This would mean the VIS is too leaky or too narrowly tuned to a single generation.

**Measurement:**

- Fraction of cuTile kernel sources that recompile unchanged
- Fraction that require source modifications, and the nature of each modification
- Performance ratio (new target / old target) after recompilation vs. after reauthoring
- Agent token budget to port the full appliance to the new target

**Note:** M3 is not testable until a second sm target is available. The architecture should be organized so that the test becomes possible as soon as hardware arrives.

## Phase 2 — Custom Silicon: Reproduce the VIS Moat with Our Own Instruction Set

Phase 2 is not yet in active execution. The following are design principles, not implementation plans.

### Core question

Given everything learned in Phase 1, can the same agentic-synthesis methodology be applied to build a Tile-IR-equivalent virtual instruction set for our own chip, and demonstrate that it becomes the economic foundation of our hardware ecosystem?

### Why this matters

The lesson from IBM S/360, NVIDIA PTX, and now Tile IR is identical: the hardware vendor that owns a stable, well-abstracted virtual instruction set controls the software investment lock-in for their entire ecosystem. leanstack Phase 1 proves the methodology on NVIDIA hardware. Phase 2 applies it to our own chip to establish our own moat.

### Design principles

**VIS specification:**

- Define our own tile-based virtual instruction set as a stable specification, versioned independently of physical chip generations
- The VIS must be: tile-native (not thread-native), portable across our chip generations, expressive enough to cover attention, linear, and normalization kernels without falling back to scalar paths
- The VIS specification should be informed by Phase 1 learnings about which Tile IR abstractions were essential and which were unnecessary

**Agent pipeline retargeting:**

- The agent-synthesis pipeline built in Phase 1 should be retargetable to our VIS as a backend
- The model-chip contract schema should accept our chip's capability description as a first-class input
- Kernel synthesis prompts should be parameterized by VIS, not hardcoded to cuTile

**Leanpack generalization:**

- leanpack format should be generalized to support our chip as a first-class target alongside sm_121
- Packed weight tensors (precision-specific, ISA-agnostic) should be shared across targets
- Compiled kernel artifacts (ISA-specific) should be factored into target-specific sections

**Economic reproduction:**

- The economic argument must be reproducible: measure agent token budget to bring up a new model on our chip vs. compatibility software cost for a generic runtime on the same chip
- The comparison must be honest: if our chip requires more compatibility software than GPU (e.g., due to immature toolchain), that should be documented, not hidden

### Success criteria (to be refined when Phase 2 becomes active)

- Our own VIS specification exists as a stable, versioned document
- The agent-synthesis pipeline can target our VIS as a backend
- A first model runs on our chip through our VIS with measured performance and recorded agent token budget
- The agent token budget for our chip is comparable to or better than the budget for a new GPU sm target (M3 comparison)
- The VIS abstraction holds across at least two revisions of our chip

### Falsification

If the agent-synthesis approach requires more engineering effort on our chip than a traditional compatibility-first runtime, then the VIS moat argument does not generalize from GPU to custom silicon. The failure mode to document:

- Was the VIS specification too ambitious (tried to cover too much)?
- Was the VIS specification too narrow (required too many escape hatches)?
- Was the toolchain maturity gap the real barrier, not the VIS design?
- Was the agent-synthesis pipeline too coupled to cuTile/Tile IR specifics to retarget?

## Milestone dependencies

```
M1 (Appliance Proof)
 ├── validates: throughput thesis, appliance architecture
 └── produces: leanpack/leanserve, comparison protocol, first contract schema
      │
M2 (Synthesis Generalization)
 ├── requires: M1 contract schema, working leanpack pipeline
 ├── validates: economic thesis (agent cost < compatibility cost)
 └── produces: formalized contract schema, agent-synthesis pipeline
      │
M3 (Cross-Generation Portability)
 ├── requires: M1 kernels, M2 contract schema, new sm hardware
 ├── validates: VIS portability thesis
 └── produces: portability evidence, leanpack versioning policy
      │
Phase 2 (Custom Silicon)
 ├── requires: M2 agent pipeline, M3 portability evidence
 ├── validates: VIS moat thesis on our own hardware
 └── produces: our VIS specification, retargeted pipeline, economic comparison
```

## Living document policy

This roadmap is updated when:

- a milestone's success criteria are met or falsified
- a new measurement changes the feasibility assessment of a future milestone
- the hardware or compiler landscape changes in a way that affects the roadmap

Updates should not change the core questions or falsification criteria of existing milestones. If those need to change, the change should be documented as a pivot with rationale.
