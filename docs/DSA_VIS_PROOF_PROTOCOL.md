# DSA VIS Proof Protocol

Date: 2026-03-12

## Purpose

This protocol defines how `leanstack` proves the importance of a virtual instruction set (VIS) on our own DSA, not just on GPU.

The target claim is:

- a stable VIS allows an agent to generate, optimize, and retarget operator code fast enough to beat compatibility-heavy stacks on both performance and software cost.

## Why this protocol exists

A throughput win alone is not enough for the DSA thesis.

For DSA, we must show three things together:

1. **Performance**: operator-first VIS-native serving can beat framework-style compatibility stacks.
2. **Regeneration cost**: agent token budget to synthesize and retarget kernels is bounded and practical.
3. **Portability**: when hardware revision changes, VIS-targeting code recompiles/re-tunes instead of being rewritten.

Without (2) and (3), the VIS argument collapses into a one-off optimization project.

## Hard constraints

1. Official serving comparisons must run through packed appliance mode (`leanpack + leanserve`), not checkpoint fallback mode.
2. Critical-path operators must be agent-owned source code targeting VIS (Tile IR in Phase 1, our VIS in Phase 2).
3. `vLLM`, `SGLang`, `llama.cpp`, TensorRT-style runtimes are baselines only.
4. Every official result must include both runtime metrics and agent synthesis cost metrics.
5. Every major hardware or compiler target change must record code churn and retarget latency.
6. Official claim-bearing runs must be tagged as `fixed-contract` and keep model/precision/profile constant.

## Falsifiable claims

### Claim A: VIS improves performance economics

Given a fixed model-chip contract:

- VIS-native appliance should reach at least `1.30x` warmed framework throughput on the primary profile.
- If this fails, document whether the blocker is kernel quality, memory system limits, scheduler limits, or VIS/toolchain gaps.

### Claim B: VIS reduces regeneration cost

Given a model update or bucket-contract update:

- agent token spend to regenerate operator set and serving artifact should remain bounded.
- manual framework-plumbing work should be strictly less than compatibility-first stacks.

### Claim C: VIS enables retargetability

Given a hardware target revision:

- operator sources should remain stable at VIS level.
- work should concentrate in compiler/backend retuning and launch metadata, not source rewrites.

## Measurement contract

All official runs must record:

- profile id (`decode_64_256`, etc.)
- model contract id
- hardware target id
- VIS target version
- throughput (`runtime tok/s`)
- latency (`TTFT`, decode latency)
- memory footprint
- operator inventory coverage ratio
- agent token budget for the iteration
- code churn by layer:
  - VIS-level kernel source changes
  - runtime glue changes
  - benchmark/control-plane changes

## Experiment matrix

For each model-chip contract, run:

1. **Framework baseline**
   - `vLLM` or `SGLang` on the same model/profile/hardware when possible.
2. **Appliance baseline**
   - packed `leanstack` without new VIS-native fused operators.
3. **VIS-native candidate**
   - agent-generated operator path (with fusion where applicable) through official VIS backend.

The VIS thesis is considered strengthened only if (3) improves both:

- runtime metrics (throughput/latency), and
- software economics (token budget + reduced compatibility surface).

## Phase alignment

### Phase 1 (GPU / Tile IR)

Goal:

- establish existence proof that VIS-native operator-first stack can beat compatibility-heavy stack under fixed contract.

Exit artifacts:

- operator coverage table
- benchmark table with official profiles
- agent token budget ledger
- retarget log across at least one target/toolchain revision

### Phase 2 (our DSA / our VIS)

Goal:

- reproduce Phase 1 methodology using our own VIS and our own compiler/runtime backend.

Exit artifacts:

- VIS specification and versioning policy
- VIS-native operator catalog for first DSA model contract
- throughput and software-cost comparison vs compatibility-style stack on DSA
- retargetability evidence across DSA revisions

## Immediate repo actions

1. Keep official benchmark path strict-packed (`leanpack` manifest required).
2. Keep compare UI and benchmark harness on packed appliance by default.
3. Continue replacing eager fallback operators in critical path with VIS-native kernels.
4. Add operator-level fusion milestones to benchmark protocol and gap registry.
5. Add token-budget accounting to official benchmark reports.

## Go / no-go policy

Go:

- if VIS-native operator-first path shows clear throughput and regeneration-cost advantages.

No-go:

- if repeated attempts only improve via framework-like complexity growth or fail to show retargetability.

In no-go cases, document exactly which abstraction boundary failed:

- VIS expressiveness
- compiler/backend maturity
- runtime scheduling model
- memory hierarchy mismatch
