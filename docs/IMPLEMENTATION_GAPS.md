# Implementation Gaps

Date verified: 2026-03-07

## Target facts

- The current remote deployment target is `NVIDIA GB10`.
- The remote machine reports compute capability `12.1`.
- The remote `tileiras` tool exposes `--gpu-name=sm_121`.
- The active first contract is `Qwen3-8B semantics + Qwen3-8B-FP4 artifact + GB10/sm_121`.
- The public remote `cuda.tile 1.1.0` install exposes `float16`, `float32`, `float64`, `bfloat16`, `tfloat32`, `float8_e4m3fn`, and `float8_e5m2`, but no visible public `FP4` or `NVFP4` dtype symbol.

## What `transformers` still provides today

The repo has already removed `device_map="auto"` from the baseline path and moved explicit weight staging and placement into `leanstack`.

Today `transformers` still provides these surfaces:

- semantic-base correctness oracles for dense Qwen execution
- tokenizer and chat-template handling
- config/model-card compatibility surfaces used as reference metadata

That means the current repo already knows how to keep framework heuristics out of the active path. The new blocker is not `device_map` or CPU offload. The new blocker is FP4 compiler feasibility.

## Legacy reference path

The repo still contains a `Qwen3-32B BF16` borrowed runtime loop and a `Qwen3-32B BF16` semantic runtime loop.

Those runs are still useful as reference data because they show:

- explicit semantic ownership is possible
- a full-model loop can run on GB10
- `~2 tokens/s` is too slow to justify a serious framework comparison

But they are no longer the active first target.

## The critical gap to close

The key engineering task is:

1. prove public FP4 compiler feasibility on `sm_121`
2. keep `Qwen3-8B` semantics and `Qwen3-8B-FP4` artifact ownership explicit
3. lower each stable semantic unit into `cuTile -> TileIR -> cubin`
4. inspect PTX and SASS for the hot kernels

## Gap matrix

### 1. FP4 compiler feasibility

- Current:
  - official external sources show Blackwell FP4 support in the broader NVIDIA stack
  - remote inspection of public `cuda.tile 1.1.0` shows visible dtypes only up to FP8
  - `tileiras` already supports `sm_121`
- Target:
  - at least one minimal FP4 or NVFP4 kernel compiles and runs through the public `cuTile`-native path on `sm_121`
- Why this matters:
  - if the compiler gate fails, the active project target is blocked before any runtime optimization matters

### 2. Semantic ownership

- Current:
  - `leanstack` already has a legacy explicit Qwen path that owns weight indexing, shard reads, GPU placement, KV state, and layer semantics
  - that path is still tied to the old `Qwen3-32B BF16` reference work
- Target:
  - `leanstack` owns `Qwen3-8B` semantics and the `Qwen3-8B-FP4` artifact layout without borrowing execution behavior from `transformers`
- Why this matters:
  - the repo already proved the general direction on a legacy path, so the next step is to shrink and retarget the same ownership pattern to the new 8B FP4 contract

### 3. Artifact ownership

- Current:
  - the repo has strong Qwen config and tokenizer handling on the legacy dense path
  - the repo does not yet own the `Qwen3-8B-FP4` tensor and scale contract
- Target:
  - explicit mapping for FP4 linears, higher-precision residual tensors, and any scale metadata needed by the artifact
- Why this matters:
  - the semantic contract and the deployment artifact must be separated before kernels can be specialized correctly

### 4. Runtime residency and KV layout

- Current:
  - the repo already has a legacy page-based KV manager and residency logic shaped around `Qwen3-32B BF16`
- Target:
  - a smaller residency plan and KV contract specialized for `Qwen3-8B-FP4` on GB10
- Why this matters:
  - the new target only makes sense if its smaller shape translates into materially simpler and faster residency behavior

### 5. Kernel catalog

- Current:
  - no FP4 kernel path is proven yet
  - the old dense Qwen path still relies on eager PyTorch math for its active semantics
- Target:
  - a small `Qwen3-8B-FP4` kernel catalog exists for:
    - FP4 linear or GEMM path
    - dequant or scale epilogue where required
    - RMSNorm
    - RoPE
    - GQA prefill and decode
    - gated MLP
    - logits projection
    - sampler
- Why this matters:
  - this is the actual bridge from model semantics and FP4 artifact structure to hardware language

### 6. Benchmark gate

- Current:
  - the benchmark harness exists
  - the legacy `Qwen3-32B` path is too slow to produce a meaningful comparison
- Target:
  - benchmark only after the FP4 compiler gate and first 8B runtime slice are working
- Why this matters:
  - otherwise the comparison measures a legacy reference path, not the active thesis

## Recommended closure order

1. prove or disprove minimal FP4 kernel feasibility on public `cuTile` for `sm_121`
2. own the `Qwen3-8B-FP4` artifact contract
3. port the old Qwen semantic path onto the new 8B FP4 contract
4. package kernels as repeatable `sm_121` artifacts
5. stand up the first runtime loop
6. only then freeze baseline configs for external frameworks

## Compiler-path policy

### Mainline

The mainline should remain:

`Qwen adapter -> cuTile Python DSL -> TileIR/tilebc -> tileiras -> cubin`

This keeps the stack inspectable and still close enough to hardware to expose the real kernel and memory decisions.

### PTX

PTX is a valid escape hatch when the DSL or TileIR surface cannot yet express a needed behavior, especially for a hotspot FP4 kernel on `sm_121`.

But PTX should not become the default authoring layer, for two reasons:

- PTX is still a virtual ISA, so the final hardware mapping is not fully under our control
- Blackwell-family architecture-accelerated features are tied to specific architecture targets, so a PTX-first path can easily miss the exact code shape we want

### SASS

SASS should be treated as a verification and analysis target, not the first implementation target.

That is the pragmatic stance because:

- the public toolchain cleanly supports disassembly and inspection of cubins into SASS
- the public, stable, forward-compatible authoring path is not SASS-first
- direct SASS ownership would sharply increase brittleness across toolkit and architecture revisions

In short:

- `cuTile/TileIR` is the mainline
- `PTX` is a controlled hotspot escape hatch
- `SASS` is the ground-truth artifact we inspect, not the default source language

## The real performance risk

Yes, there is a real possibility that the current public `cuTile/TileIR` stack either:

- does not expose FP4 at all in the authoring surface we need
- or underperforms on decisive FP4 Qwen kernels such as projections and decode attention

That should change the execution strategy, not the thesis:

- prove the compiler surface first
- measure where `cuTile/TileIR` misses on `sm_121`
- use PTX selectively only where the compiler surface is the blocker
- reserve direct SASS work for last-resort research, not first-pass implementation

If the compiler stack misses badly on the decisive kernels, that result is itself part of the research outcome.
