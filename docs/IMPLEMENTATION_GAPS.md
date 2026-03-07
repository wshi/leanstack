# Implementation Gaps

Date verified: 2026-03-07

## Target facts

- The current remote deployment target is `NVIDIA GB10`.
- The remote machine reports compute capability `12.1`.
- The remote `tileiras` tool exposes `--gpu-name=sm_121`.
- The active first contract is `Qwen/Qwen3-1.7B-Base` BF16 on `GB10 / sm_121`.
- The executable precision gate currently recommends `bfloat16` as the active public-cuTile precision on `sm_121`.
- The public remote `cuda.tile 1.1.0` install exposes `float16`, `float32`, `float64`, `bfloat16`, `tfloat32`, `float8_e4m3fn`, and `float8_e5m2`.
- The current float8 probe reaches the compiler but fails TileIR verification for both public FP8 dtypes.
- The public Python authoring surface still lacks a complete FP4 or NVFP4 path.

## What `transformers` still provides today

The repo has already removed `device_map="auto"` from the baseline path and moved explicit weight staging and placement into `leanstack`.

Today `transformers` still provides these surfaces:

- semantic-base correctness oracles for dense Qwen execution
- tokenizer and chat-template handling
- config/model-card compatibility surfaces used as reference metadata

That means the current repo already knows how to keep framework heuristics out of the active path. The new blocker is no longer `device_map` or CPU offload. The new blocker is retargeting the runtime from the slow 32B reference path to a benchmarkable 1.7B BF16 path.

## Legacy reference path

The repo still contains a `Qwen3-32B BF16` borrowed runtime loop and a `Qwen3-32B BF16` semantic runtime loop.

Those runs are still useful as reference data because they show:

- explicit semantic ownership is possible
- a full-model loop can run on GB10
- `~2 tokens/s` is too slow to justify a serious framework comparison

But they are no longer the active first target.

## The critical gap to close

The key engineering task is:

1. keep the BF16 precision gate green on `sm_121`
2. keep `Qwen3-1.7B-Base` semantics and BF16 checkpoint ownership explicit
3. lower each stable semantic unit into `cuTile -> TileIR -> cubin`
4. inspect PTX and SASS for the hot kernels

## Gap matrix

### 1. Precision gate discipline

- Current:
  - the executable precision gate returns a real remote result
  - BF16 clears on `sm_121`
  - FP8 fails TileIR verification for the current probe
  - FP4 remains blocked in the public frontend
- Target:
  - BF16 remains the active runtime precision until a narrower precision gate turns positive
- Why this matters:
  - otherwise the project will keep re-planning around blocked precision assumptions instead of shipping a benchmarkable runtime

### 2. Semantic ownership

- Current:
  - `leanstack` already has a legacy explicit Qwen path that owns weight indexing, shard reads, GPU placement, KV state, and layer semantics
  - that path is still tied to the old `Qwen3-32B BF16` reference work
- Target:
  - `leanstack` owns `Qwen3-1.7B-Base` semantics and the BF16 checkpoint layout without borrowing execution behavior from `transformers`
- Why this matters:
  - the repo already proved the general direction on a legacy path, so the next step is to shrink and retarget the same ownership pattern to the new 1.7B BF16 contract

### 3. Checkpoint ownership

- Current:
  - the repo has strong Qwen config and tokenizer handling on the legacy dense path
  - the repo does not yet own the `Qwen3-1.7B-Base` BF16 tensor contract end-to-end
- Target:
  - explicit mapping for BF16 linears, residual tensors, and logits projection
- Why this matters:
  - the semantic contract and the checkpoint contract must be separated before kernels can be specialized correctly

### 4. Runtime residency and KV layout

- Current:
  - the repo already has a legacy page-based KV manager and residency logic shaped around `Qwen3-32B BF16`
- Target:
  - a smaller residency plan and KV contract specialized for `Qwen3-1.7B-Base` BF16 on GB10
- Why this matters:
  - the new target only makes sense if its smaller shape translates into materially simpler and faster residency behavior

### 5. Kernel catalog

- Current:
  - BF16 compiles through the public `cuTile` path for the minimal probe
  - a first executable BF16 hot-kernel bundle now runs on the exact `Qwen3-1.7B-Base` geometry
  - `q_proj`, `o_proj`, `gate_up`, and `rmsnorm` already beat or match the local torch reference on GB10
  - `kv_proj` and `down_proj` are still underperforming
  - mixed tile shapes required tile-shape-specific kernel objects; a single reused generic `ct.kernel` was not safe
  - the old dense Qwen path still relies on eager PyTorch math for its active semantics
- Target:
  - a small `Qwen3-1.7B-Base` BF16 kernel catalog exists for:
    - BF16 linear or GEMM path
    - RMSNorm
    - RoPE
    - GQA prefill and decode
    - gated MLP
    - logits projection
    - sampler
- Why this matters:
  - this is the actual bridge from model semantics and checkpoint structure to hardware language

### 6. Benchmark gate

- Current:
  - the benchmark harness exists
  - the legacy `Qwen3-32B` path is too slow to produce a meaningful comparison
- Target:
  - benchmark only after the BF16 runtime slice exists for the 1.7B target
- Why this matters:
  - otherwise the comparison measures a legacy reference path, not the active thesis

## Recommended closure order

1. own the `Qwen3-1.7B-Base` BF16 checkpoint contract
2. port the old Qwen semantic path onto the new 1.7B BF16 contract
3. lower decisive BF16 kernels into repeatable `sm_121` artifacts on the cuTile path
4. stand up the first runtime loop
5. only then freeze baseline configs for external frameworks

## Compiler-path policy

### Backend policy

The official backend order should be:

1. `cuTile -> TileIR -> cubin`
2. PTX only when diagnosing a compiler miss on the cuTile path

This keeps the stack inspectable and keeps the official comparison path aligned with the project thesis.

### PTX

PTX is a valid escape hatch when the project needs to diagnose why the cuTile path misses a needed hotspot, especially for a future FP8 or FP4 kernel on `sm_121`.

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

- underperforms on decisive BF16 Qwen kernels such as projections and decode attention
- or continues to block the narrower FP8 and FP4 paths we might want later

That should change the execution strategy, not the thesis:

- prove the compiler surface first
- measure where `cuTile/TileIR` misses on `sm_121`
- use PTX selectively only where the compiler surface is the blocker
- reserve direct SASS work for last-resort research, not first-pass implementation

If the compiler stack misses badly on the decisive kernels, that result is itself part of the research outcome.
