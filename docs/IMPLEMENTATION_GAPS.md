# Implementation Gaps

Date verified: 2026-03-07

## Target facts

- The current remote deployment target is `NVIDIA GB10`.
- The remote machine reports compute capability `12.1`.
- The remote `tileiras` tool exposes `--gpu-name=sm_121`.
- The first contract remains `Qwen/Qwen3-32B + GB10/sm_121`.

## What `transformers` still provides today

The repo has already removed `device_map="auto"` from the baseline path and moved explicit weight staging and placement into `leanstack`, but the explicit probes still borrow important execution semantics from `transformers`:

- `Qwen3DecoderLayer`
- `Qwen3RotaryEmbedding`
- `Qwen3RMSNorm`
- `DynamicCache`
- tokenizer and chat-template handling

That means the current repo is no longer framework-directed in placement, but it is still framework-assisted in operator semantics.

## The critical gap to close

The key engineering task is not "replace PyTorch everywhere at once."

It is:

1. keep `transformers` only as a correctness oracle
2. pull semantic ownership into the Qwen adapter
3. lower each stable semantic unit into `cuTile -> TileIR -> cubin`
4. inspect PTX and SASS for the hot kernels

## Gap matrix

### 1. Semantic ownership

- Current:
  - `leanstack` owns weight indexing, shard reads, and GPU placement.
  - `transformers` still owns layer semantics.
- Target:
  - `leanstack` owns RMSNorm, RoPE, QKV projections, GQA attention, MLP, final norm, output projection, and sampler behavior.
- Why this matters:
  - kernel replacement is blocked until semantic ownership leaves `transformers`

### 2. Full-model residency and layout

- Current:
  - layer probes materialize tensors explicitly but only in probe-sized slices
- Target:
  - a fixed `Qwen3-32B + GB10` residency plan for all 64 layers, plus deterministic KV layout
- Why this matters:
  - performance results are meaningless if residency and movement are still fluid

### 3. KV cache ownership

- Current:
  - the explicit stack still uses `DynamicCache`
- Target:
  - a paged KV manager specialized to Qwen GQA geometry on `sm_121`
- Why this matters:
  - decode performance and memory behavior are dominated by KV layout

### 4. Kernel catalog

- Current:
  - explicit probes still inherit math execution from framework modules
- Target:
  - a small catalog of Qwen-specific kernels:
    - RMSNorm
    - RoPE
    - QKV projection
    - output projection
    - GQA prefill attention
    - GQA decode attention
    - gated MLP
    - final norm
    - logits projection
    - sampler
- Why this matters:
  - this is the actual bridge from model semantics to hardware language

### 5. Compiler packaging

- Current:
  - cuTile smoke and artifact capture exist, but the Qwen path is not yet backed by a kernel bundle
- Target:
  - per-kernel `sm_121` compilation manifests with TileIR, cubin, and SASS artifacts
- Why this matters:
  - reproducibility is required before optimization claims are credible

### 6. Runtime loop

- Current:
  - probes validate one forward / prefill / decode path
- Target:
  - a deterministic full 64-layer prefill/decode loop with explicit stop handling
- Why this matters:
  - only then does `tokens/s` comparison against `vLLM` or `SGLang` become meaningful

## Recommended closure order

1. replace `DynamicCache` with a static KV page contract
2. split the borrowed Qwen block into adapter-owned sub-operators
3. bring up `RMSNorm`, `RoPE`, and sampler in cuTile first
4. bring up GQA prefill/decode kernels
5. package kernels as repeatable `sm_121` artifacts
6. extend the explicit stack from 2 layers to all 64 layers

## Compiler-path policy

### Mainline

The mainline should remain:

`Qwen adapter -> cuTile Python DSL -> TileIR/tilebc -> tileiras -> cubin`

This keeps the stack inspectable and still close enough to hardware to expose the real kernel and memory decisions.

### PTX

PTX is a valid escape hatch when the DSL or TileIR surface cannot yet express a needed behavior, especially for a hotspot kernel on `sm_121`.

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

Yes, there is a real possibility that the current `cuTile/TileIR` stack underperforms on some Qwen kernels, especially GQA decode and large projection kernels.

That should change the execution strategy, not the thesis:

- prove the semantic path first
- measure where `cuTile/TileIR` misses on `sm_121`
- use PTX selectively only where the compiler surface is the blocker
- reserve direct SASS work for last-resort research, not first-pass implementation

If the compiler stack misses badly on the decisive kernels, that result is itself part of the research outcome.
