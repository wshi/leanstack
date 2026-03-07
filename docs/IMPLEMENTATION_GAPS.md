# Implementation Gaps

Date verified: 2026-03-07

## Target facts

- The current remote deployment target is `NVIDIA GB10`.
- The remote machine reports compute capability `12.1`.
- The remote `tileiras` tool exposes `--gpu-name=sm_121`.
- The first contract remains `Qwen/Qwen3-32B + GB10/sm_121`.

## What `transformers` still provides today

The repo has already removed `device_map="auto"` from the baseline path and moved explicit weight staging and placement into `leanstack`.

Today `transformers` still provides these surfaces:

- the borrowed reference path built from `Qwen3DecoderLayer`, `Qwen3RotaryEmbedding`, `Qwen3RMSNorm`, and `DynamicCache`
- tokenizer and chat-template handling
- config/model-card compatibility surfaces used as reference metadata

That means the current repo is no longer framework-directed in placement, and the active semantic loop is no longer framework-directed in layer semantics or KV cache behavior. The remaining dependence is mostly reference-mode correctness, tokenizer/config compatibility, and eager PyTorch math.

## The critical gap to close

The key engineering task is not "replace PyTorch everywhere at once."

It is:

1. keep `transformers` only as a correctness oracle
2. pull semantic ownership into the Qwen adapter
3. lower each stable semantic unit into `cuTile -> TileIR -> cubin`
4. inspect PTX and SASS for the hot kernels

## Measured delta on GB10 today

For the same `prompt_tokens=8`, `max_new_tokens=4` probe on the remote GB10 machine:

- borrowed full runtime loop:
  - materialization about `76.7s`
  - runtime-loop throughput about `2.24 tokens/s`
- semantic full runtime loop:
  - materialization about `290.7s`
  - runtime-loop throughput about `1.92 tokens/s`

Interpretation:

- semantic ownership and leanstack-owned KV cache are now proven at full-model scale
- the remaining gap is primarily eager PyTorch math, dense/probe-style cache layout, and layer-by-layer staging
- this is exactly the point where `cuTile/TileIR` kernel work becomes the decisive next step

## Gap matrix

### 1. Semantic ownership

- Current:
  - `leanstack` owns weight indexing, shard reads, and GPU placement.
  - an adapter-owned semantic path now runs end-to-end from layer-0 probes to the full 64-layer runtime loop.
  - the active semantic loop owns RMSNorm, RoPE, GQA attention, MLP, final norm, logits projection, and KV cache control.
  - the borrowed `transformers` path is now a correctness oracle rather than the only working full-model execution path.
- Target:
  - `leanstack` owns RMSNorm, RoPE, QKV projections, GQA attention, MLP, final norm, output projection, and sampler behavior.
- Why this matters:
  - the next blocking issue is no longer semantic ownership itself, but lowering that owned semantic surface into `cuTile/TileIR` kernels

### 2. Full-model residency and layout

- Current:
  - the explicit runtime loop can now materialize all 64 Qwen3-32B layers, final norm, and output head onto GB10 GPU memory
  - the new full semantic loop also materializes all 64 layers and reaches about `65.5 GiB` allocated after materialization on GB10
  - the current materialization path is still a probe-oriented, layer-by-layer staging flow rather than a production residency planner, which is why semantic materialization is still much slower than the borrowed reference path
- Target:
  - a fixed `Qwen3-32B + GB10` residency plan for all 64 layers, plus deterministic KV layout
- Why this matters:
  - performance results are meaningless if residency and movement are still fluid

### 3. KV cache ownership

- Current:
  - a page-based KV manager exists for the semantic block probe and now also drives the active full semantic runtime loop
  - the current implementation still uses a dense preallocated tensor and simple append logic rather than a fully paged indirection/reuse scheme
- Target:
  - a paged KV manager specialized to Qwen GQA geometry on `sm_121`
- Why this matters:
  - decode performance and memory behavior are dominated by KV layout

### 4. Kernel catalog

- Current:
  - the semantic runtime no longer inherits block execution from framework modules
  - all math still runs through eager PyTorch operators such as `F.linear`, `softmax`, and `silu`
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
  - deterministic, single-request, full 64-layer prefill/decode loops now run on the remote machine in both borrowed and semantic modes
  - the semantic mode uses leanstack-owned KV cache state and adapter-owned layer semantics, but is still a script-level loop
- Target:
  - a small runtime loop performs deterministic single-request prefill/decode first, then expands to comparable batching rules for benchmark work
- Why this matters:
  - only then does `tokens/s` comparison against `vLLM` or `SGLang` become meaningful

## Recommended closure order

1. replace the dense semantic KV tensor with a true static page contract and residency planner
2. lower `RMSNorm`, `RoPE`, logits projection, and sampler from eager PyTorch into `cuTile/TileIR`
3. lower GQA prefill/decode and projection kernels
4. package kernels as repeatable `sm_121` artifacts
5. turn the semantic single-request loop into a scheduler-ready runtime surface
6. only then freeze baseline configs for `vLLM` and `SGLang`

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
