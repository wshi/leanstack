# Execution Plan

Date: 2026-03-06

## Phase 0: Ground the environment

Deliverables:

- fixed remote workspace under `/home/pto/leanstack`
- repeatable sync and validation scripts
- environment probe for CUDA, cuTile, Python, and model-runtime dependencies

Exit gate:

- `scripts/remote_verify.sh` produces bytecode, TileIR dumps, cubin, and SASS artifacts on the remote host

## Phase 1: Build a compiler-backed kernel catalog

Deliverables:

- vector add smoke path
- RMSNorm kernel
- rotary embedding kernel
- paged attention microkernel plan
- MLP fusion plan for GLM-family blocks

Exit gate:

- each kernel has a cuTile source, a remote validation command, and captured artifacts

## Phase 2: Stand up the runtime spine

Deliverables:

- block manager
- request queue
- prefill/decode scheduler
- execution graph with explicit kernel dispatch

Exit gate:

- a synthetic token generation loop runs without the serving API

## Phase 3: Add the first model adapter

Target:

- a recent GLM-family checkpoint verified against official sources before selection

Deliverables:

- weight loader
- tokenizer wiring
- tensor layout adapter
- cache layout adapter
- kernel coverage matrix

Exit gate:

- single-request prefill and decode execute on the remote machine

## Phase 4: End-to-end serving

Deliverables:

- minimal OpenAI-compatible API
- streaming token output
- latency and throughput counters
- traceable failure logs

Exit gate:

- remote machine serves one GLM-family checkpoint through the new stack

## Phase 5: Tighten performance

Deliverables:

- scheduling policy revisions
- kernel fusion expansions
- memory planning cleanup
- benchmarking against a small baseline

Exit gate:

- the runtime remains smaller while improving real throughput

## Current blockers to clear

1. Install and verify a dedicated remote runtime environment with `torch`, `transformers`, `safetensors`, and `sentencepiece`.
2. Confirm the target GLM-family checkpoint and its loading requirements from primary sources.
3. Resolve remote access to model artifacts. As of 2026-03-06, direct `curl` access to Hugging Face model files times out on the DGX Spark machine.
4. Translate the first transformer block into explicit kernel requirements instead of importing a monolithic runtime.
