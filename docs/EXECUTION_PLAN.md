# Execution Plan

Date: 2026-03-06

## Phase 0: Ground the environment

Deliverables:

- fixed remote workspace under `/home/pto/lean`
- repeatable sync and validation scripts
- environment probe for CUDA, cuTile, Python, and model-runtime dependencies
- initial statement of the compatibility costs being intentionally deferred

Exit gate:

- `scripts/remote_verify.sh` produces bytecode, TileIR dumps, cubin, and SASS artifacts on the remote host

## Phase 1: Build a compiler-backed kernel catalog

Deliverables:

- vector add smoke path
- RMSNorm kernel
- rotary embedding kernel
- GQA paged-attention microkernel plan for `Qwen/Qwen3-32B`
- SiLU-gated MLP fusion plan for `Qwen/Qwen3-32B` blocks
- static execution assumptions for the first model-chip contract

Exit gate:

- each kernel has a cuTile source, a remote validation command, and captured artifacts

## Phase 2: Stand up the runtime spine

Deliverables:

- block manager
- request queue
- prefill/decode scheduler
- execution graph with explicit kernel dispatch
- explicit record of which generic framework features are being deferred as compatibility tax
- explicit removal plan for automatic placement and CPU-offload uncertainty in the first path

Exit gate:

- a synthetic token generation loop runs without the serving API

## Phase 3: Add the first model adapter

Target:

- `Qwen/Qwen3-32B` on Blackwell as the first target, with a second-family adapter after the Qwen path is stable

Deliverables:

- weight loader
- tokenizer wiring
- tensor layout adapter
- cache layout adapter
- kernel coverage matrix
- Qwen-specific prompt and thinking-mode handling for baseline runs

Exit gate:

- single-request prefill and decode execute for `Qwen/Qwen3-32B` on the remote machine

## Phase 4: Benchmark Against Framework Baselines

Deliverables:

- benchmark harness for `generated tokens/s`, latency, and memory
- official-baseline configs for `vLLM` and `SGLang`
- secondary deployment reference for `llama.cpp`
- comparative report that records process shape, operational complexity, and compatibility tax

Exit gate:

- `leanstack`, `vLLM`, and `SGLang` all run a comparable `Qwen/Qwen3-32B` profile on the same machine and a first comparison table is recorded

## Phase 5: Minimal Serving Surface

Deliverables:

- minimal OpenAI-compatible API
- streaming token output
- latency and throughput counters
- traceable failure logs

Exit gate:

- remote machine serves `Qwen/Qwen3-32B` through the new stack after the benchmark contract is stable

## Current blockers to clear

1. Replace the current dense semantic KV tensor and probe-style residency flow with a fixed `Qwen3-32B + GB10` page/layout contract.
2. Lower the working semantic full-model loop from eager PyTorch operators into explicit `cuTile/TileIR` kernel requirements.
3. Prepare official baseline configurations for `vLLM` and `SGLang` on the same machine and model profile.
4. Define the first benchmark table format, including software-complexity and agent-cost proxies, before adding a larger serve surface.
