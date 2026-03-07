# Execution Plan

Date: 2026-03-07

## Phase 0: Ground the environment

Deliverables:

- fixed remote workspace under `/home/pto/lean`
- repeatable sync and validation scripts
- environment probe for CUDA, cuTile, Python, and model-runtime dependencies
- initial statement of the compatibility costs being intentionally deferred

Exit gate:

- `scripts/remote_verify.sh` produces bytecode, TileIR dumps, cubin, and SASS artifacts on the remote host

## Phase 1: Prove FP4 compiler feasibility

Deliverables:

- explicit inventory of what public `cuda.tile 1.1.0` exposes on the remote host
- one minimal FP4 or NVFP4 GEMM or linear kernel attempt
- artifact capture for `TileIR`, `cubin`, and `SASS` on `sm_121`
- a written decision on whether the public cuTile-native path is viable, needs PTX help, or is blocked

Exit gate:

- a minimal FP4 kernel compiles to cubin and runs on the remote GB10, or the repo records a precise compiler-surface blocker

## Phase 2: Map the model and artifact contract

Deliverables:

- semantic contract for `Qwen/Qwen3-8B`
- artifact contract for `nvidia/Qwen3-8B-FP4`
- explicit tensor and scale mapping for FP4 linears
- explicit record of which generic framework features are being deferred as compatibility tax

Exit gate:

- the repo can parse the target FP4 artifact and its scales without leaning on a monolithic runtime

## Phase 3: Stand up the first runtime slice

Target:

- `Qwen3-8B semantics + FP4 artifact` on `GB10 / sm_121`, with a second-family adapter after the Qwen path is stable

Deliverables:

- FP4-aware weight loader
- tokenizer wiring
- tensor layout adapter
- cache layout adapter
- kernel coverage matrix for FP4 linears, norms, RoPE, and GQA
- Qwen-specific prompt and thinking-mode handling for baseline runs

Exit gate:

- single-request prefill and decode execute for `Qwen3-8B-FP4` on the remote machine

## Phase 4: Benchmark Against Framework Baselines

Deliverables:

- benchmark harness for `generated tokens/s`, latency, and memory
- exact-format baseline configs wherever possible
- `vLLM` and `SGLang` baseline configs when they can run the same or a clearly labeled equivalent format
- secondary deployment reference for `llama.cpp`
- comparative report that records process shape, operational complexity, and compatibility tax

Exit gate:

- a first comparison table exists for a comparable `Qwen3-8B-FP4` profile on the same machine, including a go / no-go conclusion

## Phase 5: Minimal Serving Surface

Deliverables:

- minimal OpenAI-compatible API
- streaming token output
- latency and throughput counters
- traceable failure logs

Exit gate:

- remote machine serves `Qwen3-8B-FP4` through the new stack after the benchmark contract is stable

## Current blockers to clear

1. Prove whether the public `cuTile` Python surface can express any credible FP4 kernel at all on `sm_121`.
2. Separate `Qwen3-8B` semantic ownership from `Qwen3-8B-FP4` artifact ownership.
3. Define the exact-format benchmark contract before rebuilding the larger runtime.
4. Keep the legacy `Qwen3-32B BF16` path as reference data only, not as the active optimization target.
