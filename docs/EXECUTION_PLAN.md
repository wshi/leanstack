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

## Phase 1: Establish the precision gate

Deliverables:

- explicit inventory of what public `cuda.tile 1.1.0` exposes on the remote host
- one minimal BF16 compiler-and-run probe
- one minimal FP8 probe for each public float8 dtype
- one explicit FP4 frontend sub-gate
- artifact capture and a written precision decision for `sm_121`

Exit gate:

- the executable precision gate recommends BF16 as the current public-cuTile primary precision on the remote GB10

## Phase 2: Map the model and checkpoint contract

Deliverables:

- semantic contract for `Qwen/Qwen3-1.7B-Base`
- checkpoint contract for the public BF16 snapshot
- explicit tensor mapping for BF16 linears and residual paths
- explicit record of which generic framework features are being deferred as compatibility tax

Exit gate:

- the repo can parse the target BF16 checkpoint contract without leaning on a monolithic runtime

## Phase 3: Stand up the first runtime slice

Target:

- `Qwen3-1.7B-Base` BF16 on `GB10 / sm_121`, with a second-family adapter after the Qwen path is stable

Deliverables:

- BF16-aware weight loader
- tokenizer wiring
- tensor layout adapter
- cache layout adapter
- kernel coverage matrix for BF16 linears, norms, RoPE, and GQA
- Qwen-specific prompt and thinking-mode handling for baseline runs

Exit gate:

- single-request prefill and decode execute for `Qwen3-1.7B-Base` BF16 on the remote machine

## Phase 4: Benchmark Against Framework Baselines

Deliverables:

- benchmark harness for `generated tokens/s`, latency, and memory
- exact-format BF16 baseline configs wherever possible
- `vLLM` and `SGLang` baseline configs when they can run the same or a clearly labeled equivalent format
- secondary deployment reference for `llama.cpp`
- comparative report that records process shape, operational complexity, and compatibility tax

Exit gate:

- a first comparison table exists for a comparable `Qwen3-1.7B-Base` BF16 profile on the same machine, including a go / no-go conclusion

## Phase 5: Minimal Serving Surface

Deliverables:

- minimal OpenAI-compatible API
- streaming token output
- latency and throughput counters
- traceable failure logs

Exit gate:

- remote machine serves `Qwen3-1.7B-Base` BF16 through the new stack after the benchmark contract is stable

## Current blockers to clear

1. Retarget the active runtime from the legacy `Qwen3-32B` work to `Qwen3-1.7B-Base` BF16.
2. Separate `Qwen3-1.7B-Base` semantic ownership from BF16 checkpoint ownership.
3. Define the exact-format BF16 benchmark contract before rebuilding the larger runtime.
4. Keep the legacy `Qwen3-32B BF16` path as reference data only, not as the active optimization target.
