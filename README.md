# leanstack

`leanstack` is a clean-slate, cuTile-native, Blackwell-first LLM inference stack.

The name is intentional:

- `lean` emphasizes compressing the deployment and serving stack to the smallest system that still remains credible.
- `stack` keeps the focus on the whole software path: compiler, runtime, and serving edge.

## Why this repo exists

The current serving ecosystem pays a large compatibility tax: too many layers, too much genericity, and too much hidden state for workloads that ultimately run one concrete model on one concrete machine.

This repo explores a different path with five constraints:

1. The execution path must stay explicit down to `cuTile -> TileIR -> cubin -> SASS`.
2. The runtime must stay small enough that an agent can regenerate and modify it cheaply.
3. The first contract is a model-chip pair, `Qwen3-8B semantics + FP4 deployment artifact` on `GB10 / sm_121`, not "all models on all hardware."
4. `vLLM`, `SGLang`, `llama.cpp`, and similar systems are compatibility-heavy baselines to compare against, not runtime dependencies.
5. Remote validation on the DGX Spark machine is part of the development loop, not an afterthought.

In the strongest form of this thesis, the only meaningful dynamic input should be the user request payload. Model geometry, chip target, quantization policy, memory layout, kernel inventory, and dispatch policy should all be fixed by the `Qwen3-8B-FP4 + GB10` contract.

## Scope

Stage 0 in this repo does five concrete things:

1. Defines the project thesis for an agent-built, model-chip-specific LLM stack.
2. Provides local and remote tooling to validate the cuTile -> TileIR -> cubin -> SASS path.
3. Separates `Qwen/Qwen3-8B` semantic ownership from the `nvidia/Qwen3-8B-FP4` deployment artifact.
4. Makes FP4 compiler feasibility on `GB10 / sm_121` the first hard gate before more runtime work.
5. Defines the benchmark contract against `vLLM`, `SGLang`, and other external baselines, including both runtime efficiency and software-stack complexity.

## Repository layout

- `docs/PROJECT_THESIS.md`: project thesis and hard constraints.
- `docs/ARCHITECTURE.md`: stack boundaries and replacement strategy.
- `docs/BENCHMARK_PLAN.md`: benchmark methodology and comparison rules.
- `docs/EXECUTION_PLAN.md`: phased build plan and verification gates.
- `docs/IMPLEMENTATION_GAPS.md`: structured gap analysis from borrowed `transformers` semantics to adapter-owned `cuTile/TileIR` kernels.
- `docs/FP4_COMPILER_GATE.md`: the current evidence and gate criteria for a public cuTile-native FP4 path on `sm_121`.
- `docs/MODEL_FIT_ANALYSIS.md`: frontier-model architecture fit versus cuTile complexity.
- `docs/MODEL_TARGETS.md`: verified model targets and hardware-fit decisions.
- `docs/REFERENCES.md`: external research, framework, model, and hardware references.
- `docs/REMOTE_VALIDATION.md`: remote workflow and artifact layout.
- `experiments/cutile/vector_add.py`: known-good cuTile smoke kernel.
- `experiments/models/hf_causal_lm_smoke.py`: baseline Hugging Face causal LM smoke path.
- `experiments/models/qwen_explicit_block_probe.py`: explicit Qwen loader and layer-0 block/prefill/decode probe.
- `experiments/models/qwen_explicit_stack_probe.py`: explicit multi-layer Qwen stack probe.
- `experiments/models/qwen_explicit_runtime_loop.py`: explicit full-model runtime loop with greedy decode accounting.
- `experiments/models/qwen_semantic_block_probe.py`: adapter-owned Qwen block semantics plus page-based KV cache probe.
- `scripts/remote_*.sh`: remote bootstrap, sync, relay, probe, install, and smoke scripts.
- `src/leanstack/`: Python control plane, planning, and repo utilities.
- `skills/leanstack/`: English Codex skill for operating the stack.

## Quick start

From `/Users/wei/work/spark/leanstack`:

```bash
PYTHONPATH=src python3 -m leanstack.cli show-plan
PYTHONPATH=src python3 -m leanstack.cli remote-env
PYTHONPATH=src python3 -m leanstack.cli show-contract --model qwen
PYTHONPATH=src python3 -m leanstack.cli show-gaps --model qwen
./scripts/remote_bootstrap.sh
./scripts/remote_sync.sh
./scripts/remote_verify.sh
./scripts/remote_fp4_inventory.sh
./scripts/remote_model_probe.sh
MODEL_ID=Qwen/Qwen3-8B ./scripts/remote_qwen_fetch.sh
MODEL_ID=Qwen/Qwen3-8B ./scripts/remote_qwen_baseline.sh
```

If remote Python runtime packages are missing:

```bash
./scripts/remote_install_runtime.sh
```

If the remote machine cannot access a site directly, download on the Mac and relay it:

```bash
./scripts/relay_url_to_remote.sh <url> <remote-path>
./scripts/push_local_file_to_remote.sh <local-path> <remote-path>
```

`remote_qwen_fetch.sh` installs `modelscope` on the remote host if needed, downloads a Qwen snapshot into `/home/pto/lean/models`, and records the resolved local snapshot path so `remote_qwen_baseline.sh` can prefer the local copy over a public model id. For the active pivot, use it first for the `Qwen/Qwen3-8B` semantic contract. The `nvidia/Qwen3-8B-FP4` deployment artifact may require relay from the Mac if the remote host cannot fetch it directly.

For a metadata-only preflight before downloading the full checkpoint:

```bash
MODEL_ALLOW_PATTERN='*.json' ./scripts/remote_qwen_fetch.sh
```

## Current status

As of 2026-03-07, the active milestone is no longer "run the largest possible Qwen on one GB10."

The active milestone is:

- prove whether the public `cuTile` stack can express one real FP4 kernel on `GB10 / sm_121`
- map `Qwen3-8B` semantics and the `Qwen3-8B-FP4` deployment artifact into an adapter-owned contract
- benchmark only after that compiler gate is cleared

Current facts:

- local repo scaffolded from zero
- remote cuTile smoke wired into the DGX Spark machine
- the public remote `cuda.tile 1.1.0` install exposes dtypes up to FP8 in `cuda/tile/_datatype.py`, but not a public `FP4` or `NVFP4` dtype symbol
- the public remote `tileiras` install does target `sm_121`, so backend code generation target coverage exists even though public frontend FP4 coverage is still unproven
- the previous `Qwen3-32B BF16` borrowed and semantic runtime loops remain in the repo as legacy reference data, not the active first target
- those legacy runs produced only about `2 tokens/s` on the remote GB10, which is not a credible starting point for a framework comparison

The next hard gate is therefore narrower and stricter than before: compile and run one minimal FP4 kernel through the public cuTile-native chain, then map `Qwen3-8B-FP4` metadata, then rebuild the runtime around that smaller contract.

The deeper hypothesis is that, once compatibility is treated as optional instead of mandatory, an agent can spend a bounded token budget to generate a more direct and efficient software path for a specific model-chip pair.

GLM remains a second-family target, but it is no longer the first bring-up path.
