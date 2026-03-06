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
3. The first contract is a model-chip pair, `Qwen/Qwen3-32B` on Blackwell, not "all models on all hardware."
4. `vLLM`, `SGLang`, `llama.cpp`, and similar systems are compatibility-heavy baselines to compare against, not runtime dependencies.
5. Remote validation on the DGX Spark machine is part of the development loop, not an afterthought.

## Scope

Stage 0 in this repo does four concrete things:

1. Defines the project thesis for an agent-built, model-chip-specific LLM stack.
2. Provides local and remote tooling to validate the cuTile -> TileIR -> cubin -> SASS path.
3. Establishes the bring-up plan for a `Qwen/Qwen3-32B` adapter on a Blackwell-class remote machine.
4. Defines the benchmark contract against `vLLM`, `SGLang`, and a secondary `llama.cpp` reference path, including both runtime efficiency and software-stack complexity.

## Repository layout

- `docs/PROJECT_THESIS.md`: project thesis and hard constraints.
- `docs/ARCHITECTURE.md`: stack boundaries and replacement strategy.
- `docs/BENCHMARK_PLAN.md`: benchmark methodology and comparison rules.
- `docs/EXECUTION_PLAN.md`: phased build plan and verification gates.
- `docs/MODEL_FIT_ANALYSIS.md`: frontier-model architecture fit versus cuTile complexity.
- `docs/MODEL_TARGETS.md`: verified model targets and hardware-fit decisions.
- `docs/REFERENCES.md`: external research, framework, model, and hardware references.
- `docs/REMOTE_VALIDATION.md`: remote workflow and artifact layout.
- `experiments/cutile/vector_add.py`: known-good cuTile smoke kernel.
- `experiments/models/hf_causal_lm_smoke.py`: baseline Hugging Face causal LM smoke path.
- `scripts/remote_*.sh`: remote bootstrap, sync, relay, probe, install, and smoke scripts.
- `src/leanstack/`: Python control plane, planning, and repo utilities.
- `skills/leanstack/`: English Codex skill for operating the stack.

## Quick start

From `/Users/wei/work/spark/leanstack`:

```bash
PYTHONPATH=src python3 -m leanstack.cli show-plan
PYTHONPATH=src python3 -m leanstack.cli remote-env
./scripts/remote_bootstrap.sh
./scripts/remote_sync.sh
./scripts/remote_verify.sh
./scripts/remote_model_probe.sh
./scripts/remote_qwen_fetch.sh
./scripts/remote_qwen_baseline.sh
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

`remote_qwen_fetch.sh` installs `modelscope` on the remote host if needed, downloads `Qwen/Qwen3-32B` into `/home/pto/lean/models`, and records the resolved local snapshot path so `remote_qwen_baseline.sh` can prefer the local copy over Hugging Face.

For a metadata-only preflight before downloading the full checkpoint:

```bash
MODEL_ALLOW_PATTERN='*.json' ./scripts/remote_qwen_fetch.sh
```

## Current status

As of 2026-03-06, the first milestone is a compiler-grounded vertical slice:

- local repo scaffolded from zero
- remote cuTile smoke wired into the DGX Spark machine
- Qwen adapter work split into explicit phases instead of being hidden inside a giant runtime
- ModelScope-based `Qwen/Qwen3-32B` fetch path validated on the remote machine

The next hard gate is a `Qwen/Qwen3-32B` adapter that can run end to end on the remote Blackwell machine and then be benchmarked against framework baselines.

The deeper hypothesis is that, once compatibility is treated as optional instead of mandatory, an agent can spend a bounded token budget to generate a more direct and efficient software path for a specific model-chip pair.

GLM remains a second-family target, but it is no longer the first bring-up path.
