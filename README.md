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

In the strongest form of this thesis, the only meaningful dynamic input should be the user request payload. Model geometry, chip target, memory layout, kernel inventory, and dispatch policy should all be fixed by the `Qwen3-32B + GB10` contract.

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
- `docs/IMPLEMENTATION_GAPS.md`: structured gap analysis from borrowed `transformers` semantics to adapter-owned `cuTile/TileIR` kernels.
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
./scripts/remote_model_probe.sh
./scripts/remote_qwen_fetch.sh
./scripts/remote_qwen_block_probe.sh
./scripts/remote_qwen_stack_probe.sh
./scripts/remote_qwen_runtime_loop.sh
./scripts/remote_qwen_semantic_block_probe.sh
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

As of 2026-03-07, the first milestone is a compiler-grounded vertical slice:

- local repo scaffolded from zero
- remote cuTile smoke wired into the DGX Spark machine
- Qwen adapter work split into explicit phases instead of being hidden inside a giant runtime
- ModelScope-based `Qwen/Qwen3-32B` fetch path validated on the remote machine
- explicit layer-0 Qwen block/prefill/decode probe runs on the remote GB10 without `device_map=\"auto\"`
- explicit multi-layer Qwen stack probe is now available, so the next extension can happen on the same runtime surface
- borrowed explicit full-model Qwen runtime loop now runs across all 64 layers on the remote GB10, with approximately `65.6 GiB` allocated after materialization and about `2.24 tokens/s` in runtime-loop throughput for the `8+4` token probe
- adapter-owned Qwen semantics now extend all the way to a full 64-layer semantic runtime loop with a page-based KV manager, approximately `65.5 GiB` allocated after materialization, and about `1.92 tokens/s` in runtime-loop throughput for the same `8+4` token probe
- the active semantic path no longer depends on borrowed `Qwen3DecoderLayer` or `DynamicCache`, so the remaining gap is now eager PyTorch math, probe-style staging, and lack of `cuTile/TileIR` kernels
- a structured gap registry now tracks the remaining code path from borrowed `transformers` semantics to `cuTile/TileIR` kernels on `sm_121`

The next hard gate is lowering the now-working full-model semantic loop from eager PyTorch operators into `cuTile/TileIR` kernels, replacing the dense probe-style cache/layout path with a true residency plan, and only then benchmarking that path against framework baselines.

The deeper hypothesis is that, once compatibility is treated as optional instead of mandatory, an agent can spend a bounded token budget to generate a more direct and efficient software path for a specific model-chip pair.

GLM remains a second-family target, but it is no longer the first bring-up path.
