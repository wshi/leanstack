# leanstack

`leanstack` is a clean-slate, cuTile-native, GB10-first LLM inference stack.

The name is intentional:

- `lean` emphasizes compressing the deployment and serving stack to the smallest system that still remains credible.
- `stack` keeps the focus on the whole software path: compiler, runtime, and serving edge.

## Why this repo exists

The current serving ecosystem pays a large compatibility tax: too many layers, too much genericity, and too much hidden state for workloads that ultimately run one concrete model on one concrete machine.

This repo explores a different path with five constraints:

1. The execution path must stay explicit down to `cuTile -> TileIR -> cubin -> SASS`.
2. The runtime must stay small enough that an agent can regenerate and modify it cheaply.
3. The first contract is a model-chip pair, `Qwen/Qwen3-1.7B-Base + BF16 checkpoint` on `GB10 / sm_121`, not "all models on all hardware."
4. `vLLM`, `SGLang`, `llama.cpp`, and similar systems are compatibility-heavy baselines to compare against, not runtime dependencies.
5. Remote validation on the DGX Spark machine is part of the development loop, not an afterthought.

In the strongest form of this thesis, the only meaningful dynamic input should be the user request payload. Model geometry, chip target, precision policy, memory layout, kernel inventory, and dispatch policy should all be fixed by the `Qwen3-1.7B-Base BF16 + GB10` contract. Backend choice is still explicit, but it is selected for throughput, not aesthetics. FP8 and FP4 stay deferred until they clearly beat the BF16 path.

## Scope

Stage 0 in this repo does five concrete things:

1. Defines the project thesis for an agent-built, model-chip-specific LLM stack.
2. Provides local and remote tooling to validate the cuTile -> TileIR -> cubin -> SASS path.
3. Keeps precision choice explicit through an executable BF16 / FP8 / FP4 gate on the remote machine.
4. Rebuilds the runtime around `Qwen/Qwen3-1.7B-Base` BF16 instead of the legacy `Qwen3-32B` reference path.
5. Allows `cuTile`, Triton, CUTLASS, or PTX on the hot path as long as ownership stays inside `leanstack` and the backend choice is explicit.
6. Defines the benchmark contract against `vLLM`, `SGLang`, and other external baselines, including both runtime efficiency and software-stack complexity.

## Repository layout

- `docs/PROJECT_THESIS.md`: project thesis and hard constraints.
- `docs/ARCHITECTURE.md`: stack boundaries and replacement strategy.
- `docs/BENCHMARK_PLAN.md`: benchmark methodology and comparison rules.
- `docs/EXECUTION_PLAN.md`: phased build plan and verification gates.
- `docs/IMPLEMENTATION_GAPS.md`: structured gap analysis from borrowed `transformers` semantics to adapter-owned `cuTile/TileIR` kernels.
- `docs/PRECISION_GATES.md`: the active BF16 / FP8 / FP4 gate results for the public cuTile stack on `sm_121`.
- `docs/FP4_COMPILER_GATE.md`: the narrower negative sub-gate for the public FP4 authoring surface.
- `docs/MODEL_FIT_ANALYSIS.md`: frontier-model architecture fit versus cuTile complexity.
- `docs/MODEL_TARGETS.md`: verified model targets and hardware-fit decisions.
- `docs/REFERENCES.md`: external research, framework, model, and hardware references.
- `docs/REMOTE_VALIDATION.md`: remote workflow and artifact layout.
- `experiments/cutile/vector_add.py`: known-good cuTile smoke kernel.
- `experiments/cutile/torch_vector_add.py`: torch-backed minimal dtype probe for BF16 and FP8 reachability.
- `experiments/cutile/precision_gate.py`: executable BF16 / FP8 / FP4 precision-gate probe for the current public cuTile install.
- `experiments/cutile/fp4_compiler_gate.py`: executable FP4 compiler-gate probe for the current public cuTile install.
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
./scripts/remote_precision_gate.sh
./scripts/remote_fp4_gate.sh
./scripts/remote_model_probe.sh
MODEL_ID=Qwen/Qwen3-1.7B-Base ./scripts/remote_qwen_fetch.sh
MODEL_ID=Qwen/Qwen3-1.7B-Base ./scripts/remote_qwen_baseline.sh
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

`remote_qwen_fetch.sh` installs `modelscope` on the remote host if needed, downloads a Qwen snapshot into `/home/pto/lean/models`, and records the resolved local snapshot path so `remote_qwen_baseline.sh` can prefer the local copy over a public model id. For the active pivot, use it first for the `Qwen/Qwen3-1.7B-Base` BF16 contract.

For a metadata-only preflight before downloading the full checkpoint:

```bash
MODEL_ALLOW_PATTERN='*.json' ./scripts/remote_qwen_fetch.sh
```

## Current status

As of 2026-03-07, the active milestone is no longer "run the largest possible Qwen on one GB10."

The active milestone is:

- keep the public BF16 `cuTile` path green on `GB10 / sm_121`
- rebuild the runtime around `Qwen/Qwen3-1.7B-Base` BF16 with adapter-owned placement, KV state, and kernel boundaries
- benchmark only after that 1.7B BF16 runtime slice exists

Current facts:

- local repo scaffolded from zero
- remote cuTile smoke wired into the DGX Spark machine
- the executable remote precision gate wrote `/home/pto/lean/artifacts/precision-gate/precision_gate_20260307T083727Z.json`
- that gate currently recommends `bfloat16` as the active public-cuTile precision on `sm_121`
- BF16 compiles and runs through the public `cuTile` path on the remote machine
- a metadata-only remote fetch for `Qwen/Qwen3-1.7B-Base` succeeded and wrote `/home/pto/lean/models/Qwen__Qwen3-1.7B-Base.path`
- the current float8 probe reaches the compiler but fails TileIR verification for both public FP8 dtypes
- the narrower FP4 sub-gate remains blocked because the public `cuda.tile` frontend does not expose a complete FP4 authoring surface
- the previous `Qwen3-32B BF16` borrowed and semantic runtime loops remain in the repo as legacy reference data, not the active first target
- those legacy runs produced only about `2 tokens/s` on the remote GB10, which is not a credible starting point for a framework comparison

The next hard gate is therefore different from the earlier FP4 pivot: own the `Qwen3-1.7B-Base` BF16 checkpoint contract, rebuild the runtime around that smaller model, and only then benchmark against exact-format BF16 baselines.

The deeper hypothesis is that, once compatibility is treated as optional instead of mandatory, an agent can spend a bounded token budget to generate a more direct and efficient software path for a specific model-chip pair.

GLM remains a second-family target, but it is no longer the first bring-up path.
