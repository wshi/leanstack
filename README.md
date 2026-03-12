# leanstack

`leanstack` is a research artifact about whether VIS-native agentic inference stacks can replace compatibility-heavy frameworks — first proven on GPU with Tile IR, then applied to custom silicon with our own virtual instruction set.

The name is intentional:

- `lean` emphasizes compressing the deployment and serving stack to the smallest system that still remains credible.
- `stack` keeps the focus on the whole software path: compiler, runtime, and serving edge.

## Why this repo exists

The dominant LLM serving stacks pay a large compatibility tax: too many layers, too much genericity, and too much hidden state for workloads that ultimately run one concrete model on one concrete machine.

leanstack tests a specific counter-thesis: **a stable virtual instruction set (VIS) enables an agent to synthesize a narrow, hardware-native inference appliance that is both simpler and faster than a compatibility-first framework.** The VIS is the enabling abstraction — it gives the agent a stable target to generate against, and gives the hardware vendor a stable contract boundary for cross-generation portability.

This is a two-phase project:

**Phase 1 (current)** validates the thesis on NVIDIA GPU using Tile IR as the VIS. The first contract is `Qwen/Qwen3-1.7B-Base BF16` on `GB10 / sm_121`, with `cuTile -> TileIR -> cubin -> SASS` as the only official execution path. The target is `>= 1.30x` warmed vLLM throughput on the primary decode profile — not as an end in itself, but as evidence that the VIS-native approach produces real advantages.

**Phase 2 (declared destination)** reproduces the methodology on custom silicon using our own tile-based VIS. The agent-synthesis pipeline, leanpack format, and economic measurement framework built in Phase 1 are designed to be retargetable.

The project measures three things simultaneously:

1. **Throughput**: can the narrow appliance beat the compatibility-heavy framework?
2. **Software cost**: how many agent tokens does the narrow appliance cost vs. how much compatibility software does it replace?
3. **VIS portability**: does the VIS abstraction boundary hold when the target changes?

## Constraints

1. The execution path must stay explicit down to `cuTile -> TileIR -> cubin -> SASS`.
2. The runtime must stay small enough that an agent can regenerate and modify it cheaply.
3. The first contract is a model-chip pair, `Qwen/Qwen3-1.7B-Base + BF16 checkpoint` on `GB10 / sm_121`, not "all models on all hardware."
4. `vLLM`, `SGLang`, `llama.cpp`, and similar systems are compatibility-heavy baselines to compare against, not runtime dependencies.
5. Remote validation on the DGX Spark machine is part of the development loop, not an afterthought.
6. The target product is not a generic runtime. It is a `leanpack + leanserve` appliance: offline serving artifacts plus a static resident decode service.
7. Agent token budget is tracked as a first-class engineering metric alongside throughput and latency.

## Scope

The repo does seven concrete things:

1. Defines a two-phase project thesis for VIS-centric agentic inference (see `docs/PROJECT_THESIS.md`).
2. Provides local and remote tooling to validate the cuTile -> TileIR -> cubin -> SASS path.
3. Keeps precision choice explicit through an executable BF16 / FP8 / FP4 gate on the remote machine.
4. Builds the runtime around `Qwen/Qwen3-1.7B-Base` BF16 with explicit ownership of execution, KV cache, and kernel dispatch.
5. Defines `leanpack` and `leanserve` as the primary build products: packed serving artifacts and a static resident decode service.
6. Defines a staged comparison protocol against `vLLM`, `SGLang`, and other external baselines, measuring both runtime efficiency and software-stack complexity.
7. Captures a milestone roadmap (M1/M2/M3 for Phase 1, design principles for Phase 2) in `docs/ROADMAP.md`.

## Repository layout

- `docs/PROJECT_THESIS.md`: two-phase project thesis (VIS-centric agentic inference on GPU, then custom silicon).
- `docs/ARCHITECTURE.md`: stack boundaries, replacement strategy, and long-term architecture principles.
- `docs/ROADMAP.md`: Phase 1 milestones (M1/M2/M3) and Phase 2 design principles.
- `docs/DSA_VIS_PROOF_PROTOCOL.md`: falsifiable protocol for proving VIS value on custom DSA using performance + regeneration-cost + retargetability together.
- `docs/BENCHMARK_PLAN.md`: benchmark methodology and comparison rules.
- `docs/COMPARISON_PROTOCOL.md`: staged comparison gates from framework baselines to cuTile kernels to full-stack results.
- `docs/THROUGHPUT_30_PLAN.md`: the stronger `>= 1.30x warmed vLLM` target, why packing alone is not enough, and the exact speculative-decode plan.
- `src/leanstack/appliance.py`: first-principles appliance reset plus `leanpack`/`leanserve` plan renderers.
- `src/leanstack/leanserve.py`: packed-artifact loader, resident buffer planner, and semantic runtime materializer for `leanserve`.
- `docs/LEANPACK_FORMAT.md`: initial serving-artifact format for `leanpack/v0`.
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
- `experiments/cutile/qwen_bf16_hot_kernels.py`: BF16 hot-kernel microbench suite for the exact Qwen3-1.7B geometry.
- `experiments/models/hf_causal_lm_smoke.py`: baseline Hugging Face causal LM smoke path.
- `experiments/models/qwen_explicit_block_probe.py`: explicit Qwen loader and layer-0 block/prefill/decode probe.
- `experiments/models/qwen_explicit_stack_probe.py`: explicit multi-layer Qwen stack probe.
- `experiments/models/qwen_explicit_runtime_loop.py`: explicit full-model runtime loop with greedy decode accounting.
- `experiments/models/qwen_semantic_block_probe.py`: adapter-owned Qwen block semantics plus page-based KV cache probe.
- `scripts/remote_*.sh`: remote bootstrap, sync, relay, probe, install, and smoke scripts.
- `scripts/remote_leanpack_build.sh`: remote builder for the first serving-only packed artifact.
- `scripts/remote_leanserve_layout.sh`: remote inspector for the resident `leanserve` layout against a real packed artifact.
- `src/leanstack/`: Python control plane, planning, and repo utilities.
- `skills/leanstack/`: English Codex skill for operating the stack.

## Quick start

From `/Users/wei/work/spark/leanstack`:

```bash
PYTHONPATH=src python3 -m leanstack.cli show-plan
PYTHONPATH=src python3 -m leanstack.cli show-comparison-plan
PYTHONPATH=src python3 -m leanstack.cli show-appliance-reset --model qwen
PYTHONPATH=src python3 -m leanstack.cli show-leanpack-plan --model qwen
PYTHONPATH=src python3 -m leanstack.cli show-leanserve-plan --model qwen
PYTHONPATH=src python3 -m leanstack.cli build-leanpack --model qwen --model-path /path/to/Qwen3-1.7B-Base --output-dir ./artifacts/leanpack-qwen --manifest-only --overwrite
PYTHONPATH=src python3 -m leanstack.cli inspect-leanpack --pack-dir ./artifacts/leanpack-qwen
PYTHONPATH=src python3 -m leanstack.cli show-leanserve-layout --model qwen --pack-dir ./artifacts/leanpack-qwen
PYTHONPATH=src python3 -m leanstack.cli list-hot-kernel-cases --default-only
PYTHONPATH=src python3 -m leanstack.cli remote-env
PYTHONPATH=src python3 -m leanstack.cli show-contract --model qwen
PYTHONPATH=src python3 -m leanstack.cli show-gaps --model qwen
./scripts/remote_bootstrap.sh
./scripts/remote_sync.sh
./scripts/remote_verify.sh
./scripts/remote_fp4_inventory.sh
./scripts/remote_precision_gate.sh
./scripts/remote_fp4_gate.sh
./scripts/remote_qwen_hot_kernel_bench.sh
./scripts/remote_model_probe.sh
MODEL_ID=Qwen/Qwen3-1.7B-Base ./scripts/remote_qwen_fetch.sh
MODEL_ID=Qwen/Qwen3-1.7B-Base OVERWRITE=1 ./scripts/remote_leanpack_build.sh
./scripts/remote_leanserve_layout.sh
MODEL_ID=Qwen/Qwen3-1.7B-Base ./scripts/remote_qwen_baseline.sh
STRICT_CONTRACT=1 STRICT_PACKED=1 PROFILE=decode_64_256 MODEL_ID=Qwen/Qwen3-1.7B-Base ./scripts/remote_leanstack_benchmark.sh
MODEL_NAME=qwen3-1.7b-base ./scripts/remote_openai_profile_sweep.sh
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

As of 2026-03-12, the active milestone remains **M1 — Appliance Proof** (see `docs/ROADMAP.md`), but with a stricter interpretation:

- official comparison claims are now locked to the fixed contract `Qwen/Qwen3-1.7B-Base + BF16 + GB10/sm_121 + decode_64_256 + packed appliance`
- compare UI and remote benchmark paths enforce strict-packed and strict-contract guards
- only this fixed-contract path is treated as evidence for the DSA VIS thesis

Current data split:

- official fixed-contract packed path: near warmed-`vLLM` parity (small lead, not yet a decisive margin)
- exploratory dual-model speculative path: can exceed `+30%`, but is not counted as core fixed-contract proof for the VIS-on-DSA argument

Current precision gates:

- BF16: cleared on `sm_121`
- FP8: blocked (TileIR verification fails)
- FP4: blocked (public `cuda.tile` frontend incomplete)

The deeper hypothesis remains: once compatibility is treated as optional instead of mandatory, an agent can spend a bounded token budget to generate a more direct and efficient software path for a specific model-chip pair — and that a stable VIS is the abstraction that makes this approach scalable across hardware targets.

See `CURRENT_STATUS.md` for detailed phase-by-phase engineering history and operational instructions.
