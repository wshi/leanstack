# tilepilot

`tilepilot` is a clean-slate, TileIR-first LLM inference stack.

The name is intentional:

- `tile` anchors the stack around cuTile and TileIR instead of a legacy CUDA runtime maze.
- `pilot` emphasizes a small control plane that schedules requests, KV movement, and kernel dispatch without turning the runtime into a monolith.

## Why this repo exists

The current serving ecosystem is too layered, too stateful, and too hard to reason about. This repo starts over with four constraints:

1. The compiler path must be explicit.
2. The runtime must stay small enough to audit.
3. Model support must come from adapters, not hidden branches.
4. Remote validation on the DGX Spark machine is part of the development loop, not an afterthought.

## Scope

Stage 0 in this repo does three concrete things:

1. Defines the target architecture for a compact LLM stack.
2. Provides local and remote tooling to validate the cuTile -> TileIR -> cubin -> SASS path.
3. Establishes the bring-up plan for a GLM-family adapter on the remote machine.

## Repository layout

- `docs/ARCHITECTURE.md`: stack boundaries and replacement strategy.
- `docs/EXECUTION_PLAN.md`: phased build plan and verification gates.
- `docs/MODEL_TARGETS.md`: verified model targets and hardware-fit decisions.
- `docs/REMOTE_VALIDATION.md`: remote workflow and artifact layout.
- `experiments/cutile/vector_add.py`: known-good cuTile smoke kernel.
- `experiments/models/hf_glm_smoke.py`: baseline Hugging Face GLM smoke path.
- `scripts/remote_*.sh`: remote bootstrap, sync, probe, install, and smoke scripts.
- `src/tilepilot/`: Python control plane, planning, and repo utilities.
- `skills/tilepilot-stack/`: English Codex skill for operating the stack.

## Quick start

From `/Users/wei/work/spark/tilepilot`:

```bash
PYTHONPATH=src python3 -m tilepilot.cli show-plan
PYTHONPATH=src python3 -m tilepilot.cli remote-env
./scripts/remote_bootstrap.sh
./scripts/remote_sync.sh
./scripts/remote_verify.sh
./scripts/remote_model_probe.sh
```

If remote Python runtime packages are missing:

```bash
./scripts/remote_install_runtime.sh
```

## Current status

As of 2026-03-06, the first milestone is a compiler-grounded vertical slice:

- local repo scaffolded from zero
- remote cuTile smoke wired into the DGX Spark machine
- GLM adapter work split into explicit phases instead of being hidden inside a giant runtime

The next hard gate is a model adapter that can run a recent GLM-family checkpoint end to end on the remote machine.

Because the latest official GLM checkpoint is larger than a single 128 GB GB10 can realistically host, the first runnable baseline on this machine should be a smaller official GLM-family checkpoint while the compiler and runtime mature.
