---
name: leanstack
description: Operate and extend the leanstack repository, a clean-slate TileIR-first LLM inference stack built to simplify and compress the serving software stack. Use this skill when Codex needs to plan the stack, add or validate cuTile kernels, sync code to the DGX Spark machine through ../remote.sh, collect remote artifacts, prepare a GLM-family adapter, or tighten the runtime without falling back to legacy vLLM-style architecture.
---

# Leanstack

Use this skill when working in `/Users/wei/work/spark/leanstack`.

## Principles

1. Keep the stack compiler-first and runtime-small.
2. Prefer explicit adapters over model-specific runtime branches.
3. Treat the DGX Spark machine as the truth source for kernel validation.
4. Keep repo-facing docs and implementation notes in English.

## First files to read

1. `README.md`
2. `docs/ARCHITECTURE.md`
3. `docs/EXECUTION_PLAN.md`
4. `docs/REMOTE_VALIDATION.md`
5. `docs/MODEL_TARGETS.md`
6. `references/remote_workflow.md`

## Default workflow

### Planning and architecture

1. Confirm whether the task touches compiler, runtime, serve, or model-adapter boundaries.
2. Extend the smallest layer that solves the task.
3. Update `docs/EXECUTION_PLAN.md` when a phase gate or blocker changes.

### Kernel bring-up

1. Implement or update a focused experiment under `experiments/cutile/`.
2. Keep the kernel independently runnable before wiring it into runtime code.
3. Capture the lowering chain on the remote machine with `scripts/remote_verify.sh`.
4. Preserve generated artifacts on the remote host, not in git.

### Remote workflow

1. Run `./scripts/remote_bootstrap.sh`.
2. Run `./scripts/remote_sync.sh`.
3. Run `./scripts/remote_verify.sh` for compiler smoke.
4. Run `./scripts/remote_model_probe.sh` before attempting a model adapter bring-up.
5. Use `./scripts/remote_install_runtime.sh` only when model-runtime dependencies are missing.
6. Use `./scripts/remote_glm_baseline.sh` to separate plain model-loading issues from runtime-design issues.

### GLM-family work

1. Verify the exact checkpoint from primary sources before calling it the current target.
2. Record the verification date in the working notes or user summary.
3. Translate the model into explicit kernel requirements before importing framework code.
4. Keep adapter rules in code or docs, never as hidden assumptions.

## Guardrails

- Do not patch legacy vLLM code into this repo.
- Do not commit remote artifacts or downloaded model weights.
- Do not claim a “latest” GLM checkpoint without a dated, primary-source verification.
- Do not hide missing kernel coverage behind a framework fallback.

## Useful commands

```bash
PYTHONPATH=src python3 -m leanstack.cli show-plan
PYTHONPATH=src python3 -m leanstack.cli remote-env
PYTHONPATH=src python3 -m leanstack.cli show-blueprint --model glm
./scripts/remote_bootstrap.sh
./scripts/remote_sync.sh
./scripts/remote_verify.sh
./scripts/remote_model_probe.sh
```
