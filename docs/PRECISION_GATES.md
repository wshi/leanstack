# Precision Gates

Date verified: 2026-03-07

## Why this document exists

The repo should not guess which precision is practical on the public `cuTile` stack.

Instead, it should record a machine-checked answer for the active remote target:

- hardware: `NVIDIA GB10 / sm_121`
- public compiler stack: `cuda.tile` plus `tileiras`
- model target: `Qwen/Qwen3-1.7B-Base`

## Current decision

The executable precision gate currently recommends:

- primary precision: `bfloat16`

Reason:

- BF16 compiles and runs through the public `cuTile` path on `sm_121`
- the current FP8 probe reaches the compiler but fails TileIR verification
- the public Python authoring surface still lacks a complete FP4 path

## Executable sources

- `experiments/cutile/torch_vector_add.py`
- `experiments/cutile/precision_gate.py`
- `experiments/cutile/fp4_compiler_gate.py`
- `scripts/remote_precision_gate.sh`
- `scripts/remote_fp4_gate.sh`

## Latest remote result

Date confirmed: 2026-03-07

The following command completed on the remote GB10:

- `./scripts/remote_precision_gate.sh`

Confirmed artifact:

- `/home/pto/lean/artifacts/precision-gate/precision_gate_20260307T083727Z.json`

Confirmed result:

- `status: cleared`
- `recommended_primary_precision: bfloat16`

## Probe-level findings

### BF16

- status: cleared
- the current minimal vector-add probe compiles and runs through the public `cuTile` path
- this is the active first precision for `leanstack`

### FP8

- status: blocked
- both public FP8 dtypes currently fail in TileIR verification for the minimal vector-add probe
- this means FP8 is visible in the frontend surface but not yet a cleared authoring path for the current probe

### FP4

- status: blocked
- the public `cuda.tile` frontend does not expose a complete FP4 authoring surface today
- see [FP4_COMPILER_GATE.md](/Users/wei/work/spark/leanstack/docs/FP4_COMPILER_GATE.md) for the narrower sub-gate

## Current policy

1. Treat `Qwen/Qwen3-1.7B-Base` BF16 as the active first deployment contract.
2. Keep rerunning the precision gate when the probe, compiler, or remote environment changes.
3. Do not restart an FP8 or FP4 runtime push until the gate turns positive for that precision, or until a narrowly scoped PTX wedge is explicitly adopted.
