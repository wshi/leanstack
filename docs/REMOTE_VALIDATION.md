# Remote Validation

The DGX Spark machine is accessed through `../remote.sh` and treated as the system-of-record for kernel validation.

As of 2026-03-07, the `Qwen3-32B BF16` runtime validations below are legacy reference data. The active target has pivoted to `Qwen/Qwen3-1.7B-Base` BF16, and the active first gate is the executable precision gate on `sm_121`.

## Remote layout

The scripts in this repo create and use:

- `/home/pto/lean/repo`
- `/home/pto/lean/artifacts`
- `/home/pto/lean/logs`
- `/home/pto/lean/models`
- `/home/pto/lean/tmp`

## Validation loop

1. `./scripts/remote_bootstrap.sh`
2. `./scripts/remote_sync.sh`
3. `./scripts/remote_verify.sh`
4. `./scripts/remote_fp4_inventory.sh`
5. `./scripts/remote_precision_gate.sh`
6. `./scripts/remote_fp4_gate.sh`
7. `./scripts/remote_model_probe.sh`
8. `MODEL_ID=Qwen/Qwen3-1.7B-Base ./scripts/remote_qwen_fetch.sh` for the semantic-base snapshot path
9. `MODEL_ID=Qwen/Qwen3-1.7B-Base ./scripts/remote_qwen_baseline.sh`
10. `./scripts/relay_url_to_remote.sh` or `./scripts/push_local_file_to_remote.sh` if the remote machine cannot download an artifact directly

## What `remote_verify.sh` checks

The smoke script does all of the following on the remote host:

1. activates `/home/pto/venv-cutile`
2. runs `experiments/cutile/vector_add.py`
3. captures `.cutile` bytecode
4. captures TileIR dumps
5. compiles the latest `.cutile` file with `tileiras`
6. dumps SASS with `cuobjdump` and `nvdisasm` when present

## Artifact contract

Each validation run writes a timestamped artifact directory:

`/home/pto/lean/artifacts/<UTC timestamp>`

The expected structure is:

- `01_bytecode/`
- `02_tileir/`
- `03_cubin/`
- `04_sass/`
- `logs/`

No generated artifact should be committed back into the repo.

## Mac relay workflow

When the remote machine cannot access a model, wheel, or archive directly:

1. download the file on the Mac
2. relay it to the remote machine
3. keep the remote path under `/home/pto/lean/models`, `/home/pto/lean/tmp`, or another explicit deployment directory

## Qwen acquisition workflow

The primary semantic-base acquisition path is:

1. `MODEL_ID=Qwen/Qwen3-1.7B-Base ./scripts/remote_qwen_fetch.sh`
2. let the script install `modelscope` remotely if it is missing
3. store the downloaded snapshot under `/home/pto/lean/models`
4. read the resolved local snapshot path from `/home/pto/lean/models/Qwen__Qwen3-1.7B-Base.path`
5. run `MODEL_ID=Qwen/Qwen3-1.7B-Base ./scripts/remote_qwen_baseline.sh`, which prefers that local snapshot path over the public model id

For a low-risk preflight, run:

- `MODEL_ID=Qwen/Qwen3-1.7B-Base MODEL_ALLOW_PATTERN='*.json' ./scripts/remote_qwen_fetch.sh`

If ModelScope or PyPI becomes unreachable from the remote host, relay a wheel, archive, or extracted model directory from the Mac into `/home/pto/lean/models` and update the path file accordingly.

Confirmed on 2026-03-07:

- metadata-only preflight for `Qwen/Qwen3-1.7B-Base` succeeded
- path file written to `/home/pto/lean/models/Qwen__Qwen3-1.7B-Base.path`

## Precision gate status

Date confirmed: 2026-03-07

The following command completed successfully on the remote machine:

- `./scripts/remote_precision_gate.sh`

Key confirmed facts:

- artifact written to `/home/pto/lean/artifacts/precision-gate/precision_gate_20260307T083727Z.json`
- gate status is `cleared`
- recommended primary precision is `bfloat16`
- BF16 compiles and runs through the public `cuTile` path on the remote machine
- both public FP8 dtypes currently fail TileIR verification for the minimal float8 vector-add probe
- the remote `tileiras` binary still reports `sm_121` coverage

Interpretation:

- backend targeting for GB10 is visible
- BF16 is the active first precision for the repo
- FP8 is visible in the frontend surface but not yet cleared for the current probe
- FP4 still needs a separate negative sub-gate because the public frontend surface is incomplete

## FP4 sub-gate status

Date confirmed: 2026-03-07

The following command completed successfully on the remote machine:

- `./scripts/remote_fp4_gate.sh`

Key confirmed facts:

- artifact written to `/home/pto/lean/artifacts/fp4-gate/fp4_gate_20260307T080849Z.json`
- gate status is `blocked`
- blocker is `public cuda.tile frontend does not expose a complete FP4 authoring surface`
- backend still reports `sm_121` support in `tileiras`

This means the repo now has an executable FP4 compiler gate. The current result is a real blocker, not just a planning note, but it is no longer the active first-format target.

## Legacy runtime references

Date confirmed: 2026-03-07

The repo still contains the following legacy reference facts for `Qwen3-32B BF16`:

- borrowed full runtime loop on GB10:
  - about `76.7s` materialization
  - about `2.24 tokens/s` runtime-loop throughput for the `8+4` probe
- semantic full runtime loop on GB10:
  - about `290.7s` materialization
  - about `1.92 tokens/s` runtime-loop throughput for the same `8+4` probe

These runs are useful only as proof that the older explicit path exists and as evidence that the old target is too slow for a meaningful benchmark-first workflow.
