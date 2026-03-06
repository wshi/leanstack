# Remote Workflow

## Purpose

This reference describes the expected remote loop for `leanstack`.

## Entry point

The shared entry point is `../remote.sh`, which currently resolves to the DGX Spark machine used for validation.

## Commands

1. `./scripts/remote_bootstrap.sh`
2. `./scripts/remote_sync.sh`
3. `./scripts/remote_verify.sh`
4. `./scripts/remote_model_probe.sh`
5. `./scripts/remote_qwen_fetch.sh`
6. `./scripts/remote_qwen_baseline.sh`

## Expected remote locations

- `/home/pto/lean/repo`
- `/home/pto/lean/artifacts`
- `/home/pto/lean/logs`
- `/home/pto/lean/models`

## Validation expectations

- compiler smoke must produce `.cutile` bytecode
- TileIR dumps must exist when dumping is enabled
- `tileiras` compilation must produce a cubin when the bytecode is valid
- SASS dumps should be captured when `cuobjdump` or `nvdisasm` is present

## Model bring-up expectations

Before attempting a Qwen-first run:

1. collect a fresh probe report
2. verify the target checkpoint from primary sources
3. confirm whether Hugging Face is reachable; if not, use `remote_qwen_fetch.sh` through ModelScope or fall back to Mac relay
4. confirm kernel coverage gaps explicitly
