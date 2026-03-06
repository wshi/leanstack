# Remote Workflow

## Purpose

This reference describes the expected remote loop for `tilepilot`.

## Entry point

The shared entry point is `../remote.sh`, which currently resolves to the DGX Spark machine used for validation.

## Commands

1. `./scripts/remote_bootstrap.sh`
2. `./scripts/remote_sync.sh`
3. `./scripts/remote_verify.sh`
4. `./scripts/remote_model_probe.sh`

## Expected remote locations

- `/home/pto/tilepilot/repo`
- `/home/pto/tilepilot/artifacts`
- `/home/pto/tilepilot/logs`
- `/home/pto/tilepilot/models`

## Validation expectations

- compiler smoke must produce `.cutile` bytecode
- TileIR dumps must exist when dumping is enabled
- `tileiras` compilation must produce a cubin when the bytecode is valid
- SASS dumps should be captured when `cuobjdump` or `nvdisasm` is present

## Model bring-up expectations

Before attempting a GLM-family run:

1. collect a fresh probe report
2. verify the target checkpoint from primary sources
3. install model-runtime dependencies if needed
4. confirm kernel coverage gaps explicitly

