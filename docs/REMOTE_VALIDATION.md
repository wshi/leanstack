# Remote Validation

The DGX Spark machine is accessed through `../remote.sh` and treated as the system-of-record for kernel validation.

## Remote layout

The scripts in this repo create and use:

- `/home/pto/tilepilot/repo`
- `/home/pto/tilepilot/artifacts`
- `/home/pto/tilepilot/logs`
- `/home/pto/tilepilot/models`
- `/home/pto/tilepilot/tmp`

## Validation loop

1. `./scripts/remote_bootstrap.sh`
2. `./scripts/remote_sync.sh`
3. `./scripts/remote_verify.sh`
4. `./scripts/remote_model_probe.sh`

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

`/home/pto/tilepilot/artifacts/<UTC timestamp>`

The expected structure is:

- `01_bytecode/`
- `02_tileir/`
- `03_cubin/`
- `04_sass/`
- `logs/`

No generated artifact should be committed back into the repo.

