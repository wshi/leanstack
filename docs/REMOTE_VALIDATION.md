# Remote Validation

The DGX Spark machine is accessed through `../remote.sh` and treated as the system-of-record for kernel validation.

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
4. `./scripts/remote_model_probe.sh`
5. `./scripts/remote_qwen_fetch.sh` for the primary model snapshot path
6. `./scripts/remote_qwen_baseline.sh`
7. `./scripts/relay_url_to_remote.sh` or `./scripts/push_local_file_to_remote.sh` if the remote machine cannot download an artifact directly

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

The primary model acquisition path is:

1. `./scripts/remote_qwen_fetch.sh`
2. let the script install `modelscope` remotely if it is missing
3. store the downloaded snapshot under `/home/pto/lean/models`
4. read the resolved local snapshot path from `/home/pto/lean/models/Qwen__Qwen3-32B.path`
5. run `./scripts/remote_qwen_baseline.sh`, which prefers that local snapshot path over the public model id

For a low-risk preflight, run:

- `MODEL_ALLOW_PATTERN='*.json' ./scripts/remote_qwen_fetch.sh`

If ModelScope or PyPI becomes unreachable from the remote host, relay a wheel, archive, or extracted model directory from the Mac into `/home/pto/lean/models` and update the path file accordingly.
