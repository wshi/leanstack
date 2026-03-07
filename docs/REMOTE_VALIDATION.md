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
6. `./scripts/remote_qwen_stack_probe.sh`
7. `./scripts/remote_qwen_runtime_loop.sh`
8. `./scripts/remote_qwen_baseline.sh`
9. `./scripts/relay_url_to_remote.sh` or `./scripts/push_local_file_to_remote.sh` if the remote machine cannot download an artifact directly

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

## Latest confirmed borrowed full runtime result

Date confirmed: 2026-03-07

The following command completed successfully on the remote GB10 machine:

```bash
python3 experiments/models/qwen_explicit_runtime_loop.py \
  --model-path /home/pto/lean/models/Qwen/Qwen3-32B \
  --num-layers 0 \
  --device cuda:0 \
  --max-prefill-tokens 8 \
  --max-new-tokens 4 \
  --disable-thinking
```

Key confirmed facts:

- all `64` decoder layers were materialized
- output head and final norm were included
- `prompt_tokens=8`
- `emitted_tokens=4`
- `cache_seq_length=12`
- materialization took about `76.7s`
- runtime loop time was about `1.79s`
- runtime-loop throughput was about `2.24 tokens/s`
- GPU allocation after materialization was about `65.5 GiB`

This confirms that the repo crossed from a multi-layer probe into a full-model, single-request runtime loop, even though this borrowed mode still used `transformers` layer semantics and cache behavior.

## Latest confirmed full semantic runtime result

Date confirmed: 2026-03-07

The following command completed successfully on the remote GB10 machine:

```bash
python3 experiments/models/qwen_explicit_runtime_loop.py \
  --model-path /home/pto/lean/models/Qwen/Qwen3-32B \
  --runtime-mode semantic \
  --num-layers 0 \
  --device cuda:0 \
  --max-prefill-tokens 8 \
  --max-new-tokens 4 \
  --disable-thinking
```

Key confirmed facts:

- all `64` decoder layers were materialized in semantic mode
- the active cache path used `KVBlockManager`, not `DynamicCache`
- `prompt_tokens=8`
- `emitted_tokens=4`
- `cache_seq_length=12`
- `page_size=16`, `used_pages=1`
- materialization took about `290.7s`
- runtime loop time was about `2.09s`
- runtime-loop throughput was about `1.92 tokens/s`
- GPU allocation after materialization was about `65.5 GiB`

This is the first remote proof that the active full-model loop can run on adapter-owned Qwen semantics and a leanstack-owned KV cache. It also exposes the next performance problem clearly: semantic ownership is in place, but the path still uses eager PyTorch math and probe-style staging, so it is not yet ready for a fair framework benchmark.

## Latest confirmed semantic block result

Date confirmed: 2026-03-07

The following command completed successfully on the remote GB10 machine:

```bash
python3 experiments/models/qwen_semantic_block_probe.py \
  --model-path /home/pto/lean/models/Qwen/Qwen3-32B \
  --layer-idx 0 \
  --device cuda:0 \
  --max-prefill-tokens 8 \
  --disable-thinking
```

Key confirmed facts:

- the adapter-owned semantic block and the borrowed block produced the same cache sequence length: `9`
- the new page-based KV manager reported `page_size=16`, `used_pages=1`, `seq_len=9`
- forward `max_abs_diff` was `0.03125`
- prefill `max_abs_diff` was `0.03125`
- decode `max_abs_diff` was `0.125`

This is the first remote block-level proof that `leanstack` can replace borrowed Qwen layer semantics and `DynamicCache` with its own explicit control surface while staying numerically close to the reference path.
