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

## Whole-model benchmark status

Date confirmed: 2026-03-09

The active full-model benchmark path now runs against the exact `Qwen/Qwen3-1.7B-Base` BF16 snapshot stored under `/home/pto/lean/models/Qwen/Qwen3-1___7B-Base`.

Key confirmed facts:

- the public ModelScope snapshot is a single-file checkpoint with `model.safetensors`, not a multi-shard HF index
- the current loader now accepts both `model.safetensors.index.json` and single-file `model.safetensors` layouts
- this checkpoint does not expose a separate `lm_head.weight`; the runtime now treats `model.embed_tokens.weight` as the tied output head
- a local comparison UI is available on the Mac through `python3 scripts/serve_compare_ui.py --port 8787`
- the UI now runs the two systems sequentially on the same remote GPU:
  1. ensure `vLLM` is ready
  2. run the `vLLM` benchmark
  3. stop `vLLM`
  4. run the `leanstack` benchmark

Measured whole-model results so far:

- `vLLM`, cold first request, `decode_64_256`:
  - `ttft_seconds ≈ 18.15`
  - `generated_tokens_per_second ≈ 10.94`
- `vLLM`, warmed request on the same loaded service, `decode_64_256`:
  - `ttft_seconds ≈ 0.233`
  - `generated_tokens_per_second ≈ 46.40`
  - `end_to_end_tokens_per_second ≈ 47.84`
- `leanstack`, earlier full semantic runtime, `decode_64_256`:
  - `materialize_seconds ≈ 15.80`
  - `prefill_seconds ≈ 0.594`
  - `runtime_tokens_per_second ≈ 29.95`
  - `full_loop_tokens_per_second ≈ 29.85`
- `leanstack`, optimized semantic runtime, `decode_64_256`:
  - `materialize_seconds ≈ 15.90`
  - `prefill_seconds ≈ 0.020`
  - `decode_loop_seconds ≈ 5.73`
  - `runtime_tokens_per_second ≈ 44.55`
  - `full_loop_tokens_per_second ≈ 44.40`
- `leanstack`, decode-only tightened semantic runtime, `decode_64_256`:
  - `materialize_seconds ≈ 15.93`
  - `prefill_seconds ≈ 0.020`
  - `decode_loop_seconds ≈ 5.72`
  - `runtime_tokens_per_second ≈ 44.61`
  - `full_loop_tokens_per_second ≈ 44.46`
- `leanpack`, remote artifact inventory:
  - path: `/home/pto/lean/packed/Qwen__Qwen3-1.7B-Base`
  - size on disk: about `3.8G`
  - files: `30`
  - tensors: `227`
- `leanserve`, resident layout from the packed artifact:
  - `resident_weights_bytes ≈ 4.06G`
  - `max_kv_cache_bytes ≈ 124.8M`
  - `resident_bytes ≈ 4.19G`
- `leanstack`, full packed semantic runtime from `leanpack`, exact `decode_64_256`:
  - `materialize_seconds ≈ 24.39`
  - `prefill_seconds ≈ 0.020`
  - `decode_loop_seconds ≈ 5.51`
  - `runtime_tokens_per_second ≈ 46.25`
  - `full_loop_tokens_per_second ≈ 46.09`
- `leanstack`, first exact speculative smoke from the packed artifact, `16-token` decode:
  - `draft=12`, `k=4`: `acceptance_ratio ≈ 0.037`, `committed_tokens_per_cycle ≈ 1.07`
  - `draft=20`, `k=2`: `acceptance_ratio ≈ 0.227`, `committed_tokens_per_cycle ≈ 1.45`
  - `draft=24`, `k=2`: `acceptance_ratio ≈ 0.368`, `committed_tokens_per_cycle ≈ 1.60`
- `UI smoke`, `max_new_tokens=16`:
  - `vLLM generated_tokens_per_second ≈ 10.56`
  - `leanstack runtime_tokens_per_second ≈ 14.64`

Interpretation:

- the active specialized runtime is now real enough to run a full 28-layer model on the target GB10
- the current `leanstack` path already beats the cold first-request framework path and a short `16-token` UI smoke
- the checkpoint-driven semantic path no longer trails warmed `vLLM` by a wide margin, but the more important fact is that the packed `leanpack -> leanserve` path now narrowly clears the warmed `vLLM` number on the main exact-bucket `decode_64_256` profile
- the packed path is therefore the new official serving path for performance work; future optimization should focus on widening that margin, not on re-optimizing the older checkpoint-driven runtime
- the first exact speculative loop is now implemented and working, but the initial acceptance ratios show that a naive early-exit draft with the shared final head is still too weak for the `30%` target

## Stage 1 hot-kernel status

Date confirmed: 2026-03-07

The following command completed successfully on the remote machine:

- `./scripts/remote_qwen_hot_kernel_bench.sh`

Key confirmed facts:

- artifact written to `/home/pto/lean/artifacts/hot-kernels/20260307T094320Z`
- `.cutile`, `cubin`, and SASS artifacts were emitted for the BF16 hot-kernel suite
- active default bundle:
  - `q_proj_prefill64`
  - `kv_proj_prefill64`
  - `o_proj_prefill64`
  - `gate_up_proj_prefill64`
  - `down_proj_prefill64`
  - `rmsnorm_prefill64`

Measured medians against local torch references:

- `q_proj_prefill64`: `20.01 TFLOPS` vs `17.28 TFLOPS`, `1.16x`
- `kv_proj_prefill64`: `10.05 TFLOPS` vs `16.09 TFLOPS`, `0.62x`
- `o_proj_prefill64`: `20.14 TFLOPS` vs `16.87 TFLOPS`, `1.19x`
- `gate_up_proj_prefill64`: `22.29 TFLOPS` vs `20.36 TFLOPS`, `1.09x`
- `down_proj_prefill64`: `6.34 TFLOPS` vs `19.19 TFLOPS`, `0.33x`
- `rmsnorm_prefill64`: `0.0795 TFLOPS` vs `0.0152 TFLOPS`, `5.23x`

Important implementation note:

- a single generic `ct.kernel` reused across multiple tile shapes produced incorrect `down_proj` results in a mixed suite
- the current repo now generates tile-shape-specific kernel objects, which makes `down_proj_prefill64` correct again in the shared benchmark run

Interpretation:

- the public BF16 path is already competitive on `q_proj`, `o_proj`, `gate_up`, and `rmsnorm`
- `kv_proj` and especially `down_proj` are still below the torch reference and remain active optimization targets
- this is now a real backend profile, not just a planning assumption

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
