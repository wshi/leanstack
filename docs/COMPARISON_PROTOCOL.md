# Comparison Protocol

Date verified: 2026-03-07

## Goal

The comparison must answer one narrow question:

Can a `cuTile/TileIR`-native stack for `Qwen/Qwen3-1.7B-Base` on `GB10 / sm_121` beat compatibility-heavy frameworks on throughput while staying simpler?

## Fixed contract

- model: `Qwen/Qwen3-1.7B-Base`
- checkpoint: exact BF16 snapshot used by every system under test
- hardware: one remote GB10 under `/home/pto/lean`
- decode mode: non-thinking
- sampling: deterministic
- backend for official `leanstack` claims: `cuTile -> TileIR -> cubin`

## Why the protocol is staged

An immature full-model runtime should not be compared directly to mature frameworks and then treated as dispositive.

So the protocol is staged:

1. establish exact-checkpoint framework baselines
2. verify decisive cuTile kernels in isolation
3. verify leanstack runtime slices
4. only then make a full end-to-end claim

## Stage 0: Exact-checkpoint baselines

Systems:

- `transformers`
- `vLLM`
- `SGLang`

Profiles:

- `decode_64_256`
- `decode_64_512`
- `prefill_1024_64`

Report:

- cold TTFT
- hot TTFT
- median decode tokens/s across repeated hot runs
- median end-to-end tokens/s
- peak GPU memory

Rule:

- if a framework cannot run the exact checkpoint, mark the mismatch explicitly and keep that result outside the main claim table

## Stage 1: cuTile kernel comparisons

The decisive kernels should be benchmarked before the full-model claim:

- Q projection GEMM
- K/V projection GEMM
- O projection GEMM
- gate/up projection GEMM
- down projection GEMM
- RMSNorm
- RoPE
- decode attention

First executable bundle for `Qwen3-1.7B-Base` BF16:

- `q_proj_prefill64`
- `kv_proj_prefill64`
- `o_proj_prefill64`
- `gate_up_proj_prefill64`
- `down_proj_prefill64`
- `rmsnorm_prefill64`

Execution path:

- `PYTHONPATH=src python3 -m leanstack.cli list-hot-kernel-cases --default-only`
- `./scripts/remote_qwen_hot_kernel_bench.sh`

Comparison rule:

- compare each cuTile kernel against the closest local torch reference for the exact tensor shape
- keep tensor shapes fixed to the `Qwen3-1.7B-Base` contract
- inspect generated cubin and SASS for every hot kernel that enters the official path

These kernel results are not the final product claim, but they are the only credible way to know whether the backend is on track.

## Stage 2: Runtime-slice comparisons

Before a full end-to-end comparison, measure these `leanstack` slices:

- single-block forward
- prefill slice
- decode slice

Rule:

- keep placement GPU-resident
- forbid CPU offload
- report slice latency and allocated GPU memory

If these slices are not stable, the project should not claim anything from a full-model table.

## Stage 3: Full-stack comparison

Only after Stages 0-2 are stable:

- run `leanstack` full path
- run `vLLM`
- run `SGLang`
- keep model, prompt profile, and decode budget identical

The official claim table should contain:

- cold TTFT
- hot TTFT
- median decode tokens/s
- median end-to-end tokens/s
- peak GPU memory
- process count
- launch/config complexity notes

## Repetition policy

For each profile:

- 1 cold run after process start
- at least 5 hot runs on the same loaded process
- report median hot values

## Stop conditions

Do not make a positive performance claim unless all of the following are true:

1. the exact-checkpoint framework baselines exist
2. the decisive hot kernels are on the cuTile path
3. the runtime slices are stable and GPU-resident
4. the full-model result beats at least one framework on a key throughput metric without hiding a major memory or complexity regression

## First measured outcome

Date confirmed: 2026-03-07

The first exact-checkpoint whole-model data point now exists for `Qwen/Qwen3-1.7B-Base` on the remote GB10:

- warmed `vLLM` on `decode_64_256`: about `46.40 generated tok/s`
- current `leanstack` semantic full runtime on `decode_64_256`: about `44.61 runtime tok/s`

So the benchmark-first conclusion is no longer strongly negative, but it is still incomplete for the main steady-state decode target:

- the specialized stack has not yet cleared the warmed-framework throughput bar
- the remaining throughput gap is now small enough that kernel ownership on the decisive decode path matters more than generic control-path cleanup
- it is therefore not enough to say that ownership is solved; the next work must target the remaining decode hot spots directly

The current repo still has positive intermediate evidence:

- cold first-request `vLLM` is much slower than the loaded specialized loop
- the `16-token` UI smoke also favors `leanstack`

But those are not the deciding metrics for the main claim. The deciding metric remains warmed full-model decode throughput.
