# Comparison Protocol

Date verified: 2026-03-16

## Goal

The official claim must answer one narrow question:

Can a `cuTile/TileIR`-native leanstack for `Qwen/Qwen3-4B-Base` on `GB10 / sm_121` deliver at least `1.30x` warmed `vLLM` throughput on `decode_64_256`, with lower stack complexity and without protocol mismatch?

## Fixed contract (official claim only)

- model: `Qwen/Qwen3-4B-Base`
- checkpoint: exact BF16 snapshot (same bytes for all systems under test)
- hardware: one remote GB10 host under `/home/pto/lean`
- profile: `decode_64_256`
- runtime mode: non-thinking
- decode policy: greedy (`temperature=0.0`)
- stopping policy: fixed-length decode (`ignore_eos=true`, emit exactly `max_new_tokens`)
- official leanstack backend: `cuTile -> TileIR -> cubin`
- serving artifact: strict packed appliance (`leanpack` manifest required)

Result labels:

- `fixed-contract`: all fields above satisfied; eligible for thesis-level claim
- `exploratory`: any deviation; useful for direction only

## Mandatory fairness gate (hard fail)

A comparison run is invalid if any of the following fails:

1. prompt tokens match exactly between systems
2. generated tokens match exactly between systems
3. generated tokens equal target `max_new_tokens`
4. decode policy is greedy on both sides
5. stopping policy is fixed-length (`ignore_eos=true`) on both sides

If fairness fails, the run must not enter summary tables.

## Stage order (go/no-go)

1. Stage 0: exact-checkpoint framework baselines (`vLLM`, optional `SGLang`/`transformers`)
2. Stage 1: decisive cuTile hot kernels
3. Stage 2: leanstack runtime slices (block/prefill/decode)
4. Stage 3: fixed-contract full-stack comparison

No stage skipping for official claim data.

## Stage 0: baseline requirements

- run same model/checkpoint/profile/prompt bucket
- report cold + hot throughput
- for vLLM report both:
  - `plain`: single run
  - `best`: best of repeated runs (default 3; report candidates)

Main throughput baseline for claim = warmed `vLLM best`.

## Stage 1: kernel requirements

Decisive kernels must be benchmarked on exact Qwen3-4B geometry:

- Q/KV/O projection
- gate/up and down projection
- RMSNorm
- RoPE
- decode attention path

Each kernel entering official path must have:

- shape-locked benchmark result
- cubin artifact
- SASS inspection record

## Stage 2: runtime-slice requirements

Before full-stack claims, leanstack must show stable:

- single-block forward
- prefill slice
- decode slice

with GPU-resident placement and no hidden CPU offload.

## Stage 3: official claim table

Required fields:

- TTFT / prefill latency
- generated tokens/s (primary)
- end-to-end tokens/s
- peak memory
- process/launch complexity summary
- fairness-gate status

## Stop conditions

Do not claim performance advantage unless all are true:

1. fairness gate passes
2. fixed-contract fields are unchanged
3. hot path stays on official backend policy
4. repeated hot-run result is stable

Do not claim thesis success unless:

`leanstack generated tok/s >= 1.30 * warmed vLLM(best) generated tok/s` on `decode_64_256`.

## Current status (for planning)

Latest fixed-contract run on 2026-03-16:

- vLLM(best-of-3): `20.3906 tok/s`
- leanstack: `20.9911 tok/s`
- ratio: `1.0294x`

Current gap to thesis bar:

- target = `26.5078 tok/s`
- remaining gap = `5.5167 tok/s`

Conclusion:

- positive but small lead exists
- protocol mismatch risk is now controlled by fairness gate
- next work should focus only on optimizations that can realistically close the remaining `~5.5 tok/s` gap
