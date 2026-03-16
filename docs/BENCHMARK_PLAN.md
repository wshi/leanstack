# Benchmark Plan

Date verified: 2026-03-16

## Core question

Can a model-chip-fixed stack (`Qwen/Qwen3-4B-Base` + `GB10/sm_121`) beat framework baselines while staying simpler, when measured under a strict fairness contract?

## Official benchmark contract

- model: `Qwen/Qwen3-4B-Base` BF16 exact checkpoint
- hardware: single remote GB10 machine under `/home/pto/lean`
- primary profile: `decode_64_256`
- decode policy: greedy (`temperature=0.0`)
- stopping policy: fixed output length (`ignore_eos=true`)
- backend policy for official leanstack claims: `cuTile -> TileIR -> cubin`

Any deviation is exploratory, not official evidence.

## Systems under test

- leanstack
- vLLM
- optional secondary: SGLang / transformers (if exact checkpoint path is stable)

## Primary metrics

- generated tokens/s (primary decision metric)
- TTFT / prefill latency
- end-to-end tokens/s
- peak GPU memory
- process shape / launch complexity

## Fairness requirements (hard gate)

Each result row must satisfy:

1. same prompt-token bucket
2. same generated token count
3. generated token count equals `max_new_tokens`
4. same decode policy (greedy)
5. same stop policy (`ignore_eos=true`)

Runs failing these checks are invalid for decision making.

## Stage execution order

1. Stage 0: baseline table (`vLLM plain` + `vLLM best-of-N`)
2. Stage 1: cuTile hot-kernel bundle
3. Stage 2: runtime slices (block/prefill/decode)
4. Stage 3: full fixed-contract table

## Throughput target

Success bar:

`leanstack >= 1.30x warmed vLLM(best)` on `decode_64_256`.

Current reference (2026-03-16):

- vLLM(best): `20.3906 tok/s`
- leanstack: `20.9911 tok/s`
- ratio: `1.0294x`
- target value: `26.5078 tok/s`

## Work direction (to avoid wasted cycles)

Only prioritize changes that can move primary throughput materially:

- reduce bytes moved/token on decode hot path
- reduce kernel launches/token with contract-safe fusion
- reduce framework/service overhead with static resident decode service
- keep protocol fairness gate green after every change

Defer or drop work that improves local elegance but cannot shift `decode_64_256` tokens/s.
