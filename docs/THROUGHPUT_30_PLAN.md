# Throughput 30% Plan

Date verified: 2026-03-16

## Target

Primary success criterion:

- `leanstack generated tok/s >= 1.30x warmed vLLM(best)` on fixed contract:
  - model: `Qwen/Qwen3-4B-Base`
  - hardware: `GB10 / sm_121`
  - profile: `decode_64_256`
  - decode: greedy
  - stop: fixed length (`ignore_eos=true`)

## Current baseline

Latest fixed-contract run:

- vLLM(best-of-3): `20.3906 tok/s`
- leanstack: `20.9911 tok/s`
- ratio: `1.0294x`

Derived target:

- target throughput: `26.5078 tok/s`
- remaining gap: `5.5167 tok/s`

## First-principles implication

The remaining gap is too large for minor cleanup.

Priority must stay on changes that reduce one of:

1. bytes moved per token
2. launches per token
3. non-kernel software overhead per token

## Execution tracks

### Track A: service/runtime overhead minimization

- static resident decode process
- fixed-shape request contract
- remove dynamic control paths in steady-state loop
- keep fairness gate hard-enabled

### Track B: hot-kernel path strengthening (cuTile)

- ensure decisive decode kernels stay on cuTile/TileIR
- fuse decode-local epilogues where contract-safe
- remove fallback eager paths on hot steps

### Track C: protocol integrity

- compare only with fairness gate pass
- report plain and best vLLM explicitly
- fail fast on prompt/generation mismatch

## Go/No-Go checkpoints

1. `>=1.10x` sustained on fixed contract: continue current path.
2. `>=1.20x` requires at least one material asymmetry win (kernel or service path).
3. `<1.10x` after multiple cycles: pause and redesign, do not burn more tuning cycles on low-leverage edits.

## Rule to avoid wasted time

Any optimization without projected impact on `decode_64_256 generated tok/s` should be deprioritized.
