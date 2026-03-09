# Throughput 30% Plan

Date verified: 2026-03-09

## Hard target

The project now has a stronger success bar than "slightly beat `vLLM`."

The new bar is:

- beat warmed `vLLM` by at least `30%` on the primary official decode profile
- keep the claim on the fixed contract:
  - model: `Qwen/Qwen3-1.7B-Base`
  - checkpoint: exact BF16 snapshot
  - hardware: `GB10 / sm_121`
  - backend: official path stays `cuTile -> TileIR -> cubin`

Current measured facts on `decode_64_256`:

- warmed `vLLM`: about `46.06 tok/s`
- current packed `leanpack -> leanserve` path: about `46.25 tok/s`

So the new target number is:

- `46.06 * 1.30 = 59.88 tok/s`

The current packed path is still about `13.63 tok/s` below that bar.

## First-principles conclusion

This target will not be reached by more runtime glue cleanup.

It also probably will not be reached by packing alone.

The remaining gap is too large to explain with:

- Python dispatch cleanup
- one more cache rewrite
- one more fused helper around eager PyTorch math
- one more small resident-service improvement

To reach `~60 tok/s`, the project needs a stronger asymmetry than "the stack is simpler."

That asymmetry must reduce one of these terms:

- bytes moved per emitted token
- decisive kernels launched per emitted token
- number of full-model decode steps per committed token

Only the third term is likely to create a `30%+` jump on top of a packed BF16 appliance.

## What is likely insufficient

These items are still worth doing, but they are not the main route to `1.3x`:

- finish all remaining BF16 decode kernels on `cuTile`
- replace the last eager `down_proj`, `kv_proj`, and logits paths
- make the resident service more static
- improve graph capture and bucket planning

Engineering judgment:

- this BF16 appliance work may move the stack from `46.25 tok/s` into roughly the low-`50s`
- that is useful and still required
- but it is not a credible standalone route to `~60 tok/s`

## The required asymmetry

The required new asymmetry is:

- exact speculative decode on a fixed model-chip appliance

That means:

- propose multiple tokens cheaply
- verify them exactly with the full model
- commit only verified tokens
- keep final output identical to the base model

This is the only currently credible route to a `30%+` throughput jump while preserving exact semantics.

## Official strategy

The project should now split into two coupled tracks.

### Track A: BF16 appliance ceiling

Goal:

- push the exact non-speculative packed appliance as far as it can go

Purpose:

- create the strongest possible verifier path
- establish the real BF16 non-spec ceiling
- reduce the amount of work the speculative verifier must do

Required work:

- `cuTile` decode kernels for `kv_proj`, `down_proj`, and logits
- fused decode epilogues where the model contract allows them
- more static per-bucket decode graphs
- zero hidden fallback to public checkpoint staging on the serving path

Track-A go/no-go gate:

- if the packed BF16 appliance cannot reach at least `52 tok/s`, the repo should assume a `30%` final win is very unlikely without changing the algorithmic path

### Track B: Exact self-speculative appliance

Goal:

- reduce full-model decode steps per committed token

Principle:

- do not introduce an unrelated second model first
- exploit the fixed `Qwen3-1.7B-Base` contract itself

Recommended first design:

- early-exit self-speculative decode
- one packed artifact
- two resident execution graphs:
  - `draft graph`
  - `verify graph`

The draft graph should use a proper prefix of the full network, for example:

- layers `0..11`
- or layers `0..15`
- or another empirically selected split

The verify graph should:

- run the remaining layers needed to validate the proposal block
- preserve exact output equivalence to the base model

Why this matches the repo thesis:

- one semantic contract
- one base checkpoint
- one GPU
- one packed artifact family
- agent-generated code specialized to one model and one machine

This is a better fit than introducing a generic draft-model stack.

## Leanpack v1.5 requirements

`leanpack` must evolve from "packed weights" into "packed serving program input."

That means the artifact should carry:

- packed full-model weights
- packed draft-path weights or layer ranges
- per-bucket draft lengths, such as `2-token` and `4-token` proposal modes
- verifier metadata
- rollback metadata
- graph-shape manifests for both draft and verify paths

The artifact should be able to answer:

- which layers are draft layers
- which layers are verifier-only layers
- which exact buckets are legal
- which proposal lengths are legal per bucket

## Leanserve v1.5 requirements

`leanserve` should evolve from "resident exact decode service" into "resident exact decode appliance."

That means:

- exact-bucket admission
- one resident verifier
- one resident draft path
- proposal/verify/commit loop in GPU-friendly chunks
- exact rollback when a proposal token is rejected

The appliance should expose and log:

- proposed tokens per cycle
- accepted tokens per cycle
- acceptance ratio
- committed tokens per verifier pass
- verifier time per committed token

## Quantitative gates

The project should now use these gates.

### Gate 1: BF16 exact appliance

Primary profile:

- `decode_64_256`

Required result:

- `>= 52 tok/s`

Reason:

- below this number, the exact verifier path is too weak to support a strong speculative appliance

### Gate 2: Self-spec viability

On the same primary profile:

- average committed tokens per verifier pass must exceed `1.7`
- average acceptance ratio should exceed `0.70`

Reason:

- below that range, the extra draft path complexity is unlikely to buy a `30%` end-to-end win

### Gate 3: Official success claim

Primary profile:

- `decode_64_256`

Required result:

- `leanstack >= 1.30x warmed vLLM`

With current known baseline:

- target is `>= 59.88 tok/s`

Secondary rule:

- do not allow a major regression on `decode_64_512`
- do not hide a large memory increase or a much more complex process shape

## Engineering order

1. Finish the packed BF16 appliance path enough to establish a real non-spec ceiling.
2. Build a draft/verifier split for self-speculative decode.
3. Add appliance metrics for proposal, verification, commit, and rollback.
4. Benchmark `decode_64_256` and `decode_64_512`.
5. Only then decide whether the `30%` claim is reachable on the official `cuTile` path.

## Failure mode to state explicitly

If the repo cannot exceed warmed `vLLM` by `30%` on the fixed contract after:

- packed weights
- resident appliance
- decisive `cuTile` decode kernels
- exact self-speculative decode

then the correct conclusion is not "optimize harder."

The correct conclusion is:

- under the public `cuTile` surface and this exact BF16 Qwen contract, the compatibility tax was not the dominant bottleneck
- the remaining bottleneck was algorithmic efficiency and kernel maturity, not framework complexity alone
