# Model Targets

Date verified: 2026-03-07

## Primary target

- semantic base: `Qwen/Qwen3-8B`
- active deployment contract: the public `Qwen/Qwen3-8B` BF16 checkpoint

Primary source:

- `https://huggingface.co/Qwen/Qwen3-8B`

Why it is the first target:

- dense causal LM
- standard grouped-query attention path
- much smaller and more benchmarkable than the legacy `Qwen3-32B` path on one GB10
- simpler cuTile kernel decomposition than frontier MoE and MLA models
- more credible as a first performance target because the model is small enough to avoid a trivial throughput collapse
- directly tests the thesis that a fixed model-chip contract can justify a much narrower runtime
- the active precision gate already clears BF16 on the public toolchain

Validated semantic configuration snapshot from the public `Qwen/Qwen3-8B` contract:

- 36 layers
- hidden size 4096
- 32 attention heads and 8 KV heads
- head dimension 128
- grouped-query attention
- Qwen3 chat / thinking controls remain part of the tokenizer or prompt contract, not the core kernel contract

Checkpoint note:

- `leanstack` should treat the semantic base and the BF16 checkpoint as separate but linked contracts
- the semantic base defines geometry, RoPE policy, and prompt semantics
- the active BF16 checkpoint defines the exact tensor layout that the runtime must own
- FP8 and FP4 remain deferred precision investigations, not the active first-format target

Current blocker:

- the runtime still needs to be retargeted from the legacy `Qwen3-32B` work to the smaller 8B geometry
- the current FP8 probe is still blocked at TileIR verification
- the current FP4 route is still blocked in the public frontend

First hard gate:

- keep one minimal BF16 kernel green through `cuTile DSL -> TileIR/tilebc -> tileiras -> cubin (sm_121)`
- do not treat FP8 or FP4 as active runtime targets until their gates are cleared

Primary benchmark mode:

- use non-thinking mode first so reasoning-length variance does not hide runtime effects
- keep exact-format baseline comparisons ahead of any serving work
- keep thinking-mode benchmarks as a second pass after the core throughput path is stable

## Second-family target to preserve

- `zai-org/glm-4-9b-hf`

Primary source:

- `https://huggingface.co/zai-org/glm-4-9b-hf`

Why it stays in scope:

- official GLM-family checkpoint
- still the cleanest GLM-family bring-up point once the Qwen path is stable

## Deferred frontier GLM target

- `zai-org/GLM-5-FP8`

Primary source:

- `https://huggingface.co/zai-org/GLM-5-FP8`

Why it is deferred:

- the model card indicates a far heavier deployment shape
- not the right first target for a lean, single-machine cuTile bring-up

## What success means

### Baseline success

- remote machine can acquire `Qwen/Qwen3-8B` directly or by relay
- one prompt can run through a simple Hugging Face semantic baseline for correctness
- one minimal BF16 kernel can run through the public compiler path on `sm_121`

### Stack success

- the new runtime reproduces the same single-request path without falling back to a monolithic serving framework
- the decisive BF16 linear path is owned by `leanstack`, not by a framework runtime
