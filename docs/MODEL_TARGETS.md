# Model Targets

Date verified: 2026-03-07

## Primary target

- semantic base: `Qwen/Qwen3-8B`
- deployment artifact: `nvidia/Qwen3-8B-FP4`

Primary source:

- `https://huggingface.co/Qwen/Qwen3-8B`
- `https://huggingface.co/nvidia/Qwen3-8B-FP4`

Why it is the first target:

- dense causal LM
- standard grouped-query attention path
- much smaller and more benchmarkable than the legacy `Qwen3-32B` path on one GB10
- simpler cuTile kernel decomposition than frontier MoE and MLA models
- more credible as a first performance target because the model is small enough to avoid a trivial throughput collapse
- directly tests the thesis that a fixed model-format-chip contract can justify a much narrower runtime

Validated semantic configuration snapshot from the public `Qwen/Qwen3-8B` contract:

- 36 layers
- hidden size 4096
- 32 attention heads and 8 KV heads
- head dimension 128
- grouped-query attention
- Qwen3 chat / thinking controls remain part of the tokenizer or prompt contract, not the core kernel contract

Artifact note:

- `leanstack` should treat the semantic base and the FP4 deployment artifact as separate but linked contracts
- the semantic base defines geometry, RoPE policy, and prompt semantics
- the FP4 artifact defines the exact linear weight and scale layout that the runtime must own

Current blocker:

- the public remote `cuda.tile 1.1.0` install exposes dtypes up to FP8, not a visible public FP4 or NVFP4 dtype
- the remote `tileiras` binary does target `sm_121`, so backend targeting is present even though frontend FP4 coverage is still unproven
- the remote machine may still need relay or mirror-based delivery for the NVIDIA FP4 artifact if direct access is blocked

First hard gate:

- prove one minimal FP4 or NVFP4 kernel through `cuTile DSL -> TileIR/tilebc -> tileiras -> cubin (sm_121)`
- do not continue the FP4 runtime plan until this gate is cleared

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

- remote machine can acquire `Qwen/Qwen3-8B` semantics and the `Qwen3-8B-FP4` deployment artifact, directly or by relay
- one prompt can run through a simple Hugging Face semantic baseline for correctness
- one minimal FP4 kernel can run through the public compiler path on `sm_121`

### Stack success

- the new runtime reproduces the same single-request path without falling back to a monolithic serving framework
- the decisive FP4 linear path is owned by `leanstack`, not by a framework runtime
