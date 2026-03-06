# Model Targets

Date verified: 2026-03-06

## Primary target

- `Qwen/Qwen3-32B`

Primary source:

- `https://huggingface.co/Qwen/Qwen3-32B`
- `https://www.modelscope.cn/models/Qwen/Qwen3-32B`

Why it is the first target:

- dense causal LM
- standard grouped-query attention path
- simpler cuTile kernel decomposition than frontier MoE and MLA models
- strong enough to be a credible first end-to-end target
- realistic fit for a single GB10-class machine compared with larger frontier MoE models

Current blocker:

- as of 2026-03-06, the DGX Spark machine times out when requesting Hugging Face model artifacts directly
- the remote machine can reach ModelScope metadata and can download the `modelscope` wheel, so relay or mirror-based download should be part of the first deployment workflow

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

- remote machine can download config, tokenizer, and weights for `Qwen/Qwen3-32B` or ingest them via relay
- one prompt can run through a simple Hugging Face baseline

### Stack success

- the new runtime reproduces the same single-request path without falling back to a monolithic serving framework
