# Model Targets

Date verified: 2026-03-06

## Official GLM checkpoints to track

### Long-term target

- `zai-org/GLM-5-FP8`

Primary source:

- `https://huggingface.co/zai-org/GLM-5-FP8`

Why it matters:

- it is the most recent official open-weight GLM-family checkpoint visible from the official `zai-org` Hugging Face organization as of 2026-03-06
- it reflects the actual frontier target for the stack

Why it is not the first runtime target on this machine:

- the model card describes multi-GPU deployment expectations
- a single GB10 with 128 GB device memory is not the right bring-up target for this checkpoint

### First runnable baseline target

- `zai-org/glm-4-9b-hf`

Primary source:

- `https://huggingface.co/zai-org/glm-4-9b-hf`

Why it is the right first baseline:

- official GLM-family checkpoint
- Hugging Face-compatible layout
- realistic fit for a single 128 GB GB10
- good enough to validate tokenizer, weight loading, prefill, and decode plumbing before scaling up

## Working decision

Use two tracks instead of forcing one target:

1. `GLM-5-FP8` remains the architecture target for kernel and runtime planning.
2. `glm-4-9b-hf` is the first practical checkpoint for remote end-to-end smoke and adapter bring-up.

## What success means

### Baseline success

- remote machine can download config, tokenizer, and weights for the selected GLM-family checkpoint
- one prompt can run through a simple Hugging Face baseline

Current blocker:

- as of 2026-03-06, the DGX Spark machine times out when requesting Hugging Face model artifacts directly, so model download needs a mirror, proxy, or pre-staged weights

### Stack success

- the new runtime reproduces the same single-request path without falling back to a monolithic serving framework
