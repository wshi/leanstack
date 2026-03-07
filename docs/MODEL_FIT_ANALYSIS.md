# Model Fit Analysis

Date verified: 2026-03-07

## Hardware envelope

The current deployment target is a single remote GB10 with about 128 GB of device memory under `/home/pto/lean`.

That hardware envelope immediately separates models into two groups:

1. models that are architecturally and memory-wise reasonable for a first cuTile-backed stack
2. models that may be strategically important but require a much richer runtime, more GPUs, or both

## Candidate models

### 1. `Qwen/Qwen3-8B` plus `nvidia/Qwen3-8B-FP4`

Why it is attractive:

- dense causal LM
- standard GQA attention
- smaller geometry that is much easier to benchmark on one GB10
- a clean separation between semantic base and deployment artifact
- strong current quality among open models

cuTile fit:

- high

Reason:

- dense transformer blocks with GQA and SwiGLU are the cleanest path for a first kernel catalog
- no MoE router, no MLA, no DSA-specific sparse path
- much more credible than the old `Qwen3-32B` path as a first performance target on one GB10

### 2. `zai-org/glm-4-9b-hf`

Why it is attractive:

- official GLM-family checkpoint
- small enough to fit comfortably
- useful as the first GLM bring-up target if keeping GLM is a hard requirement

cuTile fit:

- medium-high

Reason:

- much simpler than GLM-4.5 and GLM-5
- still requires GLM-specific adapter handling and `trust_remote_code`, but does not drag in frontier-scale MoE complexity

### 3. `zai-org/GLM-4.5-Air`

Why it matters:

- much closer to the current GLM frontier than GLM-4-9B
- official open model for agentic workloads

cuTile fit:

- medium-low

Reason:

- MoE architecture
- hybrid reasoning modes
- model-family-specific parser/runtime expectations already called out in the official card

### 4. `zai-org/GLM-5-FP8`

Why it matters:

- a recent official open-weight GLM-family checkpoint

cuTile fit:

- low for a first target

Reason:

- GLM MoE + DSA path
- official local deployment examples require tensor parallel size 8
- wrong starting point for a single-GB10 bring-up

### 5. `deepseek-ai/DeepSeek-V3`

cuTile fit:

- low

Reason:

- MLA
- DeepSeekMoE
- multi-token prediction modules
- huge total and active parameter footprint

### 6. `moonshotai/Kimi-K2-Instruct-0905`

cuTile fit:

- low

Reason:

- MLA
- large-scale MoE
- 1T total parameters
- significantly more runtime work than a dense GQA model

## Recommendation

### Best first target if cuTile simplicity is the priority

- `Qwen/Qwen3-8B` semantics with the `nvidia/Qwen3-8B-FP4` deployment artifact

### Best first target if GLM must remain the first family

- `zai-org/glm-4-9b-hf` first
- `zai-org/GLM-4.5-Air` second
- postpone `GLM-5-FP8`

### Models to postpone until the second compiler/runtime generation

- `zai-org/GLM-5-FP8`
- `deepseek-ai/DeepSeek-V3`
- `moonshotai/Kimi-K2-Instruct-0905`
