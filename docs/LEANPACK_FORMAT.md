# Leanpack Format

Date verified: 2026-03-09

## Goal

`leanpack` exists to separate the public checkpoint format from the serving format.

The public Qwen checkpoint is a staging input.
The `leanpack` artifact is the serving input for `leanserve`.

## Layout

A minimal `leanpack/v0` artifact contains:

- `manifest.json`
- `embeddings.safetensors`
- `layer-000.safetensors` through `layer-027.safetensors`
- `output.safetensors`

## Packed tensors

The current Qwen packer writes these serving tensors:

- `model.embed_tokens.weight`
- `layers.<i>.input_layernorm.weight`
- `layers.<i>.attention.qkv_proj.weight`
- `layers.<i>.attention.o_proj.weight`
- `layers.<i>.attention.q_norm.weight`
- `layers.<i>.attention.k_norm.weight`
- `layers.<i>.post_attention_layernorm.weight`
- `layers.<i>.mlp.gate_up_proj.weight`
- `layers.<i>.mlp.down_proj.weight`
- `model.norm.weight`
- `lm_head.weight`

The fused serving tensors deliberately remove some runtime concatenation work:

- `q_proj + k_proj + v_proj -> attention.qkv_proj.weight`
- `gate_proj + up_proj -> mlp.gate_up_proj.weight`

## Manifest fields

`manifest.json` records:

- source checkpoint path
- target GPU contract
- dtype
- model geometry and critical hyperparameters (`rms_norm_eps`, `rope_theta`)
- exact prompt buckets
- benchmark bucket metadata
- required kernels
- backend policy
- file inventory
- tensor inventory with role, shape, dtype, source tensors, and logical offsets

The `logical_offset_bytes` field is a serving-layout offset within each packed group, not a promise about low-level safetensors header offsets.

## Current status

`leanpack/v0` is a first executable format, not the final serving format.

What it already gives us:

- serving-only fused tensor names
- stable per-layer file boundaries
- explicit bucket metadata
- enough model geometry to rebuild a resident semantic stack without consulting Hugging Face tensor names
- a machine-readable manifest for later `leanserve` work

What it does not yet give us:

- cuTile-specific physical tiling on disk
- precomputed graph binaries
- packed KV/cache state
- direct mmap-style offset loading
