from __future__ import annotations

from .config import ModelSpec


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "glm": ModelSpec(
        key="glm",
        family="GLM-family",
        loader_hint="Confirm the exact remote target checkpoint from official sources before wiring the adapter.",
        dtype="bfloat16",
        kv_layout="grouped-query attention with explicit paged KV blocks",
        required_kernels=(
            "rmsnorm",
            "rope",
            "paged-attention-prefill",
            "paged-attention-decode",
            "gated-mlp",
            "logits-and-sampler",
        ),
        bring_up_sequence=(
            "load config and tokenizer",
            "map tensor names into adapter-owned layouts",
            "validate one block forward path",
            "bring up prefill",
            "bring up decode with KV reuse",
        ),
        notes=(
            "Do not claim a latest GLM checkpoint without verifying a dated primary source.",
            "Treat remote weight loading requirements as part of the adapter contract.",
        ),
    ),
    "llama": ModelSpec(
        key="llama",
        family="Llama-family",
        loader_hint="Useful as a fallback adapter if GLM coverage reveals missing kernels.",
        dtype="bfloat16",
        kv_layout="paged causal attention",
        required_kernels=("rmsnorm", "rope", "paged-attention", "silu-mlp", "sampler"),
    ),
    "qwen": ModelSpec(
        key="qwen",
        family="Qwen-family",
        loader_hint="Useful as a second adapter once the runtime spine is stable.",
        dtype="bfloat16",
        kv_layout="paged causal attention",
        required_kernels=("rmsnorm", "rope", "paged-attention", "gated-mlp", "sampler"),
    ),
}


def get_model_spec(key: str) -> ModelSpec:
    normalized = key.strip().lower()
    try:
        return MODEL_REGISTRY[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"unknown model '{key}'. Supported models: {supported}") from exc


def list_models() -> list[ModelSpec]:
    return [MODEL_REGISTRY[key] for key in sorted(MODEL_REGISTRY)]

