from __future__ import annotations

from .config import ModelSpec


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "qwen": ModelSpec(
        key="qwen",
        family="Qwen-family",
        loader_hint="First target: start with Qwen/Qwen3-32B on Blackwell and keep the adapter dense, explicit, and framework-light.",
        dtype="bfloat16",
        kv_layout="paged grouped-query attention (64 Q heads / 8 KV heads, head_dim 128)",
        required_kernels=(
            "rmsnorm",
            "rope",
            "gqa-paged-attention-prefill",
            "gqa-paged-attention-decode",
            "silu-gated-mlp",
            "sampler",
        ),
        bring_up_sequence=(
            "load tokenizer and config",
            "verify Qwen3-32B block layout",
            "bring up one block forward path",
            "bring up prefill",
            "bring up decode with KV reuse",
        ),
        notes=(
            "Prefer ModelScope or relay-based download if Hugging Face is unreachable from the remote host.",
            "Keep Qwen as the first adapter until the runtime spine is stable.",
            "Treat vLLM, SGLang, and llama.cpp as external baselines, not implementation dependencies.",
        ),
    ),
    "glm": ModelSpec(
        key="glm",
        family="GLM-family",
        loader_hint="Second-family target: preserve GLM work after the Qwen path is stable, starting with zai-org/glm-4-9b-hf.",
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
        loader_hint="Useful as a fallback dense adapter if Qwen coverage reveals a tooling issue unrelated to model family specifics.",
        dtype="bfloat16",
        kv_layout="paged causal attention",
        required_kernels=("rmsnorm", "rope", "paged-attention", "silu-mlp", "sampler"),
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
