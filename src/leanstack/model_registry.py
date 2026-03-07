from __future__ import annotations

from .config import ModelSpec


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "qwen": ModelSpec(
        key="qwen",
        family="Qwen-family",
        loader_hint=(
            "First target: reset to Qwen3-8B BF16 on GB10/sm_121 because the public cuTile path clears BF16 today, "
            "while FP8 and FP4 remain blocked or incomplete in the current public stack."
        ),
        semantic_model_id="Qwen/Qwen3-8B",
        artifact_model_id="Qwen/Qwen3-8B",
        num_hidden_layers=36,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        target_gpu="GB10 / sm_121",
        remote_model_key="Qwen__Qwen3-8B",
        compile_gate="BF16 compiler path clears on sm_121; FP8 and FP4 remain negative or incomplete gates in the public stack.",
        legacy_reference="Qwen3-32B BF16 borrowed and semantic runtime loops remain as larger-model legacy reference data.",
        dtype="bfloat16",
        kv_layout="paged grouped-query attention (32 Q heads / 8 KV heads, head_dim 128)",
        required_kernels=(
            "bf16-gemm",
            "rmsnorm",
            "rope",
            "gqa-paged-attention-prefill",
            "gqa-paged-attention-decode",
            "silu-gated-mlp",
            "logits-projection",
            "sampler",
        ),
        bring_up_sequence=(
            "keep the BF16 compiler path green on GB10/sm_121",
            "use the public precision gate to record FP8 and FP4 blockers explicitly",
            "load Qwen3-8B config and BF16 checkpoint metadata",
            "map dense BF16 linears into adapter-owned layouts",
            "bring up one BF16 transformer block forward path",
            "bring up prefill and decode with explicit KV reuse",
        ),
        static_contract=(
            "Model semantics are fixed to Qwen3-8B with 36 layers, hidden size 4096, and GQA geometry 32Q/8KV/128.",
            "Deployment target is fixed to the public Qwen3-8B BF16 checkpoint on GB10 / sm_121.",
            "Only the user request is intended to stay dynamic; model geometry, precision policy, kernel inventory, and dispatch order should be fixed by the model-chip contract.",
            "KV page layout, RoPE policy, BF16 linear strategy, and MLP fusion rules are fixed by the adapter.",
            "The intended execution path is GPU-resident and explicit, without framework-managed CPU offload.",
            "FP8 and FP4 stay deferred until the public precision gate reports a positive result.",
        ),
        dynamic_inputs=(
            "User request payload: prompt tokens and requested decode budget.",
            "Stopping condition derived from generated tokens or stop tokens.",
        ),
        deferred_compatibility=(
            "Cross-model compatibility outside the first Qwen3-8B BF16 contract.",
            "Cross-hardware portability beyond GB10 / sm_121.",
            "Framework-style automatic placement, fallback, and heterogeneous offload behavior.",
        ),
        notes=(
            "Treat the previous Qwen3-32B BF16 runtime work as a larger-model reference path, not the active first target.",
            "Prefer relay-based model delivery if the remote host cannot fetch the Qwen3-8B checkpoint directly.",
            "Treat TensorRT-LLM, vLLM, and SGLang as external baselines, not implementation dependencies.",
            "Treat broad compatibility as a deferred cost unless it is required by the first model-chip contract.",
            "Use the executable precision gate as the source of truth: BF16 is currently cleared, FP8 is currently blocked at TileIR verification, and FP4 lacks a complete public frontend surface.",
        ),
    ),
    "glm": ModelSpec(
        key="glm",
        family="GLM-family",
        loader_hint="Second-family target: preserve GLM work after the Qwen path is stable, starting with zai-org/glm-4-9b-hf.",
        semantic_model_id="zai-org/glm-4-9b-hf",
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
        dynamic_inputs=("User request payload.",),
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
        target_gpu="GB10 / sm_121",
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
