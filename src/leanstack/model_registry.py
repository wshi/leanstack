from __future__ import annotations

from .config import ModelSpec


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "qwen": ModelSpec(
        key="qwen",
        family="Qwen-family",
        loader_hint=(
            "First target: pivot to Qwen3-8B semantics with an NVFP4 deployment artifact on GB10/sm_121, "
            "and treat FP4 compiler feasibility as the first hard gate before building a larger runtime."
        ),
        semantic_model_id="Qwen/Qwen3-8B",
        artifact_model_id="nvidia/Qwen3-8B-FP4",
        num_hidden_layers=36,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        target_gpu="GB10 / sm_121",
        remote_model_key="Qwen__Qwen3-8B",
        compile_gate="minimal FP4 or NVFP4 kernel through `cuTile -> TileIR/tileiras -> cubin (sm_121)`",
        legacy_reference="Qwen3-32B BF16 borrowed and semantic runtime loops remain as legacy reference data.",
        dtype="nvfp4 linears with higher-precision residual path",
        kv_layout="paged grouped-query attention (32 Q heads / 8 KV heads, head_dim 128)",
        required_kernels=(
            "fp4-blockscaled-gemm",
            "rmsnorm",
            "rope",
            "gqa-paged-attention-prefill",
            "gqa-paged-attention-decode",
            "silu-gated-mlp",
            "dequant-scale-epilogue",
            "sampler",
        ),
        bring_up_sequence=(
            "verify public FP4 compiler coverage on GB10/sm_121",
            "compile one minimal FP4 GEMM through the cuTile-native path",
            "load Qwen3-8B config and FP4 artifact metadata",
            "map FP4 linears and scale tensors into adapter-owned layouts",
            "bring up one FP4 transformer block forward path",
            "bring up prefill and decode with explicit KV reuse",
        ),
        static_contract=(
            "Model semantics are fixed to Qwen3-8B with 36 layers, hidden size 4096, and GQA geometry 32Q/8KV/128.",
            "Deployment target is fixed to an NVFP4 Qwen3-8B artifact on GB10 / sm_121.",
            "Only the user request is intended to stay dynamic; model geometry, quantization policy, kernel inventory, and dispatch order should be fixed by the model-chip contract.",
            "KV page layout, RoPE policy, FP4 linear strategy, and MLP fusion rules are fixed by the adapter.",
            "The intended execution path is GPU-resident and explicit, without framework-managed CPU offload.",
            "If the public cuTile-native compiler path cannot emit the required FP4 kernels on sm_121, the FP4 route is not considered proven.",
        ),
        dynamic_inputs=(
            "User request payload: prompt tokens and requested decode budget.",
            "Stopping condition derived from generated tokens or stop tokens.",
        ),
        deferred_compatibility=(
            "Cross-model compatibility outside the first Qwen3-8B NVFP4 contract.",
            "Cross-hardware portability beyond GB10 / sm_121.",
            "Framework-style automatic placement, fallback, and heterogeneous offload behavior.",
        ),
        notes=(
            "Treat the previous Qwen3-32B BF16 runtime work as a legacy reference path, not the active first target.",
            "Prefer relay-based model delivery if the remote host cannot fetch the FP4 artifact directly.",
            "Treat TensorRT-LLM, vLLM, and SGLang as external baselines, not implementation dependencies.",
            "Treat broad compatibility as a deferred cost unless it is required by the first model-chip contract.",
            "Do not continue the FP4 runtime path unless a minimal FP4 kernel is proven through `cuTile/TileIR -> tileiras -> cubin` on sm_121.",
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
