from __future__ import annotations

from .config import ModelSpec


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "qwen": ModelSpec(
        key="qwen",
        family="Qwen-family",
        loader_hint=(
            "First target: maximize throughput on GB10/sm_121 with Qwen3-1.7B-Base BF16 while keeping the hot path "
            "inside `cuTile -> TileIR -> cubin`."
        ),
        semantic_model_id="Qwen/Qwen3-1.7B-Base",
        artifact_model_id="Qwen/Qwen3-1.7B-Base",
        num_hidden_layers=28,
        hidden_size=2048,
        intermediate_size=6144,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=151936,
        max_position_embeddings=32768,
        target_gpu="GB10 / sm_121",
        remote_model_key="Qwen__Qwen3-1.7B-Base",
        compile_gate=(
            "BF16 precision gate clears on sm_121. Official comparison path must keep decisive kernels on `cuTile -> TileIR -> cubin`."
        ),
        legacy_reference="Qwen3-32B BF16 borrowed and semantic runtime loops remain as larger-model legacy reference data.",
        dtype="bfloat16",
        kv_layout="paged grouped-query attention (16 Q heads / 8 KV heads, head_dim 128)",
        backend_policy=(
            "Keep decisive kernels on `cuTile -> TileIR -> cubin` for the official comparison path.",
            "Use PTX only as a diagnostic aid when understanding a compiler miss, not as the benchmarked steady-state backend.",
            "Inspect SASS for every hot kernel that enters the official comparison path.",
        ),
        exact_prompt_buckets=(64, 512, 1024),
        pack_contract=(
            "Convert the public BF16 checkpoint into a serving-only artifact instead of reading Hugging Face tensor names at request time.",
            "Offline-pack `QKV`, `gate/up`, `down_proj`, and `lm_head` into the exact layout consumed by the active cuTile kernels.",
            "Precompute bucket-specific metadata: exact prompt buckets, scratch sizes, KV extents, and graph-capture shapes.",
            "Treat the original checkpoint as staging input; treat the packed artifact as the serving format.",
        ),
        serve_contract=(
            "Run one resident process per GPU for one fixed model and one fixed precision policy.",
            "Accept only exact prompt buckets for the official benchmark path, with deterministic decode and fixed stopping rules.",
            "Keep weights, KV, scratch buffers, and per-bucket graphs resident on GPU after startup.",
            "Avoid framework-style automatic placement, runtime discovery, heterogeneous offload, and cross-model dispatch.",
        ),
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
            "load Qwen3-1.7B-Base config and BF16 checkpoint metadata",
            "map dense BF16 linears into adapter-owned layouts",
            "bring up one BF16 transformer block forward path through cuTile/TileIR",
            "bring up prefill and decode with explicit KV reuse",
        ),
        static_contract=(
            "Model semantics are fixed to Qwen3-1.7B-Base with 28 layers, hidden size 2048, intermediate size 6144, and GQA geometry 16Q/8KV/128.",
            "Deployment target is fixed to the public Qwen3-1.7B-Base BF16 checkpoint on GB10 / sm_121.",
            "Only the user request is intended to stay dynamic; model geometry, precision policy, kernel inventory, and dispatch order should be fixed by the model-chip contract.",
            "KV page layout, RoPE policy, BF16 linear strategy, and MLP fusion rules are fixed by the adapter.",
            "Official benchmark buckets are fixed and exact, not merely capped: 64-token decode buckets and 1024-token prefill buckets.",
            "The intended execution path is GPU-resident and explicit, without framework-managed CPU offload.",
            "FP8 and FP4 stay deferred until they beat or clearly improve upon the BF16 throughput-first path.",
        ),
        dynamic_inputs=(
            "User request payload within the fixed bucket contract: prompt content and requested decode budget.",
            "Stopping condition derived from generated tokens or stop tokens.",
        ),
        deferred_compatibility=(
            "Cross-model compatibility outside the first Qwen3-1.7B-Base BF16 contract.",
            "Cross-hardware portability beyond GB10 / sm_121.",
            "Framework-style automatic placement, fallback, and heterogeneous offload behavior.",
        ),
        notes=(
            "Treat the previous Qwen3-32B BF16 runtime work as a larger-model reference path, not the active first target.",
            "Prefer relay-based model delivery if the remote host cannot fetch the Qwen3-1.7B-Base checkpoint directly.",
            "Treat TensorRT-LLM, vLLM, and SGLang as external baselines, not implementation dependencies.",
            "Treat broad compatibility as a deferred cost unless it is required by the first model-chip contract.",
            "Treat throughput as the primary optimization target, but only count wins that stay on the cuTile/TileIR path as official project evidence.",
            "Do not expect a smaller runtime alone to beat vLLM; the project must reduce bytes moved per token, kernels launched per token, or host orchestration per token.",
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
