from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from .config import ModelSpec
from .pack import DraftHeadEntry, PackedArtifactManifest, PackedTensorEntry, load_packed_artifact_manifest
from .runtime.qwen_explicit import (
    QwenSemanticAttentionRuntime,
    QwenSemanticLayerRuntime,
    QwenSemanticMlpRuntime,
    QwenSemanticStackRuntime,
    resolve_layer_indices,
    resolve_torch_dtype,
)


def _dtype_nbytes_from_name(dtype_name: str) -> int:
    normalized = dtype_name.strip().lower()
    if normalized in {"float16", "bfloat16", "int16"}:
        return 2
    if normalized in {"float32", "int32"}:
        return 4
    if normalized in {"float64", "int64"}:
        return 8
    if normalized in {"int8", "uint8", "bool"}:
        return 1
    raise ValueError(f"unsupported dtype name: {dtype_name}")


@dataclass(frozen=True)
class QwenServeConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    head_dim: int
    rms_norm_eps: float
    rope_parameters: dict[str, float]
    _attn_implementation: str = "sdpa"


@dataclass(frozen=True)
class ScratchPlan:
    hidden_state_bytes: int
    qkv_projection_bytes: int
    attention_output_bytes: int
    mlp_activation_bytes: int
    logits_bytes: int

    @property
    def total_bytes(self) -> int:
        return (
            self.hidden_state_bytes
            + self.qkv_projection_bytes
            + self.attention_output_bytes
            + self.mlp_activation_bytes
            + self.logits_bytes
        )


@dataclass(frozen=True)
class ResidentBucketPlan:
    key: str
    prompt_tokens: int
    max_new_tokens: int
    max_seq_len: int
    kv_tokens_capacity: int
    kv_cache_bytes: int


@dataclass(frozen=True)
class ResidentBufferPlan:
    weights_bytes: int
    page_size: int
    batch_size: int
    scratch: ScratchPlan
    buckets: tuple[ResidentBucketPlan, ...]

    @property
    def max_kv_cache_bytes(self) -> int:
        return max((bucket.kv_cache_bytes for bucket in self.buckets), default=0)

    @property
    def resident_bytes(self) -> int:
        return self.weights_bytes + self.max_kv_cache_bytes + self.scratch.total_bytes


@dataclass
class LeanPackArtifact:
    root: Path
    manifest: PackedArtifactManifest

    def __post_init__(self) -> None:
        self._tensor_index: dict[str, PackedTensorEntry] = {
            entry.name: entry for entry in self.manifest.tensors
        }

    def tensor_entry(self, name: str) -> PackedTensorEntry:
        return self._tensor_index[name]

    def file_for_tensor(self, name: str) -> Path:
        return self.root / self.tensor_entry(name).file

    def load_tensors(
        self,
        file_name: str,
        tensor_names: tuple[str, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        loaded: dict[str, torch.Tensor] = {}
        with safe_open(self.root / file_name, framework="pt", device="cpu") as handle:
            for tensor_name in tensor_names:
                loaded[tensor_name] = handle.get_tensor(tensor_name).to(device=device, dtype=dtype).contiguous()
        return loaded

    def load_tensor(
        self,
        file_name: str,
        tensor_name: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.load_tensors(file_name, (tensor_name,), device=device, dtype=dtype)[tensor_name]

    def find_draft_head(self, draft_layer_count: int, key: str | None = None) -> DraftHeadEntry | None:
        matches = [
            head for head in self.manifest.draft_heads
            if head.draft_layer_count == draft_layer_count and (key is None or head.key == key)
        ]
        if not matches:
            return None
        if key is not None:
            return matches[0]
        return matches[-1]

    def qwen_config(self, fallback_model: ModelSpec | None = None) -> QwenServeConfig:
        geometry = dict(self.manifest.geometry)
        if not geometry:
            geometry = self._infer_qwen_geometry()
        if not geometry and fallback_model is not None:
            geometry = {
                "num_hidden_layers": int(fallback_model.num_hidden_layers or 0),
                "hidden_size": int(fallback_model.hidden_size or 0),
                "intermediate_size": int(fallback_model.intermediate_size or 0),
                "num_attention_heads": int(fallback_model.num_attention_heads or 0),
                "num_key_value_heads": int(fallback_model.num_key_value_heads or 0),
                "head_dim": int(fallback_model.head_dim or 0),
                "vocab_size": int(fallback_model.vocab_size or 0),
                "max_position_embeddings": int(fallback_model.max_position_embeddings or 0),
            }
        if not geometry:
            raise ValueError("leanpack manifest is missing geometry and no fallback model was provided")
        hparams = dict(self.manifest.model_hparams)
        return QwenServeConfig(
            vocab_size=int(geometry["vocab_size"]),
            hidden_size=int(geometry["hidden_size"]),
            intermediate_size=int(geometry["intermediate_size"]),
            num_hidden_layers=int(geometry["num_hidden_layers"]),
            num_attention_heads=int(geometry["num_attention_heads"]),
            num_key_value_heads=int(geometry["num_key_value_heads"]),
            max_position_embeddings=int(geometry["max_position_embeddings"]),
            head_dim=int(geometry["head_dim"]),
            rms_norm_eps=float(hparams.get("rms_norm_eps", 1e-6)),
            rope_parameters={"rope_theta": float(hparams.get("rope_theta", 1_000_000.0))},
        )

    def _infer_qwen_geometry(self) -> dict[str, int]:
        try:
            embed_entry = self.tensor_entry("model.embed_tokens.weight")
            qkv_entry = self.tensor_entry("layers.0.attention.qkv_proj.weight")
            gate_up_entry = self.tensor_entry("layers.0.mlp.gate_up_proj.weight")
            q_norm_entry = self.tensor_entry("layers.0.attention.q_norm.weight")
        except KeyError:
            return {}

        vocab_size, hidden_size = embed_entry.shape
        head_dim = int(q_norm_entry.shape[0])
        num_attention_heads = int(hidden_size // head_dim)
        qkv_out_dim = int(qkv_entry.shape[0])
        q_proj_out_dim = int(hidden_size)
        kv_proj_out_dim = int((qkv_out_dim - q_proj_out_dim) // 2)
        num_key_value_heads = int(kv_proj_out_dim // head_dim)
        intermediate_size = int(gate_up_entry.shape[0] // 2)
        layer_indices = [
            int(path.stem.split("-")[-1])
            for path in self.root.glob("layer-*.safetensors")
            if path.stem.split("-")[-1].isdigit()
        ]
        bucket_max_seq = max(
            (int(bucket.prompt_tokens) + int(bucket.max_new_tokens) for bucket in self.manifest.buckets),
            default=max(self.manifest.exact_prompt_buckets, default=0),
        )
        return {
            "num_hidden_layers": max(layer_indices) + 1 if layer_indices else 0,
            "hidden_size": int(hidden_size),
            "intermediate_size": int(intermediate_size),
            "num_attention_heads": int(num_attention_heads),
            "num_key_value_heads": int(num_key_value_heads),
            "head_dim": int(head_dim),
            "vocab_size": int(vocab_size),
            "max_position_embeddings": int(bucket_max_seq),
        }

    def validate_for_model(self, model: ModelSpec) -> None:
        if self.manifest.model_key != model.key:
            raise ValueError(
                f"leanpack model_key mismatch: manifest={self.manifest.model_key} requested={model.key}"
            )
        if model.semantic_model_id and self.manifest.semantic_model_id not in (None, model.semantic_model_id):
            raise ValueError(
                "leanpack semantic_model_id mismatch: "
                f"manifest={self.manifest.semantic_model_id} expected={model.semantic_model_id}"
            )

    def describe(self) -> str:
        lines = [f"Leanpack artifact: {self.root}"]
        lines.append(f"format: {self.manifest.format_version}")
        lines.append(f"model: {self.manifest.model_key} ({self.manifest.semantic_model_id})")
        lines.append(f"dtype: {self.manifest.dtype}")
        if self.manifest.target_gpu:
            lines.append(f"target_gpu: {self.manifest.target_gpu}")
        if self.manifest.geometry:
            lines.append(f"geometry: {self.manifest.geometry}")
        if self.manifest.exact_prompt_buckets:
            lines.append(
                "exact_prompt_buckets: " + ", ".join(str(bucket) for bucket in self.manifest.exact_prompt_buckets)
            )
        if self.manifest.speculative_modes:
            lines.append(
                "speculative_modes: "
                + ", ".join(
                    f"{mode.key}(layers={mode.draft_layer_count},k={mode.proposal_len})"
                    for mode in self.manifest.speculative_modes
                )
            )
        if self.manifest.draft_heads:
            lines.append(
                "draft_heads: "
                + ", ".join(
                    f"{head.key}(layers={head.draft_layer_count},file={head.file})"
                    for head in self.manifest.draft_heads
                )
            )
        lines.append(f"files: {len(self.manifest.files)}")
        lines.append(f"tensors: {len(self.manifest.tensors)}")
        return "\n".join(lines)


@dataclass(frozen=True)
class LeanServeAppliance:
    model: ModelSpec
    artifact: LeanPackArtifact
    config: QwenServeConfig
    device: str
    dtype: str
    buffer_plan: ResidentBufferPlan

    def render(self) -> str:
        lines = [f"Leanserve appliance for {self.model.family} ({self.model.key})"]
        lines.append(f"artifact: {self.artifact.root}")
        lines.append(f"device: {self.device}")
        lines.append(f"dtype: {self.dtype}")
        lines.append(
            "geometry: "
            f"layers={self.config.num_hidden_layers}, hidden={self.config.hidden_size}, "
            f"intermediate={self.config.intermediate_size}, heads={self.config.num_attention_heads}, "
            f"kv_heads={self.config.num_key_value_heads}, head_dim={self.config.head_dim}"
        )
        lines.append(f"resident_weights_bytes={self.buffer_plan.weights_bytes}")
        lines.append(f"max_kv_cache_bytes={self.buffer_plan.max_kv_cache_bytes}")
        lines.append(f"scratch_total_bytes={self.buffer_plan.scratch.total_bytes}")
        lines.append(f"resident_bytes={self.buffer_plan.resident_bytes}")
        lines.append("Buckets:")
        for bucket in self.buffer_plan.buckets:
            lines.append(
                "- "
                f"{bucket.key}: prompt={bucket.prompt_tokens}, new={bucket.max_new_tokens}, "
                f"max_seq={bucket.max_seq_len}, kv_tokens={bucket.kv_tokens_capacity}, "
                f"kv_cache_bytes={bucket.kv_cache_bytes}"
            )
        if self.artifact.manifest.speculative_modes:
            lines.append("Speculative modes:")
            for mode in self.artifact.manifest.speculative_modes:
                lines.append(
                    f"- {mode.key}: draft_layers={mode.draft_layer_count}, proposal_len={mode.proposal_len}"
                )
        if self.artifact.manifest.draft_heads:
            lines.append("Draft heads:")
            for head in self.artifact.manifest.draft_heads:
                lines.append(
                    f"- {head.key}: draft_layers={head.draft_layer_count}, file={head.file}, "
                    f"shape={head.output_size}x{head.input_size}, samples={head.fit_samples}"
                )
        return "\n".join(lines)


def load_leanpack_artifact(path_or_dir: str | Path) -> LeanPackArtifact:
    root = Path(path_or_dir)
    if root.name == "manifest.json":
        root = root.parent
    return LeanPackArtifact(root=root, manifest=load_packed_artifact_manifest(root))


def load_qwen_draft_head_projection(
    pack_dir: str | Path,
    *,
    draft_layer_count: int,
    key: str | None = None,
    device: str | torch.device = "cuda:0",
    dtype: str | torch.dtype = "bfloat16",
) -> torch.Tensor | None:
    artifact = load_leanpack_artifact(pack_dir)
    head = artifact.find_draft_head(draft_layer_count, key=key)
    if head is None:
        return None
    torch_dtype = resolve_torch_dtype(dtype) if isinstance(dtype, str) else dtype
    target_device = torch.device(device)
    return artifact.load_tensor(head.file, head.tensor_name, device=target_device, dtype=torch_dtype)


def build_resident_buffer_plan(
    *,
    model: ModelSpec,
    artifact: LeanPackArtifact,
    page_size: int = 16,
    batch_size: int = 1,
) -> ResidentBufferPlan:
    config = artifact.qwen_config(fallback_model=model)
    dtype_bytes = _dtype_nbytes_from_name(artifact.manifest.dtype)
    weights_bytes = sum(int(file_entry.size_bytes) for file_entry in artifact.manifest.files)
    scratch = ScratchPlan(
        hidden_state_bytes=config.hidden_size * dtype_bytes,
        qkv_projection_bytes=(config.num_attention_heads + (2 * config.num_key_value_heads)) * config.head_dim * dtype_bytes,
        attention_output_bytes=config.hidden_size * dtype_bytes,
        mlp_activation_bytes=(2 * config.intermediate_size) * dtype_bytes,
        logits_bytes=config.vocab_size * dtype_bytes,
    )
    buckets: list[ResidentBucketPlan] = []
    for bucket in artifact.manifest.buckets:
        max_seq_len = int(bucket.prompt_tokens) + int(bucket.max_new_tokens)
        kv_tokens_capacity = ceil(max_seq_len / page_size) * page_size
        kv_cache_bytes = (
            batch_size
            * config.num_hidden_layers
            * 2
            * config.num_key_value_heads
            * config.head_dim
            * kv_tokens_capacity
            * dtype_bytes
        )
        buckets.append(
            ResidentBucketPlan(
                key=bucket.key,
                prompt_tokens=int(bucket.prompt_tokens),
                max_new_tokens=int(bucket.max_new_tokens),
                max_seq_len=max_seq_len,
                kv_tokens_capacity=kv_tokens_capacity,
                kv_cache_bytes=int(kv_cache_bytes),
            )
        )
    return ResidentBufferPlan(
        weights_bytes=weights_bytes,
        page_size=page_size,
        batch_size=batch_size,
        scratch=scratch,
        buckets=tuple(buckets),
    )


def build_leanserve_appliance(
    *,
    model: ModelSpec,
    pack_dir: str | Path,
    device: str = "cuda:0",
    dtype: str | None = None,
    page_size: int = 16,
    batch_size: int = 1,
) -> LeanServeAppliance:
    artifact = load_leanpack_artifact(pack_dir)
    artifact.validate_for_model(model)
    config = artifact.qwen_config(fallback_model=model)
    buffer_plan = build_resident_buffer_plan(
        model=model,
        artifact=artifact,
        page_size=page_size,
        batch_size=batch_size,
    )
    return LeanServeAppliance(
        model=model,
        artifact=artifact,
        config=config,
        device=device,
        dtype=dtype or artifact.manifest.dtype,
        buffer_plan=buffer_plan,
    )


def _load_qwen_layer_from_leanpack(
    artifact: LeanPackArtifact,
    config: QwenServeConfig,
    layer_idx: int,
    device: torch.device,
    dtype: torch.dtype,
) -> QwenSemanticLayerRuntime:
    file_name = f"layer-{layer_idx:03d}.safetensors"
    names = (
        f"layers.{layer_idx}.input_layernorm.weight",
        f"layers.{layer_idx}.attention.qkv_proj.weight",
        f"layers.{layer_idx}.attention.o_proj.weight",
        f"layers.{layer_idx}.attention.q_norm.weight",
        f"layers.{layer_idx}.attention.k_norm.weight",
        f"layers.{layer_idx}.post_attention_layernorm.weight",
        f"layers.{layer_idx}.mlp.gate_up_proj.weight",
        f"layers.{layer_idx}.mlp.down_proj.weight",
    )
    staged = artifact.load_tensors(file_name, names, device=device, dtype=dtype)
    q_proj_out_dim = config.num_attention_heads * config.head_dim
    kv_proj_out_dim = config.num_key_value_heads * config.head_dim
    q_proj_weight, k_proj_weight, v_proj_weight = torch.split(
        staged[f"layers.{layer_idx}.attention.qkv_proj.weight"],
        (q_proj_out_dim, kv_proj_out_dim, kv_proj_out_dim),
        dim=0,
    )
    gate_proj_weight, up_proj_weight = torch.split(
        staged[f"layers.{layer_idx}.mlp.gate_up_proj.weight"],
        (config.intermediate_size, config.intermediate_size),
        dim=0,
    )
    return QwenSemanticLayerRuntime(
        layer_idx=layer_idx,
        input_layernorm_weight=staged[f"layers.{layer_idx}.input_layernorm.weight"],
        attention=QwenSemanticAttentionRuntime(
            layer_idx=layer_idx,
            q_proj_weight=q_proj_weight.contiguous(),
            k_proj_weight=k_proj_weight.contiguous(),
            v_proj_weight=v_proj_weight.contiguous(),
            qkv_proj_weight=staged[f"layers.{layer_idx}.attention.qkv_proj.weight"],
            q_proj_out_dim=q_proj_out_dim,
            k_proj_out_dim=kv_proj_out_dim,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            num_key_value_groups=config.num_attention_heads // config.num_key_value_heads,
            head_dim=config.head_dim,
            attention_scale=config.head_dim**-0.5,
            o_proj_weight=staged[f"layers.{layer_idx}.attention.o_proj.weight"],
            q_norm_weight=staged[f"layers.{layer_idx}.attention.q_norm.weight"],
            k_norm_weight=staged[f"layers.{layer_idx}.attention.k_norm.weight"],
        ),
        post_attention_layernorm_weight=staged[f"layers.{layer_idx}.post_attention_layernorm.weight"],
        mlp=QwenSemanticMlpRuntime(
            gate_proj_weight=gate_proj_weight.contiguous(),
            up_proj_weight=up_proj_weight.contiguous(),
            gate_up_proj_weight=staged[f"layers.{layer_idx}.mlp.gate_up_proj.weight"],
            gate_proj_out_dim=config.intermediate_size,
            down_proj_weight=staged[f"layers.{layer_idx}.mlp.down_proj.weight"],
        ),
    )


def materialize_qwen_semantic_stack_from_leanpack(
    pack_dir: str | Path,
    *,
    model: ModelSpec | None = None,
    layer_indices: tuple[int, ...] | None = None,
    device: str | torch.device = "cuda:0",
    dtype: str | torch.dtype = "bfloat16",
    include_output_head: bool = False,
) -> QwenSemanticStackRuntime:
    artifact = load_leanpack_artifact(pack_dir)
    if model is not None:
        artifact.validate_for_model(model)
    config = artifact.qwen_config(fallback_model=model)
    resolved_layer_indices = resolve_layer_indices(config, layer_indices)
    torch_dtype = resolve_torch_dtype(dtype) if isinstance(dtype, str) else dtype
    target_device = torch.device(device)
    rope_theta = config.rope_parameters["rope_theta"]
    rope_inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, config.head_dim, 2, device=target_device, dtype=torch.float32) / config.head_dim)
    )
    embed_tensors = artifact.load_tensors(
        "embeddings.safetensors",
        ("model.embed_tokens.weight",),
        device=target_device,
        dtype=torch_dtype,
    )
    layers = tuple(
        _load_qwen_layer_from_leanpack(
            artifact,
            config,
            layer_idx,
            target_device,
            torch_dtype,
        )
        for layer_idx in resolved_layer_indices
    )
    final_norm_weight: torch.Tensor | None = None
    lm_head_weight: torch.Tensor | None = None
    if include_output_head:
        output_tensors = artifact.load_tensors(
            "output.safetensors",
            ("model.norm.weight", "lm_head.weight"),
            device=target_device,
            dtype=torch_dtype,
        )
        final_norm_weight = output_tensors["model.norm.weight"]
        lm_head_weight = output_tensors["lm_head.weight"]
    return QwenSemanticStackRuntime(
        model_path=artifact.root,
        config=config,
        layer_indices=resolved_layer_indices,
        embed_tokens_weight=embed_tensors["model.embed_tokens.weight"],
        layers=layers,
        rope_inv_freq=rope_inv_freq,
        attention_scaling=1.0,
        final_norm_weight=final_norm_weight,
        lm_head_weight=lm_head_weight,
        device=target_device,
        dtype=torch_dtype,
    )


def materialize_qwen_full_semantic_runtime_from_leanpack(
    pack_dir: str | Path,
    *,
    model: ModelSpec | None = None,
    device: str | torch.device = "cuda:0",
    dtype: str | torch.dtype = "bfloat16",
    include_output_head: bool = True,
) -> QwenSemanticStackRuntime:
    return materialize_qwen_semantic_stack_from_leanpack(
        pack_dir,
        model=model,
        layer_indices=None,
        device=device,
        dtype=dtype,
        include_output_head=include_output_head,
    )
