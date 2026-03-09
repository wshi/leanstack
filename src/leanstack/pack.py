from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from .benchmark import list_benchmark_profiles
from .config import ModelSpec
from .runtime.qwen_explicit import (
    load_qwen_config,
    qwen_has_separate_lm_head,
    qwen_layer_tensor_names,
    stage_tensors_to_cpu,
)


def _dtype_nbytes(dtype: torch.dtype) -> int:
    if dtype in (torch.float32, torch.int32):
        return 4
    if dtype in (torch.float16, torch.bfloat16, torch.int16):
        return 2
    if dtype in (torch.int8, torch.uint8, torch.bool):
        return 1
    if dtype in (torch.int64, torch.float64):
        return 8
    raise ValueError(f"unsupported dtype for size accounting: {dtype}")


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


@dataclass(frozen=True)
class PackedTensorEntry:
    name: str
    role: str
    file: str
    dtype: str
    shape: list[int]
    numel: int
    logical_offset_bytes: int
    size_bytes: int
    source_tensors: list[str]


@dataclass(frozen=True)
class PackedFileEntry:
    file: str
    tensor_count: int
    size_bytes: int


@dataclass(frozen=True)
class BucketEntry:
    key: str
    prompt_tokens: int
    max_new_tokens: int


@dataclass(frozen=True)
class SpeculativeModeEntry:
    key: str
    draft_layer_count: int
    proposal_len: int


@dataclass(frozen=True)
class PackedArtifactManifest:
    format_version: str
    created_at_utc: str
    model_key: str
    semantic_model_id: str | None
    source_model_path: str
    target_gpu: str | None
    dtype: str
    geometry: dict[str, int]
    model_hparams: dict[str, Any]
    exact_prompt_buckets: list[int]
    buckets: list[BucketEntry]
    speculative_modes: list[SpeculativeModeEntry]
    required_kernels: list[str]
    backend_policy: list[str]
    files: list[PackedFileEntry]
    tensors: list[PackedTensorEntry]

    def as_payload(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "created_at_utc": self.created_at_utc,
            "model_key": self.model_key,
            "semantic_model_id": self.semantic_model_id,
            "source_model_path": self.source_model_path,
            "target_gpu": self.target_gpu,
            "dtype": self.dtype,
            "geometry": self.geometry,
            "model_hparams": self.model_hparams,
            "exact_prompt_buckets": self.exact_prompt_buckets,
            "buckets": [asdict(bucket) for bucket in self.buckets],
            "speculative_modes": [asdict(mode) for mode in self.speculative_modes],
            "required_kernels": self.required_kernels,
            "backend_policy": self.backend_policy,
            "files": [asdict(file_entry) for file_entry in self.files],
            "tensors": [asdict(tensor) for tensor in self.tensors],
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "PackedArtifactManifest":
        return cls(
            format_version=payload["format_version"],
            created_at_utc=payload["created_at_utc"],
            model_key=payload["model_key"],
            semantic_model_id=payload.get("semantic_model_id"),
            source_model_path=payload["source_model_path"],
            target_gpu=payload.get("target_gpu"),
            dtype=payload["dtype"],
            geometry=dict(payload.get("geometry", {})),
            model_hparams=dict(payload.get("model_hparams", {})),
            exact_prompt_buckets=[int(bucket) for bucket in payload.get("exact_prompt_buckets", [])],
            buckets=[BucketEntry(**bucket) for bucket in payload.get("buckets", [])],
            speculative_modes=[
                SpeculativeModeEntry(**mode) for mode in payload.get("speculative_modes", [])
            ],
            required_kernels=list(payload.get("required_kernels", [])),
            backend_policy=list(payload.get("backend_policy", [])),
            files=[PackedFileEntry(**file_entry) for file_entry in payload.get("files", [])],
            tensors=[PackedTensorEntry(**tensor_entry) for tensor_entry in payload.get("tensors", [])],
        )


def load_packed_artifact_manifest(path_or_dir: str | Path) -> PackedArtifactManifest:
    manifest_path = Path(path_or_dir)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return PackedArtifactManifest.from_payload(payload)


def _qwen_bucket_entries(model: ModelSpec) -> list[BucketEntry]:
    buckets: list[BucketEntry] = []
    for profile in list_benchmark_profiles():
        if profile.target_prompt_tokens not in model.exact_prompt_buckets:
            continue
        buckets.append(
            BucketEntry(
                key=profile.key,
                prompt_tokens=profile.target_prompt_tokens,
                max_new_tokens=profile.max_new_tokens,
            )
        )
    return buckets


def _qwen_speculative_modes(model: ModelSpec) -> list[SpeculativeModeEntry]:
    total_layers = int(model.num_hidden_layers or 0)
    default_draft = min(12, max(total_layers - 1, 1))
    return [
        SpeculativeModeEntry(
            key="draft12_k4",
            draft_layer_count=default_draft,
            proposal_len=4,
        ),
        SpeculativeModeEntry(
            key="draft12_k2",
            draft_layer_count=default_draft,
            proposal_len=2,
        ),
    ]


def _record_tensor_entries(file_name: str, tensors: dict[str, torch.Tensor], roles: dict[str, str], sources: dict[str, list[str]]) -> tuple[list[PackedTensorEntry], int]:
    entries: list[PackedTensorEntry] = []
    logical_offset = 0
    for name, tensor in tensors.items():
        size_bytes = int(tensor.numel()) * _dtype_nbytes(tensor.dtype)
        entries.append(
            PackedTensorEntry(
                name=name,
                role=roles[name],
                file=file_name,
                dtype=_dtype_name(tensor.dtype),
                shape=list(tensor.shape),
                numel=int(tensor.numel()),
                logical_offset_bytes=logical_offset,
                size_bytes=size_bytes,
                source_tensors=sources[name],
            )
        )
        logical_offset += size_bytes
    return entries, logical_offset


def _write_group(output_dir: Path, file_name: str, tensors: dict[str, torch.Tensor], roles: dict[str, str], sources: dict[str, list[str]], *, write_tensors: bool) -> tuple[PackedFileEntry, list[PackedTensorEntry]]:
    file_path = output_dir / file_name
    if write_tensors:
        save_file(tensors, str(file_path))
        size_bytes = file_path.stat().st_size
    else:
        _, size_bytes = _record_tensor_entries(file_name, tensors, roles, sources)
    tensor_entries, _ = _record_tensor_entries(file_name, tensors, roles, sources)
    return (
        PackedFileEntry(file=file_name, tensor_count=len(tensors), size_bytes=int(size_bytes)),
        tensor_entries,
    )


def build_qwen_leanpack(
    *,
    model: ModelSpec,
    model_path: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
    write_tensors: bool = True,
) -> PackedArtifactManifest:
    output_root = Path(output_dir)
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"output_dir already exists: {output_root}")
    else:
        output_root.mkdir(parents=True, exist_ok=True)

    config = load_qwen_config(model_path)
    files: list[PackedFileEntry] = []
    tensors: list[PackedTensorEntry] = []

    embed_weight = stage_tensors_to_cpu(model_path, ("model.embed_tokens.weight",))["model.embed_tokens.weight"]
    embed_tensors = {"model.embed_tokens.weight": embed_weight}
    embed_roles = {"model.embed_tokens.weight": "embedding"}
    embed_sources = {"model.embed_tokens.weight": ["model.embed_tokens.weight"]}
    file_entry, tensor_entries = _write_group(
        output_root,
        "embeddings.safetensors",
        embed_tensors,
        embed_roles,
        embed_sources,
        write_tensors=write_tensors,
    )
    files.append(file_entry)
    tensors.extend(tensor_entries)

    for layer_idx in range(config.num_hidden_layers):
        staged = stage_tensors_to_cpu(model_path, qwen_layer_tensor_names(layer_idx))
        prefix = f"model.layers.{layer_idx}."
        q_proj_name = f"{prefix}self_attn.q_proj.weight"
        k_proj_name = f"{prefix}self_attn.k_proj.weight"
        v_proj_name = f"{prefix}self_attn.v_proj.weight"
        gate_proj_name = f"{prefix}mlp.gate_proj.weight"
        up_proj_name = f"{prefix}mlp.up_proj.weight"

        layer_tensors = {
            f"layers.{layer_idx}.input_layernorm.weight": staged[f"{prefix}input_layernorm.weight"].contiguous(),
            f"layers.{layer_idx}.attention.qkv_proj.weight": torch.cat(
                (staged[q_proj_name], staged[k_proj_name], staged[v_proj_name]),
                dim=0,
            ).contiguous(),
            f"layers.{layer_idx}.attention.o_proj.weight": staged[f"{prefix}self_attn.o_proj.weight"].contiguous(),
            f"layers.{layer_idx}.attention.q_norm.weight": staged[f"{prefix}self_attn.q_norm.weight"].contiguous(),
            f"layers.{layer_idx}.attention.k_norm.weight": staged[f"{prefix}self_attn.k_norm.weight"].contiguous(),
            f"layers.{layer_idx}.post_attention_layernorm.weight": staged[
                f"{prefix}post_attention_layernorm.weight"
            ].contiguous(),
            f"layers.{layer_idx}.mlp.gate_up_proj.weight": torch.cat(
                (staged[gate_proj_name], staged[up_proj_name]),
                dim=0,
            ).contiguous(),
            f"layers.{layer_idx}.mlp.down_proj.weight": staged[f"{prefix}mlp.down_proj.weight"].contiguous(),
        }
        layer_roles = {
            f"layers.{layer_idx}.input_layernorm.weight": "input_layernorm",
            f"layers.{layer_idx}.attention.qkv_proj.weight": "attention_qkv_fused",
            f"layers.{layer_idx}.attention.o_proj.weight": "attention_output_projection",
            f"layers.{layer_idx}.attention.q_norm.weight": "attention_q_norm",
            f"layers.{layer_idx}.attention.k_norm.weight": "attention_k_norm",
            f"layers.{layer_idx}.post_attention_layernorm.weight": "post_attention_layernorm",
            f"layers.{layer_idx}.mlp.gate_up_proj.weight": "mlp_gate_up_fused",
            f"layers.{layer_idx}.mlp.down_proj.weight": "mlp_down_projection",
        }
        layer_sources = {
            f"layers.{layer_idx}.input_layernorm.weight": [f"{prefix}input_layernorm.weight"],
            f"layers.{layer_idx}.attention.qkv_proj.weight": [q_proj_name, k_proj_name, v_proj_name],
            f"layers.{layer_idx}.attention.o_proj.weight": [f"{prefix}self_attn.o_proj.weight"],
            f"layers.{layer_idx}.attention.q_norm.weight": [f"{prefix}self_attn.q_norm.weight"],
            f"layers.{layer_idx}.attention.k_norm.weight": [f"{prefix}self_attn.k_norm.weight"],
            f"layers.{layer_idx}.post_attention_layernorm.weight": [f"{prefix}post_attention_layernorm.weight"],
            f"layers.{layer_idx}.mlp.gate_up_proj.weight": [gate_proj_name, up_proj_name],
            f"layers.{layer_idx}.mlp.down_proj.weight": [f"{prefix}mlp.down_proj.weight"],
        }
        file_entry, tensor_entries = _write_group(
            output_root,
            f"layer-{layer_idx:03d}.safetensors",
            layer_tensors,
            layer_roles,
            layer_sources,
            write_tensors=write_tensors,
        )
        files.append(file_entry)
        tensors.extend(tensor_entries)

    lm_head_tensor_name = "lm_head.weight" if qwen_has_separate_lm_head(model_path) else "model.embed_tokens.weight"
    output_staged = stage_tensors_to_cpu(model_path, ("model.norm.weight", lm_head_tensor_name))
    output_tensors = {
        "model.norm.weight": output_staged["model.norm.weight"].contiguous(),
        "lm_head.weight": output_staged[lm_head_tensor_name].contiguous(),
    }
    output_roles = {
        "model.norm.weight": "final_norm",
        "lm_head.weight": "lm_head",
    }
    output_sources = {
        "model.norm.weight": ["model.norm.weight"],
        "lm_head.weight": [lm_head_tensor_name],
    }
    file_entry, tensor_entries = _write_group(
        output_root,
        "output.safetensors",
        output_tensors,
        output_roles,
        output_sources,
        write_tensors=write_tensors,
    )
    files.append(file_entry)
    tensors.extend(tensor_entries)

    manifest = PackedArtifactManifest(
        format_version="leanpack/v0",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        model_key=model.key,
        semantic_model_id=model.semantic_model_id,
        source_model_path=str(Path(model_path)),
        target_gpu=model.target_gpu,
        dtype=model.dtype,
        geometry={
            "num_hidden_layers": int(config.num_hidden_layers),
            "hidden_size": int(config.hidden_size),
            "intermediate_size": int(config.intermediate_size),
            "num_attention_heads": int(config.num_attention_heads),
            "num_key_value_heads": int(config.num_key_value_heads),
            "head_dim": int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)),
            "vocab_size": int(config.vocab_size),
            "max_position_embeddings": int(config.max_position_embeddings),
        },
        model_hparams={
            "rms_norm_eps": float(config.rms_norm_eps),
            "rope_theta": float(config.rope_parameters["rope_theta"]),
        },
        exact_prompt_buckets=list(model.exact_prompt_buckets),
        buckets=_qwen_bucket_entries(model),
        speculative_modes=_qwen_speculative_modes(model),
        required_kernels=list(model.required_kernels),
        backend_policy=list(model.backend_policy),
        files=files,
        tensors=tensors,
    )
    (output_root / "manifest.json").write_text(json.dumps(manifest.as_payload(), indent=2) + "\n", encoding="utf-8")
    return manifest
