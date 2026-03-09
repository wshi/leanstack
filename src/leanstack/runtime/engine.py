from __future__ import annotations

from dataclasses import dataclass

from leanstack.config import ModelSpec


@dataclass(frozen=True)
class ModelGeometry:
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int

    def render(self) -> tuple[str, ...]:
        return (
            f"layers={self.num_hidden_layers}",
            f"hidden_size={self.hidden_size}",
            f"intermediate_size={self.intermediate_size}",
            f"attention_heads={self.num_attention_heads}",
            f"kv_heads={self.num_key_value_heads}",
            f"head_dim={self.head_dim}",
            f"vocab_size={self.vocab_size}",
            f"max_position_embeddings={self.max_position_embeddings}",
        )


@dataclass(frozen=True)
class RuntimeBlueprint:
    model: ModelSpec
    geometry: ModelGeometry | None

    def render(self) -> str:
        lines = [f"Runtime blueprint for {self.model.family} ({self.model.key})"]
        if self.model.semantic_model_id:
            lines.append(f"Semantic model: {self.model.semantic_model_id}")
        if self.model.artifact_model_id:
            lines.append(f"Deployment artifact: {self.model.artifact_model_id}")
        if self.model.target_gpu:
            lines.append(f"Target GPU: {self.model.target_gpu}")
        if self.model.remote_model_key:
            lines.append(f"Preferred remote path file: /home/pto/lean/models/{self.model.remote_model_key}.path")
        if self.geometry is not None:
            lines.append("Geometry:")
            for item in self.geometry.render():
                lines.append(f"- {item}")
        lines.append(f"KV layout: {self.model.kv_layout}")
        lines.append(f"DType: {self.model.dtype}")
        if self.model.compile_gate:
            lines.append(f"Compile gate: {self.model.compile_gate}")
        if self.model.backend_policy:
            lines.append("Backend policy:")
            for item in self.model.backend_policy:
                lines.append(f"- {item}")
        if self.model.exact_prompt_buckets:
            lines.append(
                "Exact prompt buckets: "
                + ", ".join(str(bucket) for bucket in self.model.exact_prompt_buckets)
            )
        if self.model.pack_contract:
            lines.append("Leanpack contract:")
            for item in self.model.pack_contract:
                lines.append(f"- {item}")
        if self.model.serve_contract:
            lines.append("Leanserve contract:")
            for item in self.model.serve_contract:
                lines.append(f"- {item}")
        lines.append("Required kernels:")
        for kernel in self.model.required_kernels:
            lines.append(f"- {kernel}")
        if self.model.bring_up_sequence:
            lines.append("Bring-up sequence:")
            for step in self.model.bring_up_sequence:
                lines.append(f"- {step}")
        if self.model.legacy_reference:
            lines.append(f"Legacy reference: {self.model.legacy_reference}")
        return "\n".join(lines)


@dataclass(frozen=True)
class StaticInferenceContract:
    model: ModelSpec

    def render(self) -> str:
        lines = [f"Static inference contract for {self.model.family} ({self.model.key})"]
        lines.append("Fixed:")
        for item in self.model.static_contract:
            lines.append(f"- {item}")
        if self.model.backend_policy:
            lines.append("Backend policy:")
            for item in self.model.backend_policy:
                lines.append(f"- {item}")
        if self.model.exact_prompt_buckets:
            lines.append(
                "Exact prompt buckets: "
                + ", ".join(str(bucket) for bucket in self.model.exact_prompt_buckets)
            )
        if self.model.pack_contract:
            lines.append("Leanpack contract:")
            for item in self.model.pack_contract:
                lines.append(f"- {item}")
        if self.model.serve_contract:
            lines.append("Leanserve contract:")
            for item in self.model.serve_contract:
                lines.append(f"- {item}")
        lines.append("Dynamic:")
        for item in self.model.dynamic_inputs:
            lines.append(f"- {item}")
        if self.model.deferred_compatibility:
            lines.append("Deferred compatibility:")
            for item in self.model.deferred_compatibility:
                lines.append(f"- {item}")
        return "\n".join(lines)


def _geometry_from_model(model: ModelSpec) -> ModelGeometry | None:
    values = (
        model.num_hidden_layers,
        model.hidden_size,
        model.intermediate_size,
        model.num_attention_heads,
        model.num_key_value_heads,
        model.head_dim,
        model.vocab_size,
        model.max_position_embeddings,
    )
    if any(value is None for value in values):
        return None
    return ModelGeometry(
        num_hidden_layers=model.num_hidden_layers or 0,
        hidden_size=model.hidden_size or 0,
        intermediate_size=model.intermediate_size or 0,
        num_attention_heads=model.num_attention_heads or 0,
        num_key_value_heads=model.num_key_value_heads or 0,
        head_dim=model.head_dim or 0,
        vocab_size=model.vocab_size or 0,
        max_position_embeddings=model.max_position_embeddings or 0,
    )


def build_runtime_blueprint(model: ModelSpec) -> RuntimeBlueprint:
    return RuntimeBlueprint(
        model=model,
        geometry=_geometry_from_model(model),
    )


def build_static_inference_contract(model: ModelSpec) -> StaticInferenceContract:
    return StaticInferenceContract(model=model)
