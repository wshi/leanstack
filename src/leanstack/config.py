from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RemoteEndpoint:
    user_host: str
    port: int
    workspace: str = "/home/pto/lean"
    cutile_env: str = "/home/pto/venv-cutile"
    ssh_command: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelSpec:
    key: str
    family: str
    loader_hint: str
    dtype: str
    kv_layout: str
    required_kernels: tuple[str, ...]
    semantic_model_id: str | None = None
    artifact_model_id: str | None = None
    num_hidden_layers: int | None = None
    hidden_size: int | None = None
    intermediate_size: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    head_dim: int | None = None
    vocab_size: int | None = None
    max_position_embeddings: int | None = None
    target_gpu: str | None = None
    remote_model_key: str | None = None
    compile_gate: str | None = None
    legacy_reference: str | None = None
    backend_policy: tuple[str, ...] = field(default_factory=tuple)
    bring_up_sequence: tuple[str, ...] = field(default_factory=tuple)
    static_contract: tuple[str, ...] = field(default_factory=tuple)
    dynamic_inputs: tuple[str, ...] = field(default_factory=tuple)
    deferred_compatibility: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)
