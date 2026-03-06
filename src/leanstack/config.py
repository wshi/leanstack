from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RemoteEndpoint:
    user_host: str
    port: int
    workspace: str = "/home/pto/leanstack"
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
    bring_up_sequence: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)
