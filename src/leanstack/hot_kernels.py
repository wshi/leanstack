from __future__ import annotations

import json
import shlex
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class HotKernelCase:
    key: str
    kernel_kind: str
    role: str
    description: str
    model_id: str
    dtype: str
    m: int
    n: int | None = None
    k: int | None = None
    hidden_size: int | None = None
    m_tile: int | None = None
    n_tile: int | None = None
    k_tile: int | None = None
    eps: float | None = None
    default_enabled: bool = False

    def as_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["estimated_flops"] = self.estimated_flops()
        return payload

    def estimated_flops(self) -> int | None:
        if self.kernel_kind == "gemm" and self.n is not None and self.k is not None:
            return 2 * self.m * self.n * self.k
        if self.kernel_kind == "rmsnorm" and self.hidden_size is not None:
            return 5 * self.m * self.hidden_size
        return None

    def render(self) -> str:
        lines = [f"{self.key}: {self.role}"]
        lines.append(f"  kind: {self.kernel_kind}")
        lines.append(f"  model_id: {self.model_id}")
        lines.append(f"  dtype: {self.dtype}")
        if self.kernel_kind == "gemm":
            lines.append(f"  shape: M={self.m}, K={self.k}, N={self.n}")
            lines.append(f"  tile: M={self.m_tile}, K={self.k_tile}, N={self.n_tile}")
        else:
            lines.append(f"  rows: {self.m}")
            lines.append(f"  hidden_size: {self.hidden_size}")
            lines.append(f"  eps: {self.eps}")
        lines.append(f"  default_enabled: {self.default_enabled}")
        lines.append(f"  description: {self.description}")
        return "\n".join(lines)

    def render_shell(self) -> str:
        env = {
            "CASE_KEY": self.key,
            "KERNEL_KIND": self.kernel_kind,
            "ROLE": self.role,
            "MODEL_ID": self.model_id,
            "DTYPE": self.dtype,
            "M": str(self.m),
            "N": "" if self.n is None else str(self.n),
            "K": "" if self.k is None else str(self.k),
            "HIDDEN_SIZE": "" if self.hidden_size is None else str(self.hidden_size),
            "M_TILE": "" if self.m_tile is None else str(self.m_tile),
            "N_TILE": "" if self.n_tile is None else str(self.n_tile),
            "K_TILE": "" if self.k_tile is None else str(self.k_tile),
            "EPS": "" if self.eps is None else str(self.eps),
        }
        return "\n".join(f"{key}={shlex.quote(value)}" for key, value in env.items())

    def render_json(self) -> str:
        return json.dumps(self.as_payload(), indent=2)


QWEN_4B_MODEL_ID = "Qwen/Qwen3-4B-Base"
QWEN_4B_DTYPE = "bfloat16"


HOT_KERNEL_CASES: dict[str, HotKernelCase] = {
    "q_proj_prefill64": HotKernelCase(
        key="q_proj_prefill64",
        kernel_kind="gemm",
        role="Q projection GEMM",
        description="Prefill bucket for the dense Q projection: [64, 2560] x [2560, 4096].",
        model_id=QWEN_4B_MODEL_ID,
        dtype=QWEN_4B_DTYPE,
        m=64,
        n=4096,
        k=2560,
        m_tile=64,
        n_tile=128,
        k_tile=64,
        default_enabled=True,
    ),
    "q_proj_prefill1024": HotKernelCase(
        key="q_proj_prefill1024",
        kernel_kind="gemm",
        role="Q projection GEMM",
        description="Long-prefill bucket for the dense Q projection: [1024, 2560] x [2560, 4096].",
        model_id=QWEN_4B_MODEL_ID,
        dtype=QWEN_4B_DTYPE,
        m=1024,
        n=4096,
        k=2560,
        m_tile=64,
        n_tile=128,
        k_tile=64,
    ),
    "kv_proj_prefill64": HotKernelCase(
        key="kv_proj_prefill64",
        kernel_kind="gemm",
        role="KV projection GEMM",
        description="Prefill bucket for K/V projections with grouped-query output width: [64, 2560] x [2560, 1024].",
        model_id=QWEN_4B_MODEL_ID,
        dtype=QWEN_4B_DTYPE,
        m=64,
        n=1024,
        k=2560,
        m_tile=64,
        n_tile=128,
        k_tile=64,
        default_enabled=True,
    ),
    "o_proj_prefill64": HotKernelCase(
        key="o_proj_prefill64",
        kernel_kind="gemm",
        role="O projection GEMM",
        description="Prefill bucket for the dense output projection: [64, 4096] x [4096, 2560].",
        model_id=QWEN_4B_MODEL_ID,
        dtype=QWEN_4B_DTYPE,
        m=64,
        n=2560,
        k=4096,
        m_tile=64,
        n_tile=128,
        k_tile=64,
        default_enabled=True,
    ),
    "gate_up_proj_prefill64": HotKernelCase(
        key="gate_up_proj_prefill64",
        kernel_kind="gemm",
        role="Gate/Up projection GEMM",
        description="Prefill bucket for the gated MLP expansion: [64, 2560] x [2560, 9728].",
        model_id=QWEN_4B_MODEL_ID,
        dtype=QWEN_4B_DTYPE,
        m=64,
        n=9728,
        k=2560,
        m_tile=64,
        n_tile=128,
        k_tile=64,
        default_enabled=True,
    ),
    "down_proj_prefill64": HotKernelCase(
        key="down_proj_prefill64",
        kernel_kind="gemm",
        role="Down projection GEMM",
        description="Prefill bucket for the MLP contraction: [64, 9728] x [9728, 2560].",
        model_id=QWEN_4B_MODEL_ID,
        dtype=QWEN_4B_DTYPE,
        m=64,
        n=2560,
        k=9728,
        m_tile=64,
        n_tile=128,
        k_tile=256,
        default_enabled=True,
    ),
    "rmsnorm_prefill64": HotKernelCase(
        key="rmsnorm_prefill64",
        kernel_kind="rmsnorm",
        role="RMSNorm",
        description="Prefill bucket for the transformer RMSNorm over hidden size 2560 with 64 rows.",
        model_id=QWEN_4B_MODEL_ID,
        dtype=QWEN_4B_DTYPE,
        m=64,
        hidden_size=2560,
        eps=1e-6,
        default_enabled=True,
    ),
    "rmsnorm_prefill1024": HotKernelCase(
        key="rmsnorm_prefill1024",
        kernel_kind="rmsnorm",
        role="RMSNorm",
        description="Long-prefill RMSNorm bucket over hidden size 2560 with 1024 rows.",
        model_id=QWEN_4B_MODEL_ID,
        dtype=QWEN_4B_DTYPE,
        m=1024,
        hidden_size=2560,
        eps=1e-6,
    ),
}


def get_hot_kernel_case(key: str) -> HotKernelCase:
    normalized = key.strip().lower()
    try:
        return HOT_KERNEL_CASES[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(HOT_KERNEL_CASES))
        raise KeyError(f"unknown hot-kernel case '{key}'. Supported cases: {supported}") from exc


def list_hot_kernel_cases(*, default_only: bool = False) -> tuple[HotKernelCase, ...]:
    cases = tuple(HOT_KERNEL_CASES[key] for key in sorted(HOT_KERNEL_CASES))
    if not default_only:
        return cases
    return tuple(case for case in cases if case.default_enabled)
