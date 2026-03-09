from __future__ import annotations

from dataclasses import dataclass

from .config import ModelSpec


@dataclass(frozen=True)
class ApplianceReset:
    model: ModelSpec

    def render(self) -> str:
        lines = [f"Appliance reset for {self.model.family} ({self.model.key})"]
        lines.append("First-principles conclusions:")
        for item in (
            "Single-request decode throughput is dominated by bytes moved per token, decisive kernel quality, KV traffic, and host intervention frequency.",
            "A smaller codebase does not automatically reduce HBM traffic or kernel launches.",
            "Resident service mode improves cold start and orchestration cost, but cannot by itself create a large steady-state throughput win.",
            "To beat a mature framework, the stack must exploit asymmetries the framework cannot use as aggressively.",
        ):
            lines.append(f"- {item}")
        lines.append("Required asymmetries:")
        for item in (
            "offline-packed serving artifacts instead of framework checkpoint layouts",
            "exact prompt-token buckets instead of loose caps",
            "one resident process per GPU with preallocated KV, scratch, and graphs",
            "real cuTile kernels on the decisive decode path, not only microbenchmarks",
        ):
            lines.append(f"- {item}")
        lines.append("Reset build order:")
        for item in (
            "build `leanpack` to convert the public checkpoint into serving-only artifacts",
            "build `leanserve` as a static resident decode appliance",
            "compare appliance-mode leanstack against warmed vLLM only after the exact bucket contract is enforced",
        ):
            lines.append(f"- {item}")
        return "\n".join(lines)


@dataclass(frozen=True)
class LeanPackPlan:
    model: ModelSpec

    def render(self) -> str:
        lines = [f"Leanpack plan for {self.model.family} ({self.model.key})"]
        lines.append("Outputs:")
        for item in (
            "packed weight shards in kernel-consumption order",
            "tensor manifest with shapes, dtypes, and offsets",
            "bucket manifest with prompt buckets, decode budgets, scratch sizes, and KV extents",
            "compiler artifact manifest for the official cuTile kernels",
        ):
            lines.append(f"- {item}")
        if self.model.pack_contract:
            lines.append("Contract:")
            for item in self.model.pack_contract:
                lines.append(f"- {item}")
        return "\n".join(lines)


@dataclass(frozen=True)
class LeanServePlan:
    model: ModelSpec

    def render(self) -> str:
        lines = [f"Leanserve plan for {self.model.family} ({self.model.key})"]
        lines.append("Service shape:")
        for item in (
            "single model",
            "single GPU",
            "single precision policy",
            "resident weights and buffers",
            "exact prompt buckets on the official path",
            "deterministic decode on the official benchmark path",
        ):
            lines.append(f"- {item}")
        if self.model.serve_contract:
            lines.append("Contract:")
            for item in self.model.serve_contract:
                lines.append(f"- {item}")
        return "\n".join(lines)


def render_appliance_reset(model: ModelSpec) -> str:
    return ApplianceReset(model).render()


def render_leanpack_plan(model: ModelSpec) -> str:
    return LeanPackPlan(model).render()


def render_leanserve_plan(model: ModelSpec) -> str:
    return LeanServePlan(model).render()
