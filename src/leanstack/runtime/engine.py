from __future__ import annotations

from dataclasses import dataclass

from leanstack.config import ModelSpec


@dataclass(frozen=True)
class RuntimeComponent:
    name: str
    responsibility: str
    exit_criterion: str


@dataclass(frozen=True)
class RuntimeBlueprint:
    model: ModelSpec
    components: tuple[RuntimeComponent, ...]

    def render(self) -> str:
        lines = [f"Runtime blueprint for {self.model.family} ({self.model.key})"]
        for component in self.components:
            lines.append(f"- {component.name}: {component.responsibility}")
            lines.append(f"  Exit criterion: {component.exit_criterion}")
        return "\n".join(lines)


def build_runtime_blueprint(model: ModelSpec) -> RuntimeBlueprint:
    return RuntimeBlueprint(
        model=model,
        components=(
            RuntimeComponent(
                name="Block manager",
                responsibility="Owns paged KV blocks, reuse, and eviction policy.",
                exit_criterion="Can allocate and reclaim KV pages without leaking blocks.",
            ),
            RuntimeComponent(
                name="Scheduler",
                responsibility="Separates prefill and decode waves and emits execution batches.",
                exit_criterion="Produces deterministic work packets for one-step decode.",
            ),
            RuntimeComponent(
                name="Dispatcher",
                responsibility="Maps execution packets onto explicit kernels from the catalog.",
                exit_criterion="Every model step names the kernel path it used.",
            ),
            RuntimeComponent(
                name="Sampler",
                responsibility="Transforms logits into token choices without hiding policy in the runtime.",
                exit_criterion="Supports deterministic greedy sampling before probabilistic policies.",
            ),
        ),
    )
