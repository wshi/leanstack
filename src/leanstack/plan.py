from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlanPhase:
    name: str
    goal: str
    exit_gate: str


DEFAULT_PHASES: tuple[PlanPhase, ...] = (
    PlanPhase(
        name="Phase 0",
        goal="Create a repeatable remote bring-up loop and capture compiler artifacts.",
        exit_gate="Remote smoke produces bytecode, TileIR, cubin, and SASS.",
    ),
    PlanPhase(
        name="Phase 1",
        goal="Build a kernel catalog around the transformer block instead of a framework runtime.",
        exit_gate="Core kernels exist as explicit cuTile programs with remote validation commands.",
    ),
    PlanPhase(
        name="Phase 2",
        goal="Implement a small runtime spine for batching, KV blocks, and dispatch.",
        exit_gate="A synthetic decode loop runs without an API layer.",
    ),
    PlanPhase(
        name="Phase 3",
        goal="Land the first model adapter, starting with GLM once the checkpoint is verified.",
        exit_gate="Single-request prefill and decode execute on the remote machine.",
    ),
    PlanPhase(
        name="Phase 4",
        goal="Expose a serving surface only after the execution path is stable.",
        exit_gate="A remote endpoint serves one model through the new stack.",
    ),
)


def render_plan() -> str:
    lines: list[str] = []
    for phase in DEFAULT_PHASES:
        lines.append(f"{phase.name}: {phase.goal}")
        lines.append(f"  Exit gate: {phase.exit_gate}")
    return "\n".join(lines)

