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
        goal="Create a repeatable remote bring-up loop, capture compiler artifacts, and state the compatibility costs being deferred.",
        exit_gate="Remote smoke produces bytecode, TileIR, cubin, and SASS.",
    ),
    PlanPhase(
        name="Phase 1",
        goal="Keep the BF16 compiler path green on GB10/sm_121 and record FP8/FP4 blockers explicitly before building a larger runtime.",
        exit_gate="The executable precision gate recommends BF16 as the current public-cuTile primary precision on the remote GB10.",
    ),
    PlanPhase(
        name="Phase 2",
        goal="Map Qwen3-1.7B-Base BF16 checkpoint metadata into an adapter-owned contract and keep deferred compatibility features explicit.",
        exit_gate="The repo can parse the Qwen3-1.7B-Base BF16 checkpoint contract without a monolithic runtime.",
    ),
    PlanPhase(
        name="Phase 3",
        goal="Stand up a throughput-first runtime for Qwen3-1.7B-Base BF16 once the precision and checkpoint contracts are proven.",
        exit_gate="Single-request prefill and decode execute for the Qwen3-1.7B-Base BF16 target on the remote machine.",
    ),
    PlanPhase(
        name="Phase 4",
        goal="Benchmark leanstack against exact-format external baselines and stop if the specialized stack does not show a real advantage.",
        exit_gate="A first comparison table exists for a comparable Qwen3-1.7B-Base BF16 profile on the same machine, including complexity proxies and a go/no-go conclusion.",
    ),
    PlanPhase(
        name="Phase 5",
        goal="Expose a serving surface only after the execution path and specialization-versus-compatibility story are stable.",
        exit_gate="A remote endpoint serves one model through the new stack.",
    ),
)


def render_plan() -> str:
    lines: list[str] = []
    for phase in DEFAULT_PHASES:
        lines.append(f"{phase.name}: {phase.goal}")
        lines.append(f"  Exit gate: {phase.exit_gate}")
    return "\n".join(lines)
