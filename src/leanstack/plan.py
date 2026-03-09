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
        goal="Map Qwen3-1.7B-Base BF16 checkpoint metadata into an adapter-owned semantic contract and define the future serving artifact format.",
        exit_gate="The repo can parse the Qwen3-1.7B-Base BF16 checkpoint contract without a monolithic runtime.",
    ),
    PlanPhase(
        name="Phase 3",
        goal="Build `leanpack`: convert the public BF16 checkpoint into serving-only artifacts with exact bucket metadata and kernel-friendly layouts.",
        exit_gate="A serving artifact exists for Qwen3-1.7B-Base BF16, with packed weights, manifests, and exact prompt-bucket metadata.",
    ),
    PlanPhase(
        name="Phase 4",
        goal="Build `leanserve`: a static resident decode appliance for the packed artifact on GB10/sm_121.",
        exit_gate="The resident appliance holds weights, KV, scratch, and exact-bucket decode state on GPU for Qwen3-1.7B-Base BF16.",
    ),
    PlanPhase(
        name="Phase 5",
        goal="Benchmark appliance-mode leanstack against warmed external baselines and stop if the specialized appliance still does not show a real advantage.",
        exit_gate="A first exact-bucket comparison table exists for the packed-appliance path on the same machine, including a go/no-go conclusion.",
    ),
    PlanPhase(
        name="Phase 6",
        goal="Expose a serving surface only after the appliance path and specialization-versus-compatibility story are stable.",
        exit_gate="A remote endpoint serves one fixed packed model through the new appliance path.",
    ),
)


def render_plan() -> str:
    lines: list[str] = []
    for phase in DEFAULT_PHASES:
        lines.append(f"{phase.name}: {phase.goal}")
        lines.append(f"  Exit gate: {phase.exit_gate}")
    return "\n".join(lines)
