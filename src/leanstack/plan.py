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
        goal="Prove that the cuTile-native compiler path can emit one real FP4 kernel for GB10/sm_121 before building a larger runtime.",
        exit_gate="A minimal FP4 GEMM or FP4 linear kernel compiles to cubin and runs on the remote GB10.",
    ),
    PlanPhase(
        name="Phase 2",
        goal="Map Qwen3-8B FP4 artifact metadata into an adapter-owned contract and keep deferred compatibility features explicit.",
        exit_gate="The repo can parse the target FP4 artifact, its scales, and its linear-layer contract without a monolithic runtime.",
    ),
    PlanPhase(
        name="Phase 3",
        goal="Stand up a small Blackwell-first runtime for Qwen3-8B NVFP4 once the compiler and artifact contracts are proven.",
        exit_gate="Single-request prefill and decode execute for the Qwen3-8B FP4 target on the remote machine.",
    ),
    PlanPhase(
        name="Phase 4",
        goal="Benchmark leanstack against exact-format external baselines and stop if the specialized stack does not show a real advantage.",
        exit_gate="A first comparison table exists for a comparable Qwen3-8B FP4 profile on the same machine, including complexity proxies and a go/no-go conclusion.",
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
