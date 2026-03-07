from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ComparisonStage:
    name: str
    question: str
    scope: tuple[str, ...]
    metrics: tuple[str, ...]
    gate: str

    def render(self) -> str:
        lines = [f"{self.name}: {self.question}"]
        lines.append("  Scope:")
        for item in self.scope:
            lines.append(f"  - {item}")
        lines.append("  Metrics:")
        for item in self.metrics:
            lines.append(f"  - {item}")
        lines.append(f"  Gate: {self.gate}")
        return "\n".join(lines)


COMPARISON_STAGES: tuple[ComparisonStage, ...] = (
    ComparisonStage(
        name="Stage 0",
        question="What is the exact-checkpoint throughput ceiling on GB10 before leanstack enters the comparison?",
        scope=(
            "Run `Qwen/Qwen3-1.7B-Base` BF16 on the same remote GB10 with identical prompts and decode limits.",
            "Collect exact-checkpoint baselines from `transformers`, `vLLM`, and `SGLang` when each can run the same BF16 snapshot.",
            "Use single-request, non-thinking, deterministic decode first. Concurrency stays deferred until the single-stream path is stable.",
        ),
        metrics=(
            "TTFT",
            "decode tokens/s",
            "end-to-end tokens/s",
            "peak GPU memory",
        ),
        gate="A stable exact-checkpoint baseline table exists for the selected benchmark profiles.",
    ),
    ComparisonStage(
        name="Stage 1",
        question="Can the decisive Qwen3-1.7B kernels stay competitive when authored through cuTile/TileIR?",
        scope=(
            "Microbenchmark QKV GEMM, O projection, gate/up projection, down projection, RMSNorm, RoPE, and decode attention.",
            "Keep tensor shapes fixed to the Qwen3-1.7B contract on GB10/sm_121.",
            "Compare cuTile kernels against the closest torch/cuBLAS-backed reference implementation for the same tensor shape.",
        ),
        metrics=(
            "kernel latency",
            "effective TFLOP/s or bandwidth proxy",
            "generated cubin and inspected SASS",
        ),
        gate="The hot kernels that dominate decode and prefill are implemented on the cuTile path and do not show an obvious regression against the local torch reference.",
    ),
    ComparisonStage(
        name="Stage 2",
        question="Does the leanstack runtime add avoidable overhead between the cuTile kernels and the request loop?",
        scope=(
            "Benchmark single-block forward, prefill slice, and decode slice inside leanstack.",
            "Keep placement GPU-resident and forbid framework-managed CPU offload.",
            "Measure partial-runtime slices before making any full-model throughput claim.",
        ),
        metrics=(
            "slice latency",
            "slice tokens/s where meaningful",
            "allocated GPU memory",
        ),
        gate="The runtime slices are stable and the overhead outside the cuTile kernels is small enough to justify a full-model comparison.",
    ),
    ComparisonStage(
        name="Stage 3",
        question="Does the full specialized stack beat framework baselines on the fixed model-chip contract?",
        scope=(
            "Run full `leanstack` against `vLLM` and `SGLang` on the same Qwen3-1.7B-Base BF16 checkpoint.",
            "Use the agreed benchmark profiles and report both cold and hot runs.",
            "Only count this as official evidence if the hot path remains on the cuTile/TileIR backend.",
        ),
        metrics=(
            "cold TTFT",
            "hot TTFT",
            "median decode tokens/s over repeated hot runs",
            "median end-to-end tokens/s",
            "peak GPU memory",
            "process shape and launch complexity",
        ),
        gate="A full-table result exists and clearly shows whether leanstack wins, loses, or remains incomplete on the fixed contract.",
    ),
)


def render_comparison_plan() -> str:
    return "\n\n".join(stage.render() for stage in COMPARISON_STAGES)
