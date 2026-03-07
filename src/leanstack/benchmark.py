from __future__ import annotations

import json
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _long_prefill_prompt() -> str:
    clause = (
        "A fixed Qwen3-1.7B-Base and GB10 contract removes runtime discovery, minimizes compatibility heuristics, "
        "and lets an agent specialize the cuTile path around one dense GQA model."
    )
    return " ".join(clause for _ in range(96))


@dataclass(frozen=True)
class BenchmarkProfile:
    key: str
    title: str
    goal: str
    prompt: str
    prompt_format: str
    max_prefill_tokens: int
    max_new_tokens: int
    concurrency: int

    def as_payload(self) -> dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        lines = [f"{self.key}: {self.title}"]
        lines.append(f"  goal: {self.goal}")
        lines.append(f"  prompt_format: {self.prompt_format}")
        lines.append(f"  max_prefill_tokens: {self.max_prefill_tokens}")
        lines.append(f"  max_new_tokens: {self.max_new_tokens}")
        lines.append(f"  concurrency: {self.concurrency}")
        lines.append(f"  prompt_chars: {len(self.prompt)}")
        return "\n".join(lines)

    def render_shell(self) -> str:
        env = {
            "PROFILE_KEY": self.key,
            "PROFILE_TITLE": self.title,
            "PROFILE_GOAL": self.goal,
            "PROMPT": self.prompt,
            "PROMPT_FORMAT": self.prompt_format,
            "MAX_PREFILL_TOKENS": str(self.max_prefill_tokens),
            "MAX_NEW_TOKENS": str(self.max_new_tokens),
            "CONCURRENCY": str(self.concurrency),
        }
        return "\n".join(f"{key}={shlex.quote(value)}" for key, value in env.items())


BENCHMARK_PROFILES: dict[str, BenchmarkProfile] = {
    "decode_64_256": BenchmarkProfile(
        key="decode_64_256",
        title="Decode 64/256",
        goal="Primary single-request decode-throughput profile for the fixed Qwen3-1.7B-Base contract.",
        prompt="Explain how a fixed Qwen3-1.7B-Base and GB10 contract can reduce inference overhead.",
        prompt_format="raw",
        max_prefill_tokens=64,
        max_new_tokens=256,
        concurrency=1,
    ),
    "decode_64_512": BenchmarkProfile(
        key="decode_64_512",
        title="Decode 64/512",
        goal="Stress the steady-state decode path with a longer emission window on one request.",
        prompt="Summarize why specialization can outperform compatibility when the model and hardware are fixed.",
        prompt_format="raw",
        max_prefill_tokens=64,
        max_new_tokens=512,
        concurrency=1,
    ),
    "prefill_1024_64": BenchmarkProfile(
        key="prefill_1024_64",
        title="Prefill 1024/64",
        goal="Expose prefill cost, staging overhead, and KV/cache setup behavior with a long prompt bucket.",
        prompt=_long_prefill_prompt(),
        prompt_format="raw",
        max_prefill_tokens=1024,
        max_new_tokens=64,
        concurrency=1,
    ),
}


def list_benchmark_profiles() -> tuple[BenchmarkProfile, ...]:
    return tuple(BENCHMARK_PROFILES[key] for key in sorted(BENCHMARK_PROFILES))


def get_benchmark_profile(key: str) -> BenchmarkProfile:
    normalized = key.strip().lower()
    try:
        return BENCHMARK_PROFILES[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(BENCHMARK_PROFILES))
        raise KeyError(f"no benchmark profile for '{key}'. Supported profiles: {supported}") from exc


@dataclass(frozen=True)
class NormalizedBenchmarkResult:
    system: str
    variant: str
    profile: str
    prompt_tokens: int | None
    emitted_tokens: int | None
    ttft_seconds: float | None
    runtime_seconds: float | None
    end_to_end_seconds: float | None
    runtime_tokens_per_second: float | None
    end_to_end_tokens_per_second: float | None
    peak_gpu_gib: float | None
    source_path: str


def _bytes_to_gib(value: int | None) -> float | None:
    if value is None:
        return None
    return float(value) / (1024**3)


def normalize_benchmark_result(payload: dict[str, Any], source_path: str) -> NormalizedBenchmarkResult:
    if "runtime_mode" in payload:
        memory = payload.get("memory", {})
        peak_allocated = None
        for key in ("after_runtime_loop", "after_materialize"):
            section = memory.get(key) or {}
            candidate = section.get("max_allocated_bytes") or section.get("allocated_bytes")
            if candidate is not None:
                peak_allocated = max(int(candidate), int(peak_allocated or 0))
        return NormalizedBenchmarkResult(
            system="leanstack",
            variant=str(payload.get("runtime_mode", "unknown")),
            profile=str(payload.get("benchmark_profile", "unlabeled")),
            prompt_tokens=payload.get("prompt_tokens"),
            emitted_tokens=payload.get("emitted_tokens"),
            ttft_seconds=None,
            runtime_seconds=(payload.get("timings") or {}).get("runtime_loop_seconds"),
            end_to_end_seconds=(payload.get("timings") or {}).get("full_loop_seconds"),
            runtime_tokens_per_second=(payload.get("throughput") or {}).get("runtime_tokens_per_second"),
            end_to_end_tokens_per_second=(payload.get("throughput") or {}).get("full_loop_tokens_per_second"),
            peak_gpu_gib=_bytes_to_gib(peak_allocated),
            source_path=source_path,
        )

    if "base_url" in payload:
        memory = payload.get("memory") or {}
        peak_allocated = memory.get("max_allocated_bytes") or memory.get("allocated_bytes")
        return NormalizedBenchmarkResult(
            system=str(payload.get("system", "framework")),
            variant=str(payload.get("variant", "openai")),
            profile=str(payload.get("benchmark_profile", "unlabeled")),
            prompt_tokens=payload.get("prompt_tokens"),
            emitted_tokens=payload.get("generated_tokens"),
            ttft_seconds=payload.get("ttft_seconds"),
            runtime_seconds=payload.get("stream_seconds"),
            end_to_end_seconds=payload.get("end_to_end_seconds"),
            runtime_tokens_per_second=payload.get("generated_tokens_per_second"),
            end_to_end_tokens_per_second=payload.get("end_to_end_tokens_per_second"),
            peak_gpu_gib=_bytes_to_gib(peak_allocated),
            source_path=source_path,
        )

    raise ValueError(f"unsupported benchmark result shape in {source_path}")


def load_benchmark_result(path: str | Path) -> NormalizedBenchmarkResult:
    resolved = Path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    return normalize_benchmark_result(payload, str(resolved))


def render_benchmark_report(results: list[NormalizedBenchmarkResult]) -> str:
    header = (
        "| system | variant | profile | prompt_tokens | emitted_tokens | ttft_s | runtime_s | end_to_end_s | "
        "runtime_tok_s | end_to_end_tok_s | peak_gpu_gib | source |\n"
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
    )
    rows = [header]
    for result in results:
        rows.append(
            "| {system} | {variant} | {profile} | {prompt_tokens} | {emitted_tokens} | {ttft} | {runtime} | {e2e} | {rtps} | {etps} | {gpu} | {source} |".format(
                system=result.system,
                variant=result.variant,
                profile=result.profile,
                prompt_tokens=result.prompt_tokens if result.prompt_tokens is not None else "-",
                emitted_tokens=result.emitted_tokens if result.emitted_tokens is not None else "-",
                ttft=f"{result.ttft_seconds:.4f}" if result.ttft_seconds is not None else "-",
                runtime=f"{result.runtime_seconds:.4f}" if result.runtime_seconds is not None else "-",
                e2e=f"{result.end_to_end_seconds:.4f}" if result.end_to_end_seconds is not None else "-",
                rtps=(
                    f"{result.runtime_tokens_per_second:.4f}"
                    if result.runtime_tokens_per_second is not None
                    else "-"
                ),
                etps=(
                    f"{result.end_to_end_tokens_per_second:.4f}"
                    if result.end_to_end_tokens_per_second is not None
                    else "-"
                ),
                gpu=f"{result.peak_gpu_gib:.2f}" if result.peak_gpu_gib is not None else "-",
                source=result.source_path,
            )
        )
    return "\n".join(rows)
