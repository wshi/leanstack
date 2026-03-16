#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from leanstack.compare_runner import (
    DEFAULT_PROFILE,
    DEFAULT_VLLM_BASELINE_MODE,
    DEFAULT_VLLM_BASELINE_RUNS,
    build_comparison_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed-contract side-by-side benchmark and write a markdown report."
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Explain how fixing Qwen3-4B-Base and GB10 lets an agent-generated "
            "inference stack reduce runtime overhead."
        ),
    )
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--vllm-baseline-mode",
        choices=("plain", "best"),
        default=DEFAULT_VLLM_BASELINE_MODE,
    )
    parser.add_argument("--vllm-baseline-runs", type=int, default=DEFAULT_VLLM_BASELINE_RUNS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports") / "benchmarks",
    )
    return parser.parse_args()


def _render_report(payload: dict) -> str:
    contract = payload.get("official_contract") or {}
    vllm = payload.get("vllm") or {}
    leanstack = payload.get("leanstack") or {}
    leanstack_throughput = leanstack.get("throughput") or {}
    delta = payload.get("delta") or {}
    fairness = payload.get("fairness_gate") or {}

    lines = [
        "# Contract Benchmark Report",
        "",
        "## Contract",
        "",
        f"- id: `{contract.get('id')}`",
        f"- model: `{contract.get('model_id')}`",
        f"- profile: `{contract.get('profile')}`",
        f"- dtype: `{contract.get('dtype')}`",
        f"- device: `{contract.get('device')}`",
        f"- vLLM baseline mode: `{vllm.get('baseline_mode', 'plain')}`",
        f"- vLLM baseline runs: `{vllm.get('baseline_runs', 1)}`",
        "",
        "## Summary",
        "",
        "| System | Runtime tok/s | End-to-end tok/s | TTFT / Prefill (s) |",
        "| --- | ---: | ---: | ---: |",
        "| vLLM | {vllm_rt} | {vllm_e2e} | {vllm_ttft} |".format(
            vllm_rt=_fmt(vllm.get("generated_tokens_per_second")),
            vllm_e2e=_fmt(vllm.get("end_to_end_tokens_per_second")),
            vllm_ttft=_fmt(vllm.get("ttft_seconds")),
        ),
        "| leanstack | {ls_rt} | {ls_e2e} | {ls_prefill} |".format(
            ls_rt=_fmt(leanstack_throughput.get("runtime_tokens_per_second")),
            ls_e2e=_fmt(leanstack_throughput.get("full_loop_tokens_per_second")),
            ls_prefill=_fmt((leanstack.get("timings") or {}).get("prefill_seconds")),
        ),
        "",
        "## Fairness Gate",
        "",
        f"- passed: `{fairness.get('passed')}`",
        f"- decode policy: `{fairness.get('decode_policy')}`",
        f"- temperature: `{_fmt(fairness.get('temperature'))}`",
        f"- ignore_eos: `{fairness.get('ignore_eos')}`",
        f"- prompt tokens (matched): `{fairness.get('prompt_tokens')}`",
        f"- generated tokens (matched): `{fairness.get('generated_tokens')}`",
        f"- max_new_tokens: `{fairness.get('max_new_tokens')}`",
        "",
        "## Ratios",
        "",
        f"- leanstack/vLLM runtime tokens/s: `{_fmt(delta.get('runtime_tokens_per_second_ratio'))}x`",
        f"- leanstack prefill / vLLM TTFT: `{_fmt(delta.get('prefill_to_vllm_ttft_ratio'))}x`",
        "",
        "## Raw Payload",
        "",
        "```json",
        json.dumps(payload, indent=2),
        "```",
        "",
    ]
    return "\n".join(lines)


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def main() -> int:
    args = parse_args()
    payload = build_comparison_payload(
        prompt=args.prompt,
        profile=args.profile,
        max_new_tokens=args.max_new_tokens,
        vllm_baseline_mode=args.vllm_baseline_mode,
        vllm_baseline_runs=args.vllm_baseline_runs,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output_dir / f"contract_benchmark_{stamp}.md"
    output_path.write_text(_render_report(payload), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
