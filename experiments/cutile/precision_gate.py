from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from fp4_compiler_gate import build_gate_report


def _run_torch_vector_add_probe(dtype: str) -> dict[str, Any]:
    script = Path(__file__).with_name("torch_vector_add.py")
    command = [sys.executable, str(script), "--dtype", dtype]
    result = subprocess.run(command, capture_output=True, text=True)
    payload: dict[str, Any] = {
        "command": command,
        "returncode": result.returncode,
    }
    if result.returncode == 0:
        payload["status"] = "cleared"
        payload["result"] = json.loads(result.stdout)
        return payload

    payload["status"] = "blocked"
    stderr = result.stderr.strip()
    stdout = result.stdout.strip()
    payload["stderr_tail"] = stderr.splitlines()[-20:] if stderr else []
    payload["stdout_tail"] = stdout.splitlines()[-20:] if stdout else []
    return payload


def build_precision_gate_report(tileiras: str) -> dict[str, Any]:
    bf16 = _run_torch_vector_add_probe("bfloat16")
    fp8_e4m3fn = _run_torch_vector_add_probe("float8_e4m3fn")
    fp8_e5m2 = _run_torch_vector_add_probe("float8_e5m2")
    fp4 = build_gate_report(tileiras)

    recommended_primary_precision = "undetermined"
    rationale = "No precision target has been proven yet."
    if bf16["status"] == "cleared":
        recommended_primary_precision = "bfloat16"
        rationale = (
            "BF16 compiles and runs through the public cuTile path on sm_121, while FP8 remains blocked in the "
            "current public stack and FP4 lacks a complete public authoring surface."
        )

    return {
        "gate": "precision_gate",
        "status": "cleared" if recommended_primary_precision != "undetermined" else "blocked",
        "recommended_primary_precision": recommended_primary_precision,
        "rationale": rationale,
        "probes": {
            "bfloat16": bf16,
            "float8_e4m3fn": fp8_e4m3fn,
            "float8_e5m2": fp8_e5m2,
            "fp4_frontend": fp4,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BF16, FP8, and FP4 readiness for the public cuTile path.")
    parser.add_argument("--tileiras", default="tileiras")
    parser.add_argument("--output")
    parser.add_argument("--require-success", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_precision_gate_report(args.tileiras)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    if args.require_success and report["status"] != "cleared":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
