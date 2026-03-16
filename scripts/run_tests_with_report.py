#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CheckResult:
    name: str
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def passed(self) -> bool:
        return self.returncode == 0


def _run_check(name: str, command: list[str], *, extra_env: dict[str, str] | None = None) -> CheckResult:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    return CheckResult(
        name=name,
        command=tuple(command),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _render_markdown(results: list[CheckResult], timestamp: str) -> str:
    overall_passed = all(result.passed for result in results)
    status = "PASSED" if overall_passed else "FAILED"
    lines: list[str] = [
        "# Test Report",
        "",
        f"- Timestamp (UTC): {timestamp}",
        f"- Overall status: **{status}**",
        "",
        "| Check | Status | Exit Code |",
        "| --- | --- | ---: |",
    ]
    for result in results:
        lines.append(
            f"| {result.name} | {'PASS' if result.passed else 'FAIL'} | {result.returncode} |"
        )

    for result in results:
        lines.extend(
            [
                "",
                f"## {result.name}",
                "",
                f"- Command: `{' '.join(result.command)}`",
                f"- Exit code: `{result.returncode}`",
                "",
                "### Stdout",
                "",
                "```text",
                (result.stdout or "").rstrip(),
                "```",
                "",
                "### Stderr",
                "",
                "```text",
                (result.stderr or "").rstrip(),
                "```",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local tests and generate a markdown report.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "reports" / "tests",
        help="Directory for markdown test reports.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    checks = [
        _run_check(
            "Compile",
            [sys.executable, "-m", "compileall", "src/leanstack"],
            extra_env={"PYTHONPYCACHEPREFIX": "/tmp/leanstack-pycache"},
        ),
        _run_check(
            "Unit Tests",
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            extra_env={"PYTHONPATH": "src"},
        ),
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f"test_report_{timestamp}.md"
    report_path.write_text(_render_markdown(checks, timestamp), encoding="utf-8")
    print(report_path)

    return 0 if all(check.passed for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
