#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


REQUIRED = (
    "README.md",
    "docs/ARCHITECTURE.md",
    "docs/EXECUTION_PLAN.md",
    "docs/REMOTE_VALIDATION.md",
    "experiments/cutile/vector_add.py",
    "scripts/remote_verify.sh",
)


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    missing = [path for path in REQUIRED if not (root / path).exists()]
    if missing:
        for path in missing:
            print(f"missing: {path}")
        return 1
    print("leanstack repo layout is present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
