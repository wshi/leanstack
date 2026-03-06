#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


def command_output(argv: list[str]) -> str | None:
    try:
        result = subprocess.run(argv, check=True, capture_output=True, text=True)
    except Exception:
        return None
    return result.stdout.strip()


def module_version(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return f"MISSING ({type(exc).__name__}: {exc})"
    return getattr(module, "__version__", "present")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect remote environment metadata.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = {
        "host": platform.node(),
        "python": platform.python_version(),
        "cwd": os.getcwd(),
        "modules": {
            name: module_version(name)
            for name in ("cuda.tile", "cupy", "torch", "transformers", "accelerate", "sentencepiece", "safetensors")
        },
        "tools": {
            name: shutil.which(name)
            for name in ("git", "python3", "pip", "tileiras", "cuobjdump", "nvdisasm", "nvidia-smi")
        },
        "gpu": command_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
        ),
    }

    payload = json.dumps(report, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{payload}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

