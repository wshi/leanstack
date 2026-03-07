from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load_cuda_tile() -> tuple[Any | None, str | None]:
    try:
        return importlib.import_module("cuda.tile"), None
    except Exception as exc:  # pragma: no cover - exercised via runtime environment
        return None, f"{type(exc).__name__}: {exc}"


def _public_dtypes(tile_module: Any) -> list[str]:
    names: list[str] = []
    for name in dir(tile_module):
        if name.startswith("_"):
            continue
        value = getattr(tile_module, name)
        class_name = getattr(getattr(value, "__class__", None), "__name__", "")
        if class_name in {"ArithmeticDType", "NumericDType", "DType"}:
            names.append(name)
    return sorted(names)


def _tileiras_targets(tileiras: str) -> tuple[bool, list[str]]:
    try:
        result = subprocess.run(
            [tileiras, "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return False, []
    targets = [line.strip() for line in result.stdout.splitlines() if "sm_12" in line]
    return any("sm_121" in line for line in targets), targets


def build_gate_report(tileiras: str) -> dict[str, Any]:
    tile_module, import_error = _load_cuda_tile()
    backend_sm121_available, tileiras_targets = _tileiras_targets(tileiras)
    if tile_module is None:
        return {
            "gate": "fp4_compiler_gate",
            "status": "blocked",
            "gate_cleared": False,
            "blocker": "cuda.tile import failed",
            "import_error": import_error,
            "backend_sm121_available": backend_sm121_available,
            "tileiras_targets": tileiras_targets,
            "next_action": "Install or repair the remote cuTile environment before attempting any FP4 kernel authoring.",
        }

    module_path = Path(tile_module.__file__)
    datatype_text = module_path.with_name("_datatype.py").read_text(encoding="utf-8", errors="ignore")
    bytecode_type_text = module_path.with_name("_bytecode").joinpath("type.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )
    public_dtypes = _public_dtypes(tile_module)
    public_has_fp4_symbol = any("fp4" in name.lower() or "float4" in name.lower() for name in public_dtypes)
    datatype_mentions_fp4 = "fp4" in datatype_text.lower() or "float4" in datatype_text.lower()
    bytecode_mentions_fp4 = "fp4" in bytecode_type_text.lower() or "float4" in bytecode_type_text.lower()
    authorable_fp4_surface = public_has_fp4_symbol and datatype_mentions_fp4 and bytecode_mentions_fp4

    gate_cleared = authorable_fp4_surface and backend_sm121_available
    status = "cleared" if gate_cleared else "blocked"
    blocker = None
    next_action = "Author and compile a minimal FP4 GEMM kernel through the public cuTile path."
    if not authorable_fp4_surface:
        blocker = "public cuda.tile frontend does not expose a complete FP4 authoring surface"
        next_action = (
            "Treat the FP4 route as blocked on the public frontend, or add a tightly scoped PTX wedge before proceeding."
        )
    elif not backend_sm121_available:
        blocker = "tileiras does not advertise sm_121"
        next_action = "Repair the backend compiler target before attempting FP4 kernel generation."

    return {
        "gate": "fp4_compiler_gate",
        "status": status,
        "gate_cleared": gate_cleared,
        "blocker": blocker,
        "cuda_tile_module": str(module_path),
        "cuda_tile_version": module_path.with_name("VERSION").read_text(encoding="utf-8").strip(),
        "public_dtypes": public_dtypes,
        "public_has_fp4_symbol": public_has_fp4_symbol,
        "datatype_mentions_fp4": datatype_mentions_fp4,
        "bytecode_mentions_fp4": bytecode_mentions_fp4,
        "backend_sm121_available": backend_sm121_available,
        "tileiras_targets": tileiras_targets,
        "next_action": next_action,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate whether the current public cuTile install clears the FP4 compiler gate.")
    parser.add_argument("--tileiras", default="tileiras")
    parser.add_argument("--output")
    parser.add_argument("--require-success", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_gate_report(args.tileiras)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    if args.require_success and not report["gate_cleared"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
