#!/usr/bin/env zsh
set -eu
set -o pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_SCRIPT="${REMOTE_SCRIPT:-$ROOT/../remote.sh}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/home/pto/venv-cutile/bin/activate}"
source "$ROOT/scripts/remote_helpers.sh"

load_remote_cmd "$REMOTE_SCRIPT"

COMMAND="set -euo pipefail; \
if [[ -f \"$VENV_ACTIVATE\" ]]; then source \"$VENV_ACTIVATE\"; fi; \
python3 -c \"import importlib, json; from pathlib import Path; tile=importlib.import_module(\\\"cuda.tile\\\"); \
public=sorted(name for name in dir(tile) if not name.startswith(\\\"_\\\") and getattr(getattr(tile, name), \\\"__class__\\\", None).__name__ in (\\\"ArithmeticDType\\\", \\\"NumericDType\\\", \\\"DType\\\")); \
datatype_text=Path(tile.__file__).with_name(\\\"_datatype.py\\\").read_text(encoding=\\\"utf-8\\\", errors=\\\"ignore\\\"); \
bytecode_type_text=Path(tile.__file__).with_name(\\\"_bytecode\\\").joinpath(\\\"type.py\\\").read_text(encoding=\\\"utf-8\\\", errors=\\\"ignore\\\"); \
payload={\\\"cuda_tile_module\\\": tile.__file__, \\\"cuda_tile_version\\\": Path(tile.__file__).with_name(\\\"VERSION\\\").read_text(encoding=\\\"utf-8\\\").strip(), \
\\\"public_dtypes\\\": public, \\\"public_has_fp4_symbol\\\": any(\\\"fp4\\\" in name.lower() or \\\"float4\\\" in name.lower() for name in public), \
\\\"datatype_mentions_fp4\\\": \\\"fp4\\\" in datatype_text.lower() or \\\"float4\\\" in datatype_text.lower(), \
\\\"bytecode_mentions_fp4\\\": \\\"fp4\\\" in bytecode_type_text.lower() or \\\"float4\\\" in bytecode_type_text.lower()}; \
print(json.dumps(payload, indent=2))\"; \
printf \"\n-- tileiras targets --\n\"; \
tileiras --help | grep \"sm_12\" || true"

run_remote_script "$COMMAND"
