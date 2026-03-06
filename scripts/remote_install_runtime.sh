#!/usr/bin/env zsh
set -eu
set -o pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_SCRIPT="${REMOTE_SCRIPT:-$ROOT/../remote.sh}"
REMOTE_HOME="${REMOTE_HOME:-/home/pto/tilepilot}"
RUNTIME_VENV="${RUNTIME_VENV:-$REMOTE_HOME/runtime-venv}"
TORCH_SPEC="${TORCH_SPEC:-torch}"
source "$ROOT/scripts/remote_helpers.sh"

load_remote_cmd "$REMOTE_SCRIPT"

COMMAND="set -euo pipefail; \
RUNTIME_VENV=\"$RUNTIME_VENV\"; \
TORCH_SPEC=\"$TORCH_SPEC\"; \
python3 -m venv \"\$RUNTIME_VENV\"; \
source \"\$RUNTIME_VENV/bin/activate\"; \
python3 -m pip install --upgrade pip setuptools wheel; \
python3 -m pip install \"\$TORCH_SPEC\" transformers accelerate sentencepiece safetensors huggingface_hub; \
python3 -c \"import importlib; names=('torch','transformers','accelerate','sentencepiece','safetensors','huggingface_hub'); [print(name, getattr(importlib.import_module(name), '__version__', 'present')) for name in names]\""

run_remote_script "$COMMAND"
