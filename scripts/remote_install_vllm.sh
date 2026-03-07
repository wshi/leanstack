#!/usr/bin/env zsh
set -eu
set -o pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_SCRIPT="${REMOTE_SCRIPT:-$ROOT/../remote.sh}"
REMOTE_HOME="${REMOTE_HOME:-/home/pto/lean}"
VLLM_VENV="${VLLM_VENV:-$REMOTE_HOME/venv-vllm}"
VLLM_SPEC="${VLLM_SPEC:-vllm}"
VLLM_TORCH_BACKEND="${VLLM_TORCH_BACKEND:-auto}"
source "$ROOT/scripts/remote_helpers.sh"

load_remote_cmd "$REMOTE_SCRIPT"

COMMAND="set -euo pipefail; \
python3 -m venv \"$VLLM_VENV\"; \
source \"$VLLM_VENV/bin/activate\"; \
python3 -m pip install --upgrade pip setuptools wheel uv; \
uv pip install --torch-backend=\"$VLLM_TORCH_BACKEND\" \"$VLLM_SPEC\"; \
python3 -c \"import torch, vllm; print(\\\"vllm\\\", vllm.__version__); print(\\\"torch\\\", torch.__version__)\""

run_remote_script "$COMMAND"
