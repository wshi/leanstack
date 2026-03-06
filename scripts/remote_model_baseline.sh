#!/usr/bin/env zsh
set -eu
set -o pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_SCRIPT="${REMOTE_SCRIPT:-$ROOT/../remote.sh}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-32B}"
MODEL_SOURCE_FILE="${MODEL_SOURCE_FILE:-}"
PROMPT="${PROMPT:-Summarize the design goals of a TileIR-first serving stack.}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
source "$ROOT/scripts/remote_helpers.sh"

"$ROOT/scripts/remote_sync.sh"
load_remote_cmd "$REMOTE_SCRIPT"

COMMAND="set -euo pipefail; \
source /home/pto/venv-cutile/bin/activate; \
export PYTHONPATH=/home/pto/lean/repo/src; \
MODEL_REF=\"$MODEL_ID\"; \
if [[ -n \"$MODEL_SOURCE_FILE\" && -f \"$MODEL_SOURCE_FILE\" ]]; then MODEL_REF=\$(<\"$MODEL_SOURCE_FILE\"); fi; \
python3 /home/pto/lean/repo/experiments/models/hf_causal_lm_smoke.py --model-id \"\$MODEL_REF\" --prompt \"$PROMPT\" --max-new-tokens \"$MAX_NEW_TOKENS\""

run_remote_script "$COMMAND"
