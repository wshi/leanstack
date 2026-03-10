#!/usr/bin/env zsh
# Build a leanpack artifact for the draft model (Qwen3-0.6B-Base)
# used in dual-model speculative decode.
set -eu
set -o pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_SCRIPT="${REMOTE_SCRIPT:-$ROOT/../remote.sh}"
REMOTE_HOME="${REMOTE_HOME:-/home/pto/lean}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-0.6B-Base}"
MODEL_KEY="${MODEL_ID//\//__}"
MODEL_PATH_FILE="${MODEL_PATH_FILE:-$REMOTE_HOME/models/$MODEL_KEY.path}"
MODEL_PATH="${MODEL_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$REMOTE_HOME/packed/$MODEL_KEY}"
MANIFEST_ONLY="${MANIFEST_ONLY:-0}"
OVERWRITE="${OVERWRITE:-0}"
SKIP_REMOTE_SYNC="${SKIP_REMOTE_SYNC:-0}"
source "$ROOT/scripts/remote_helpers.sh"

if [[ "$SKIP_REMOTE_SYNC" != "1" ]]; then
  "$ROOT/scripts/remote_sync.sh"
fi
load_remote_cmd "$REMOTE_SCRIPT"

COMMAND="set -euo pipefail; \
source /home/pto/venv-cutile/bin/activate; \
export PYTHONPATH=/home/pto/lean/repo/src; \
MODEL_REF=\"$MODEL_PATH\"; \
if [[ -z \"\$MODEL_REF\" ]]; then MODEL_REF=\$(<\"$MODEL_PATH_FILE\"); fi; \
mkdir -p \"$(dirname "$OUTPUT_DIR")\"; \
EXTRA_ARGS=\"\"; \
if [[ \"$MANIFEST_ONLY\" == \"1\" ]]; then EXTRA_ARGS=\"\$EXTRA_ARGS --manifest-only\"; fi; \
if [[ \"$OVERWRITE\" == \"1\" ]]; then EXTRA_ARGS=\"\$EXTRA_ARGS --overwrite\"; fi; \
python3 -m leanstack.cli build-leanpack \
  --model qwen-draft \
  --model-path \"\$MODEL_REF\" \
  --output-dir \"$OUTPUT_DIR\" \
  \$EXTRA_ARGS"

run_remote_script "$COMMAND"
