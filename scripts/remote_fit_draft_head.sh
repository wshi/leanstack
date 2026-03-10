#!/usr/bin/env zsh
set -eu
set -o pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_SCRIPT="${REMOTE_SCRIPT:-$ROOT/../remote.sh}"
REMOTE_HOME="${REMOTE_HOME:-/home/pto/lean}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-1.7B-Base}"
MODEL_KEY="${MODEL_ID//\//__}"
MODEL_PATH_FILE="${MODEL_PATH_FILE:-$REMOTE_HOME/models/$MODEL_KEY.path}"
MODEL_PATH="${MODEL_PATH:-}"
PACK_DIR="${PACK_DIR:-$REMOTE_HOME/packed/$MODEL_KEY}"
DRAFT_LAYER_COUNT="${DRAFT_LAYER_COUNT:-24}"
KEY="${KEY:-draft_l${DRAFT_LAYER_COUNT}_proj}"
CHUNK_TOKENS="${CHUNK_TOKENS:-128}"
MAX_CHUNKS="${MAX_CHUNKS:-32}"
RIDGE_LAMBDA="${RIDGE_LAMBDA:-0.1}"
CALIBRATION_MODE="${CALIBRATION_MODE:-prefill}"
DECODE_STEPS="${DECODE_STEPS:-16}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"
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
python3 -m leanstack.cli fit-draft-head \
  --model-path \"\$MODEL_REF\" \
  --pack-dir \"$PACK_DIR\" \
  --draft-layer-count \"$DRAFT_LAYER_COUNT\" \
  --key \"$KEY\" \
  --chunk-tokens \"$CHUNK_TOKENS\" \
  --max-chunks \"$MAX_CHUNKS\" \
  --ridge-lambda \"$RIDGE_LAMBDA\" \
  --calibration-mode \"$CALIBRATION_MODE\" \
  --decode-steps \"$DECODE_STEPS\" \
  --device \"$DEVICE\" \
  --dtype \"$DTYPE\" \
  --repo-root /home/pto/lean/repo"

run_remote_script "$COMMAND"
