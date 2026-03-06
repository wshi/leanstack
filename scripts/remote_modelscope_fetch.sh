#!/usr/bin/env zsh
set -eu
set -o pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_SCRIPT="${REMOTE_SCRIPT:-$ROOT/../remote.sh}"
REMOTE_HOME="${REMOTE_HOME:-/home/pto/lean}"
REMOTE_REPO="$REMOTE_HOME/repo"
MODEL_ID="${MODEL_ID:?set MODEL_ID to a ModelScope model id}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$REMOTE_HOME/models}"
MODEL_REVISION="${MODEL_REVISION:-}"
MODELSCOPE_SPEC="${MODELSCOPE_SPEC:-modelscope}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/home/pto/venv-cutile/bin/activate}"
MODEL_KEY="${MODEL_ID//\//__}"
MODEL_PATH_FILE="${MODEL_PATH_FILE:-$MODEL_CACHE_DIR/$MODEL_KEY.path}"
source "$ROOT/scripts/remote_helpers.sh"

"$ROOT/scripts/remote_sync.sh"
load_remote_cmd "$REMOTE_SCRIPT"

COMMAND="set -euo pipefail; \
REMOTE_REPO=\"$REMOTE_REPO\"; \
MODEL_ID=\"$MODEL_ID\"; \
MODEL_CACHE_DIR=\"$MODEL_CACHE_DIR\"; \
MODEL_REVISION=\"$MODEL_REVISION\"; \
MODEL_PATH_FILE=\"$MODEL_PATH_FILE\"; \
MODELSCOPE_SPEC=\"$MODELSCOPE_SPEC\"; \
VENV_ACTIVATE=\"$VENV_ACTIVATE\"; \
mkdir -p \"\$MODEL_CACHE_DIR\"; \
if [[ -f \"\$VENV_ACTIVATE\" ]]; then source \"\$VENV_ACTIVATE\"; fi; \
python3 -c \"import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('modelscope') else 1)\" || python3 -m pip install \"\$MODELSCOPE_SPEC\"; \
FETCH_ARGS=(--model-id \"\$MODEL_ID\" --cache-dir \"\$MODEL_CACHE_DIR\" --path-file \"\$MODEL_PATH_FILE\"); \
if [[ -n \"\$MODEL_REVISION\" ]]; then FETCH_ARGS+=(--revision \"\$MODEL_REVISION\"); fi; \
python3 \"\$REMOTE_REPO/scripts/fetch_modelscope_snapshot.py\" \"\${FETCH_ARGS[@]}\"; \
printf \"Resolved model path file: %s\n\" \"\$MODEL_PATH_FILE\"; \
cat \"\$MODEL_PATH_FILE\""

run_remote_script "$COMMAND"
