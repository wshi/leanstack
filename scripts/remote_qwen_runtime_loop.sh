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
RUNTIME_MODE="${RUNTIME_MODE:-borrowed}"
NUM_LAYERS="${NUM_LAYERS:-0}"
PROMPT="${PROMPT:-Explain why fixing the model-chip contract simplifies inference software.}"
PROMPT_FORMAT="${PROMPT_FORMAT:-chat}"
THINKING_MODE="${THINKING_MODE:-disable}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4}"
PAGE_SIZE="${PAGE_SIZE:-16}"
DEVICE="${DEVICE:-cuda:0}"
IGNORE_EOS="${IGNORE_EOS:-0}"
SKIP_FINAL_CACHE_ADVANCE="${SKIP_FINAL_CACHE_ADVANCE:-0}"
COMPILE="${COMPILE:-0}"
COMPILE_MODE="${COMPILE_MODE:-default}"
source "$ROOT/scripts/remote_helpers.sh"

"$ROOT/scripts/remote_sync.sh"
load_remote_cmd "$REMOTE_SCRIPT"

COMMAND="set -euo pipefail; \
source /home/pto/venv-cutile/bin/activate; \
export PYTHONPATH=/home/pto/lean/repo/src; \
MODEL_REF=\"$MODEL_PATH\"; \
if [[ -z \"\$MODEL_REF\" ]]; then MODEL_REF=\$(<\"$MODEL_PATH_FILE\"); fi; \
EXTRA_ARGS=\"\"; \
if [[ \"$THINKING_MODE\" == \"enable\" ]]; then EXTRA_ARGS=\"--enable-thinking\"; fi; \
if [[ \"$THINKING_MODE\" == \"disable\" ]]; then EXTRA_ARGS=\"--disable-thinking\"; fi; \
if [[ \"$IGNORE_EOS\" == \"1\" ]]; then EXTRA_ARGS=\"\$EXTRA_ARGS --ignore-eos\"; fi; \
if [[ \"$SKIP_FINAL_CACHE_ADVANCE\" == \"1\" ]]; then EXTRA_ARGS=\"\$EXTRA_ARGS --skip-final-cache-advance\"; fi; \
if [[ \"$COMPILE\" == \"1\" ]]; then EXTRA_ARGS=\"\$EXTRA_ARGS --compile --compile-mode $COMPILE_MODE\"; fi; \
python3 /home/pto/lean/repo/experiments/models/qwen_explicit_runtime_loop.py --model-path \"\$MODEL_REF\" --runtime-mode \"$RUNTIME_MODE\" --num-layers \"$NUM_LAYERS\" --device \"$DEVICE\" --prompt \"$PROMPT\" --prompt-format \"$PROMPT_FORMAT\" --max-prefill-tokens \"$MAX_PREFILL_TOKENS\" --max-new-tokens \"$MAX_NEW_TOKENS\" --page-size \"$PAGE_SIZE\" \$EXTRA_ARGS"

run_remote_script "$COMMAND"
