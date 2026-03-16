#!/usr/bin/env zsh
set -eu
set -o pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_SCRIPT="${REMOTE_SCRIPT:-$ROOT/../remote.sh}"
REMOTE_HOME="${REMOTE_HOME:-/home/pto/lean}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Base}"
MODEL_KEY="${MODEL_ID//\//__}"
MODEL_PATH_FILE="${MODEL_PATH_FILE:-$REMOTE_HOME/models/$MODEL_KEY.path}"
MODEL_PATH="${MODEL_PATH:-}"
NUM_LAYERS="${NUM_LAYERS:-2}"
PROMPT="${PROMPT:-Explain why fixing the model-chip contract simplifies inference software.}"
PROMPT_FORMAT="${PROMPT_FORMAT:-chat}"
THINKING_MODE="${THINKING_MODE:-disable}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-16}"
DEVICE="${DEVICE:-cuda:0}"
WITH_OUTPUT_HEAD="${WITH_OUTPUT_HEAD:-0}"
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
if [[ \"$WITH_OUTPUT_HEAD\" == \"1\" ]]; then EXTRA_ARGS=\"\$EXTRA_ARGS --with-output-head\"; fi; \
python3 /home/pto/lean/repo/experiments/models/qwen_explicit_stack_probe.py --model-path \"\$MODEL_REF\" --num-layers \"$NUM_LAYERS\" --device \"$DEVICE\" --prompt \"$PROMPT\" --prompt-format \"$PROMPT_FORMAT\" --max-prefill-tokens \"$MAX_PREFILL_TOKENS\" \$EXTRA_ARGS"

run_remote_script "$COMMAND"
