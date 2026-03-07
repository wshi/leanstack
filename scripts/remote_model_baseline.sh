#!/usr/bin/env zsh
set -eu
set -o pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_SCRIPT="${REMOTE_SCRIPT:-$ROOT/../remote.sh}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-1.7B-Base}"
REMOTE_HOME="${REMOTE_HOME:-/home/pto/lean}"
MODEL_KEY="${MODEL_ID//\//__}"
MODEL_SOURCE_FILE="${MODEL_SOURCE_FILE:-$REMOTE_HOME/models/$MODEL_KEY.path}"
MODEL_PATH="${MODEL_PATH:-}"
PROMPT="${PROMPT:-Summarize why an agent-built, Qwen3-on-Blackwell software path may beat a compatibility-heavy serving stack.}"
PROMPT_FORMAT="${PROMPT_FORMAT:-auto}"
THINKING_MODE="${THINKING_MODE:-auto}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
source "$ROOT/scripts/remote_helpers.sh"

"$ROOT/scripts/remote_sync.sh"
load_remote_cmd "$REMOTE_SCRIPT"

COMMAND="set -euo pipefail; \
source /home/pto/venv-cutile/bin/activate; \
export PYTHONPATH=/home/pto/lean/repo/src; \
MODEL_REF=\"$MODEL_PATH\"; \
if [[ -z \"\$MODEL_REF\" ]]; then MODEL_REF=\"$MODEL_ID\"; fi; \
if [[ -z \"$MODEL_PATH\" && -n \"$MODEL_SOURCE_FILE\" && -f \"$MODEL_SOURCE_FILE\" ]]; then MODEL_REF=\$(<\"$MODEL_SOURCE_FILE\"); fi; \
EXTRA_ARGS=\"\"; \
if [[ \"$THINKING_MODE\" == \"enable\" ]]; then EXTRA_ARGS=\"--enable-thinking\"; fi; \
if [[ \"$THINKING_MODE\" == \"disable\" ]]; then EXTRA_ARGS=\"--disable-thinking\"; fi; \
python3 /home/pto/lean/repo/experiments/models/hf_causal_lm_smoke.py --model-id \"\$MODEL_REF\" --prompt \"$PROMPT\" --prompt-format \"$PROMPT_FORMAT\" --max-new-tokens \"$MAX_NEW_TOKENS\" \$EXTRA_ARGS"

run_remote_script "$COMMAND"
