#!/usr/bin/env zsh
set -eu
set -o pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_SCRIPT="${REMOTE_SCRIPT:-$ROOT/../remote.sh}"
REMOTE_HOME="${REMOTE_HOME:-/home/pto/lean}"
PROFILE="${PROFILE:-decode_64_256}"
SYSTEM_LABEL="${SYSTEM_LABEL:-framework}"
VARIANT_LABEL="${VARIANT_LABEL:-openai}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
MODEL_NAME="${MODEL_NAME:?set MODEL_NAME to the served model name}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Base}"
MODEL_KEY="${MODEL_ID//\//__}"
TOKENIZER_MODEL_PATH_FILE="${TOKENIZER_MODEL_PATH_FILE:-$REMOTE_HOME/models/$MODEL_KEY.path}"
API_KEY="${API_KEY:-EMPTY}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-600}"
RESULT_DIR="${RESULT_DIR:-$REMOTE_HOME/benchmarks}"
BENCHMARK_TAG="${BENCHMARK_TAG:-}"
PROMPT_OVERRIDE="${PROMPT_OVERRIDE:-}"
MAX_NEW_TOKENS_OVERRIDE="${MAX_NEW_TOKENS_OVERRIDE:-}"
TEMPERATURE="${TEMPERATURE:-0.0}"
IGNORE_EOS="${IGNORE_EOS:-1}"
SKIP_REMOTE_SYNC="${SKIP_REMOTE_SYNC:-0}"
source "$ROOT/scripts/remote_helpers.sh"

if [[ "$SKIP_REMOTE_SYNC" != "1" ]]; then
  "$ROOT/scripts/remote_sync.sh"
fi
load_remote_cmd "$REMOTE_SCRIPT"

COMMAND="set -euo pipefail; \
source /home/pto/venv-cutile/bin/activate; \
export PYTHONPATH=/home/pto/lean/repo/src; \
eval \"\$(python3 -m leanstack.cli show-benchmark-profile --profile \"$PROFILE\" --format shell)\"; \
if [[ -n \"$PROMPT_OVERRIDE\" ]]; then PROMPT=\"$PROMPT_OVERRIDE\"; fi; \
if [[ -n \"$MAX_NEW_TOKENS_OVERRIDE\" ]]; then MAX_NEW_TOKENS=\"$MAX_NEW_TOKENS_OVERRIDE\"; fi; \
TOKENIZER_MODEL_PATH=\$(<\"$TOKENIZER_MODEL_PATH_FILE\"); \
mkdir -p \"$RESULT_DIR\"; \
if [[ -z \"$BENCHMARK_TAG\" ]]; then BENCHMARK_TAG=\$(date -u +%Y%m%dT%H%M%SZ); fi; \
OUTPUT_PATH=\"$RESULT_DIR/${SYSTEM_LABEL}_${VARIANT_LABEL}_${PROFILE}_\${BENCHMARK_TAG}.json\"; \
EXTRA_ARGS=\"\"; \
if [[ \"$IGNORE_EOS\" == \"1\" ]]; then EXTRA_ARGS=\"\$EXTRA_ARGS --ignore-eos\"; fi; \
python3 /home/pto/lean/repo/experiments/models/openai_compatible_benchmark.py \
  --base-url \"$BASE_URL\" \
  --model \"$MODEL_NAME\" \
  --system \"$SYSTEM_LABEL\" \
  --variant \"$VARIANT_LABEL\" \
  --benchmark-profile \"$PROFILE\" \
  --prompt \"\$PROMPT\" \
  --tokenizer-model-path \"\$TOKENIZER_MODEL_PATH\" \
  --exact-prompt-tokens \"\$TARGET_PROMPT_TOKENS\" \
  --max-new-tokens \"\$MAX_NEW_TOKENS\" \
  --temperature \"$TEMPERATURE\" \
  --api-key \"$API_KEY\" \
  --request-timeout \"$REQUEST_TIMEOUT\" \
  \$EXTRA_ARGS \
  --output \"\$OUTPUT_PATH\" > /dev/null; \
cat \"\$OUTPUT_PATH\""

run_remote_script "$COMMAND"
