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
API_KEY="${API_KEY:-EMPTY}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-600}"
RESULT_DIR="${RESULT_DIR:-$REMOTE_HOME/benchmarks}"
BENCHMARK_TAG="${BENCHMARK_TAG:-}"
source "$ROOT/scripts/remote_helpers.sh"

"$ROOT/scripts/remote_sync.sh"
load_remote_cmd "$REMOTE_SCRIPT"

COMMAND="set -euo pipefail; \
export PYTHONPATH=/home/pto/lean/repo/src; \
eval \"\$(python3 -m leanstack.cli show-benchmark-profile --profile \"$PROFILE\" --format shell)\"; \
mkdir -p \"$RESULT_DIR\"; \
if [[ -z \"$BENCHMARK_TAG\" ]]; then BENCHMARK_TAG=\$(date -u +%Y%m%dT%H%M%SZ); fi; \
OUTPUT_PATH=\"$RESULT_DIR/${SYSTEM_LABEL}_${VARIANT_LABEL}_${PROFILE}_\${BENCHMARK_TAG}.json\"; \
python3 /home/pto/lean/repo/experiments/models/openai_compatible_benchmark.py \
  --base-url \"$BASE_URL\" \
  --model \"$MODEL_NAME\" \
  --system \"$SYSTEM_LABEL\" \
  --variant \"$VARIANT_LABEL\" \
  --benchmark-profile \"$PROFILE\" \
  --prompt \"\$PROMPT\" \
  --max-new-tokens \"\$MAX_NEW_TOKENS\" \
  --api-key \"$API_KEY\" \
  --request-timeout \"$REQUEST_TIMEOUT\" \
  --output \"\$OUTPUT_PATH\" > /dev/null; \
cat \"\$OUTPUT_PATH\""

run_remote_script "$COMMAND"
