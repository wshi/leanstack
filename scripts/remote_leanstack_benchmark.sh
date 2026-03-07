#!/usr/bin/env zsh
set -eu
set -o pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_SCRIPT="${REMOTE_SCRIPT:-$ROOT/../remote.sh}"
REMOTE_HOME="${REMOTE_HOME:-/home/pto/lean}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
MODEL_KEY="${MODEL_ID//\//__}"
MODEL_PATH_FILE="${MODEL_PATH_FILE:-$REMOTE_HOME/models/$MODEL_KEY.path}"
MODEL_PATH="${MODEL_PATH:-}"
PROFILE="${PROFILE:-single_stream_short}"
RUNTIME_MODE="${RUNTIME_MODE:-semantic}"
NUM_LAYERS="${NUM_LAYERS:-0}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"
RESULT_DIR="${RESULT_DIR:-$REMOTE_HOME/benchmarks}"
BENCHMARK_TAG="${BENCHMARK_TAG:-}"
source "$ROOT/scripts/remote_helpers.sh"

"$ROOT/scripts/remote_sync.sh"
load_remote_cmd "$REMOTE_SCRIPT"

COMMAND="set -euo pipefail; \
source /home/pto/venv-cutile/bin/activate; \
export PYTHONPATH=/home/pto/lean/repo/src; \
MODEL_REF=\"$MODEL_PATH\"; \
if [[ -z \"\$MODEL_REF\" ]]; then MODEL_REF=\$(<\"$MODEL_PATH_FILE\"); fi; \
eval \"\$(python3 -m leanstack.cli show-benchmark-profile --profile \"$PROFILE\" --format shell)\"; \
mkdir -p \"$RESULT_DIR\"; \
if [[ -z \"$BENCHMARK_TAG\" ]]; then BENCHMARK_TAG=\$(date -u +%Y%m%dT%H%M%SZ); fi; \
OUTPUT_PATH=\"$RESULT_DIR/leanstack_${RUNTIME_MODE}_${PROFILE}_\${BENCHMARK_TAG}.json\"; \
python3 /home/pto/lean/repo/experiments/models/qwen_explicit_runtime_loop.py \
  --model-path \"\$MODEL_REF\" \
  --runtime-mode \"$RUNTIME_MODE\" \
  --benchmark-profile \"$PROFILE\" \
  --num-layers \"$NUM_LAYERS\" \
  --device \"$DEVICE\" \
  --dtype \"$DTYPE\" \
  --prompt \"\$PROMPT\" \
  --prompt-format \"\$PROMPT_FORMAT\" \
  --max-prefill-tokens \"\$MAX_PREFILL_TOKENS\" \
  --max-new-tokens \"\$MAX_NEW_TOKENS\" \
  --output \"\$OUTPUT_PATH\" > /dev/null; \
cat \"\$OUTPUT_PATH\""

run_remote_script "$COMMAND"
