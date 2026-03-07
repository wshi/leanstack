#!/usr/bin/env zsh
set -eu
set -o pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_SCRIPT="${REMOTE_SCRIPT:-$ROOT/../remote.sh}"
REMOTE_HOME="${REMOTE_HOME:-/home/pto/lean}"
VLLM_VENV="${VLLM_VENV:-$REMOTE_HOME/venv-vllm}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
MODEL_KEY="${MODEL_ID//\//__}"
MODEL_PATH_FILE="${MODEL_PATH_FILE:-$REMOTE_HOME/models/$MODEL_KEY.path}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
DTYPE="${DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-8b}"
LOG_DIR="${LOG_DIR:-$REMOTE_HOME/logs}"
WAIT_SECONDS="${WAIT_SECONDS:-180}"
source "$ROOT/scripts/remote_helpers.sh"

load_remote_cmd "$REMOTE_SCRIPT"

COMMAND="set -euo pipefail; \
source \"$VLLM_VENV/bin/activate\"; \
MODEL_PATH=\$(<\"$MODEL_PATH_FILE\"); \
mkdir -p \"$LOG_DIR\"; \
PID_FILE=\"$LOG_DIR/vllm_${PORT}.pid\"; \
LOG_FILE=\"$LOG_DIR/vllm_${PORT}.log\"; \
if [[ -f \"\$PID_FILE\" ]] && kill -0 \$(<\"\$PID_FILE\") 2>/dev/null; then \
  echo \"vLLM already running with pid \$(<\"\$PID_FILE\")\"; \
else \
  nohup vllm serve \"\$MODEL_PATH\" \
    --host \"$VLLM_HOST\" \
    --port \"$PORT\" \
    --dtype \"$DTYPE\" \
    --served-model-name \"$SERVED_MODEL_NAME\" \
    --gpu-memory-utilization \"$GPU_MEMORY_UTILIZATION\" \
    --max-model-len \"$MAX_MODEL_LEN\" \
    > \"\$LOG_FILE\" 2>&1 < /dev/null & \
  echo \$! > \"\$PID_FILE\"; \
fi; \
READY=0; \
for _ in \$(seq 1 $((WAIT_SECONDS / 2))); do \
  if curl -fsS \"http://$VLLM_HOST:$PORT/v1/models\"; then READY=1; break; fi; \
  sleep 2; \
done; \
if [[ \"\$READY\" != \"1\" ]]; then \
  echo \"--- vLLM log tail ---\" >&2; \
  tail -n 80 \"\$LOG_FILE\" >&2 || true; \
  echo \"vLLM server did not become ready in $WAIT_SECONDS seconds\" >&2; \
  exit 1; \
fi"

run_remote_script "$COMMAND"
