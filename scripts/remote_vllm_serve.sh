#!/usr/bin/env zsh
set -eu
set -o pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_SCRIPT="${REMOTE_SCRIPT:-$ROOT/../remote.sh}"
REMOTE_HOME="${REMOTE_HOME:-/home/pto/lean}"
REMOTE_REPO="${REMOTE_REPO:-$REMOTE_HOME/repo}"
VLLM_VENV="${VLLM_VENV:-$REMOTE_HOME/venv-vllm}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Base}"
MODEL_KEY="${MODEL_ID//\//__}"
MODEL_PATH_FILE="${MODEL_PATH_FILE:-$REMOTE_HOME/models/$MODEL_KEY.path}"
MODEL_PATH="${MODEL_PATH:-}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
DTYPE="${DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-4b-base}"
LOG_DIR="${LOG_DIR:-$REMOTE_HOME/logs}"
WAIT_SECONDS="${WAIT_SECONDS:-180}"
PYTHON_DEV_ROOT="${PYTHON_DEV_ROOT:-$REMOTE_HOME/tmp/pydev_probe/extracted}"
source "$ROOT/scripts/remote_helpers.sh"

load_remote_cmd "$REMOTE_SCRIPT"

COMMAND="set -euo pipefail; \
cleanup_stale_vllm() { \
  local model_ref_local=\"$MODEL_PATH\"; \
  local vllm_exec=\"$VLLM_VENV/bin/vllm\"; \
  local vllm_python=\"$VLLM_VENV/bin/python3 $VLLM_VENV/bin/vllm\"; \
  if [[ -z \"\$model_ref_local\" && -f \"$MODEL_PATH_FILE\" ]]; then model_ref_local=\$(<\"$MODEL_PATH_FILE\"); fi; \
  ps -eo pid=,args= | while read -r pid rest; do \
    [[ \"\$rest\" == *\"\$vllm_exec serve\"* || \"\$rest\" == *\"\$vllm_python serve\"* ]] || continue; \
    [[ \"\$rest\" == *\"--port $PORT\"* ]] || continue; \
    if [[ -n \"\$model_ref_local\" && \"\$rest\" != *\"\$model_ref_local\"* ]]; then continue; fi; \
    kill \"\$pid\" 2>/dev/null || true; \
  done; \
}; \
source \"$VLLM_VENV/bin/activate\"; \
if [[ -f \"$PYTHON_DEV_ROOT/usr/include/python3.12/Python.h\" ]]; then \
  export LEANSTACK_PYTHON_DEV_ROOT=\"$PYTHON_DEV_ROOT\"; \
  export PYTHONPATH=\"$REMOTE_REPO/runtime_support:\${PYTHONPATH:-}\"; \
  export CPATH=\"$PYTHON_DEV_ROOT/usr/include/python3.12:$PYTHON_DEV_ROOT/usr/include/aarch64-linux-gnu/python3.12:\${CPATH:-}\"; \
  export C_INCLUDE_PATH=\"$PYTHON_DEV_ROOT/usr/include/python3.12:$PYTHON_DEV_ROOT/usr/include/aarch64-linux-gnu/python3.12:\${C_INCLUDE_PATH:-}\"; \
  export CPLUS_INCLUDE_PATH=\"$PYTHON_DEV_ROOT/usr/include/python3.12:$PYTHON_DEV_ROOT/usr/include/aarch64-linux-gnu/python3.12:\${CPLUS_INCLUDE_PATH:-}\"; \
  export LIBRARY_PATH=\"$PYTHON_DEV_ROOT/usr/lib/aarch64-linux-gnu:\${LIBRARY_PATH:-}\"; \
fi; \
MODEL_REF=\"$MODEL_PATH\"; \
if [[ -z \"\$MODEL_REF\" ]]; then MODEL_REF=\$(<\"$MODEL_PATH_FILE\"); fi; \
mkdir -p \"$LOG_DIR\"; \
PID_FILE=\"$LOG_DIR/vllm_${PORT}.pid\"; \
LOG_FILE=\"$LOG_DIR/vllm_${PORT}.log\"; \
if [[ -f \"\$PID_FILE\" ]] && kill -0 \$(<\"\$PID_FILE\") 2>/dev/null; then \
  if curl -fsS \"http://$VLLM_HOST:$PORT/v1/models\" >/dev/null 2>&1; then \
    echo \"vLLM already running with pid \$(<\"\$PID_FILE\")\"; \
  else \
    kill \$(<\"\$PID_FILE\") 2>/dev/null || true; \
    rm -f \"\$PID_FILE\"; \
    cleanup_stale_vllm; \
    sleep 2; \
    nohup vllm serve \"\$MODEL_REF\" \
      --host \"$VLLM_HOST\" \
      --port \"$PORT\" \
      --dtype \"$DTYPE\" \
      --served-model-name \"$SERVED_MODEL_NAME\" \
      --gpu-memory-utilization \"$GPU_MEMORY_UTILIZATION\" \
      --max-model-len \"$MAX_MODEL_LEN\" \
      > \"\$LOG_FILE\" 2>&1 < /dev/null & \
    echo \$! > \"\$PID_FILE\"; \
  fi; \
else \
  cleanup_stale_vllm; \
  sleep 2; \
  nohup vllm serve \"\$MODEL_REF\" \
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
