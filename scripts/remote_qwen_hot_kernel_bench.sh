#!/usr/bin/env zsh
set -eu
set -o pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_SCRIPT="${REMOTE_SCRIPT:-$ROOT/../remote.sh}"
REMOTE_HOME="${REMOTE_HOME:-/home/pto/lean}"
REMOTE_REPO="$REMOTE_HOME/repo"
RESULT_DIR="${RESULT_DIR:-$REMOTE_HOME/artifacts/hot-kernels}"
CASE_KEYS="${CASE_KEYS:-q_proj_prefill64 kv_proj_prefill64 o_proj_prefill64 gate_up_proj_prefill64 down_proj_prefill64 rmsnorm_prefill64}"
WARMUP="${WARMUP:-3}"
REPEATS="${REPEATS:-10}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
source "$ROOT/scripts/remote_helpers.sh"

"$ROOT/scripts/remote_sync.sh"

load_remote_cmd "$REMOTE_SCRIPT"

COMMAND="set -euo pipefail; \
REMOTE_HOME=\"$REMOTE_HOME\"; \
REMOTE_REPO=\"$REMOTE_REPO\"; \
RESULT_DIR=\"$RESULT_DIR\"; \
CASE_KEYS=\"$CASE_KEYS\"; \
WARMUP=\"$WARMUP\"; \
REPEATS=\"$REPEATS\"; \
RUN_ID=\"$RUN_ID\"; \
ARTIFACT_ROOT=\"\$RESULT_DIR/\$RUN_ID\"; \
mkdir -p \"\$ARTIFACT_ROOT\"/{01_bytecode,02_tileir,03_cubin,04_sass,logs}; \
source /home/pto/venv-cutile/bin/activate; \
export PATH=\"/usr/local/cuda-13.1/bin:/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:\$PATH\"; \
export PYTHONPATH=\"\$REMOTE_REPO/src\"; \
export CUDA_TILE_DUMP_BYTECODE=\"\$ARTIFACT_ROOT/01_bytecode\"; \
export CUDA_TILE_DUMP_TILEIR=\"\$ARTIFACT_ROOT/02_tileir\"; \
EXTRA_CASE_ARGS=\"\"; \
if [[ -n \"\$CASE_KEYS\" ]]; then EXTRA_CASE_ARGS=\"--cases \$CASE_KEYS\"; fi; \
python3 \"\$REMOTE_REPO/experiments/cutile/qwen_bf16_hot_kernels.py\" --warmup \"\$WARMUP\" --repeats \"\$REPEATS\" \$EXTRA_CASE_ARGS --output \"\$ARTIFACT_ROOT/qwen_bf16_hot_kernels.json\" > \"\$ARTIFACT_ROOT/logs/qwen_bf16_hot_kernels.stdout.json\"; \
shopt -s nullglob; \
for CUTILE in \"\$ARTIFACT_ROOT\"/01_bytecode/*.cutile; do \
  STEM=\$(basename \"\$CUTILE\" .cutile); \
  /usr/local/cuda/bin/tileiras --gpu-name=sm_121 -O3 -o \"\$ARTIFACT_ROOT/03_cubin/\${STEM}.cubin\" \"\$CUTILE\" > \"\$ARTIFACT_ROOT/logs/\${STEM}_tileiras.log\" 2>&1; \
  if [[ -x /usr/local/cuda-13.1/bin/cuobjdump ]]; then /usr/local/cuda-13.1/bin/cuobjdump --dump-sass \"\$ARTIFACT_ROOT/03_cubin/\${STEM}.cubin\" > \"\$ARTIFACT_ROOT/04_sass/\${STEM}_cuobjdump.sass\" 2> \"\$ARTIFACT_ROOT/logs/\${STEM}_cuobjdump.log\" || true; fi; \
  if [[ -x /usr/local/cuda-13.1/bin/cuobjdump ]]; then /usr/local/cuda-13.1/bin/cuobjdump --dump-resource-usage \"\$ARTIFACT_ROOT/03_cubin/\${STEM}.cubin\" > \"\$ARTIFACT_ROOT/04_sass/\${STEM}_resource.txt\" 2>> \"\$ARTIFACT_ROOT/logs/\${STEM}_cuobjdump.log\" || true; fi; \
  if [[ -x /usr/local/cuda-13.0/bin/nvdisasm ]]; then /usr/local/cuda-13.0/bin/nvdisasm -g -hex -c \"\$ARTIFACT_ROOT/03_cubin/\${STEM}.cubin\" > \"\$ARTIFACT_ROOT/04_sass/\${STEM}_nvdisasm.sass\"; fi; \
done; \
printf \"artifact_root=%s\n\" \"\$ARTIFACT_ROOT\"; \
cat \"\$ARTIFACT_ROOT/qwen_bf16_hot_kernels.json\""

run_remote_script "$COMMAND"
