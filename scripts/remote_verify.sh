#!/usr/bin/env zsh
set -eu
set -o pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_SCRIPT="${REMOTE_SCRIPT:-$ROOT/../remote.sh}"
REMOTE_HOME="${REMOTE_HOME:-/home/pto/leanstack}"
REMOTE_REPO="$REMOTE_HOME/repo"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
source "$ROOT/scripts/remote_helpers.sh"

"$ROOT/scripts/remote_sync.sh"

load_remote_cmd "$REMOTE_SCRIPT"

COMMAND="set -euo pipefail; \
REMOTE_HOME=\"$REMOTE_HOME\"; \
REMOTE_REPO=\"$REMOTE_REPO\"; \
RUN_ID=\"$RUN_ID\"; \
ARTIFACT_ROOT=\"\$REMOTE_HOME/artifacts/\$RUN_ID\"; \
mkdir -p \"\$ARTIFACT_ROOT\"/{01_bytecode,02_tileir,03_cubin,04_sass,logs}; \
source /home/pto/venv-cutile/bin/activate; \
export PYTHONPATH=\"\$REMOTE_REPO/src\"; \
export CUDA_TILE_DUMP_BYTECODE=\"\$ARTIFACT_ROOT/01_bytecode\"; \
export CUDA_TILE_DUMP_TILEIR=\"\$ARTIFACT_ROOT/02_tileir\"; \
python3 \"\$REMOTE_REPO/experiments/cutile/vector_add.py\" --artifact-dir \"\$ARTIFACT_ROOT\" > \"\$ARTIFACT_ROOT/logs/vector_add.log\"; \
CUTILE=\$(ls -1t \"\$ARTIFACT_ROOT\"/01_bytecode/*.cutile 2>/dev/null | head -n 1 || true); \
if [[ -n \"\$CUTILE\" ]]; then /usr/local/cuda/bin/tileiras --gpu-name=sm_121 -O3 -o \"\$ARTIFACT_ROOT/03_cubin/vector_add.cubin\" --print-before-all --print-after-all --print-module-scope \"\$CUTILE\" > \"\$ARTIFACT_ROOT/logs/tileiras.log\" 2>&1; fi; \
if [[ -x /usr/local/cuda-13.1/bin/cuobjdump && -f \"\$ARTIFACT_ROOT/03_cubin/vector_add.cubin\" ]]; then /usr/local/cuda-13.1/bin/cuobjdump --dump-sass \"\$ARTIFACT_ROOT/03_cubin/vector_add.cubin\" > \"\$ARTIFACT_ROOT/04_sass/vector_add_cuobjdump.sass\"; fi; \
if [[ -x /usr/local/cuda-13.1/bin/cuobjdump && -f \"\$ARTIFACT_ROOT/03_cubin/vector_add.cubin\" ]]; then /usr/local/cuda-13.1/bin/cuobjdump --dump-resource-usage \"\$ARTIFACT_ROOT/03_cubin/vector_add.cubin\" > \"\$ARTIFACT_ROOT/04_sass/vector_add_resource.txt\"; fi; \
if [[ -x /usr/local/cuda-13.0/bin/nvdisasm && -f \"\$ARTIFACT_ROOT/03_cubin/vector_add.cubin\" ]]; then /usr/local/cuda-13.0/bin/nvdisasm -g -hex -c \"\$ARTIFACT_ROOT/03_cubin/vector_add.cubin\" > \"\$ARTIFACT_ROOT/04_sass/vector_add_nvdisasm.sass\"; fi; \
python3 \"\$REMOTE_REPO/scripts/collect_remote_probe.py\" --output \"\$ARTIFACT_ROOT/logs/remote_probe.json\" > \"\$ARTIFACT_ROOT/logs/remote_probe_stdout.json\"; \
printf \"Artifact root: %s\n\" \"\$ARTIFACT_ROOT\"; \
find \"\$ARTIFACT_ROOT\" -maxdepth 2 -type f | sort"

run_remote_script "$COMMAND"
