#!/usr/bin/env zsh
set -eu
set -o pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_SCRIPT="${REMOTE_SCRIPT:-$ROOT/../remote.sh}"
REMOTE_HOME="${REMOTE_HOME:-/home/pto/lean}"
RESULT_DIR="${RESULT_DIR:-$REMOTE_HOME/artifacts/fp4-gate}"
REQUIRE_SUCCESS="${REQUIRE_SUCCESS:-0}"
source "$ROOT/scripts/remote_helpers.sh"

"$ROOT/scripts/remote_sync.sh"
load_remote_cmd "$REMOTE_SCRIPT"

COMMAND="set -euo pipefail; \
source /home/pto/venv-cutile/bin/activate; \
export PYTHONPATH=/home/pto/lean/repo/src; \
mkdir -p \"$RESULT_DIR\"; \
STAMP=\$(date -u +%Y%m%dT%H%M%SZ); \
OUTPUT_PATH=\"$RESULT_DIR/fp4_gate_\${STAMP}.json\"; \
EXTRA_ARGS=\"\"; \
if [[ \"$REQUIRE_SUCCESS\" == \"1\" ]]; then EXTRA_ARGS=\"--require-success\"; fi; \
python3 /home/pto/lean/repo/experiments/cutile/fp4_compiler_gate.py --output \"\$OUTPUT_PATH\" \$EXTRA_ARGS > /dev/null; \
printf \"output_path=%s\n\" \"\$OUTPUT_PATH\"; \
cat \"\$OUTPUT_PATH\""

run_remote_script "$COMMAND"
