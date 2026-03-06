#!/usr/bin/env zsh
set -eu
set -o pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_SCRIPT="${REMOTE_SCRIPT:-$ROOT/../remote.sh}"
REMOTE_HOME="${REMOTE_HOME:-/home/pto/tilepilot}"
REMOTE_REPO="$REMOTE_HOME/repo"
source "$ROOT/scripts/remote_helpers.sh"

"$ROOT/scripts/remote_sync.sh"

load_remote_cmd "$REMOTE_SCRIPT"

COMMAND="set -euo pipefail; \
REMOTE_HOME=\"$REMOTE_HOME\"; \
REMOTE_REPO=\"$REMOTE_REPO\"; \
REPORT=\"\$REMOTE_HOME/logs/model_probe_\$(date -u +%Y%m%dT%H%M%SZ).json\"; \
if [[ -d /home/pto/venv-cutile ]]; then source /home/pto/venv-cutile/bin/activate; fi; \
export PYTHONPATH=\"\$REMOTE_REPO/src\"; \
python3 \"\$REMOTE_REPO/scripts/collect_remote_probe.py\" --output \"\$REPORT\"; \
printf \"Model probe report: %s\n\" \"\$REPORT\""

run_remote_script "$COMMAND"
