#!/usr/bin/env zsh
set -eu
set -o pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_SCRIPT="${REMOTE_SCRIPT:-$ROOT/../remote.sh}"
REMOTE_HOME="${REMOTE_HOME:-/home/pto/lean}"
source "$ROOT/scripts/remote_helpers.sh"

load_remote_cmd "$REMOTE_SCRIPT"

COMMAND="set -euo pipefail; \
REMOTE_HOME=\"$REMOTE_HOME\"; \
mkdir -p \"\$REMOTE_HOME\"/{repo,artifacts,logs,models,tmp}; \
printf \"Remote workspace ready: %s\n\" \"\$REMOTE_HOME\"; \
find \"\$REMOTE_HOME\" -maxdepth 1 -mindepth 1 -type d | sort"

run_remote_script "$COMMAND"
