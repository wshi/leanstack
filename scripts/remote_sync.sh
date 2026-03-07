#!/usr/bin/env zsh
set -eu
set -o pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_SCRIPT="${REMOTE_SCRIPT:-$ROOT/../remote.sh}"
REMOTE_HOME="${REMOTE_HOME:-/home/pto/lean}"
REMOTE_REPO="$REMOTE_HOME/repo"
source "$ROOT/scripts/remote_helpers.sh"

"$ROOT/scripts/remote_bootstrap.sh"

load_remote_cmd "$REMOTE_SCRIPT"

COPYFILE_DISABLE=1 tar \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='.pycache' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='*.pyc' \
  --exclude='artifacts' \
  --exclude='.DS_Store' \
  --exclude='._*' \
  -C "$ROOT" \
  -cf - \
  . | "${REMOTE_CMD[@]}" "mkdir -p '$REMOTE_REPO' && tar -xf - -C '$REMOTE_REPO'"

COMMAND="set -euo pipefail; \
REMOTE_REPO=\"$REMOTE_REPO\"; \
printf \"Remote repo synced to: %s\n\" \"\$REMOTE_REPO\"; \
find \"\$REMOTE_REPO\" -maxdepth 2 -type f | sort | sed -n 1,120p"

run_remote_script "$COMMAND"
