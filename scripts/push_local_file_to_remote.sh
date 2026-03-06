#!/usr/bin/env zsh
set -eu
set -o pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_SCRIPT="${REMOTE_SCRIPT:-$ROOT/../remote.sh}"
REMOTE_HOME="${REMOTE_HOME:-/home/pto/lean}"
LOCAL_PATH="${1:?usage: push_local_file_to_remote.sh <local-path> <remote-path>}"
REMOTE_PATH_INPUT="${2:?usage: push_local_file_to_remote.sh <local-path> <remote-path>}"
source "$ROOT/scripts/remote_helpers.sh"

load_remote_cmd "$REMOTE_SCRIPT"

if [[ ! -e "$LOCAL_PATH" ]]; then
  printf 'local path not found: %s\n' "$LOCAL_PATH" >&2
  exit 1
fi

if [[ "$REMOTE_PATH_INPUT" == /* ]]; then
  REMOTE_PATH="$REMOTE_PATH_INPUT"
else
  REMOTE_PATH="$REMOTE_HOME/$REMOTE_PATH_INPUT"
fi

if [[ -d "$LOCAL_PATH" ]]; then
  tar -C "$LOCAL_PATH" -cf - . | "${REMOTE_CMD[@]}" "mkdir -p '$REMOTE_PATH' && tar -xf - -C '$REMOTE_PATH'"
  printf 'pushed directory %s -> %s\n' "$LOCAL_PATH" "$REMOTE_PATH"
else
  REMOTE_DIR="$(dirname "$REMOTE_PATH")"
  cat "$LOCAL_PATH" | "${REMOTE_CMD[@]}" "mkdir -p '$REMOTE_DIR' && cat > '$REMOTE_PATH'"
  printf 'pushed file %s -> %s\n' "$LOCAL_PATH" "$REMOTE_PATH"
fi

