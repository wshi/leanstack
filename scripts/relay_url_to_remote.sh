#!/usr/bin/env zsh
set -eu
set -o pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
URL="${1:?usage: relay_url_to_remote.sh <url> <remote-path> [filename]}"
REMOTE_PATH="${2:?usage: relay_url_to_remote.sh <url> <remote-path> [filename]}"
FILENAME="${3:-}"
CACHE_DIR="${CACHE_DIR:-$ROOT/.relay-cache}"

mkdir -p "$CACHE_DIR"

if [[ -z "$FILENAME" ]]; then
  FILENAME="$(python3 -c 'import os, sys, urllib.parse; print(os.path.basename(urllib.parse.urlparse(sys.argv[1]).path) or "download.bin")' "$URL")"
fi

LOCAL_FILE="$CACHE_DIR/$FILENAME"
curl -L "$URL" -o "$LOCAL_FILE"
"$ROOT/scripts/push_local_file_to_remote.sh" "$LOCAL_FILE" "$REMOTE_PATH"
printf 'downloaded and relayed %s -> %s\n' "$URL" "$REMOTE_PATH"

