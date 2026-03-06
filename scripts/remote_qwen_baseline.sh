#!/usr/bin/env zsh
set -eu
set -o pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-32B}"
REMOTE_HOME="${REMOTE_HOME:-/home/pto/lean}"
MODEL_SOURCE_FILE="${MODEL_SOURCE_FILE:-$REMOTE_HOME/models/Qwen__Qwen3-32B.path}"
PROMPT="${PROMPT:-Summarize the design goals of a lean, TileIR-first serving stack.}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"

MODEL_ID="$MODEL_ID" MODEL_SOURCE_FILE="$MODEL_SOURCE_FILE" PROMPT="$PROMPT" MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  "$ROOT/scripts/remote_model_baseline.sh"
