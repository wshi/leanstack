#!/usr/bin/env zsh
set -eu
set -o pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
REMOTE_HOME="${REMOTE_HOME:-/home/pto/lean}"
MODEL_KEY="${MODEL_ID//\//__}"
MODEL_SOURCE_FILE="${MODEL_SOURCE_FILE:-$REMOTE_HOME/models/$MODEL_KEY.path}"
MODEL_PATH="${MODEL_PATH:-}"
PROMPT="${PROMPT:-Summarize why an agent-built, cuTile-native Qwen3-8B stack may outperform a compatibility-heavy framework stack on Blackwell.}"
PROMPT_FORMAT="${PROMPT_FORMAT:-chat}"
THINKING_MODE="${THINKING_MODE:-disable}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"

MODEL_ID="$MODEL_ID" MODEL_SOURCE_FILE="$MODEL_SOURCE_FILE" MODEL_PATH="$MODEL_PATH" PROMPT="$PROMPT" PROMPT_FORMAT="$PROMPT_FORMAT" THINKING_MODE="$THINKING_MODE" MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  "$ROOT/scripts/remote_model_baseline.sh"
