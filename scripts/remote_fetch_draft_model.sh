#!/usr/bin/env zsh
# Fetch the draft model (Qwen3-0.6B-Base) via ModelScope to the remote machine.
# This is a prerequisite for running dual-model speculative decode.
#
# Usage:
#   ./scripts/remote_fetch_draft_model.sh
set -eu
set -o pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-0.6B-Base}" \
  exec "$ROOT/scripts/remote_modelscope_fetch.sh" "$@"
