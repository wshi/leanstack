#!/usr/bin/env zsh
set -eu
set -o pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROFILES=(${=PROFILES:-"decode_64_256 decode_64_512 prefill_1024_64"})
SYSTEM_LABEL="${SYSTEM_LABEL:-framework}"
VARIANT_LABEL="${VARIANT_LABEL:-openai}"
RUNS="${RUNS:-1}"

if [[ -z "${MODEL_NAME:-}" ]]; then
  echo "set MODEL_NAME to the served model name" >&2
  exit 1
fi

for profile in "${PROFILES[@]}"; do
  for run_idx in $(seq 1 "$RUNS"); do
    PROFILE="$profile" \
    SYSTEM_LABEL="$SYSTEM_LABEL" \
    VARIANT_LABEL="$VARIANT_LABEL" \
    BENCHMARK_TAG="${profile}_run${run_idx}" \
    "$ROOT/scripts/remote_openai_backend_benchmark.sh"
  done
done
