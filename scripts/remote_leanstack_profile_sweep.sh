#!/usr/bin/env zsh
set -eu
set -o pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROFILES=(${=PROFILES:-"decode_64_256 decode_64_512 prefill_1024_64"})
RUNS="${RUNS:-1}"

for profile in "${PROFILES[@]}"; do
  for run_idx in $(seq 1 "$RUNS"); do
    PROFILE="$profile" \
    BENCHMARK_TAG="${profile}_run${run_idx}" \
    "$ROOT/scripts/remote_leanstack_benchmark.sh"
  done
done
