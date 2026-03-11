#!/usr/bin/env zsh
# Run dual-model speculative decode benchmark:
#   Draft:    Qwen3-0.6B-Base (packed artifact)
#   Verifier: Qwen3-1.7B-Base (packed artifact)
#
# Usage:
#   # Default: k=5, decode_64_256 profile
#   ./scripts/remote_dual_spec_benchmark.sh
#
#   # Custom proposal length
#   PROPOSAL_LEN=8 ./scripts/remote_dual_spec_benchmark.sh
#
#   # Custom prompt
#   PROMPT_OVERRIDE="Your custom prompt here" ./scripts/remote_dual_spec_benchmark.sh
set -eu
set -o pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_SCRIPT="${REMOTE_SCRIPT:-$ROOT/../remote.sh}"
REMOTE_HOME="${REMOTE_HOME:-/home/pto/lean}"

# Verifier (main model)
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-1.7B-Base}"
MODEL_KEY="${MODEL_ID//\//__}"
PACK_DIR="${PACK_DIR:-$REMOTE_HOME/packed/$MODEL_KEY}"
MODEL_PATH_FILE="${MODEL_PATH_FILE:-$REMOTE_HOME/models/$MODEL_KEY.path}"
MODEL_PATH="${MODEL_PATH:-}"

# Draft model
DRAFT_MODEL_ID="${DRAFT_MODEL_ID:-Qwen/Qwen3-0.6B-Base}"
DRAFT_MODEL_KEY="${DRAFT_MODEL_ID//\//__}"
DRAFT_PACK_DIR="${DRAFT_PACK_DIR:-$REMOTE_HOME/packed/$DRAFT_MODEL_KEY}"

# Speculative decode parameters
PROPOSAL_LEN="${PROPOSAL_LEN:-5}"

# Benchmark profile
PROFILE="${PROFILE:-decode_64_256}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
PROMPT="${PROMPT:-Explain why fixing the model-chip contract simplifies inference software.}"
PROMPT_FORMAT="${PROMPT_FORMAT:-chat}"
PROMPT_OVERRIDE="${PROMPT_OVERRIDE:-}"
THINKING_MODE="${THINKING_MODE:-disable}"

# Resident/warmup
RESIDENT_REQUESTS="${RESIDENT_REQUESTS:-3}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-1}"

DEVICE="${DEVICE:-cuda:0}"
SEMANTIC_LOGITS_BACKEND="${SEMANTIC_LOGITS_BACKEND:-auto}"
SKIP_REMOTE_SYNC="${SKIP_REMOTE_SYNC:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-$REMOTE_HOME/benchmarks}"

source "$ROOT/scripts/remote_helpers.sh"

if [[ "$SKIP_REMOTE_SYNC" != "1" ]]; then
  "$ROOT/scripts/remote_sync.sh"
fi
load_remote_cmd "$REMOTE_SCRIPT"

TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
OUTPUT_FILE="$OUTPUT_DIR/dual_spec_${PROFILE}_k${PROPOSAL_LEN}_${TIMESTAMP}.json"

if [[ -n "$PROMPT_OVERRIDE" ]]; then
  PROMPT="$PROMPT_OVERRIDE"
fi

COMMAND="set -euo pipefail; \
source /home/pto/venv-cutile/bin/activate; \
export PYTHONPATH=/home/pto/lean/repo/src; \
export LEANSTACK_SEMANTIC_LOGITS_BACKEND=\"$SEMANTIC_LOGITS_BACKEND\"; \
MODEL_REF=\"$MODEL_PATH\"; \
if [[ -z \"\$MODEL_REF\" ]]; then MODEL_REF=\$(<\"$MODEL_PATH_FILE\"); fi; \
EXTRA_ARGS=\"\"; \
if [[ \"$THINKING_MODE\" == \"enable\" ]]; then EXTRA_ARGS=\"--enable-thinking\"; fi; \
if [[ \"$THINKING_MODE\" == \"disable\" ]]; then EXTRA_ARGS=\"--disable-thinking\"; fi; \
mkdir -p \"$OUTPUT_DIR\"; \
python3 /home/pto/lean/repo/experiments/models/qwen_explicit_runtime_loop.py \
  --model-path \"\$MODEL_REF\" \
  --runtime-mode semantic \
  --pack-dir \"$PACK_DIR\" \
  --dual-model-speculative \
  --draft-pack-dir \"$DRAFT_PACK_DIR\" \
  --proposal-len \"$PROPOSAL_LEN\" \
  --device \"$DEVICE\" \
  --prompt \"$PROMPT\" \
  --prompt-format \"$PROMPT_FORMAT\" \
  --max-prefill-tokens \"$MAX_PREFILL_TOKENS\" \
  --max-new-tokens \"$MAX_NEW_TOKENS\" \
  --exact-prefill-bucket \
  --ignore-eos \
  --benchmark-profile \"$PROFILE\" \
  --resident-requests \"$RESIDENT_REQUESTS\" \
  --warmup-requests \"$WARMUP_REQUESTS\" \
  --output \"$OUTPUT_FILE\" \
  \$EXTRA_ARGS"

run_remote_script "$COMMAND"
