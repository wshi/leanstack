#!/usr/bin/env zsh
set -eu
set -o pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_SCRIPT="${REMOTE_SCRIPT:-$ROOT/../remote.sh}"
REMOTE_HOME="${REMOTE_HOME:-/home/pto/lean}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-1.7B-Base}"
MODEL_KEY="${MODEL_ID//\//__}"
MODEL_PATH_FILE="${MODEL_PATH_FILE:-$REMOTE_HOME/models/$MODEL_KEY.path}"
MODEL_PATH="${MODEL_PATH:-}"
PACK_DIR="${PACK_DIR:-$REMOTE_HOME/packed/$MODEL_KEY}"
STRICT_PACKED="${STRICT_PACKED:-1}"
STRICT_CONTRACT="${STRICT_CONTRACT:-0}"
PROFILE="${PROFILE:-decode_64_256}"
RUNTIME_MODE="${RUNTIME_MODE:-semantic}"
NUM_LAYERS="${NUM_LAYERS:-0}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"
RESULT_DIR="${RESULT_DIR:-$REMOTE_HOME/benchmarks}"
BENCHMARK_TAG="${BENCHMARK_TAG:-}"
PROMPT_OVERRIDE="${PROMPT_OVERRIDE:-}"
PROMPT_FORMAT_OVERRIDE="${PROMPT_FORMAT_OVERRIDE:-}"
MAX_PREFILL_TOKENS_OVERRIDE="${MAX_PREFILL_TOKENS_OVERRIDE:-}"
MAX_NEW_TOKENS_OVERRIDE="${MAX_NEW_TOKENS_OVERRIDE:-}"
RESIDENT_REQUESTS="${RESIDENT_REQUESTS:-3}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-1}"
IGNORE_EOS="${IGNORE_EOS:-1}"
SPECULATIVE="${SPECULATIVE:-0}"
DRAFT_LAYER_COUNT="${DRAFT_LAYER_COUNT:-12}"
PROPOSAL_LEN="${PROPOSAL_LEN:-4}"
DRAFT_HEAD_KEY="${DRAFT_HEAD_KEY:-}"
COMPILE="${COMPILE:-0}"
COMPILE_MODE="${COMPILE_MODE:-default}"
SEMANTIC_LOGITS_BACKEND="${SEMANTIC_LOGITS_BACKEND:-auto}"
SKIP_REMOTE_SYNC="${SKIP_REMOTE_SYNC:-0}"
OFFICIAL_MODEL_ID="Qwen/Qwen3-1.7B-Base"
OFFICIAL_PROFILE="decode_64_256"
OFFICIAL_RUNTIME_MODE="semantic"
OFFICIAL_DTYPE="bfloat16"
source "$ROOT/scripts/remote_helpers.sh"

if [[ "$STRICT_CONTRACT" == "1" ]]; then
  if [[ "$MODEL_ID" != "$OFFICIAL_MODEL_ID" ]]; then
    echo "strict contract requires MODEL_ID=$OFFICIAL_MODEL_ID (got $MODEL_ID)" >&2
    exit 1
  fi
  if [[ "$PROFILE" != "$OFFICIAL_PROFILE" ]]; then
    echo "strict contract requires PROFILE=$OFFICIAL_PROFILE (got $PROFILE)" >&2
    exit 1
  fi
  if [[ "$RUNTIME_MODE" != "$OFFICIAL_RUNTIME_MODE" ]]; then
    echo "strict contract requires RUNTIME_MODE=$OFFICIAL_RUNTIME_MODE (got $RUNTIME_MODE)" >&2
    exit 1
  fi
  if [[ "$DTYPE" != "$OFFICIAL_DTYPE" ]]; then
    echo "strict contract requires DTYPE=$OFFICIAL_DTYPE (got $DTYPE)" >&2
    exit 1
  fi
  if [[ "$NUM_LAYERS" != "0" ]]; then
    echo "strict contract requires NUM_LAYERS=0 (got $NUM_LAYERS)" >&2
    exit 1
  fi
  if [[ "$SPECULATIVE" != "0" ]]; then
    echo "strict contract requires SPECULATIVE=0 (got $SPECULATIVE)" >&2
    exit 1
  fi
  if [[ "$STRICT_PACKED" != "1" ]]; then
    echo "strict contract requires STRICT_PACKED=1 (got $STRICT_PACKED)" >&2
    exit 1
  fi
fi

if [[ "$SKIP_REMOTE_SYNC" != "1" ]]; then
  "$ROOT/scripts/remote_sync.sh"
fi
load_remote_cmd "$REMOTE_SCRIPT"

COMMAND="set -euo pipefail; \
source /home/pto/venv-cutile/bin/activate; \
export PYTHONPATH=/home/pto/lean/repo/src; \
export LEANSTACK_SEMANTIC_LOGITS_BACKEND=\"$SEMANTIC_LOGITS_BACKEND\"; \
MODEL_REF=\"$MODEL_PATH\"; \
if [[ -z \"\$MODEL_REF\" ]]; then MODEL_REF=\$(<\"$MODEL_PATH_FILE\"); fi; \
PACK_REF=\"$PACK_DIR\"; \
if [[ \"$STRICT_PACKED\" == \"1\" ]]; then \
  if [[ -z \"\$PACK_REF\" ]] || [[ ! -f \"\$PACK_REF/manifest.json\" ]]; then \
    echo \"packed appliance required: missing manifest at \$PACK_REF/manifest.json\" >&2; \
    exit 1; \
  fi; \
fi; \
eval \"\$(python3 -m leanstack.cli show-benchmark-profile --profile \"$PROFILE\" --format shell)\"; \
if [[ -n \"$PROMPT_OVERRIDE\" ]]; then PROMPT=\"$PROMPT_OVERRIDE\"; fi; \
if [[ -n \"$PROMPT_FORMAT_OVERRIDE\" ]]; then PROMPT_FORMAT=\"$PROMPT_FORMAT_OVERRIDE\"; fi; \
if [[ -n \"$MAX_PREFILL_TOKENS_OVERRIDE\" ]]; then MAX_PREFILL_TOKENS=\"$MAX_PREFILL_TOKENS_OVERRIDE\"; fi; \
if [[ -n \"$MAX_NEW_TOKENS_OVERRIDE\" ]]; then MAX_NEW_TOKENS=\"$MAX_NEW_TOKENS_OVERRIDE\"; fi; \
mkdir -p \"$RESULT_DIR\"; \
if [[ -z \"$BENCHMARK_TAG\" ]]; then BENCHMARK_TAG=\$(date -u +%Y%m%dT%H%M%SZ); fi; \
OUTPUT_PATH=\"$RESULT_DIR/leanstack_${RUNTIME_MODE}_${PROFILE}_\${BENCHMARK_TAG}.json\"; \
EXTRA_ARGS=\"\"; \
if [[ \"$IGNORE_EOS\" == \"1\" ]]; then EXTRA_ARGS=\"\$EXTRA_ARGS --ignore-eos\"; fi; \
EXTRA_ARGS=\"\$EXTRA_ARGS --exact-prefill-bucket\"; \
if [[ -n \"\$PACK_REF\" ]]; then EXTRA_ARGS=\"\$EXTRA_ARGS --pack-dir \$PACK_REF\"; fi; \
if [[ \"$SPECULATIVE\" == \"1\" ]]; then EXTRA_ARGS=\"\$EXTRA_ARGS --speculative --draft-layer-count $DRAFT_LAYER_COUNT --proposal-len $PROPOSAL_LEN\"; fi; \
if [[ -n \"$DRAFT_HEAD_KEY\" ]]; then EXTRA_ARGS=\"\$EXTRA_ARGS --draft-head-key $DRAFT_HEAD_KEY\"; fi; \
if [[ \"$COMPILE\" == \"1\" ]]; then EXTRA_ARGS=\"\$EXTRA_ARGS --compile --compile-mode $COMPILE_MODE\"; fi; \
python3 /home/pto/lean/repo/experiments/models/qwen_explicit_runtime_loop.py \
  --model-path \"\$MODEL_REF\" \
  --runtime-mode \"$RUNTIME_MODE\" \
  --benchmark-profile \"$PROFILE\" \
  --num-layers \"$NUM_LAYERS\" \
  --device \"$DEVICE\" \
  --dtype \"$DTYPE\" \
  --prompt \"\$PROMPT\" \
  --prompt-format \"\$PROMPT_FORMAT\" \
  --max-prefill-tokens \"\$MAX_PREFILL_TOKENS\" \
  --max-new-tokens \"\$MAX_NEW_TOKENS\" \
  --resident-requests \"$RESIDENT_REQUESTS\" \
  --warmup-requests \"$WARMUP_REQUESTS\" \
  \$EXTRA_ARGS \
  --output \"\$OUTPUT_PATH\" > /dev/null; \
cat \"\$OUTPUT_PATH\""

run_remote_script "$COMMAND"
