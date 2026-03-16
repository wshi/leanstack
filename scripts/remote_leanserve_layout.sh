#!/bin/zsh
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)

if [[ "${SKIP_REMOTE_SYNC:-0}" != "1" ]]; then
  COPYFILE_DISABLE=1 tar \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='.pycache' \
    --exclude='__pycache__' \
    --exclude='.pytest_cache' \
    --exclude='*.pyc' \
    --exclude='artifacts' \
    --exclude='.relay-cache' \
    -C "$ROOT_DIR" -cf - . | ssh pto@kb119856792y.vicp.fun -p 33402 'mkdir -p /home/pto/lean/repo && tar -xf - -C /home/pto/lean/repo'
fi

PACK_DIR=${PACK_DIR:-/home/pto/lean/packed/Qwen__Qwen3-4B-Base}
PAGE_SIZE=${PAGE_SIZE:-16}
BATCH_SIZE=${BATCH_SIZE:-1}
DEVICE=${DEVICE:-cuda:0}
MODEL_ID=${MODEL_ID:-qwen}

ssh pto@kb119856792y.vicp.fun -p 33402 "bash -lc '
set -euo pipefail
cd /home/pto/lean/repo
if [ -d /home/pto/venv-cutile ]; then
  . /home/pto/venv-cutile/bin/activate
fi
export PYTHONPATH=/home/pto/lean/repo/src
python3 -m leanstack.cli show-leanserve-layout \
  --model \"$MODEL_ID\" \
  --pack-dir \"$PACK_DIR\" \
  --device \"$DEVICE\" \
  --page-size \"$PAGE_SIZE\" \
  --batch-size \"$BATCH_SIZE\"
'"
