#!/usr/bin/env bash
set -euo pipefail

# One-command cross-task evaluation of the 12 checkpoints in 1014_ckpt.
# Override CROSS_TASK_EVAL_CHECKPOINTS, CROSS_TASK_EVAL_SOURCE,
# CROSS_TASK_EVAL_OUTPUT and CROSS_TASK_EVAL_GPU when running elsewhere.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
CHECKPOINTS="${CROSS_TASK_EVAL_CHECKPOINTS:-/home/sai/zsq/FactileLDM/1014_ckpt}"
SOURCE="${CROSS_TASK_EVAL_SOURCE:-/home/sai/zx/openpi-sim-runtime/third_party/ContactWorld-bilateral-probe}"
OUTPUT="${CROSS_TASK_EVAL_OUTPUT:-$ROOT/runs/contactworld_cross_task_eval_$(date +%Y%m%d)}"
GPU="${CROSS_TASK_EVAL_GPU:-0}"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" sim/scripts/run_contactworld_cross_task_eval.py \
  --checkpoints "$CHECKPOINTS" \
  --source "$SOURCE" \
  --output "$OUTPUT" \
  --gpu "$GPU" \
  "$@"

