#!/usr/bin/env bash
# Full conversion only after the corresponding smoke review has been approved.
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${OPENPI_PYTHON:-$PWD/.venv/bin/python}"
[[ -x "$PYTHON" ]] || { echo "Missing Python interpreter: $PYTHON" >&2; exit 2; }
TASK="${1:-}"
if [[ -z "$TASK" ]]; then
  echo "usage: $0 TASK [archive.tar.gz] --approved; TASK is peg, barbed_flat, or valve" >&2
  exit 2
fi
shift
ARCHIVE=""
if [[ "${1:-}" != "--approved" ]]; then
  ARCHIVE="${1:-}"
  shift || true
fi
if [[ "${1:-}" != "--approved" ]]; then
  echo "Refusing bulk replay: inspect the smoke review first, then pass --approved as argument 3." >&2
  exit 2
fi
RUNTIME="/home/sai/zx/openpi-sim-runtime"
RUN="$RUNTIME/runs/contactworld_cross_${TASK}_full"
DATA="data/contactworld_${TASK}_v3_bilateral"
if [[ -e "$RUN" || -e "$DATA" ]]; then
  echo "Refusing to overwrite existing $RUN or $DATA" >&2
  exit 2
fi
PREP=("$PYTHON" sim/scripts/prepare_contactworld_cross_task_inputs.py --task "$TASK" --source "$RUNTIME/data/contactworld" --output "$RUN/input" --limit 100000)
if [[ -n "$ARCHIVE" ]]; then PREP+=(--archive "$ARCHIVE"); fi
"${PREP[@]}"
"$PYTHON" sim/scripts/run_contactworld_cross_task_replay.py --task "$TASK" --input "$RUN/input" --output "$RUN/replay" --review-approved
"$PYTHON" sim/scripts/export_contactworld_cross_task_lerobot.py --task "$TASK" --input "$RUN/input" --replay "$RUN/replay" --output "$DATA"
bash sim/scripts/prepare_contactworld_cross_task_training.sh "$TASK"
echo "Complete dataset: $DATA"
echo "Then train all approved task matrices with:"
echo "$PYTHON sim/scripts/launch_contactworld_cross_task_matrix.py --tasks peg barbed_flat valve --label cw_cross_first --gpus 0,1,2,3,4,5,6,7"
