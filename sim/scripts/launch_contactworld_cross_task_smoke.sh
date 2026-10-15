#!/usr/bin/env bash
# One official demonstration -> isolated bilateral replay -> LeRobot -> review.
# This stops after exactly one source episode.  Do not run the full script until
# the generated review page has been accepted.
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${OPENPI_PYTHON:-$PWD/.venv/bin/python}"
[[ -x "$PYTHON" ]] || { echo "Missing Python interpreter: $PYTHON" >&2; exit 2; }
TASK="${1:-}"
if [[ -z "$TASK" ]]; then
  echo "usage: $0 TASK [archive.tar.gz]; TASK is peg, barbed_flat, or valve" >&2
  exit 2
fi
ARCHIVE="${2:-}"
RUNTIME="/home/sai/zx/openpi-sim-runtime"
RUN="$RUNTIME/runs/contactworld_cross_${TASK}_smoke"
DATA="data/contactworld_${TASK}_v3_bilateral_smoke"
if [[ -e "$RUN" || -e "$DATA" ]]; then
  echo "Refusing to overwrite existing $RUN or $DATA; inspect it or choose a clean workspace." >&2
  exit 2
fi
PREP=("$PYTHON" sim/scripts/prepare_contactworld_cross_task_inputs.py --task "$TASK" --source "$RUNTIME/data/contactworld" --output "$RUN/input" --limit 1)
if [[ -n "$ARCHIVE" ]]; then PREP+=(--archive "$ARCHIVE"); fi
"${PREP[@]}"
"$PYTHON" sim/scripts/run_contactworld_cross_task_replay.py --task "$TASK" --input "$RUN/input" --output "$RUN/replay"
"$PYTHON" sim/scripts/export_contactworld_cross_task_lerobot.py --task "$TASK" --input "$RUN/input" --replay "$RUN/replay" --output "$DATA" --max-episodes 1
"$PYTHON" sim/scripts/verify_contactworld_cross_task_lerobot.py --task "$TASK" "$DATA"
"$PYTHON" sim/scripts/render_contactworld_cross_task_review.py --task "$TASK" "$DATA" "$RUN/review"
echo "Review: $RUN/review/index.html"
