#!/usr/bin/env bash
# Compute normalization statistics once per reviewed task.  Run after the
# single-episode review has been approved and after full conversion finishes.
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${OPENPI_PYTHON:-$PWD/.venv/bin/python}"
[[ -x "$PYTHON" ]] || { echo "Missing Python interpreter: $PYTHON" >&2; exit 2; }
TASK="${1:-}"
if [[ -z "$TASK" ]]; then
  echo "usage: $0 TASK; TASK is peg, barbed_flat, or valve" >&2
  exit 2
fi
case "$TASK" in
  peg|barbed_flat|valve) ;;
  *) echo "unknown task: $TASK" >&2; exit 2 ;;
esac
"$PYTHON" scripts/compute_norm_stats.py --config-name="pi0_cw3_${TASK}_01_rgb"
"$PYTHON" sim/scripts/verify_contactworld_cross_task_lerobot.py --task "$TASK" "data/contactworld_${TASK}_v3_bilateral"
JAX_PLATFORM_NAME=cpu "$PYTHON" sim/scripts/audit_contactworld_cross_task_inputs.py --task "$TASK" "data/contactworld_${TASK}_v3_bilateral"
