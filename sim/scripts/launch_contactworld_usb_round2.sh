#!/usr/bin/env bash
# Launch the six USB V3 representation ablations after the contact-input audit.
#
# Example:
#   OPENPI_CONTACTWORLD_BASE_WEIGHTS=/path/to/pi0_base/params \
#   OPENPI_CONTACTWORLD_GPUS=0,1,2,3,4,5 \
#   bash sim/scripts/launch_contactworld_usb_round2.sh cw3_usb_round2 --check-only
#
# Remove --check-only only after reviewing the generated contact-representation
# audit page.  The Python launcher performs an additional full sidecar verify
# and a real data-to-model input audit before it starts any worker.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 LABEL [--check-only] [--wandb]" >&2
  exit 2
fi

LABEL="$1"
shift
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
GPUS="${OPENPI_CONTACTWORLD_GPUS:-0,1,2,3,4,5}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON" sim/scripts/launch_contactworld_v2_matrix.py \
  --version v3 \
  --label "$LABEL" \
  --gpus "$GPUS" \
  --configs \
    pi0_cw3_usb_14_base_pooled_no_pc \
    pi0_cw3_usb_15_base_map_matched_no_pc \
    pi0_cw3_usb_16_contact_centered_point \
    pi0_cw3_usb_17_contact_centered_map \
    pi0_cw3_usb_18_base_pooled_relation \
    pi0_cw3_usb_19_base_map_matched_relation \
  "$@"
