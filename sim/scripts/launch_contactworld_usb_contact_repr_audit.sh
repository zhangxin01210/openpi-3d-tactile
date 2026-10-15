#!/usr/bin/env bash
set -euo pipefail

# Builds the derived contact sidecar once, then renders a review page.  The
# page is the required gate before training configurations 16--19.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
DATA="${OPENPI_CONTACTWORLD_USB_V3_DATA:-$ROOT/data/contactworld_usb_v3_bilateral}"
RUN="${OPENPI_CONTACTWORLD_CONTACT_REPR_RUN:-$ROOT/runs/contactworld_usb_contact_repr_audit_$(date +%Y%m%d)}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$RUN"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/openpi-matplotlib}"
"$PYTHON" sim/scripts/prepare_contactworld_usb_contact_repr.py \
  --dataset "$DATA" \
  --output "$DATA/spatial/contact_repr_v1" \
  | tee "$RUN/prepare.log"
"$PYTHON" sim/scripts/verify_contactworld_usb_contact_repr.py \
  --dataset "$DATA" \
  --repr-root "$DATA/spatial/contact_repr_v1" \
  --output "$RUN/contact_repr_verify.json" \
  | tee "$RUN/verify.log"
"$PYTHON" sim/scripts/audit_contactworld_usb_contact_repr.py \
  --dataset "$DATA" \
  --repr-root "$DATA/spatial/contact_repr_v1" \
  --output "$RUN" \
  | tee "$RUN/audit.log"
echo "Review: $RUN/index.html"
