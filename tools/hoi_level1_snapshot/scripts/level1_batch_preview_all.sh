#!/usr/bin/env bash
set -euo pipefail

HOI="${HOI:-/workspace/mnt/sqzhang26/hoi}"
REPO="${REPO:-/workspace/mnt/sqzhang26/openpi-3d-tactile}"
PY="${PY:-$REPO/.venv/bin/python}"
OUTROOT="$HOI/outputs"
LOGDIR="$OUTROOT/level1_batch_logs"
mkdir -p "$LOGDIR"

cd "$REPO"

echo "== [1/4] Spread reviewed representative prompts to all episodes =="
PYTHONPATH=src "$PY" \
  "$HOI/scripts/level1_manual_six_positions_v4.py" \
  spread --episodes 1-69 | tee "$LOGDIR/spread.log"

echo
echo "== [2/4] Prepare missing frame directories =="
for ep in $(seq 0 69); do
  frame="$HOI/data/level1_sequence_e${ep}/frames/00000.jpg"
  dir="$HOI/data/level1_sequence_e${ep}"

  if [[ -s "$frame" ]]; then
    echo "[SKIP PREPARE] e$ep"
    continue
  fi

  if [[ -d "$dir" ]]; then
    echo "[ERROR] e$ep directory exists but frame0 is missing: $dir" | tee -a "$LOGDIR/prepare_errors.log"
    exit 1
  fi

  echo "[PREPARE] e$ep"
  bash "$HOI/scripts/level1_ep.sh" "$ep" prepare \
    >"$LOGDIR/e${ep}_prepare.log" 2>&1 || {
      echo "[FAILED PREPARE] e$ep; see $LOGDIR/e${ep}_prepare.log"
      exit 1
    }
done

echo
echo "== [3/4] Generate frame-0 SAM2 previews =="
failed=()
for ep in $(seq 0 69); do
  echo "[PREVIEW] e$ep"
  if ! bash "$HOI/scripts/level1_ep.sh" "$ep" preview \
      >"$LOGDIR/e${ep}_preview.log" 2>&1; then
    echo "[FAILED PREVIEW] e$ep"
    failed+=("$ep")
  fi
done

if (( ${#failed[@]} > 0 )); then
  printf '%s\n' "${failed[@]}" > "$LOGDIR/preview_failed.txt"
  echo "[WARN] preview failures: ${failed[*]}"
else
  rm -f "$LOGDIR/preview_failed.txt"
  echo "[OK] all 70 previews generated"
fi

echo
echo "== [4/4] Build one 70-episode contact sheet =="
"$PY" - <<'PY'
from pathlib import Path
import math

root = Path("/workspace/mnt/sqzhang26/hoi/outputs")
paths = [root / f"level1_sequence_e{ep}" / "mask_preview_frame0.jpg" for ep in range(70)]
existing = [(ep, p) for ep, p in enumerate(paths) if p.exists()]

try:
    import cv2
    import numpy as np

    thumbs = []
    w, h = 320, 240
    for ep, p in existing:
        im = cv2.imread(str(p))
        if im is None:
            continue
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(im, (0, 0), (92, 30), (0, 0, 0), -1)
        cv2.putText(im, f"e{ep}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        thumbs.append((ep, im))

    cols = 5
    rows = math.ceil(len(thumbs) / cols) if thumbs else 1
    canvas = np.full((rows*h, cols*w, 3), 235, dtype=np.uint8)
    for k, (_, im) in enumerate(thumbs):
        r, c = divmod(k, cols)
        canvas[r*h:(r+1)*h, c*w:(c+1)*w] = im

    out = root / "level1_all70_frame0_preview_contact_sheet.jpg"
    cv2.imwrite(str(out), canvas)
    print(f"[CONTACT SHEET] {out}")
    print(f"[FOUND] {len(thumbs)}/70 previews")
except Exception as e:
    print("[ERROR] Could not build contact sheet:", repr(e))
    raise
PY

echo
echo "Done."
echo "Inspect:"
echo "  $OUTROOT/level1_all70_frame0_preview_contact_sheet.jpg"
echo "Logs:"
echo "  $LOGDIR"
