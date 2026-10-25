#!/usr/bin/env bash
# Reuse successful funnel, change only target flask to the reviewed f0 geometric anchor.
set -euo pipefail
REPO="${REPO:-/workspace/mnt/sqzhang26/openpi-3d-tactile}"
HOI="${HOI:-/workspace/mnt/sqzhang26/hoi}"
D="$REPO/scripts/object_geometry/pose_tracking"
cd "$REPO"
SOURCE="${SOURCE_POSE_RUN:-$(find "$REPO/output/pose_eval" -maxdepth 1 -type d -name 'e23_funnel_kept_flask_pivot_*' | sort | tail -1)}"
[[ -n "$SOURCE" && -s "$SOURCE/funnel_poses.jsonl" && -s "$SOURCE/target_flask_poses.jsonl" ]] || { echo "Missing source: $SOURCE" >&2;exit 2; }
TAG="$(date +%Y%m%d_%H%M%S)"
OUT="$REPO/output/pose_eval/e23_funnel_kept_flask_static_${TAG}"
VIZ="$REPO/output/hand_scene_qa/e23_funnel_kept_flask_static_${TAG}"
PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$REPO/.venv/bin/python" -u "$D/freeze_target_flask_anchor.py" \
  --source-pose-run "$SOURCE" --output "$OUT" --episode 23
PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$REPO/.venv/bin/python" -u "$D/visualize_rgbtrack_full_episode.py" \
  --repo-root "$REPO" --dataset "$REPO/data/1018_peg_all" --episode 23 \
  --pose-run "$OUT" --output "$VIZ" \
  --frames-dir "$HOI/data/level1_sequence_e23/frames" --viz-stride 1
printf '\n[HTML] %s\n[REPORT] %s\n' "$VIZ/hand_scene_dynamic_e23.html" "$OUT/flask_static_prior_report.json"
