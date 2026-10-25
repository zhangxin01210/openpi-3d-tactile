#!/usr/bin/env bash
# Keep the SUCCESSFUL funnel 468-frame pose unchanged; fix only target flask.
set -euo pipefail
REPO="${REPO:-/workspace/mnt/sqzhang26/openpi-3d-tactile}"
HOI="${HOI:-/workspace/mnt/sqzhang26/hoi}"
D="$REPO/scripts/object_geometry/pose_tracking"
cd "$REPO"
SOURCE="${SOURCE_POSE_RUN:-$(find "$REPO/output/pose_eval" -maxdepth 1 -type d -name 'e23_maskfix_native_upright_*' | sort | tail -1)}"
[[ -n "$SOURCE" && -s "$SOURCE/funnel_poses.jsonl" && -s "$SOURCE/target_flask_poses.jsonl" ]] || { echo "Missing successful source run: $SOURCE" >&2;exit 2; }
for f in "$D/rgbtrack_flask_pivot.py" "$D/visualize_rgbtrack_full_episode.py" "$HOI/envs/sam3d/bin/python" "$REPO/.venv/bin/python"; do
  [[ -e "$f" ]] || { echo "Missing: $f" >&2;exit 2; }
done
export PYTHONPATH="$REPO/src:$REPO/data/pose_eval/rgbtrack_pydeps:$REPO/data/pose_eval/rgbtrack_sources/bop_toolkit-master${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$REPO/data/pose_eval/rgbtrack_bin:$PATH"
export TORCH_EXTENSIONS_DIR="$REPO/data/pose_eval/torch_extensions_rgbtrack_native"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="8.0" MAX_JOBS=4
TAG="$(date +%Y%m%d_%H%M%S)"
OUT="$REPO/output/pose_eval/e23_funnel_kept_flask_pivot_${TAG}"
VIZ="$REPO/output/hand_scene_qa/e23_funnel_kept_flask_pivot_${TAG}"
echo "[KEEP FUNNEL] $SOURCE"
echo "[FLASK ONLY] Upright rank selection frame 0; deterministic pivot filter 468 frames"
"$HOI/envs/sam3d/bin/python" -u "$D/rgbtrack_flask_pivot.py" \
  --repo-root "$REPO" --dataset "$REPO/data/1018_peg_all" --episode 23 \
  --mask-root "$HOI/outputs/level1_sequence_e23_maskfix_v1/masks" \
  --source-pose-run "$SOURCE" --output "$OUT" \
  --funnel-glb "$REPO/assets/object_geometry/synthware_19_22/funnel_F181975.glb" \
  --flask-glb "$REPO/assets/object_geometry/synthware_19_22/flask_F301950.glb" \
  --scale-manifest "$REPO/data/pose_eval/hand_scene_metric_mesh_scales.json" \
  --front-calibration-profile "$REPO/configs/ur7e_xhand/front_calibration_candidate.json" \
  --flask-mouth-axis auto --flask-init-tilt-deg 20 \
  --flask-max-tilt-deg 20 --flask-max-axis-jump-deg 12 \
  --flask-max-pivot-shift-m 0.025 --flask-axis-alpha 0.25
"$REPO/.venv/bin/python" -u "$D/visualize_rgbtrack_full_episode.py" \
  --repo-root "$REPO" --dataset "$REPO/data/1018_peg_all" --episode 23 \
  --pose-run "$OUT" --output "$VIZ" \
  --frames-dir "$HOI/data/level1_sequence_e23/frames" --viz-stride 1
printf '\n[OPEN HTML] %s\n[QA REPORT] %s\n' "$VIZ/hand_scene_dynamic_e23.html" "$OUT/flask_pivot_report.json"
