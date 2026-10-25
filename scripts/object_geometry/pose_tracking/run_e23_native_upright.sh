#!/usr/bin/env bash
# Compare RGBTrack native recovery with recovery-time upright hypotheses/gates.
set -euo pipefail
REPO="${REPO:-/workspace/mnt/sqzhang26/openpi-3d-tactile}"
HOI="${HOI:-/workspace/mnt/sqzhang26/hoi}"
cd "$REPO"
D="$REPO/scripts/object_geometry/pose_tracking"
MASK="$HOI/outputs/level1_sequence_e23_maskfix_v1/masks"
for p in "$D/rgbtrack_native_upright.py" "$D/rgbtrack_upright_full.py" "$D/rgbtrack_adapter.py" "$D/visualize_rgbtrack_full_episode.py" \
 "$HOI/envs/sam3d/bin/python" "$REPO/.venv/bin/python" "$MASK/funnel/00000.png" "$MASK/funnel/00467.png" \
 "$REPO/data/pose_eval/hand_scene_metric_mesh_scales.json"; do
  test -e "$p" || { echo "Missing: $p" >&2; exit 2; }
done
export PYTHONPATH="$REPO/src:$REPO/data/pose_eval/rgbtrack_pydeps:$REPO/data/pose_eval/rgbtrack_sources/bop_toolkit-master${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$REPO/data/pose_eval/rgbtrack_bin:$PATH"
export TORCH_EXTENSIONS_DIR="$REPO/data/pose_eval/torch_extensions_rgbtrack_native"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=4
TAG="$(date +%Y%m%d_%H%M%S)"
OUT="$REPO/output/pose_eval/e23_maskfix_native_upright_${TAG}"
VIZ="$REPO/output/hand_scene_qa/e23_maskfix_native_upright_${TAG}"
echo "[RUN] $OUT"
"$HOI/envs/sam3d/bin/python" -u "$D/rgbtrack_native_upright.py" \
 --follow-max-tilt-deg 35 \
 --recovery-center-max-px 90 \
 --recovery-depth-l1-max-m 0.18 \
 --repo-root "$REPO" --dataset "$REPO/data/1018_peg_all" --episode 23 \
 --mask-root "$MASK" \
 --funnel-glb "$REPO/assets/object_geometry/synthware_19_22/funnel_F181975.glb" \
 --flask-glb "$REPO/assets/object_geometry/synthware_19_22/flask_F301950.glb" \
 --scale-manifest "$REPO/data/pose_eval/hand_scene_metric_mesh_scales.json" \
 --front-calibration-profile "$REPO/configs/ur7e_xhand/front_calibration_candidate.json" \
 --output "$OUT" --objects funnel target_flask --mode depth \
 --viz-frames 0,100,127,141,148,181,203,234,357,424,456 \
 --upright-max-tilt-deg 25 --upright-max-rank 150
"$REPO/.venv/bin/python" -u "$D/visualize_rgbtrack_full_episode.py" \
 --repo-root "$REPO" --dataset "$REPO/data/1018_peg_all" --episode 23 \
 --pose-run "$OUT" --output "$VIZ" \
 --frames-dir "$HOI/data/level1_sequence_e23/frames" --viz-stride 1
printf '\n[REPORT] %s\n[HTML] %s\n' "$OUT/native_upright_report.json" "$VIZ/hand_scene_dynamic_e23.html"
