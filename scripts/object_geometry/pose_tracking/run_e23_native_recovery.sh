#!/usr/bin/env bash
# Run original RGBTrack recovery mechanism (with minimal adapter bug fixes)
# and the already-verified interactive HTML player. No temporal constraints.
set -euo pipefail
REPO="${REPO:-/workspace/mnt/sqzhang26/openpi-3d-tactile}"
HOI="${HOI:-/workspace/mnt/sqzhang26/hoi}"
cd "$REPO"
SCRIPT_DIR="$REPO/scripts/object_geometry/pose_tracking"
for p in "$SCRIPT_DIR/rgbtrack_native_recovery.py" "$SCRIPT_DIR/audit_rgbtrack_mask_recovery.py" "$SCRIPT_DIR/rgbtrack_upright_full.py" "$SCRIPT_DIR/visualize_rgbtrack_full_episode.py" "$HOI/envs/sam3d/bin/python" "$REPO/.venv/bin/python" "$REPO/data/pose_eval/hand_scene_metric_mesh_scales.json"; do
  if [[ ! -e "$p" ]]; then echo "[MISSING] $p" >&2; exit 2; fi
done
export PYTHONPATH="$REPO/src:$REPO/data/pose_eval/rgbtrack_pydeps:$REPO/data/pose_eval/rgbtrack_sources/bop_toolkit-master${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$REPO/data/pose_eval/rgbtrack_bin:$PATH"
export TORCH_EXTENSIONS_DIR="$REPO/data/pose_eval/torch_extensions_rgbtrack_native"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=4
TAG="$(date +%Y%m%d_%H%M%S)"
MASK_AUDIT="$REPO/output/hand_scene_qa/e23_mask_native_${TAG}"
echo "[0/2] SAM2 mask validity/occlusion audit: $MASK_AUDIT"
"$REPO/.venv/bin/python" -u "$SCRIPT_DIR/audit_rgbtrack_mask_recovery.py" \
 --dataset "$REPO/data/1018_peg_all" --episode 23 \
 --mask-root "$HOI/outputs/level1_sequence_e23/masks" --output "$MASK_AUDIT"

POSE_DIR="$REPO/output/pose_eval/e23_upright_native_${TAG}"
HTML_DIR="$REPO/output/hand_scene_qa/e23_upright_native_${TAG}"
echo "[1/2] Official RGBTrack recovery: $POSE_DIR"
"$HOI/envs/sam3d/bin/python" -u "$SCRIPT_DIR/rgbtrack_native_recovery.py" \
 --repo-root "$REPO" --dataset "$REPO/data/1018_peg_all" --episode 23 \
 --mask-root "$HOI/outputs/level1_sequence_e23/masks" \
 --funnel-glb "$REPO/assets/object_geometry/synthware_19_22/funnel_F181975.glb" \
 --flask-glb "$REPO/assets/object_geometry/synthware_19_22/flask_F301950.glb" \
 --scale-manifest "$REPO/data/pose_eval/hand_scene_metric_mesh_scales.json" \
 --front-calibration-profile "$REPO/configs/ur7e_xhand/front_calibration_candidate.json" \
 --output "$POSE_DIR" --objects funnel target_flask --mode depth \
 --viz-frames 0,100,141,148,155,181,203,357,456 \
 --upright-max-tilt-deg 25 --upright-max-rank 150
 echo "[2/2] Interactive HTML: $HTML_DIR"
"$REPO/.venv/bin/python" -u "$SCRIPT_DIR/visualize_rgbtrack_full_episode.py" \
 --repo-root "$REPO" --dataset "$REPO/data/1018_peg_all" --episode 23 \
 --pose-run "$POSE_DIR" --output "$HTML_DIR" \
 --frames-dir "$HOI/data/level1_sequence_e23/frames" --viz-stride 1
printf '\n[MASK AUDIT] %s\n[DONE] %s\n[REPORT] %s\n[HTML] %s\n' "$MASK_AUDIT/funnel_sam2_mask_contact_sheet.png" "$POSE_DIR" \
 "$POSE_DIR/native_recovery_report.json" "$HTML_DIR/hand_scene_dynamic_e23.html"
