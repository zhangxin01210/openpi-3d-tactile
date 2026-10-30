#!/usr/bin/env bash
# Run full RGBTrack episode 23 using first-frame upright funnel candidate, then
# generate the same interactive 3D HTML as the original full-run QA.
# Place beside rgbtrack_upright_full.py and visualize_rgbtrack_full_episode.py
# in scripts/object_geometry/pose_tracking/ of the official repository.
set -euo pipefail

REPO="${REPO:-/workspace/mnt/sqzhang26/openpi-3d-tactile}"
HOI="${HOI:-/workspace/mnt/sqzhang26/hoi}"
cd "$REPO"
SCRIPT_DIR="$REPO/scripts/object_geometry/pose_tracking"

for path in \
  "$SCRIPT_DIR/rgbtrack_upright_full.py" \
  "$SCRIPT_DIR/visualize_rgbtrack_full_episode.py" \
  "$REPO/data/pose_eval/hand_scene_metric_mesh_scales.json" \
  "$HOI/envs/sam3d/bin/python" \
  "$REPO/.venv/bin/python"; do
    if [[ ! -e "$path" ]]; then echo "[MISSING] $path" >&2; exit 2; fi
done

export PYTHONPATH="$REPO/src:$REPO/data/pose_eval/rgbtrack_pydeps:$REPO/data/pose_eval/rgbtrack_sources/bop_toolkit-master${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$REPO/data/pose_eval/rgbtrack_bin:$PATH"
export TORCH_EXTENSIONS_DIR="$REPO/data/pose_eval/torch_extensions_rgbtrack_native"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=4

RUN_ID="$(date +%Y%m%d_%H%M%S)"
POSE_DIR="$REPO/output/pose_eval/e23_upright_full_${RUN_ID}"
HTML_DIR="$REPO/output/hand_scene_qa/e23_upright_full_${RUN_ID}"

printf '\n[1/2] Full 468-frame 6D tracking: %s\n' "$POSE_DIR"
"$HOI/envs/sam3d/bin/python" -u "$SCRIPT_DIR/rgbtrack_upright_full.py" \
  --repo-root "$REPO" \
  --dataset "$REPO/data/1018_peg_all" \
  --episode 23 \
  --mask-root "$HOI/outputs/level1_sequence_e23/masks" \
  --funnel-glb "$REPO/assets/object_geometry/synthware_19_22/funnel_F181975.glb" \
  --flask-glb "$REPO/assets/object_geometry/synthware_19_22/flask_F301950.glb" \
  --scale-manifest "$REPO/data/pose_eval/hand_scene_metric_mesh_scales.json" \
  --front-calibration-profile "$REPO/configs/ur7e_xhand/front_calibration_candidate.json" \
  --output "$POSE_DIR" \
  --objects funnel target_flask \
  --mode depth \
  --viz-frames 0,100,141,144,151,155,203,358,467 \
  --upright-max-tilt-deg 25 \
  --upright-max-rank 150

printf '\n[2/2] Interactive 3D RGB/mesh HTML: %s\n' "$HTML_DIR"
"$REPO/.venv/bin/python" -u "$SCRIPT_DIR/visualize_rgbtrack_full_episode.py" \
  --repo-root "$REPO" \
  --dataset "$REPO/data/1018_peg_all" \
  --episode 23 \
  --pose-run "$POSE_DIR" \
  --output "$HTML_DIR" \
  --frames-dir "$HOI/data/level1_sequence_e23/frames" \
  --viz-stride 1

printf '\n[DONE] Pose JSONL: %s\n' "$POSE_DIR"
printf '[DONE] 3D HTML: %s/hand_scene_dynamic_e23.html\n' "$HTML_DIR"
printf '[DONE] Quality report: %s/tracking_quality_report.json\n' "$HTML_DIR"
