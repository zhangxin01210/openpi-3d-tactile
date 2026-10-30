#!/usr/bin/env bash
set -euo pipefail
REPO="${REPO:-/workspace/mnt/sqzhang26/openpi-3d-tactile}"
HOI="${HOI:-/workspace/mnt/sqzhang26/hoi}"
cd "$REPO"
DIR="$REPO/scripts/object_geometry/pose_tracking"
for path in "$DIR/rgbtrack_upright_full.py" "$DIR/rgbtrack_upright_temporal.py" "$DIR/visualize_rgbtrack_full_episode.py" "$REPO/data/pose_eval/hand_scene_metric_mesh_scales.json" "$HOI/envs/sam3d/bin/python" "$REPO/.venv/bin/python"; do
 test -e "$path" || { echo "Missing: $path" >&2; exit 1; }
done
export PYTHONPATH="$REPO/src:$REPO/data/pose_eval/rgbtrack_pydeps:$REPO/data/pose_eval/rgbtrack_sources/bop_toolkit-master${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$REPO/data/pose_eval/rgbtrack_bin:$PATH"
export TORCH_EXTENSIONS_DIR="$REPO/data/pose_eval/torch_extensions_rgbtrack_native"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=4
ID="$(date +%Y%m%d_%H%M%S)"
OUT="$REPO/output/pose_eval/e23_upright_temporal_${ID}"
HTML="$REPO/output/hand_scene_qa/e23_upright_temporal_${ID}"
echo "[1/2] Full episode, upright init and causal temporal gate: $OUT"
"$HOI/envs/sam3d/bin/python" -u "$DIR/rgbtrack_upright_temporal.py" \
 --repo-root "$REPO" --dataset "$REPO/data/1018_peg_all" --episode 23 \
 --mask-root "$HOI/outputs/level1_sequence_e23/masks" \
 --funnel-glb "$REPO/assets/object_geometry/synthware_19_22/funnel_F181975.glb" \
 --flask-glb "$REPO/assets/object_geometry/synthware_19_22/flask_F301950.glb" \
 --scale-manifest "$REPO/data/pose_eval/hand_scene_metric_mesh_scales.json" \
 --front-calibration-profile "$REPO/configs/ur7e_xhand/front_calibration_candidate.json" \
 --output "$OUT" --objects funnel target_flask --mode depth \
 --viz-frames 0,100,141,144,151,155,181,203,357,456,467 \
 --upright-max-tilt-deg 25 --upright-max-rank 150 \
 --temporal-max-tilt-deg 35 \
 --temporal-max-axis-step-deg 25 \
 --temporal-max-rot-step-deg 50 \
 --temporal-max-center-step-m 0.06

echo "[2/2] Interactive 3D review: $HTML"
"$REPO/.venv/bin/python" -u "$DIR/visualize_rgbtrack_full_episode.py" \
 --repo-root "$REPO" --dataset "$REPO/data/1018_peg_all" --episode 23 \
 --pose-run "$OUT" --output "$HTML" --frames-dir "$HOI/data/level1_sequence_e23/frames" --viz-stride 1
printf '[DONE] %s\n[DONE] %s/hand_scene_dynamic_e23.html\n' "$OUT" "$HTML"
