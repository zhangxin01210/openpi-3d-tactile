#!/usr/bin/env bash
# Hand-Scene cross-position trial: reuses the confirmed e23 algorithms and parameters.
# Run: bash scripts/object_geometry/pose_tracking/run_cross_episode_hand_scene.sh EP [check|run]
set -euo pipefail
EP="${1:-}"
MODE="${2:-check}"
[[ "$EP" =~ ^[0-9]+$ ]] && (( 10#$EP <= 69 )) || { echo 'Usage: bash run_cross_episode_hand_scene.sh EP[0..69] [check|run]' >&2; exit 2; }
[[ "$MODE" == check || "$MODE" == run ]] || { echo 'Mode must be check or run' >&2; exit 2; }
EP=$((10#$EP))
REPO="${REPO:-/workspace/mnt/sqzhang26/openpi-3d-tactile}"
HOI="${HOI:-/workspace/mnt/sqzhang26/hoi}"
D="$REPO/scripts/object_geometry/pose_tracking"
FRAMES="$HOI/data/level1_sequence_e${EP}/frames"
MASK="${MASK_ROOT:-$HOI/outputs/level1_sequence_e${EP}/masks}"
CAMERA="$REPO/configs/ur7e_xhand/front_calibration_candidate.json"
METRIC="$REPO/data/pose_eval/hand_scene_metric_mesh_scales.json"
DATASET="$REPO/data/1018_peg_all"
FUNNEL="$REPO/assets/object_geometry/synthware_19_22/funnel_F181975.glb"
FLASK="$REPO/assets/object_geometry/synthware_19_22/flask_F301950.glb"
RUNNER="$HOI/envs/sam3d/bin/python"
PY="$REPO/.venv/bin/python"
required=("$FRAMES" "$MASK/funnel" "$MASK/target_flask" "$MASK/funnel/00000.png" "$MASK/target_flask/00000.png" "$METRIC" "$FUNNEL" "$FLASK" "$CAMERA" "$RUNNER" "$PY" "$D/rgbtrack_native_upright.py" "$D/rgbtrack_upright_full.py" "$D/rgbtrack_adapter.py" "$D/rgbtrack_flask_pivot.py" "$D/freeze_target_flask_anchor.py" "$D/visualize_rgbtrack_full_episode.py")
for f in "${required[@]}"; do [[ -e "$f" ]] || { echo "[MISSING] $f" >&2;exit 2; }; done
n_frames=$(find "$FRAMES" -maxdepth 1 -type f -name '*.jpg' | wc -l)
n_funnel=$(find "$MASK/funnel" -maxdepth 1 -type f -name '*.png' | wc -l)
n_flask=$(find "$MASK/target_flask" -maxdepth 1 -type f -name '*.png' | wc -l)
(( n_frames > 1 && n_funnel == n_frames && n_flask == n_frames )) || {
  echo "[MISMATCH] e${EP} RGB frames=${n_frames}, funnel masks=${n_funnel}, target masks=${n_flask}. Re-run/fix SAM2 first." >&2; exit 2;
}
last=$(printf '%05d' "$((n_frames-1))")
for f in "$MASK/funnel/$last.png" "$MASK/target_flask/$last.png"; do [[ -s "$f" ]] || { echo "[MISSING LAST MASK] $f" >&2;exit 2; };done
printf '[CHECK] e%s frames=%s, funnel_masks=%s, flask_masks=%s\n' "$EP" "$n_frames" "$n_funnel" "$n_flask"
printf '[MASK] %s\n' "$MASK"
printf '[NOTE] Masks from other episodes may predate the corrected e23 prompts. Examine contact sheet/first-frame preview BEFORE treating results as accepted.\n'
if [[ "$MODE" == check ]]; then echo '[CHECK-ONLY] No inference launched.';exit 0;fi
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO/data/pose_eval/rgbtrack_pydeps:$REPO/data/pose_eval/rgbtrack_sources/bop_toolkit-master${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$REPO/data/pose_eval/rgbtrack_bin:$PATH"
export TORCH_EXTENSIONS_DIR="$REPO/data/pose_eval/torch_extensions_rgbtrack_native"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=4
TAG="$(date +%Y%m%d_%H%M%S)_$$"
BASE="$REPO/output/pose_eval"
RAW="$BASE/e${EP}_crosscheck_native_upright_${TAG}"
PIVOT="$BASE/e${EP}_crosscheck_flask_pivot_${TAG}"
FINAL="$BASE/e${EP}_crosscheck_static_target_${TAG}"
VIZ="$REPO/output/hand_scene_qa/e${EP}_crosscheck_static_target_${TAG}"
for f in "$RAW" "$PIVOT" "$FINAL" "$VIZ"; do [[ ! -e "$f" ]] || { echo "Refuse overwrite: $f" >&2; exit 2; };done
# Same algorithm and 90px gate as the successful e23 run. No per-episode tuning.
echo "[1/4] e${EP} RGBTrack upright native recovery: $RAW"
"$RUNNER" -u "$D/rgbtrack_native_upright.py" \
  --follow-max-tilt-deg 35 --recovery-center-max-px 90 --recovery-depth-l1-max-m 0.18 \
  --repo-root "$REPO" --dataset "$DATASET" --episode "$EP" --mask-root "$MASK" \
  --funnel-glb "$FUNNEL" --flask-glb "$FLASK" \
  --scale-manifest "$METRIC" --front-calibration-profile "$CAMERA" \
  --output "$RAW" --objects funnel target_flask --mode depth \
  --viz-frames 0 --upright-max-tilt-deg 25 --upright-max-rank 150
# Keep funnel unchanged, redo only flask first-frame upright init.
echo "[2/4] e${EP} independent flask first-frame upright registration: $PIVOT"
"$RUNNER" -u "$D/rgbtrack_flask_pivot.py" \
  --repo-root "$REPO" --dataset "$DATASET" --episode "$EP" --mask-root "$MASK" \
  --source-pose-run "$RAW" --output "$PIVOT" \
  --funnel-glb "$FUNNEL" --flask-glb "$FLASK" \
  --scale-manifest "$METRIC" --front-calibration-profile "$CAMERA" \
  --flask-mouth-axis auto --flask-init-tilt-deg 20 \
  --flask-max-tilt-deg 20 --flask-max-axis-jump-deg 12 \
  --flask-max-pivot-shift-m 0.025 --flask-axis-alpha 0.25
# Freeze the initial flask rather than allow false tilt drift. Dynamic funnel bytes preserved.
echo "[3/4] e${EP} freeze initial flask anchor: $FINAL"
"$PY" -u "$D/freeze_target_flask_anchor.py" \
  --source-pose-run "$PIVOT" --output "$FINAL" --episode "$EP" --anchor-frame 0
# Standard interactive full-frame QA: existing renderer, no retraining/model additions.
echo "[4/4] e${EP} interactive 3D QA: $VIZ"
"$PY" -u "$D/visualize_rgbtrack_full_episode.py" \
  --repo-root "$REPO" --dataset "$DATASET" --episode "$EP" \
  --pose-run "$FINAL" --output "$VIZ" --frames-dir "$FRAMES" --viz-stride 1
# The existing renderer hardcodes e23 in its filename even when --episode differs.
# Rename only this NEW output so reviewers cannot accidentally confuse episodes.
HTML_SRC="$VIZ/hand_scene_dynamic_e23.html"
HTML_DST="$VIZ/hand_scene_dynamic_e${EP}.html"
[[ -s "$HTML_SRC" ]] || { echo "Missing rendered HTML: $HTML_SRC" >&2; exit 2; }
mv -- "$HTML_SRC" "$HTML_DST"
"$PY" "$D/summarize_cross_episode_hand_scene.py" --pose-run "$FINAL" --output "$FINAL/crosscheck_summary.json"
printf '\n[POSE FINAL] %s\n[HTML] %s\n[SUMMARY] %s\n' "$FINAL" "$HTML_DST" "$FINAL/crosscheck_summary.json"
