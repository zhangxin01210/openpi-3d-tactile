#!/usr/bin/env bash
# Single command on sai: Episode 23 -> SAME perception -> trained Pi0 -> GT.
# Offline ONLY: no robot SDK, websocket, or actuator commands.
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OLD="$HOME/zx/openpi-3d-tactile"
DATASET="$OLD/data/1018_peg_all"
RUN="$ROOT/output/deploy/qa/e023_action_compare_v1"
FLASK_RUN="$ROOT/output/deploy/qa/e023_flask_training_init_v1"
PROFILE="$ROOT/configs/ur7e_xhand/front_calibration_candidate.json"
FRAMES=64
EPISODE=23
if [ -n "$RUN_DIR" ]; then RUN="$RUN_DIR"; fi
if [ -e "$RUN" ]; then
  echo "Refusing to overwrite prior comparison: $RUN" >&2; exit 2
fi
PER_PY="$OLD/.venv-perception/bin/python"
PI_PY="$OLD/.venv/bin/python"
for f in "$PER_PY" "$PI_PY" "$PROFILE" "$DATASET/meta/info.json"; do
  if [ ! -e "$f" ]; then echo "Missing: $f" >&2; exit 2; fi
done
mkdir -p "$RUN"
ANCHOR="$RUN/reviewed_episode23_flask_anchor.json"
if [ -n "$ANCHOR_JSON" ]; then ANCHOR="$ANCHOR_JSON"; fi
if [ ! -f "$ANCHOR" ]; then
  PROJECTION="$FLASK_RUN/target_init/rgb_overlays/f000000.png"
  if [ ! -f "$PROJECTION" ]; then
    echo "Prior reviewed Episode 23 overlay missing: $PROJECTION" >&2
    echo "Pass existing anchor as ANCHOR_JSON=/absolute/path/to/anchor.json" >&2
    exit 2
  fi
  echo "[1/4] Reuse reviewed Episode 23 frame-0 flask pose"
  PYTHONPATH="$ROOT/src:$ROOT" "$PER_PY" \
    scripts/deploy/review_flask_anchor.py \
    --flask-pivot-run "$FLASK_RUN" --episode "$EPISODE" \
    --front-calibration-profile "$PROFILE" \
    --approve-projection "$PROJECTION" --output "$ANCHOR"
fi

echo "[2/4] Read original training RGB, state and depth with SAME registration"
PYTHONPATH="$ROOT/src:$ROOT:$ROOT/scripts/object_geometry/pose_tracking" \
 "$PER_PY" -u scripts/deploy/export_training_episode_packets.py \
 --dataset "$DATASET" --episode "$EPISODE" --max-frames "$FRAMES" \
 --front-calibration-profile "$PROFILE" --output-dir "$RUN/raw_frames"

echo "[3/4] SAM2 + RGBTrack + spatial/tactile/hand-object geometry"
PATH="$OLD/.venv-perception/bin:$PATH" \
PYTHONPATH="$ROOT/src:$ROOT:$ROOT/scripts/object_geometry/pose_tracking" \
 "$PER_PY" -u scripts/deploy/run_online_perception.py \
 --repo-root "$ROOT" --replay-dir "$RUN/raw_frames" \
 --output-dir "$RUN/perception" \
 --sam2-checkpoint "$OLD/checkpoints/sam2/sam2.1_hiera_small.pt" \
 --weights-root "$OLD/checkpoints/rgbtrack" \
 --scale-manifest "$OLD/data/pose_eval/hand_scene_metric_mesh_scales.json" \
 --front-calibration-profile "$PROFILE" \
 --episode "$EPISODE" --position-template-episode "$EPISODE" \
 --template-reviewed-for-current-scene \
 --static-target-anchor-json "$ANCHOR" \
 --static-target-anchor-source human_reviewed --max-frames "$FRAMES"

echo "[4/4] Trained Pi0 at 16-frame intervals: compare with real 18D actions"
CHECKPOINT="$OLD/checkpoints/pi0_xhand_spatial_structured_suffix_hand_scene_oo_v1/m2_b8_20k_1026/19999"
PYTHONPATH="$ROOT/src:$ROOT" "$PI_PY" -u \
 scripts/deploy/run_policy_dry_run.py \
 --repo-root "$ROOT" --frames-dir "$RUN/perception/frames" \
 --output-dir "$RUN/policy" \
 --config pi0_xhand_spatial_structured_suffix_hand_scene_oo_v1 \
 --checkpoint "$CHECKPOINT" --frame-stride 16 \
 --gt-dataset "$DATASET" --gt-episode "$EPISODE" \
 --prompt "Move the funnel over the target flask and insert it."

echo "===== ACTION COMPARISON ====="
"$PI_PY" - "$RUN/policy/summary.json" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))
print("Pi0 queries:",s["frames"],"action horizon:",s["action_horizon"])
print("Mean inference ms:",round(s["mean_inference_ms"],1))
for k in ("first_arm6_mae_rad","first_hand12_mae_rad",
          "chunk_arm6_mae_rad","chunk_hand12_mae_rad"):
    print(k,round(s["ground_truth"][k],5))
print("Result:",sys.argv[1])
PY
