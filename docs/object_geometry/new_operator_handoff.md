# New operator handoff — SAM2 → RGBTrack → Hand-Scene Sidecar

**Status:** operator handoff for the reviewed `1018_peg_all` workflow. This is NOT an automatic pipeline for a new camera, object or position. Begin with 1 representative episode; do not launch a new large batch until its masks, pose overlay, calibration and QA have been reviewed.

## 0. What is reused?

- `funnel` and `target_flask` are the important objects. The conical flask is only an initial support.
- For the **same physical funnel/flask** reuse the metric CAD meshes under `assets/object_geometry/synthware_19_22/`. Do **not** run SAM-3D reconstruction independently on every episode. `mesh-build` and `mesh-calibrate` intentionally skip already-approved assets.
- The source of truth for segmentation is the reviewed HOI Level-1 SAM2 pipeline, not a new deployment-side mask implementation.
- The source of truth for offline pose is the reviewed RGBTrack pipeline and its JSONL/manifest outputs, not an independently implemented tracker.
- The sidecar producer in `openpi.object_geometry.cli` converts reviewed masks and pose runs into the fixed-size model input. It does not estimate fresh poses itself.

## 1. Legacy, already-validated data (70 episodes, 6 position groups)

Run from the repository root. Set the paths to the actual machine; examples below assume the training environment has been installed.

```bash
export REPO="$PWD"
export HOI_ROOT="/actual/path/to/hoi"
export HOI="$HOI_ROOT"
export OPENPI_1018_DATASET_ROOT="/actual/path/to/1018_peg_all"
export PYTHONPATH="$REPO/src"
CFG=configs/object_geometry/datasets/1018_peg_all/config.json
PY="$REPO/.venv/bin/python"
```

**Step A: inspect the first frame/masks.** For the legacy HOI checkout, the original Level-1 command sequence is:

```bash
# WARNING: the historical Level-1 snapshot wrapper has machine-specific absolute paths.
# Check tools/hoi_level1_snapshot/scripts/level1_ep.sh before using it on a new host.
bash scripts/object_geometry/level1_ep.sh 23 preview
# Correct the seed prompt/masks if the funnel includes hand or support.
bash scripts/object_geometry/level1_ep.sh 23 track
```

For a fresh or changed setup, independently review the first-frame prompt, funnel/target_flask mask and several occluded frames. An automatically produced mask is not an approval.

**Step B: validate mask/frame counts without tracking:**

```bash
bash scripts/object_geometry/pose_tracking/run_cross_episode_hand_scene.sh 23 check
```

**Step C: track one reviewed episode (fast by default):**

```bash
RENDER_HTML=0 RUN_TAG=operator_trial_e23 \
  bash scripts/object_geometry/pose_tracking/run_cross_episode_hand_scene.sh 23 run
```

This produces the usual RAW → flask-pivot → static-anchor pose runs, plus `crosscheck_summary.json`. It **does not** render the expensive full-episode interactive HTML. Output directories are write-once: choose a fresh RUN_TAG rather than overwrite old runs.

**Step D: render an inspection HTML only for selected episodes, WITHOUT re-running RGBTrack:**

```bash
POSE_RUN="$REPO/output/pose_eval/e23_crosscheck_static_target_operator_trial_e23"
"$PY" scripts/object_geometry/pose_tracking/visualize_rgbtrack_full_episode.py \
  --repo-root "$REPO" --dataset "$OPENPI_1018_DATASET_ROOT" --episode 23 \
  --pose-run "$POSE_RUN" --frames-dir "$HOI_ROOT/data/level1_sequence_e23/frames" \
  --output "$REPO/output/hand_scene_qa/manual_e23_01" --viz-stride 5
```

The historical renderer may name the HTML `hand_scene_dynamic_e23.html` regardless of episode; trust the requested output directory/manifest, not that basename. Use a NEW output directory for each review.

**Step E: produce and QA the existing data sidecar:**

```bash
# Keep the validated version immutable; choose a NEW output version for new/modified data.
"$PY" -m openpi.object_geometry.cli batch --config "$CFG" \
  --episodes 23 --resume --auto-qa --output-version v1_hand_scene_OPERATOR_TRIAL \
  --pose-run 23="$POSE_RUN"
"$PY" -m openpi.object_geometry.cli qa --config "$CFG" \
  --episodes 23 --output-version v1_hand_scene_OPERATOR_TRIAL
```

For episode 23 the config intentionally selects the corrected `maskfix_v1` mask root. Change the mask-root override explicitly if your actual masks differ. Batch may mark an episode `pending_review` rather than export it if calibration, masks, tactile coverage, pose provenance, or other checks fail. Do not bypass a failed gate by inventing a valid frame.

## 2. Large batch after pilot approval

Use `batch --episodes <range> --resume --auto-qa --output-version <NEW_VERSION>`. By default the RGBTrack wrapper now **skips per-episode interactive HTML**, but still emits a JSON summary and the formal CLI still writes a tiny HTML diagnostic only for failed QA episodes. A fast QA summary is NOT proof of metric pose accuracy. Sample the first/contact/occluded/insertion/last frames at representative episodes and at every changed position/camera.

If you need the old interactive HTML during tracking, explicitly use `RENDER_HTML=1` for a **single** episode; never enable it for a large unattended batch.

## 3. New data are NOT automatically covered by legacy configuration

Current reproducibility limits in the repository:

1. `configs/object_geometry/datasets/1018_peg_all/config.json` enumerates episodes **0–69**, with six approved position groups. A new episode `70+` lacks a position group; `batch` will reject it rather than safely guess its initialization.
2. `tools/hoi_level1_snapshot/scripts/level1_batch_preview_all.sh` hardcodes `0..69` and `1..69`, and one absolute HOI output directory.
3. `tools/hoi_level1_snapshot/scripts/level1_ep.sh` contains computer-specific absolute `HOI`/`REPO` paths.
4. The front calibration and static-target anchor assumption are setup-specific. Moving a camera, flask or fixture may invalidate the established profile or pose templates.
5. A new object requires a reviewed metric mesh/scale/axis and a new asset/config version. Reuse approved CAD only when the object is the same.

**Before accepting the new collection:** create a separate dataset config / position-group review; parameterize the historical HOI wrapper and batch range; record actual camera calibration and metric CAD provenance; run one representative pilot; inspect mask overlays and selected 3D frames; only then enable the unattended batch. Do not copy over old `v1_hand_scene_q2` outputs.

## 4. Failure triage

| Symptom | First check |
| --- | --- |
| Hand/support is in funnel mask | Correct the Level-1 seed prompt or masks, inspect first/contact/occluded frames, then rerun SAM2 tracking. |
| Frame-count mismatch | Compare RGB JPG count and the two PNG mask counts; preserve zero-based frame IDs. |
| Funnel mesh rotated/lying down | Check metric CAD axis, upright init and camera extrinsics; don't change pose convention ad hoc. |
| Point cloud includes background | Verify depth is aligned to RGB and the correct intrinsics/extrinsics are used. |
| Tracking jumps after occlusion | Examine mask and native recovery diagnostics; don't silently fill invalid poses. |
| Target flask drifts | The final static-anchor prior is intentional for the validated task; inspect frame-0 anchor. |
| QA rejects data | Read `scratch/object_geometry/1018_peg_all/batch/batch_report.json` and per-episode failures. |
| Same output version already exists | Use `--resume` only for provenance-identical data; otherwise choose a NEW version. |
| Run is slow after RGBTrack | Check if `RENDER_HTML=1` and whether full-episode HTML is being generated. |

**Not yet verified in this integration branch:** syntax/tests and real A800/SAM2/RGBTrack runs must be carried out on the data machine. This branch intentionally does not change trained token contracts or calibration settings.
