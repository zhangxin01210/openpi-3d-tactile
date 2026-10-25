# Object Geometry `v1_hand_scene` production

This is the production boundary for `1018_peg_all`. It consumes existing SAM2
masks and the reviewed RGBTrack JSONL pose format; it does not run a second
mask reader or tracker. Generated data lives in
`<LeRobot root>/object_geometry/v1_hand_scene_q2/` by the current CLI (the
older `v1_hand_scene/` remains readable and is never overwritten).

The `batch` command applies the reviewed six-position setup, runs deterministic
quality checks, and exports only `auto_qa_passed` episodes. `human_accepted` is
an independent offline review status; automatic QA is not a claim of metric
pose accuracy.

## Contract

The dataset config is
`configs/object_geometry/datasets/1018_peg_all/config.json`. Paths are
checkout-relative unless overridden by environment variables. The validated
metric GLBs in `assets/object_geometry/synthware_19_22/` are reused; their
explicit `scale_m_per_unit: 1.0` values are required even though they are
already in metres. A normalized SAM3D scale is never inferred.

For each accepted pose frame the producer deterministically samples 512
surface points in the object-local metric mesh frame and applies the row's
`T_base_link_object`. An invalid dynamic funnel row produces a zero, invalid
cloud. The target flask is accepted as a static geometry prior only when the
reviewed stream has a valid frame-0 anchor and explicitly tags subsequent rows
as `flask_static_anchor_prior`/`static_geometry_prior`.

The sidecar contains the existing mmap fields plus Hand-Scene V0 fields: 600
force magnitudes, a causal first-valid reference, previous-frame delta, contact
validity, geometry quality, and an optional current funnel mask. Without a
tactile or hand-offset file the corresponding values are zero/false; no
confidence or relation label is fabricated. The manifest records the front
calibration digest, mesh digests/scales/sampler, pose-run digest, and
`future_frame_used: false`.

`geometry_quality` is structural validity (approved finite Mesh plus accepted
tracker/prior state), not calibrated pose accuracy. The RGBTrack mask-area
heuristic is retained only as raw QA provenance. `grasp_quality` stays zero
(unknown) unless an explicit stability label is supplied; finite tactile data
alone is not a grasp-quality label.

## Computer 4 workflow

Use the existing environments; do not copy weights or generated data into the
repository. `HOI_ROOT` points at the existing HOI checkout.

```bash
cd /workspace/mnt/sqzhang26/openpi-3d-tactile
export HOI_ROOT=/workspace/mnt/sqzhang26/hoi
export PYTHONPATH="$PWD/src"
# The existing SAM3D/RGBTrack environment supplies trimesh/Open3D for GLB
# sampling. Override this only with another environment that has those same
# dependencies.
PY="${OBJECT_GEOMETRY_PYTHON:-$HOI_ROOT/envs/sam3d/bin/python}"
CFG=configs/object_geometry/datasets/1018_peg_all/config.json

$PY -m openpi.object_geometry.cli init --config "$CFG"
$PY -m openpi.object_geometry.cli mesh-build --config "$CFG"
$PY -m openpi.object_geometry.cli mesh-calibrate --config "$CFG"
$PY -m openpi.object_geometry.cli mesh-approve --config "$CFG" --approved-by reviewer
$PY -m openpi.object_geometry.cli mask-preview --config "$CFG" --episodes 23,38,65
```

`mesh-build` and `mesh-calibrate` intentionally report a no-op for this
dataset: the assets are already human-approved. If a mask preview is wrong,
use the existing HOI `annotate_frame0.html`, replace that episode's prompt
file, and rerun the same HOI preview/track stage. No Python source edit is
needed. An unknown position group remains pending rather than being forced
into a wrong template.

Run the already validated upright/native-recovery/static-flask pipeline only
when an exact pose directory is not already present. `RUN_TAG` makes output
names deterministic and avoids selecting a “latest” directory. Episode 23 must
use the corrected `maskfix_v1` root:

```bash
RUN_TAG="e23_validated" \
  MASK_ROOT="$HOI_ROOT/outputs/level1_sequence_e23_maskfix_v1/masks" \
  bash scripts/object_geometry/pose_tracking/run_cross_episode_hand_scene.sh 23 run
```

For e38/e65, pass their exact existing pose directories to `batch --pose-run`;
if one is missing, batch can run the same deterministic tracker with
`--run-tracking`. The old review command remains available for recording a
human decision, but it is no longer required for batch export:

```bash
for ep in 23 38 65; do
  $PY -m openpi.object_geometry.cli pose-preview --config "$CFG" \
    --episode "$ep" \
    --pose-run "output/pose_eval/e${ep}_crosscheck_static_target_e${ep}_validated" \
    --review-status accepted --approved-by reviewer \
    --review-note "RGB+3D review: f0, f120, f200, f300, f400, f440, f460"
done
```

Produce the first three formal shards and verify the mmap reader. Use a new
output version when quality/tactile semantics change; never overwrite an older
version or rely on `--resume` to update it:

```bash
$PY -m openpi.object_geometry.cli batch --config "$CFG" \
  --episodes 23,38,65 --resume --auto-qa --no-run-tracking \
  --output-version v1_hand_scene_q2 \
  --mask-root 23="$HOI_ROOT/outputs/level1_sequence_e23_maskfix_v1/masks" \
  --pose-run 23="$PWD/output/pose_eval/e23_crosscheck_static_target_e23_validated" \
  --pose-run 38="/exact/existing/e38/pose/run" \
  --pose-run 65="/exact/existing/e65/pose/run"
$PY -m openpi.object_geometry.cli qa --config "$CFG" \
  --episodes 23,38,65 --output-version v1_hand_scene_q2
```

`batch` writes `scratch/object_geometry/1018_peg_all/batch/batch_report.json`
and an HTML failure report for every pending episode. It records the selected
position group/template, actual mask root and per-file SHA256, pose manifest
SHA256, calibration/mesh provenance, frame checks, tracker failures/recovery,
static-flask drift, raw RGBTrack heuristic statistics, tactile coverage, and
the no-future-frame check. Existing shards are skipped only when their
provenance still matches; otherwise the episode remains pending and a new
`--output-version` is required.

## Full 70-episode run

The following is the direct computer-4 command. It reuses complete masks and
the six configured position groups, skips provenance-matching shards, runs the
validated tracker for missing pose runs, and exports only automatic-QA passes.
One bad episode does not stop the other episodes; pending episodes retain an
HTML inspection entry and are not silently exported.

```bash
E23_MASK_ROOT="$HOI_ROOT/outputs/level1_sequence_e23_maskfix_v1/masks"
$PY -m openpi.object_geometry.cli batch --config "$CFG" \
  --episodes 0-69 --resume --auto-qa \
  --output-version v1_hand_scene_q2 \
  --mask-root 23="$E23_MASK_ROOT" \
  --pose-run 23="$PWD/output/pose_eval/e23_crosscheck_static_target_e23_validated" \
  --pose-run 38="$PWD/output/pose_eval/e38_crosscheck_static_target_e38_validated" \
  --pose-run 65="$PWD/output/pose_eval/e65_crosscheck_static_target_e65_validated"
$PY -m openpi.object_geometry.cli qa --config "$CFG" \
  --episodes 0-69 --output-version v1_hand_scene_q2
```

To inspect without running RGBTrack, add `--no-run-tracking`; this produces
pending reports for missing pose runs. To continue after an interruption, rerun
the same command with `--resume`. Do not use shell globbing or “newest output”
selection. If tactile sidecar loading fails, the formal config rejects the
episode (`allow_missing_tactile: false`); the output remains pending rather
than receiving fabricated grasp labels. `grasp_quality=0` means unknown, while
`tactile_valid_fraction` reports source integrity separately.

`qa` reports all requested episodes, actual versus LeRobot frame counts,
funnel and static-flask validity, point validity, tactile coverage, relation
validity, failures/recoveries/jumps, automatic/human/pending/rejected status,
missing/version mismatches, and future-frame usage. It does not call a pose
run “accurate”.

## Existing readers and compatibility

`openpi.object_geometry_dataset.ObjectGeometryDerivedDataset` is the only
training-side mmap reader. The legacy `export_hand_scene_sidecar.py` remains
available for old calibrated A/B sidecars; it is intentionally not used for
Mesh+6D production because it cannot create metric mesh points. No M0/M1
encoder, training config, robot action interface, front baseline, or RGBTrack
submodule file is changed by this pipeline.

Computer 1 can run the deterministic sampler, pose-contract and mmap tests,
but has no real e23/e38/e65 masks, pose runs, RGBTrack CUDA stack, or A800.
Real tracking/export success must therefore be reported from the computer-4
commands above, not inferred from local tests.
