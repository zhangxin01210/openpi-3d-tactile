# Object Geometry `v1_hand_scene` production

This is the production boundary for `1018_peg_all`. It consumes existing SAM2
masks and reviewed RGBTrack JSONL pose runs; it does not run a second mask
reader or tracker. Generated data lives in
`<LeRobot root>/object_geometry/v1_hand_scene/` and is not checked into Git.

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

Run the already validated upright/native-recovery/static-flask pipeline for
the regression episodes. `RUN_TAG` makes output names deterministic and avoids
selecting a “latest” directory:

```bash
for ep in 23 38 65; do
  RUN_TAG="e${ep}_validated" \
    bash scripts/object_geometry/pose_tracking/run_cross_episode_hand_scene.sh "$ep" run
done
```

After visually reviewing the generated HTML and representative frames, record
each exact final directory. The explicit review command is the approval gate:

```bash
for ep in 23 38 65; do
  $PY -m openpi.object_geometry.cli pose-preview --config "$CFG" \
    --episode "$ep" \
    --pose-run "output/pose_eval/e${ep}_crosscheck_static_target_e${ep}_validated" \
    --review-status accepted --approved-by reviewer \
    --review-note "RGB+3D review: f0, f120, f200, f300, f400, f440, f460"
done
```

Produce the first three formal shards and verify the mmap reader:

```bash
$PY -m openpi.object_geometry.cli export --config "$CFG" \
  --episodes 23,38,65 --resume
$PY -m openpi.object_geometry.cli qa --config "$CFG" --episodes 23,38,65
```

The exporter refuses to overwrite an existing episode shard, checks that the
pose manifest has not changed since approval, and can be rerun after a failed
episode. It requires `trimesh` in the existing computer-4 RGBTrack/SAM3D
environment for GLB sampling; the OpenPI environment only needs to read the
result.

## Full 70-episode run

Run/review masks and the same validated pose pipeline for each episode. The
following uses deterministic output names and never discovers a newest
directory. A reviewer must execute `pose-preview --review-status accepted` for
every episode before export.

```bash
for ep in $(seq 0 69); do
  RUN_TAG="e${ep}_validated" \
    bash scripts/object_geometry/pose_tracking/run_cross_episode_hand_scene.sh "$ep" run
  $PY -m openpi.object_geometry.cli pose-preview --config "$CFG" \
    --episode "$ep" \
    --pose-run "output/pose_eval/e${ep}_crosscheck_static_target_e${ep}_validated" \
    --review-status pending --approved-by reviewer
done
# After inspection, rerun each pending review with --review-status accepted.
$PY -m openpi.object_geometry.cli export --config "$CFG" --episodes 0-69 --resume
$PY -m openpi.object_geometry.cli qa --config "$CFG" --episodes 0-69
```

`qa` reports frame/episode coverage and checks finite `[512,3]` clouds,
Hand-Scene fields, calibration provenance, and the no-future-frame flag. It
does not call a pose run “accurate”; visual approval remains a required human
record in `scratch/object_geometry/1018_peg_all/pose_reviews/`.

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
