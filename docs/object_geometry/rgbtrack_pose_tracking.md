# RGBTrack 6D pose adapter

`third_party/RGBTrack` remains an unmodified git submodule at commit
`23e43f832b82f4813d6ab72773779bf8d2ee4d83`.  The adapter lives in
`scripts/object_geometry/pose_tracking/` and is intentionally headless:
there is no `cv2.imshow`, no future-frame smoothing, and no SAM2 inference.

## What the adapter does

`rgbtrack_pose_eval.py` joins the front RGB video and front depth rows through
`RawSpatialDataset`, resolves existing SAM2 PNG masks, and tracks each object
with an independent RGBTrack estimator.  The accepted object names are
`funnel` and `target_flask`; selecting both does **not** tie the flask pose to
the robot hand or to the funnel pose.

RGBTrack's camera-frame result is `T_color_object` (`ob_in_cam`).  The active
front calibration is loaded through the same profiled path as the spatial
preprocessor, including the candidate `[0, -0.012, +0.005]` m base-frame
translation and color intrinsics.  The output is:

```
T_base_link_object = T_base_color @ T_color_object
```

Translations are metres.  Every JSONL row contains both 4x4 matrices,
`valid`, `confidence`, `status`, `elapsed_ms`, `mask_area_px`, the selected
mode, and (for unknown-scale mode) the recovered scale.

The mask is treated as an **observed visible mask**, not an amodal mask.  A
small/empty mask produces an invalid causal short-term prediction with zero
confidence.  It is never silently promoted to a valid object pose.

## Modes are deliberately separate

* `--mode depth` calls upstream `register` followed by `track_one_new` and
  uses the supplied depth in metres.
* `--mode no_depth` calls `register_without_depth` followed by
  `track_one_new_without_depth`.  Upstream searches synthetic depth samples;
  this mode does not recover scene scale.
* `--mode unknown_scale` calls upstream `binary_search_scale` on the first
  RGB-D frame.  It is opt-in and must not be used to pretend that a normalized
  SAM3D mesh is metric.

The mesh conversion always requires an explicit metres-per-GLB-unit value.
There is no inference from GLB units, extents, or SAM3D output.  Converted
meshes are written under a new content/scale-named directory and the source
GLB, existing e23 `cloud_frames`, and existing `abc_v1` are untouched.

## Computer 4 preparation

Create `data/pose_eval/rgbtrack_scales_e23.json` using independently known
metric dimensions (do not copy the normalized SAM3D scale):

```json
{
  "schema_version": 1,
  "objects": {
    "funnel": <metres per funnel.glb unit>,
    "target_flask": <metres per flask.glb unit>
  }
}
```

Then run the complete two-object RGB-D evaluation:

```bash
cd /workspace/mnt/sqzhang26/openpi-3d-tactile
PYTHONPATH=src python scripts/object_geometry/pose_tracking/rgbtrack_pose_eval.py \
  --repo-root /workspace/mnt/sqzhang26/openpi-3d-tactile \
  --dataset /workspace/mnt/sqzhang26/openpi-3d-tactile/data/1018_peg_all \
  --episode 23 \
  --mask-root /workspace/mnt/sqzhang26/hoi/outputs/level1_sequence_e23/masks \
  --funnel-glb /workspace/mnt/sqzhang26/hoi/outputs/sam3d_funnel_212/funnel.glb \
  --flask-glb /workspace/mnt/sqzhang26/openpi-3d-tactile/output/pose_eval/meshes/target_flask_e23_f000/flask.glb \
  --scale-manifest /workspace/mnt/sqzhang26/openpi-3d-tactile/data/pose_eval/rgbtrack_scales_e23.json \
  --front-calibration-profile /workspace/mnt/sqzhang26/openpi-3d-tactile/configs/ur7e_xhand/front_calibration_candidate.json \
  --output /workspace/mnt/sqzhang26/openpi-3d-tactile/output/pose_eval/rgbtrack_e23_front_candidate \
  --objects funnel target_flask \
  --mode depth \
  --viz-frames 120,200,300,400,440,460
```

For a single-object bring-up, use the same command with
`--objects funnel`.  `--prepare-only` converts the GLBs and writes the
calibration/mesh manifest without importing RGBTrack or reading the dataset.

Outputs include one JSONL pose stream per object, `manifest.json`, selected
RGB reprojection images under `rgb_overlays/`, transformed base-link bbox PLY
files under `bbox_base_link/`, the converted mesh and
`mesh_conversion.json`, plus a dependency-free `preview.html` containing the
provenance.  The manifest records the active calibration profile and its
SHA-256, the RGBTrack submodule commit, explicit mesh scales, and the causal
mode.

## Upstream environment caveat

At the pinned RGBTrack commit, `estimater.py` imports `Utils.py`,
`predict_score.py`, and `predict_pose_refine.py` at module import time.  Those
modules import/use CUDA, nvdiffrast, Open3D, PyTorch3D and CUDA tensors.  The
prediction modules also import the TensorRT model wrappers even though their
`USE_TRT` flags are `False`; TensorRT is not lazily isolated upstream.  The
adapter therefore delays the import until actual inference.  Computer 1 can
run the synthetic tests and preparation boundary checks, but real e23
inference requires the existing computer-4 RGBTrack environment and weights.
