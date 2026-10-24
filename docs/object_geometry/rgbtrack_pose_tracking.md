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

RGBTrack internally subtracts the mesh AABB center in `reset_object()`.  Its
`register`/`track_one_new` return path multiplies
`get_tf_to_centered_mesh()` before returning, so the adapter receives a pose
for the **original mesh frame**.  It does not apply that center translation a
second time; the same original vertices are used for BBox projection and the
base-link PLY checks.

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

## Computer-4 runtime compatibility

The adapter does not edit tracked files in `third_party/RGBTrack`.  The
upstream predictors, however, look for weights in the legacy path
`third_party/RGBTrack/weights`.  Create a local symlink to the main-repository
checkpoint directory once per checkout:

```bash
cd /workspace/mnt/sqzhang26/openpi-3d-tactile
PYTHONPATH=src /workspace/mnt/sqzhang26/hoi/envs/sam3d/bin/python \
  scripts/object_geometry/pose_tracking/check_rgbtrack_runtime.py \
  --repo-root "$PWD" \
  --weights-root "$PWD/checkpoints/rgbtrack" \
  --setup-weights-link
```

The command refuses to replace an existing non-symlink directory.  It only
creates `third_party/RGBTrack/weights -> checkpoints/rgbtrack`; this is a
local untracked submodule artifact and must not be committed.

### Dependency matrix

| Dependency | When it is needed | Computer-4 action |
| --- | --- | --- |
| `torch`, `pytorch3d`, `open3d`, `trimesh`, `omegaconf`, `kornia`, `einops`, `transformations`, `ruamel.yaml` | import and first estimator initialization | already supplied by the SAM3D environment / `rgbtrack_pydeps` |
| `nvdiffrast.torch` | every estimator initialization and CUDA render | build the pinned source with the command below |
| `mycpp.build.mycpp` with `cluster_poses` | every `FoundationPose` initialization; `make_rotation_grid()` calls it unconditionally | build the C++ extension with the command below |
| `warp` (`warp-lang`) | depth mode `register`/`track_one_new`, for depth erosion and bilateral filtering | install the compatible wheel if the checker reports it missing |
| `filterpy` | upstream `PoseTracker` construction and occlusion recovery | install this small dependency if missing |
| `bop_toolkit_lib` | imported by `tools.py`; only metric/evaluation helpers use it | install from the local tarball, or keep the checker result as an explicit blocked optional import |
| TensorRT | not needed for normal PyTorch inference | the adapter installs an import-only stub when TensorRT is absent; instantiating a TRT class still fails explicitly |
| `bundlesdf.mycuda.common`, XMem, SAM2 | not used by this RGBTrack path | safe to skip |

Do not install the entire upstream `requirements.txt`: it pins an older CUDA
and PyTorch stack.  The checker reports imports separately from the CUDA/model
phases.

### NVDiffRast build (CUDA 12.4 / A800)

Use the existing source archive or an equivalent pinned checkout.  This does
not change the RGBTrack submodule:

```bash
cd /workspace/mnt/sqzhang26/openpi-3d-tactile
NR_TMP=$(mktemp -d)
tar -xzf nvdiffrast-v0.3.5.tar.gz -C "$NR_TMP"
cd "$NR_TMP/nvdiffrast-0.3.5"
TORCH_CUDA_ARCH_LIST=8.0 \
  /workspace/mnt/sqzhang26/hoi/envs/sam3d/bin/python -m pip install \
  --no-build-isolation .
```

Verify the extension in the same interpreter:

```bash
/workspace/mnt/sqzhang26/hoi/envs/sam3d/bin/python -c \
  "import torch, nvdiffrast.torch as dr; assert torch.cuda.is_available(); dr.RasterizeCudaContext(); print('nvdiffrast CUDA OK')"
```

### mycpp build

`mycpp.cluster_poses` is not optional.  The extension requires CMake, a C++14
compiler, Boost `system`/`program_options`, Eigen3, OpenMP and Python
`pybind11` headers.  On a Debian/Ubuntu computer 4, install only the system
development packages if the checker says they are absent, then build:

```bash
sudo apt-get update
sudo apt-get install -y cmake g++ libboost-system-dev libboost-program-options-dev \
  libeigen3-dev libomp-dev

cd /workspace/mnt/sqzhang26/openpi-3d-tactile
/workspace/mnt/sqzhang26/hoi/envs/sam3d/bin/python -m pip install --no-deps pybind11
PYBIND11_CMAKE=$(/workspace/mnt/sqzhang26/hoi/envs/sam3d/bin/python -m pybind11 --cmakedir)
cmake -S third_party/RGBTrack/mycpp -B third_party/RGBTrack/mycpp/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE=/workspace/mnt/sqzhang26/hoi/envs/sam3d/bin/python \
  -Dpybind11_DIR="$PYBIND11_CMAKE"
cmake --build third_party/RGBTrack/mycpp/build --parallel "$(nproc)"

PYTHONPATH=third_party/RGBTrack \
  /workspace/mnt/sqzhang26/hoi/envs/sam3d/bin/python -c \
  "import mycpp.build.mycpp as m; assert callable(m.cluster_poses); print('mycpp cluster_poses OK')"
```

If `python -m pybind11 --cmakedir` fails, install `pybind11` in the same SAM3D
interpreter before configuring CMake.  Do not use the old `build_all_conda.sh`:
it assumes the upstream CUDA 11/PyTorch 2.0 environment.

If phase A reports the small runtime-only imports as missing, install only
these packages (still in the SAM3D interpreter):

```bash
PY=/workspace/mnt/sqzhang26/hoi/envs/sam3d/bin/python
$PY -m pip install --no-deps filterpy warp-lang
BOP_TMP=$(mktemp -d)
tar -xzf /workspace/mnt/sqzhang26/openpi-3d-tactile/bop_toolkit.tar.gz -C "$BOP_TMP"
$PY -m pip install --no-deps -e "$BOP_TMP/bop_toolkit-master"
```

`filterpy` is used by RGBTrack's causal `PoseTracker`; `warp-lang` supplies
the CUDA depth erosion/bilateral kernels.  `bop_toolkit_lib` is imported by
the upstream `tools.py` even though its metric functions are not used during
normal tracking.  TensorRT is the one intentional exception: its import is
shimmed for `USE_TRT=False`, and no TensorRT engine is loaded.

## Staged smoke tests

Run the checker with the SAM3D interpreter:

```bash
PYTHONPATH=src /workspace/mnt/sqzhang26/hoi/envs/sam3d/bin/python \
  scripts/object_geometry/pose_tracking/check_rgbtrack_runtime.py \
  --repo-root /workspace/mnt/sqzhang26/openpi-3d-tactile \
  --weights-root /workspace/mnt/sqzhang26/openpi-3d-tactile/checkpoints/rgbtrack \
  --phase A
```

The phases are intentionally not conflated:

* **A** checks Python imports and reports optional imports; it does not prove
  CUDA, weights, or tracking.
* **B** imports RGBTrack, checks `cluster_poses`, and constructs the estimator
  with the configured PyTorch checkpoints.
* **C** runs a tiny CUDA-only NVDiffRast triangle render without robot data.
* **D** is the real e23 single-object initialization command below.
* **E** is the same command without `--max-frames 1`, producing causal
  continuous tracking.

For the first real bring-up, initialize the funnel only when it becomes
visible while allowing the flask to initialize from frame zero:

```bash
PYTHONPATH=src /workspace/mnt/sqzhang26/hoi/envs/sam3d/bin/python \
  scripts/object_geometry/pose_tracking/rgbtrack_pose_eval.py \
  --repo-root /workspace/mnt/sqzhang26/openpi-3d-tactile \
  --dataset /workspace/mnt/sqzhang26/openpi-3d-tactile/data/1018_peg_all \
  --episode 23 \
  --mask-root /workspace/mnt/sqzhang26/hoi/outputs/level1_sequence_e23/masks \
  --funnel-glb /workspace/mnt/sqzhang26/hoi/outputs/sam3d_funnel_212/funnel.glb \
  --flask-glb /workspace/mnt/sqzhang26/openpi-3d-tactile/output/pose_eval/meshes/target_flask_e23_f000/flask.glb \
  --scale-manifest /workspace/mnt/sqzhang26/openpi-3d-tactile/data/pose_eval/rgbtrack_scales_e23.json \
  --weights-root /workspace/mnt/sqzhang26/openpi-3d-tactile/checkpoints/rgbtrack \
  --front-calibration-profile /workspace/mnt/sqzhang26/openpi-3d-tactile/configs/ur7e_xhand/front_calibration_candidate.json \
  --output /workspace/mnt/sqzhang26/openpi-3d-tactile/output/pose_eval/rgbtrack_e23_funnel_smoke \
  --objects funnel \
  --init-frame funnel=212 \
  --mode depth \
  --max-frames 213
```

After D succeeds, remove `--max-frames 213`, add
`--objects funnel target_flask --init-frame target_flask=0`, and use a new
output directory for E.  The adapter records requested and actual
initialization frames; frames before an object's requested frame are invalid
and do not initialize that object.

## Upstream environment caveat

At the pinned RGBTrack commit, `estimater.py` imports `Utils.py`,
`predict_score.py`, and `predict_pose_refine.py` at module import time.  Those
modules import/use CUDA, nvdiffrast, Open3D, PyTorch3D and CUDA tensors.  The
prediction modules also import the TensorRT model wrappers even though their
`USE_TRT` flags are `False`; TensorRT is not lazily isolated upstream.  The
adapter therefore delays the import until actual inference.  Computer 1 can
run the synthetic tests and preparation boundary checks, but real e23
inference requires the existing computer-4 RGBTrack environment and weights.
