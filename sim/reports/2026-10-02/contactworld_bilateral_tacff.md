# ContactWorld USB bilateral TacFF implementation

The released ContactWorld USB Zarr has only `tactile_force_field_right`.
This report covers an **isolated simulator extension** that computes a new
left field during replay. It does not revise or retroactively complete the
released dataset.

## Isolation and reproducibility

- Original checkout: `/home/sai/zx/openpi-sim-runtime/third_party/ContactWorld-sm120`
  (untouched).
- Modified checkout: `/home/sai/zx/openpi-sim-runtime/third_party/ContactWorld-bilateral-probe`.
- Reproducible source diff: [`contactworld_usb_bilateral_tacff.patch`](../../patches/contactworld_usb_bilateral_tacff.patch).
  `patch --dry-run -p1` against the original checkout passes. Apply the patch
  only to a private copy of the simulator.
- The legacy Isaac Gym/PyTorch/CUDA build environment is reused, not rebuilt.

## Sensor repair

The USB environment now instantiates the configured left and right pad fields.
The source task exposes both as separate 10×14×3 observation keys. TacSL's
sampling grid, SDF and pose caches were keyed by the plug rigid-body ID;
left and right share that ID, so enabling both without changing the cache
would overwrite one grid. The isolated patch keys those caches by sensor name
and uses each pad's own link ID for the velocity-bias lookup. Same-frame world
taxel poses are retained by sensor name so each field can be rotated into the
Franka base independently.

The force model itself is unchanged: SDF penetration normal plus a
relative-velocity shear approximation, with synthetic, uncalibrated units.
This is two-finger plug/pad tactile sensing, not a calibrated measurement of
the plug/socket axial load.

## Completed checks

- Eight zero-action steps produced finite, nonzero 10×14×3 force fields on
  both pads. The initial pad centroid separation was about 25.8 mm.
- A complete 82-frame USB demonstration produced both sidecars; each pad has
  140 distinct sample positions. The two force fields differ and the pad
  positions occupy opposite sides of the plug.
- On the same initial state, the isolated extension and original checkout
  yielded **bitwise identical** right force, RGB, joint/EE/plug states,
  4096-point clouds and executed actions. Final insertion success and plug
  error were also identical. The original source was not edited.
- The one-episode bilateral LeRobot export and its capture-alignment/video
  verifier passed: 82 frames, both force fields, per-frame camera depth and
  calibration, and actual executed actions.
- Bilateral 280-taxel data-loader and encoder smoke checks passed; the
  separate-finger encoder emits one token per pad.

The [bilateral visualization](/home/sai/zx/openpi-sim-runtime/visualizations/contactworld_bilateral_audit_20261002/index.html)
shows both pad positions, all three raw force channels, and base-frame XYZ
component timelines.

## Training interpretation

The bilateral dataset will use a separate `contactworld-spatial-v3-bilateral`
manifest and new `pi0_cw3_usb_*` configs. The paired matrix compares RGB,
fused point cloud, right-only, left-only, both pads pooled into one token,
both pads as separate tokens, per-pad summary, and front-only point cloud.
Every config will use the same accepted replay demonstrations and split.
Comparing pooled versus separate tests the encoding of finger identity;
comparing full versus summary tests whether fine contact distribution helps.
The summary model also changes the small tactile encoder, so its result is a
representation-plus-encoder comparison. None of these experiments alone
establishes reliable axial jam-force measurement or closed-loop correction.

## User review and V3 data contract

The user accepted the complete 82-frame source episode 001 bilateral review,
then requested two additional checks before training: the exported LeRobot
dataset must visibly retain both tactile sides and the actual OpenPI inputs
must contain the requested modalities. The prior smoke export had one video
named `observation.images.tactile` because it inherited the released
right-finger tactile RGB camera stream. That video did not represent the new
left force field and was misleading for bilateral review.

The corrected V3 export stores six separate numerical sidecars: local
10×14×3 force grids, base-frame 140×3 taxel positions and base-frame 140×3
forces for **each** of `left` and `right`. The only tactile videos in the
corrected V3 LeRobot dataset are a *pair* of explicitly named
`tacff_left_preview` and `tacff_right_preview` streams. These are diagnostic
RGB encodings of local normal/shear X/shear Y with a fixed ±0.003 scale and
zero=128. They are not raw tactile camera images and are masked out of all
current OpenPI image inputs. The unpaired released right tactile RGB and
right-only tactile depth remain available in older V1/V2 datasets, but are
excluded from V3. Front/wrist RGB and front/fused 4096-point clouds remain.

The corrected single-episode dataset at
`data/contactworld_usb_v3_bilateral_smoke_final_20261002` passed the verifier:
all 82 rows, source-aligned action/state/pointcloud/force arrays, both camera
calibrations, paired H.264/yuv420p videos, preview-to-force encoding and
sampled RGB-to-capture image comparisons. The
[LeRobot review page](/home/sai/zx/openpi-sim-runtime/visualizations/contactworld_bilateral_dataset_v3_final_20261002/index.html)
reads only the exported dataset, shows the four video streams and interactive
clouds from the saved model input arrays. A separate audit instantiated all
eight `pi0_cw3_usb_*` training configs with that LeRobot dataset and ran their
real data transforms and spatial encoders. It checked the exact point/force
arrays, masks, model input specs, token counts, and left/right independent
token response for separate-finger configs.

The original premature batch stopped at 43 archives. After the user accepted
the single demonstration and the corrected export/input checks passed, the
remaining replay batch was resumed with `--review-approved`.

## Full conversion checkpoint — stopped before training

- All 145 selected source demonstrations were replayed and captured with both
  sensors; no archive was incomplete. Across all 145 captures, both pads had
  finite, nonzero, nonidentical force fields. Measured pad-center separation
  across frames was 25.6–34.0 mm.
- Twenty-five replays did not satisfy terminal insertion success and are
  listed by source ID in `data/contactworld_usb_v3_bilateral/spatial/manifest.json`.
  They remain in the replay archive and were excluded from the positive
  LeRobot dataset. The accepted dataset has **120 episodes / 8347 frames**:
  train 95 / 6489 frames, val 15 / 1104, test 10 / 754.
- The full exported dataset is `data/contactworld_usb_v3_bilateral` (about
  2.6 GB). `sim/scripts/verify_contactworld_v2_lerobot.py` passed all 120
  episodes against source capture archives: action/state, both clouds, both
  tactile force/position pairs and camera geometry; MP4 codec, frame count,
  sampled RGB-to-capture alignment and preview-to-force alignment. For each
  accepted episode, sampled frames additionally rotate the saved left/right
  base-frame force vectors back through the saved sensor/base quaternions and
  recover the original normal/shear-X/shear-Y channels within 1e-6; this
  checks 3D axis order and signs in the exported artifact.
- `sim/scripts/audit_contactworld_v3_inputs.py` passed all eight V3 configs
  through the actual LeRobot loader, OpenPI transforms and spatial encoder.
  It verified exact sidecar-to-model XYZ/force arrays, left/right taxel IDs,
  the masked third image slot, model input specs and token shapes, and the
  train-only sample index selection. The separate-finger tokens respond
  independently to a left-force intervention.
- Forty-eight episodes were available in both the earlier unmodified-sensor
  V2 archive and the new bilateral replay. Original right TacFF, RGB, both
  clouds, state, action and plug pose were bitwise identical in all 48.
- The [full-dataset review page](/home/sai/zx/openpi-sim-runtime/visualizations/contactworld_bilateral_full_dataset_20261002/index.html)
  reads the exported LeRobot files and overlays the fused point cloud,
  both pad taxel positions and 3D force vectors in robot-base coordinates.
  Vector lengths use an explicitly stated display multiplier because the
  simulator force values have no Newton calibration. Visible cloud-to-taxel
  distances of several millimeters are expected from 4096-point sampling
  and occlusion of the inside pad surfaces.

At the data-review checkpoint, conversion stopped before any training. The
subsequent input-design work is documented in
[`contactworld_v3_training_matrix.md`](contactworld_v3_training_matrix.md).
That work computed train-only state/action normalization statistics and added
13 V3 configurations with structured point/taxel encoders and dedicated tactile
map ViTs. Training has still not started. The older eight-config audit above
records the checks at the conversion checkpoint; the newer 12-config audit is
saved as `contactworld_v3_input_audit.json`.
