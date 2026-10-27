# Hand-Scene online perception and Pi0 dry-run

This deployment path consumes one current observation at a time. The default
SAM2 backend is a version-pinned causal VideoPredictor adapter using memory of
past frames only (same algorithm family as reviewed training Level-1 masks).
The former per-frame image predictor remains available as `legacy_image` for
A/B QA only. Video-memory parity on real data is **not yet validated**.
Two object tracks remain independent: `funnel` and `target_flask`.

Current test evidence and the remaining checkpoint/hardware blockers are listed
in [validation_report.md](validation_report.md).

The policy process stays in the repository's existing `.venv` (JAX GPU). Vision
dependencies belong to `.venv-perception`, which inherits the already tested
`lerobot5090` Torch 2.10+cu128 installation read-only and adds missing packages
to the project overlay. Neither the OpenPI environment nor the base Conda
environment is modified.

## Audit and setup

Run the read-only host and dependency audit:

```bash
.venv/bin/python scripts/deploy/check_environment.py
.venv/bin/python scripts/deploy/check_assets.py
```

Prepare the perception overlay and pin the SAM2 source checkout:

```bash
bash scripts/deploy/setup_perception.sh
```

The audit report is written to `docs/deploy/environment_audit.md` and
`docs/deploy/environment_audit.json`. The scale manifest is explicit and
content-addressed; `check_assets.py` checks the mesh SHA-256 and verifies that
the CAD bounds agree with the recorded millimetre dimensions.

Download SAM2.1 Small from the official checkpoint URL and record its byte
count and SHA-256:

```bash
.venv/bin/python scripts/deploy/download_weights.py --download-sam2
```

ModelScope's public search endpoint tested in this setup returned HTTP 404 and
the search page returned only its JavaScript shell, so no model ID or converted
checkpoint was assumed. RGBTrack's project-specific scorer/refiner weights
were not present locally. When the verified computer-4 directory is mounted,
copy only the two expected files and verify their hashes:

```bash
.venv/bin/python scripts/deploy/download_weights.py \
  --rgbtrack-source-dir /path/to/computer4/checkpoints/rgbtrack
```

The required relative files are `2024-01-11-20-02-45/model_best.pth` and
`2023-10-28-18-33-37/model_best.pth`. This command refuses to overwrite them.

Build the RGBTrack extensions in the same isolated perception environment:

```bash
bash scripts/deploy/setup_rgbtrack.sh
```

This pins and builds NVDiffRast and PyTorch3D for `sm_120`, installs the
runtime imports used by RGBTrack, builds the local `mycpp` extension, and runs
phases A/C. It skips Phase B with an explicit `BLOCKED` message until both
scorer/refiner files are present. It does not install the upstream requirements
file or change the OpenPI `.venv`.

## Causal input packet

Replay uses a directory containing `frame_000000.npz`, `frame_000001.npz`, … .
Each packet contains only one frame and these arrays:

| Key | Contract |
| --- | --- |
| `front_rgb` | `uint8 [H,W,3]`, RGB order |
| `left_rgb`, `right_rgb` | `uint8 [H,W,3]`, Pi0 image slots |
| `front_depth_native` | numeric `[Hd,Wd]`, original depth-optical pixels and native units; passed to canonical training `SpatialPreprocessor` |
| `front_depth_color_aligned` | numeric `[Hc,Wc]` on front RGB pixels, original sensor units; passed only to RGBTrack (K_color) |
| `robot_state` | finite `float32 [1972]`, existing UR7e + XHand state contract |
| `frame_index`, `timestamp_s` | monotonic current-frame identifiers |
| `tactile_force_norm` | optional finite `[600]`; if omitted, the existing spatial tactile preprocessor derives it from `robot_state` |
| `funnel_seed_mask`, `target_flask_seed_mask` | optional full-resolution boolean masks on the configured SAM2 seed frame |

Both objects need a manual seed prompt. Example `data/deploy/seed_prompts.json`:

```json
{
  "seed_frame": 0,
  "init_frames": {"funnel": 0, "target_flask": 0},
  "objects": {
    "funnel": {"box_xyxy": [120, 80, 360, 410]},
    "target_flask": {"positive": [[460, 250]], "negative": [[300, 240]]}
  }
}
```

Coordinates use the front RGB pixels. A seed mask array in that frame packet
can replace the corresponding box/points. Later corrections use
`manual_corrections.json` keyed by frame number, for example
`{"120":{"funnel":{"box_xyxy":[... ]}}}`. A correction resets that object's
RGBTrack instance on that current frame. Implausible area, large frame-to-frame
area jumps, and object-mask overlap are marked invalid and shown in the overlay.

## Run causal replay or a live packet bridge

```bash
PYTHONPATH=src .venv-perception/bin/python scripts/deploy/run_online_perception.py \
  --replay-dir data/deploy/replay \
  --output-dir output/deploy/hand_scene_online \
  --prompts data/deploy/seed_prompts.json \
  --masks-only
```

`--masks-only` runs SAM2 and emits the correct fixed-size but invalid object
geometry when RGBTrack is not yet installable. Full pose tracking removes that
flag and uses the two independent RGBTrack objects. The script requires the
front calibration, BOTH native and color-aligned depth, explicit metric scale manifest,
RGBTrack dependencies, and both checkpoint files. It converts the meshes
through the existing `rgbtrack_adapter.py`; the adapter applies RGBTrack's
mesh-centering compensation once and composes
`T_base_link_object = T_base_color @ T_color_object`.

For a live camera/robot SDK bridge, write each complete packet to a temporary
name and atomically rename it to the next `frame_*.npz`, then add `--follow`.
The repository does not contain the UR/XHand camera SDK capture client, so the
packet writer remains the hardware-specific boundary. `right_rgb` is kept for
the trained Pi0 image slot and is never used for geometry. No action is sent
to the robot.

Perception and policy can run as separate processes over the enriched NPZ
directory. Start the dry-run policy in a second terminal before the perception
writer if it should wait for the first frame:

```bash
PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
  scripts/deploy/run_policy_dry_run.py \
  --frames-dir output/deploy/hand_scene_online/frames \
  --config pi0_xhand_spatial_structured_suffix_hand_scene_oo_v1 \
  --follow --output-dir output/deploy/pi0_m2_dry_run
```

Perception publishes each enriched packet with an atomic rename. The policy
watcher consumes completed current-frame packets and writes actions to disk;
it has no robot command transport.

Outputs include per-object binary masks, an RGB overlay with contour, pixel
area, validity and recovery status, per-frame NPZ packets with nested
`spatial.object_geometry` fields flattened under `object_geometry__...`, a
causal manifest, and calibration/weight provenance. A funnel mask-hand offset
is invalid unless a calibrated projector supplies it; this is deliberate and
matches the training producer's invalid representation.

## Pi0 dry-run

The three local trained checkpoints use an Orbax step-root layout, with
`params` and the `1018_peg_all` normalization stats at the step root. The
policy loader accepts this layout as well as the standard `params/` and
`assets/` directories. No checkpoint is synthesized from Pi0 Base.

```bash
PYTHONPATH=src .venv/bin/python scripts/deploy/run_policy_dry_run.py \
  --frames-dir output/deploy/hand_scene_online/frames \
  --config pi0_xhand_spatial_structured_suffix_hand_scene_oo_v1 \
  --output-dir output/deploy/pi0_m2_dry_run
```

The script runs the normal policy transforms and trained model, checks the
actual configured action horizon and 18 returned XHand joints, records finite
actions and inference latency, and hard-codes `robot_command_sent: false`.
The current three checkpoints use a 50-step action horizon, so output is
`[50,18]`; the model's padded internal action dimension remains 32. M2-OC
preserves all three ordinary Pi0 RGB slots while its spatial scene branch uses
object-centric geometry.

Compare online packets with an existing training Sidecar, when one is
available:

```bash
PYTHONPATH=src .venv/bin/python scripts/deploy/compare_online_offline.py \
  --online-frames output/deploy/hand_scene_online/frames \
  --sidecar-root data/1018_peg_all \
  --version v1_hand_scene_q2 --episode 23 \
  --output output/deploy/online_offline_comparison.json
```

The comparison reports `NOT_TESTED` and exits nonzero if the Sidecar is absent.
It does not treat newly estimated RGBTrack poses as geometry-parity evidence.

## RGBTrack runtime phases

Use the pinned submodule and run phases separately in the perception overlay:

```bash
PYTHONPATH=src .venv-perception/bin/python scripts/object_geometry/pose_tracking/check_rgbtrack_runtime.py \
  --repo-root "$PWD" --weights-root "$PWD/checkpoints/rgbtrack" --phase A
```

Phase A checks imports, including importing upstream `FoundationPose`, and
reports the weight files. It does not prove checkpoint-backed tracking. Phase B
also initializes the estimator and requires both trained RGBTrack weights and
the C++ `mycpp.cluster_poses` extension. Phase C runs a NVDiffRast CUDA render
on `sm_120`.


## Important 2026-10-27 parity correction

The original replay key `front_depth_raw` was ambiguous and is rejected by the new contract. The source camera/replay writer must produce *two distinct depth products*: `front_depth_native` (original depth optical pixel grid, used by the canonical spatial visual encoder) and `front_depth_color_aligned` (registered to color pixels, used by RGBTrack with color intrinsics). Do not rename or duplicate the old aligned array to invent native depth. Verify native intrinsics, registered alignment, scale, and `T_color_depth` against the training calibration. The current packet reader requires both, even in mask-only mode, to avoid silently building invalid geometry downstream.

`src/openpi/policies/hand_scene_online.py` now exists in the draft PR and delegates to the existing training-side deterministic mesh sampler, object geometry record producer and causal Hand-Scene state. Run `scripts/deploy/test_hand_scene_online.py` and `scripts/deploy/test_depth_contract.py` before moving to Episode 23 replay. The restored source and packet changes are not real-world validated.
