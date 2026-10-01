# ContactWorld USB: OpenPI training handoff

The first matrix uses only USB and one training seed. Each model receives the same
117 positive training demonstrations (7,881 source frames), the same train-only
normalization statistics, the same π0 checkpoint and the same 16-action horizon.
The original source IDs are in the dataset `spatial/manifest.json`. Source endpoint
pass is a pose proxy, not an independently measured physical insertion label.

## Transfer and launch

Transfer the code checkout **and the whole** `data/contactworld_usb_positive_all/`
directory. The latter contains LeRobot v2.1 `data/`, `videos/`, `meta/`, the
`spatial/` arrays, and `assets/norm/norm_stats.json`; transfer of the 12 GB
source Zarr, source Isaac Gym runtime, audit CSV or geometry report is not needed
for training. The machine must already have the existing OpenPI dependencies
and π0 base checkpoint. Configs search `checkpoints/pi0_base/params` and the
`/workspace/mnt/sqzhang26/hf_weight/pi0_base/params` path used by existing XHand
configs. Set `OPENPI_CONTACTWORLD_BASE_WEIGHTS` to another existing server path
if needed.

From the repository root on the A800 machine:

```bash
cd data/contactworld_usb_positive_all && sha256sum -c SHA256SUMS --status && cd ../..
uv run --no-sync python sim/scripts/verify_contactworld_lerobot.py data/contactworld_usb_positive_all
uv run --no-sync python sim/scripts/launch_contactworld_matrix.py --label cw_usb_first --gpus 0,1,2,3,4,5,6,7 --check-only
uv run --no-sync python sim/scripts/launch_contactworld_matrix.py --label cw_usb_first --gpus 0,1,2,3,4,5,6,7
```

The launcher checks required inputs and existing output names before starting.
It runs one config per GPU and fills freed GPUs until all 11 are complete. Each
process writes a log under `runs/contactworld_usb_train/cw_usb_first/` and a
checkpoint under `checkpoints/<config>/cw_usb_first/`. W&B is off by default;
add `--wandb` to enable it. To run one config directly:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/train.py \
  pi0_cw_usb_04_rgb_pc_ff --exp-name cw_usb_first --no-wandb-enabled
```

## Matrix

| Config | Observation difference |
| --- | --- |
| `pi0_cw_usb_01_rgb` | front + wrist RGB |
| `pi0_cw_usb_02_rgb_pc` | + camera-visible XYZ point cloud |
| `pi0_cw_usb_03_rgb_ff` | + signed right-sensor force grid |
| `pi0_cw_usb_04_rgb_pc_ff` | + both geometry and force |
| `pi0_cw_usb_05_ff_summary` | force replaced by signed/absolute/RMS means and contact fraction |
| `pi0_cw_usb_06_ff_ee3d_proxy` | EE-centred transformed taxel grid and force; approximate geometry |
| `pi0_cw_usb_07_tacrgb` | tactile RGB in third image slot; no force branch |
| `pi0_cw_usb_08_tacdepth` | tactile depth converted to 8-bit grayscale third image slot; no force branch |
| `pi0_cw_usb_09_prefix` | point and force tokens routed to VLM prefix |
| `pi0_cw_usb_10_split` | point token to VLM prefix; force token to action suffix |
| `pi0_cw_usb_11_pcnoise` | 2 mm Gaussian cloud noise at train-time |

Other spatial configs route their tokens to the action suffix. The force grid
uses source signed units, scaled by 0.002 in the encoder; this scale is a fixed
model convention, not a Newton calibration. The force summary and force grid
use different encoders, so their comparison is a representation-plus-encoder
comparison. The EE3D variant places a 20 mm plane at the recorded EE pose:
it is explicitly a proxy, not calibrated right-pad taxel geometry. The point
cloud was corrected using the fixed local camera geometry report; historical
per-frame camera calibration is unavailable. The LeRobot FPS=10 is an index-time
convention for action-chunk sampling and video playback; the historical
collection frequency was not verified.

The separate ContactWorld policy expects online RGB keys
`observation/front_rgb`, `observation/wrist_rgb`, 18D
`observation/state` (nine joint positions followed by nine velocities), and
optionally `observation/tactile_rgb` or `observation/tactile_depth`. Spatial
variants also expect the `spatial` visual/tactile schema emitted by
`ContactWorldLeRobotSpatial`; online code must reconstruct it from the current
simulator observation, including the same fixed cloud correction and force
conventions. The policy returns the original six action components. This
contract is prepared for later closed-loop evaluation; no learned policy has
yet been run in the simulator.

## Checks completed locally

- Full export: 145 pose-positive USB episodes and 9,977 frames across
  train/val/test, about 1.7 GB. The 117 train episodes contain 7,881 frames.
  The training loader uses only complete 16-step horizons from train episodes
  (6,126 training rows).
- `verify_contactworld_lerobot.py`: every parquet episode, sidecar array and
  H.264/yuv420p video has matching frame count.
- LeRobot API loaded the complete dataset; ContactWorld configs resolved and
  the normalized data loader produced batches for RGB, joint spatial, depth,
  and split-routing cases.
- Dummy-size π0 forward-loss probes completed for all eleven configurations.
  Geometry+force and split-routing variants also completed reverse-mode
  gradients with finite values. The normalized real-data loader produced batches
  for representative RGB, geometry+force, tactile depth and split cases.
- The isolated online policy input transform accepted a current-frame RGB,
  18D joint state and joined spatial sample, producing the expected 32D padded
  state, 224px image and 1024-point spatial input.
- Using an existing local XHand π0 checkpoint as a loader compatibility probe,
  RGB merged 70 parameter leaves; the geometry+force loader found 52 shared
  parameter leaves, 28 intentionally new spatial leaves, zero unexpected
  missing leaves and zero shape mismatches. The server's selected base checkpoint
  has not been inspected here.

The local machine has no accessible copy of the server's base π0 checkpoint.
The first server run should check weight loading before scheduling all 11; the
launcher rejects a missing path, and one direct config can be used as a short
pilot. These checks do not establish closed-loop insertion success.
