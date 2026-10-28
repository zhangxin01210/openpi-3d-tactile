# Live sensor discovery, WITHOUT PI0 commands — staged robot-host validation

Current status: GitHub source prepared; no actual UR7e/XHand/RealSense SDK
observation or camera registration verified in this development session.

## Why this gate precedes the online perception bridge

The established training geometry preprocessor consumes NATIVE front depth
pixels and the calibrated depth intrinsics, while RGBTrack consumes DEPTH
REGISTERED TO FRONT RGB and the color intrinsics. The older LeRobot client
`get_depth()` does not identify which coordinate system its output represents.
Identical 480x640 image sizes are **not** evidence of identical pixel frames.
The 1972-dim state includes 12 XHand joint fields labelled torque (SDK
actually exposes motor current in mA), NOT UR arm joint torque. The current
single-file client enables UR RTDE TCP force/torque in UR7eConfig; its observed
`arm_tcp_force` (3) and `arm_tcp_torque` (3) values are reported separately.
The physical reference frame, bias and freshness remain unverified.

## Hardware-host read-only discovery (operator present)

The deployment host runs the EXISTING standalone LeRobot client. Copy
ONLY the updated `deploy/ur7e_xhand_deploy_3d_tactile_multi.py` from PR #1
to `~/code/lerobot/scripts/ur7e_xhand/ur7e_xhand_deploy_3d_tactile_multi.py`.
Keep a backup of the robot host's previous script. No `contact_safety.py` or
other openpi repository modules are needed for this read-only mode.

Run in the robot machine's **existing** LeRobot / EtherCAT Python environment.
The `--dataset-dir` supplies the original `meta/info.json` feature ordering:

```bash
cd ~/code/lerobot
sudo -E "$(which python)" \
  scripts/ur7e_xhand/ur7e_xhand_deploy_3d_tactile_multi.py \
  --dataset-dir data/1018_peg_all \
  --capture-observation-audit output/deploy/live_sensor_audit_001 \
  --capture-audit-frames 5 \
  --spatial-cameras front
```

Do **not** use `--check-config` together with audit. The audit branch is
executed before any WebSocket inference connection or policy-control branch;
it NEVER calls `reset_to_home()`, tactile zero, `send_action()` or policy
inference. It connects SDK devices and reads observations, then disconnects.
Observe robot safety precautions nonetheless; use no glass or fragile objects.

Outputs are in a NEW directory:
- `observation_schema.json`: all top-level and arm-observation field names,
  array shapes/dtypes, possible force/torque key hints, acquisition monotonic
  times, raw XHand peak diagnostics and status.
- `audit_0000.npz`, ...: 5 RGB samples, raw 1972 vector, and any available
  *unverified* front-depth array.
- The depth candidate key is deliberately named
  `front_depth_coordinate_system_UNVERIFIED`. These files do NOT have a
  `front_depth_native` field and are **NOT** valid input to
  `run_online_perception.py`.

After an audit, confirm from the real camera driver source/SDK:
- Native depth pixel frame and depth intrinsics; whether hardware/SDK has
  already aligned/filtered depth to RGB.
- Registered-to-RGB depth construction using EXACT shared calibration,
  depth scale in original sensor units, camera timestamps and latency.
- The enabled `arm_tcp_force` / `arm_tcp_torque` RTDE output shapes, units,
  reference frame, timestamp freshness and whether it is filtered/bias-corrected.
- The XHand SDK read success/error, last-good-value fallback and sample
  freshness at the driver layer: first five samples had **all 600 taxels
  exactly zero**, which cannot establish a working sensor.
- `hand_joint_i.torque` is sourced from motor current (mA), not an arm torque.
- The exact state ordering, simultaneous RGB slots, no uncalibrated wrist
  camera in 3D preprocessing, and independent hardware safety stop.

Do not make a live policy packet adapter until the native-vs-aligned depth
contract is verified; otherwise a visually plausible point cloud can
produce false hand–object relationships. After the data boundary is confirmed,
reuse `_rgb_registered_depth` and the existing
`run_online_perception.py --follow` plus
`run_policy_dry_run.py --follow` exactly as in the replay chain.

The control client's existing synchronous inference and 15 Hz loop remain
insufficient to guarantee detection of impacts between sensor samples.
No glass actuation is authorized by this audit.

## Why a NEW audit is needed

The October 28 five-frame first audit showed three RGB cameras, 1972 state,
front uint16 depth, but no UR wrench because the deployment script had not
set `record_tcp_force_torque=True`. It has now been set by the single-file
client, and its safety guard explicitly reads **separate 3D force and 3D
torque** rather than inventing a 6D key. Use a new audit output directory,
confirm both measured fields, then investigate XHand raw zero/fallback and
front RGB-vs-depth geometry BEFORE wiring the policy or allowing motion.


## Follow-up read-only tactile response check

The 5-frame audit found nonzero measured UR TCP wrench values (roughly 8 N)
but ALL FIVE XHand raw arrays were zero. This does not prove an SDK defect:
a truly unloaded array may be all zero. To distinguish nonresponsive data
from genuine no-contact readings, the standalone robot script now reports
per-finger nonzero taxel counts, calculated-force norms and whether raw taxel
values changed relative to the previous frame. **No controls or tactile
sensor zeroing are issued by this audit.**

With the robot stationary and glass removed, keep a trained operator present.
Acquire 90 frames (~6 s at 15 FPS) while gently touching/releasing ONE
fingerpad with a soft, non-fragile material, leaving other fingers free.
Never touch near joints, squeeze the hand, or allow the robot to move.

```bash
sudo -E "$(which python)" \
 scripts/ur7e_xhand/ur7e_xhand_deploy_3d_tactile_multi.py \
 --dataset-dir data/1018_peg_all \
 --capture-observation-audit output/deploy/live_sensor_audit_004 \
 --capture-audit-frames 90 --spatial-cameras front
```

If no finger ever changes or nonzero taxels remain zero during deliberate
gentle contact, block live control and inspect XHand SDK read_state error /
last-good-value fallback, actual EtherCAT read success and sensor wiring.
Unchanging samples do not establish a healthy XHand sensor, and no configured
tactile threshold can fix unavailable sensor signals. The report's SDK
freshness flag stays FALSE; only a real driver-level per-read health/time
signal can establish freshness. The report includes the exact 1972-state
start-name for each raw tactile block so that training field ordering can
be checked against metadata before doing anything else.

Depth remains unverified: obtain the robot-host RealSense driver's source
for RGB vs depth alignment, the actual depth scale, the intrinsic/extrinsic
tables and any filtering/alignment calls. Do not rename this `uint16`
candidate as native depth merely because its shape matches RGB. Keep the
two-stage training+RGBTrack depth contracts separate.
