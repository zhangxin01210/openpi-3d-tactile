# Two-host live LeRobot -> trained depth geometry -> Pi0 dry-run

The ROBOT receives exactly ONE modified control script. The inference host
receives 48 original LeRobot front depth/three RGB/state frames, makes its own
color-registered depth using the SAME rgbtrack_adapter._rgb_registered_depth
as training, and reuses run_online_perception + run_policy_dry_run.
NO real robot actions, no homing, no tactile zeroing.

## 1. On inference host (sai)

Fetch PR #1, test locally:

~~~bash
cd ~/zx/openpi-3d-tactile-pr1
git fetch origin fix/sam2-video-memory-parity-20261027
git switch --detach origin/fix/sam2-video-memory-parity-20261027
OLD="$HOME/zx/openpi-3d-tactile"
PYTHONPATH="$HOME/.cache/openpi-pr1-audit-packages:$PWD/src:$PWD" \
 "$OLD/.venv/bin/python" -m pytest -q \
 scripts/deploy/test_receive_live_packets.py \
 deploy/live_observation_audit_test.py \
 deploy/standalone_client_test.py \
 deploy/contact_safety_test.py \
 scripts/deploy/test_depth_contract.py
~~~

In a separate terminal start the authenticated HTTP packet receiver, 
replacing the token with a long random one shared only on the robot LAN:

~~~bash
cd ~/zx/openpi-3d-tactile-pr1
OLD="$HOME/zx/openpi-3d-tactile"
export OPENPI_LIVE_PACKET_TOKEN="REPLACE_WITH_LONG_SHARED_SECRET"
PYTHONPATH="$PWD/src:$PWD:$PWD/scripts/object_geometry/pose_tracking" \
 "$OLD/.venv-perception/bin/python" \
 scripts/deploy/receive_live_packets.py \
 --output-dir "$PWD/output/deploy/live_e23_raw_001" \
 --front-calibration-profile "$PWD/configs/ur7e_xhand/front_calibration_candidate.json" \
 --bind 0.0.0.0 --port 8765 --max-frames 48
~~~

The receiver calls the existing training RGBTrack depth registration. Native
raw uint16 depth stays unchanged and is also supplied to the exact
SpatialPreprocessor. It creates two DISTINCT image-coordinate depth arrays.
Never replace native by aligned or align twice.

## 2. On robot host (amax)

Back up and replace ONLY the existing
scripts/ur7e_xhand/ur7e_xhand_deploy_3d_tactile_multi.py from PR #1.

~~~bash
cd ~/code/lerobot
export OPENPI_LIVE_PACKET_TOKEN="REPLACE_WITH_THE_SAME_SHARED_SECRET"
sudo -E "$(which python)" \
 scripts/ur7e_xhand/ur7e_xhand_deploy_3d_tactile_multi.py \
 --dataset-dir data/1018_peg_all \
 --live-packet-url http://SAI_PRIVATE_IP:8765/frame \
 --live-packet-frames 48 --spatial-cameras front
~~~

This branch DOES NOT connect to PI0 websocket, reset/home the robot,
or issue any control action. It sends the original LeRobot
observation.depths.cam_front input field in its uint16 form and original
1972-dimensional state, as recorded by the training dataset.

The GPU receiver must show frames 0..47 received.

## 3. Inference host: original training-equivalent perception

Start only after first frame exists, and only with prompts and a reviewed
static flask anchor for the SAME current scene. If the flask position has
changed, the reviewed Episode 23 base pose is NOT valid and a new reviewed
anchor is required before full pose processing.

~~~bash
cd ~/zx/openpi-3d-tactile-pr1
OLD="$HOME/zx/openpi-3d-tactile"
RAW="$PWD/output/deploy/live_e23_raw_001"
OUT="$PWD/output/deploy/live_e23_perception_001"
PATH="$OLD/.venv-perception/bin:$PATH" \
PYTHONPATH="$PWD/src:$PWD:$PWD/scripts/object_geometry/pose_tracking" \
 "$OLD/.venv-perception/bin/python" -u \
 scripts/deploy/run_online_perception.py \
 --repo-root "$PWD" --replay-dir "$RAW" --output-dir "$OUT" \
 --sam2-checkpoint "$OLD/checkpoints/sam2/sam2.1_hiera_small.pt" \
 --weights-root "$OLD/checkpoints/rgbtrack" \
 --scale-manifest "$OLD/data/pose_eval/hand_scene_metric_mesh_scales.json" \
 --front-calibration-profile "$PWD/configs/ur7e_xhand/front_calibration_candidate.json" \
 --episode 23 --position-template-episode 23 \
 --template-reviewed-for-current-scene \
 --static-target-anchor-json /ABS/PATH/TO/CURRENT_SCENE_REVIEWED_ANCHOR.json \
 --static-target-anchor-source human_reviewed \
 --follow --max-frames 48
~~~

The SAM2 package source is resolved from the existing installed default
(~/.cache/openpi_deploy/sam2); use the actual deployment checkpoint paths
if they differ. Do not download model weights again merely for this command.

## 4. Inference host: original trained policy dry run

~~~bash
cd ~/zx/openpi-3d-tactile-pr1
OLD="$HOME/zx/openpi-3d-tactile"
PYTHONPATH="$PWD/src:$PWD" \
 "$OLD/.venv/bin/python" -u \
 scripts/deploy/run_policy_dry_run.py \
 --repo-root "$PWD" \
 --frames-dir "$PWD/output/deploy/live_e23_perception_001/frames" \
 --output-dir "$PWD/output/deploy/live_e23_policy_dryrun_001" \
 --max-frames 4 --follow --idle-timeout-s 60
~~~

Existing output: actions.jsonl, actions_NNNNNN.npz, summary.json.
No robot action is sent in this stage. The robot-side wrist RGB is only
retained for trained raw RGB schema; NO wrist depth or uncalibrated geometry
enters the new front spatial point-cloud path.

This file-based bridge is for real sensor and model dry-run validation; it
is NOT a deterministic real-time control loop. Before actual glass motion,
hardware protective stop, calibrated thresholds and true stopping behavior
need separate validation.
