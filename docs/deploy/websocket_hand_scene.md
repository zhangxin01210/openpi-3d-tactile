# True original WebSocket flow: robot -> SAM2/RGBTrack -> trained Pi0

The robot uses the SAME single-file UR7e/XHand script; the inference
computer runs the existing WebsocketPolicyServer with a small wrapper.
There is **NO HTTP sensor receiver** and no new communication protocol.
Geometry uses the SAME reviewed online-perception code as the Episode 23
64-frame replay with the correct task prompt.

CURRENT MODE: **dry-run only**. The blocking perception worker may take
seconds for a 16-frame batch; it is NOT yet a real-time safe servo controller.
Robot action sending is explicitly rejected in the client for this server.

## A. sai GPU / inference computer: start Pi0 WebSocket server

Use a NEW output directory, correct preinstalled checkpoint paths, and
the static target anchor already reviewed for the SAME current physical
scene. Episode 23's previously reviewed anchor is valid ONLY if the scene
and camera have not moved. If not, review the new frame-0 anchor.

~~~bash
cd ~/zx/openpi-3d-tactile-pr1
git fetch origin fix/sam2-video-memory-parity-20261027
git switch --detach origin/fix/sam2-video-memory-parity-20261027
OLD="$HOME/zx/openpi-3d-tactile"

# This anchor was already reviewed for offline Episode 23; do not reuse it
# for a relocated flask/camera.
ANCHOR="$PWD/output/deploy/qa/e023_action_compare_v1/reviewed_episode23_flask_anchor.json"

PYTHONPATH="$PWD/src:$PWD" "$OLD/.venv/bin/python" -u \
 scripts/serve_hand_scene_policy.py \
 --repo-root "$PWD" --port 8990 \
 --config pi0_xhand_spatial_structured_suffix_hand_scene_oo_v1 \
 --checkpoint "$OLD/checkpoints/pi0_xhand_spatial_structured_suffix_hand_scene_oo_v1/m2_b8_20k_1026/19999" \
 --perception-python "$OLD/.venv-perception/bin/python" \
 --workdir "$PWD/output/deploy/qa/websocket_scene_dryrun_001" \
 --static-target-anchor-json "$ANCHOR" \
 --episode 23 --position-template-episode 23 \
 --sam2-checkpoint "$OLD/checkpoints/sam2/sam2.1_hiera_small.pt" \
 --rgbtrack-weights "$OLD/checkpoints/rgbtrack" \
 --scale-manifest "$OLD/data/pose_eval/hand_scene_metric_mesh_scales.json"
~~~

Keep the terminal running. This launches the REAL trained Pi0 and the
ORIGINAL WebsocketPolicyServer at port 8990. On the first robot request,
it starts a separate causal worker in .venv-perception that reuses the
EXACT run_online_perception.py (video-memory SAM2, RGBTrack, static flask,
training geometry, native depth and color-alignment contract).

## B. amax robot computer: replace only one standalone control .py

Back up the existing script and copy only the updated
deploy/ur7e_xhand_deploy_3d_tactile_multi.py into
~/code/lerobot/scripts/ur7e_xhand/ur7e_xhand_deploy_3d_tactile_multi.py.

Connect to sai's WebSocket port 8990 (NOT the old HTTP 8765 port), with a
**no-motion** run; no mechanical homing, no tactile zeroing and no actions:

~~~bash
cd ~/code/lerobot
sudo -E "$(which python)" \
 scripts/ur7e_xhand/ur7e_xhand_deploy_3d_tactile_multi.py \
 --dataset-dir data/1018_peg_all \
 --server-ip 192.168.10.5 --server-port 8990 \
 --spatial-cameras front \
 --dry-run --no-home --no-tactile-reset \
 --run-mode single --duration 12 \
 --max-action-chunk-size 16
~~~

This sends the original RGB/state/front depth to the existing WebSocket
server. When the server advertises hand_scene_geometry_service=true,
the client includes all intermediate raw frames since the previous Pi0
query in hand_scene_history. The SERVER (not the robot) constructs
object_geometry and calls trained policy.infer. Pi0 returns normal 18D
joint targets, shown as [dry-run would_send], but NEVER executed.

The first request may take much longer due to JAX model compilation.
Each later request consumes multiple SAM2/RGBTrack frames and so may
still take seconds. That is NOT a 15 Hz motion loop. Stop after first
meaningful Pi0 outputs; do not remove --dry-run.

## Errors and stage boundaries

- A static flask reference from Episode 23 is NOT approval for another
  physical pose. A mismatch intentionally blocks geometric inference.
- The worker writes internal raw/perception NPZ under the server workdir to
  call the VERIFIED training-parity implementation, NOT as a cross-host
  HTTP transfer. Logs: workdir/perception.log.
- It requires sequential frame history and exact robot state/front RGB/depth
  matching the most recent geometry frame. It never fabricates missing
  hand-scene geometry.
- The wrist image is retained as one of Pi0's three trained RGB slots,
  but is excluded from pose estimation and point-cloud geometry.
- A second robot session requires a server restart, avoiding carryover of
  SAM2/RGBTrack/tactile EMA states.
- Physical glass actuation remains blocked until independent calibrated
  protection, bounded inference latency and measured robot stop behavior
  have been verified. The correct 64-frame action replay is a SOFTWARE
  correctness result, not evidence of safe contact force.
