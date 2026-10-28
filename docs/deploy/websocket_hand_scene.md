# Six scene configurations selected on robot (one persistent Pi0 WebSocket)

Robot host continues to run ONE standalone script. Inference host runs the
original WebsocketPolicyServer port 8990. The robot sets
--scene-position [1..6]; no server restart when switching runs.

| Robot --scene-position | Training reference | Flask 2D points | Funnel 2D points |
| --- | --- | --- | --- |
| 1 | E0 | E0 | E23 |
| 2 | E23 | E23 | E23 |
| 3 | E38 | E38 | E38 |
| 4 | E49 | E49 | E49 |
| 5 | E58 | E58 | E58 |
| 6 | E65 | E65 | E65 |

**Only the previously hand-clicked positive/negative 2D points are reused.**
The current camera image is segmented LIVE, the static flask pose is
estimated on the CURRENT first frame by RGBTrack, and is then held for the
rest of the run. Reference episode 0 has no funnel points, so scene 1 uses
the funnel clicks from Episode 23.

The preview images below include reference training first-frame pictures
only to help identify each selection; none of those pictures or pre-recorded
object 6D poses are read by live Pi0 inference.

## 0. Create six reference previews once (sai)

~~~bash
cd ~/zx/openpi-3d-tactile-pr1
OLD="$HOME/zx/openpi-3d-tactile"
PYTHONPATH="$PWD/src:$PWD" "$OLD/.venv-perception/bin/python" \
 scripts/deploy/render_scene_position_previews.py \
 --dataset "$OLD/data/1018_peg_all" \
 --output-dir "$PWD/output/deploy/scene_position_previews_v1"
~~~

Find output/deploy/scene_position_previews_v1/scene_positions_overview.png,
the six individual scene_position_N_reference_points.png files and the
scene_positions_manifest.json for provenance.

## 1. Start server ONCE on sai

~~~bash
cd ~/zx/openpi-3d-tactile-pr1
OLD="$HOME/zx/openpi-3d-tactile"
PYTHONPATH="$PWD/src:$PWD" "$OLD/.venv/bin/python" -u \
 scripts/serve_hand_scene_policy.py \
 --repo-root "$PWD" --port 8990 \
 --checkpoint "$OLD/checkpoints/pi0_xhand_spatial_structured_suffix_hand_scene_oo_v1/m2_b8_20k_1026/19999" \
 --perception-python "$OLD/.venv-perception/bin/python" \
 --workdir "$PWD/output/deploy/qa/websocket_scene_six_positions_001" \
 --sam2-checkpoint "$OLD/checkpoints/sam2/sam2.1_hiera_small.pt" \
 --rgbtrack-weights "$OLD/checkpoints/rgbtrack" \
 --scale-manifest "$OLD/data/pose_eval/hand_scene_metric_mesh_scales.json"
~~~

No fixed anchor, no fixed template episode on server. Per-session CURRENT
frames, SAM2 masks and RGBTrack overlays are saved under the server workdir
in distinct scene_N_session_NNNN directories.

## 2. amax: run robot client with a remembered position number

Back up and replace ONLY the existing robot-side standalone
scripts/ur7e_xhand/ur7e_xhand_deploy_3d_tactile_multi.py.

~~~bash
cd ~/code/lerobot
sudo -E "$(which python)" \
 scripts/ur7e_xhand/ur7e_xhand_deploy_3d_tactile_multi.py \
 --dataset-dir data/1018_peg_all \
 --server-ip SAI_LAN_IP --server-port 8990 \
 --scene-position 1 \
 --dry-run --no-home --no-tactile-reset \
 --run-mode single --duration 60 --max-action-chunk-size 16
~~~

To switch scenes, change ONLY --scene-position 1 to another number 2..6,
and rerun the robot client. The sai server stays up and the model weights
remain loaded; the old causal SAM2/RGBTrack perception session is reset.

This integration is currently DRY-RUN ONLY. Perception may block for seconds;
physical actions with glass are not authorized without measured latency,
independent hardware protection and actual verified stopping behavior.
