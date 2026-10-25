Hand-Scene cross Episode trial, 2026-10-25

Copy the two scripts into the EXISTING repository directory:
  /workspace/mnt/sqzhang26/openpi-3d-tactile/scripts/object_geometry/pose_tracking/

Dependencies: all earlier e23 scripts must be in that directory.

First check existing masks (no GPU):
  bash scripts/object_geometry/pose_tracking/run_cross_episode_hand_scene.sh 38 check
  bash scripts/object_geometry/pose_tracking/run_cross_episode_hand_scene.sh 65 check

Then run each: replace check with run. Do not overwrite Episode 23 results.
Inspect first frame, recovery, insertion, release. Other position groups: 49, 58, 0.
