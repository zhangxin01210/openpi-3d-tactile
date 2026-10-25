Episode 23 / target-flask fixed-bottom-pivot constraint
=======================================================

WHAT CHANGES
- Exactly ZERO modifications to the successful funnel_poses.jsonl: byte-for-byte copy.
- Runs target flask RGBTrack ONLY at frame 0 and selects a top-ranked upright CAD candidate.
- For all 468 frames: keep target bottom pivot fixed; admit small measured tilt; reject improbable
  rotation/pivot drift; suppress symmetric yaw flips. The raw target pose is retained separately
  in the source run and in the CSV audit, not overwritten.
- Outputs a new complete pose run usable by visualize_rgbtrack_full_episode.py.
- The 3D pose in a prior-fallback frame is an approximate STATIC GEOMETRY PRIOR, NOT an observation.

INSTALL
Put rgbtrack_flask_pivot.py and run_e23_flask_pivot.sh alongside existing scripts
in openpi-3d-tactile/scripts/object_geometry/pose_tracking/

RUN
cd /workspace/mnt/sqzhang26/openpi-3d-tactile
bash scripts/object_geometry/pose_tracking/run_e23_flask_pivot.sh

The runner automatically finds the latest successful e23_maskfix_native_upright_* run.
If it picks the wrong one, explicitly set:
SOURCE_POSE_RUN=/.../output/pose_eval/e23_maskfix_native_upright_YYYYMMDD_HHMMSS bash scripts/object_geometry/pose_tracking/run_e23_flask_pivot.sh

OUTPUT
- output/pose_eval/e23_funnel_kept_flask_pivot_*/flask_pivot_report.json
- output/pose_eval/e23_funnel_kept_flask_pivot_*/flask_pivot_qa.csv
- output/hand_scene_qa/e23_funnel_kept_flask_pivot_*/hand_scene_dynamic_e23.html

IMPORTANT
- Run uses the existing SAM3D+RGBTrack Python environment ONLY for target frame-0 registration.
  No full-episode tracking rerun; no new network or model weights.
- Target mouth-axis local sign is inferred by comparing radial mesh widths at ±Z. This is a
  heuristic, and the inferred mouth direction and first-frame projection MUST be reviewed.
  Override with --flask-mouth-axis +z or -z if clearly wrong.
- Fallback pose frames have status flask_pivot_prior_* and confidence 0.08; valid=True means
  a usable approximate geometric prior only, NOT that RGBTrack observed the true 6D pose.
- Constant-pivot is a task-specific approximation for an almost-fixed flask. Do not apply
  it to objects that translate or are freely manipulated.
