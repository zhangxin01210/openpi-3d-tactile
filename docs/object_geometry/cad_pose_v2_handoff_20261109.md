# RGB-D Single-CAD Pose Transformer V2 Handoff

Date: 2026-11-09
Branch: feat/unified-deployment

## Model

RGB-D + one CAD (512 points) -> one 6D pose.

One shared model for four XHand tasks.
Each frame uses two independent forward passes:
manipulated CAD and target CAD.

Training implementation:
cad_pose_rgbd_single_v2.py

Cloud checkpoint:
output/cad_pose_rgbd_v2/train_20261109_162628/best.pt

## Evaluation

V1 overall translation error: 35.84 mm
V2 overall translation error: 18.696 mm
V2 overall ADD: 37.379 mm
V2 overall wrong-object-closer rate: 3.5%

V2 task translation errors:
- 1018_peg_all: 41.912 mm
- 1028_press_button: 9.080 mm
- 1030_yellow_socket: 7.370 mm
- 1102_eraser: 12.070 mm

Some legacy GT poses are visibly incorrect.
Human inspection of full-episode HTML indicates V2
predictions are generally usable.

Rotation around symmetric object axes may be unobservable.
Raw rotation error is not the sole acceptance criterion.

## Full Geometry Export

Script:
v2_geometry_full_export.py

Cloud output:
output/cad_pose_rgbd_v2/geometry_full_candidate/

Completed and verified:
328 episodes, 112939 frames.

Task coverage:
- 1018_peg_all: 70 episodes
- 1028_press_button: 88 episodes (cleaned dataset)
- 1030_yellow_socket: 111 episodes
- 1102_eraser: 59 episodes

Each frame contains predicted manipulated and target poses
and two corresponding 512-point CAD geometries.

Original object_geometry/v1 is unchanged.

Prediction validity means finite model output.
It does not prove visibility or detection confidence.

## Integration Status

Candidate reader smoke: PASS on four tasks.

Real pi0 DataLoader monkeypatch smoke:
Button PASS, 3 sampled indices.
Socket was interrupted before completion.

Existing Codex V1 integration:
src/openpi/cad_pose/sidecar.py
scripts/cad_pose/export_sidecar.py
src/openpi/training/data_loader.py
src/openpi/training/config.py
src/openpi/training/checkpoints.py

The existing official integration checks V1-specific
checkpoint and sidecar provenance.

## Next Task

Extend the existing official sidecar and DataLoader
mechanism to accept the V2 candidate geometry.

Reuse existing code. Preserve V1 behavior.

Do not modify pi0 architecture, tactile encoder,
scene encoder, HO/OO encoder, state/action or
normalization.

Do not overwrite legacy geometry, checkpoints,
or existing manifests.

Do not use GT poses or GT visibility masks to
construct V2 predicted geometry.

Do not commit generated datasets or model weights.

First validate native Sidecar publication and real
four-task DataLoader loading, then train pi0.
