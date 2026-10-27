# Deployment parity audit — 2026-10-27 (in progress)

Base: `feat/object-geometry-pipeline@3260d63`. All modifications are on a
separate review branch; no robot SDK or actuator code was touched.

## Verified by comparing committed source

| Stage | Training reference | Deployment path | Finding |
|---|---|---|---|
| SAM2 | `tools/hoi_level1_snapshot/scripts/level1_sam2_multi.py`: native video predictor / reviewed clicks | `scripts/deploy/run_online_perception.py`: image predictor and synthetic re-prompts | **DIVERGED**; new optional video-memory adapter restores native memory calls, but replay validation pending. |
| RGBTrack | `scripts/object_geometry/pose_tracking/rgbtrack_native_upright.py` and `rgbtrack_upright_full.py` | `run_online_perception.py` calls `rgbtrack_native_upright.main(--install-runtime-only)` | Uses training hook installer; verify scorer/refiner weights and exact phase B + causal occlusion recovery on real frames. |
| Mesh sampling | `openpi.object_geometry.production.load_mesh_surface_points` | `OnlineObjectGeometryBuilder.from_meshes` | **SOURCE RESTORED IN PR**: added `src/openpi/policies/hand_scene_online.py`, delegating to training `load_mesh_surface_points`, `_geometry_record`, and `CausalHandSceneState` rather than reimplementing mesh sampling/confidence/source semantics. Does not establish real-scene parity; run the existing online/offline unit tests. |
| RGB-D | `openpi.spatial.SpatialPreprocessor.preprocess` takes native role depth with calibrated depth intrinsics / T_color_depth | replay requires `front_depth_raw` to already have RGB color dimensions; RGBTrack treats it as color-aligned depth with K_color | **FIXED INPUT BOUNDARY IN PR**: require both `front_depth_native` (for canonical `SpatialPreprocessor`) and `front_depth_color_aligned` (for `RGBTrackObject` using K_color). Ambiguous legacy `front_depth_raw` is no longer a substitute. Acquisition must publish two genuinely registered/correct streams; merely duplicating an array is incorrect. |
| Base visual cloud | `openpi.spatial.preprocess.SpatialPreprocessor` | `XHandSpatialOnlinePreprocess` | Same core implementation, but verify camera roles and native-depth contract at the runtime boundary. |
| Tactile | `SpatialPreprocessor.preprocess_tactile` and `CausalTactileEMA` | same in `XHandSpatialOnlinePreprocess`; geometry builder handles additional grasp fields | Check there is *one* EMA update per observation; avoid divergent duplicate filters. |
| Object geometry passed to policy | `spatial.object_geometry` sidecar join in training, consumed by `Observation.from_dict` | `run_policy_dry_run.py` puts geometry at **top level** while `XHandSpatialOnlinePreprocess` omitted it | **CONFIRMED BUG** fixed in this PR: strict canonical schema validation and forwarding to `spatial.object_geometry`, with fail-closed policy config for geometry checkpoints. |
| Actor | trained Pi0 policy output [50,18] | `deploy/ur7e_xhand_deploy_3d_tactile_multi.py` | Existing control path untouched. Current scripts/deploy dry-run only saves actions to disk. Do NOT claim hardware integration. |

## Status and remaining blockers

1. The current PR is draft and **must not be merged or deployed for actuation**. New source and depth-contract edits have not run on the deployment machine.
2. Compare the restored `hand_scene_online.py` against any previously untracked deployment-host implementation; run `scripts/deploy/test_hand_scene_online.py`. Never silently overwrite a divergent local file; verify its intended physics/semantics first.
3. Update replay/camera packet producers to supply *both* `front_depth_native` and `front_depth_color_aligned` in original raw units, obtained through the camera SDK or its verified registration pipeline. Validate in `scripts/deploy/test_depth_contract.py`; do not relabel a color-aligned array as native.
4. Run all new tests plus existing dry-run on the deployment machine. No real hardware and no dataset are available in this editing session.
5. Run Episode 23 with both segmentation backends and visually inspect the masks and 3-D viewer before any robot movement.
6. Verify each of the three checkpoint configs actually routes nonempty geometry tokens to the encoder (not only that action tensors have shape [50,18]).

## Update: source and depth boundary changes (GitHub-only)

The restored online geometry builder reuses the training geometry functions for deterministic surface sampling, object validity, source age and structural confidence. It consumes the shared tactile hand-scene state and causal tactile EMA; it is not an independent learned or calibrated contact estimator. Unknown hand projection remains invalid.

Replay packets now explicitly require two front depth arrays. The old name `front_depth_raw` was semantically overloaded; old packet bundles need conversion from their **original sensor data**, not an automated rename. Their two depth coordinate systems can have different image dimensions.

Neither this PR's new depth regression tests nor the geometry builder parity tests has been executed against real Episode 23 on computer 2. Do not treat a compiled model, finite action shape, or synthetic mask as evidence of trained/online geometric equivalence.
