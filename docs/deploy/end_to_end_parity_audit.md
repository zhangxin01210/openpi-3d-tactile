# Deployment parity audit — 2026-10-27 (in progress)

Base: `feat/object-geometry-pipeline@3260d63`. All modifications are on a
separate review branch; no robot SDK or actuator code was touched.

## Verified by comparing committed source

| Stage | Training reference | Deployment path | Finding |
|---|---|---|---|
| SAM2 | `tools/hoi_level1_snapshot/scripts/level1_sam2_multi.py`: native video predictor / reviewed clicks | `scripts/deploy/run_online_perception.py`: image predictor and synthetic re-prompts | **DIVERGED**; new optional video-memory adapter restores native memory calls, but replay validation pending. |
| RGBTrack | `scripts/object_geometry/pose_tracking/rgbtrack_native_upright.py` and `rgbtrack_upright_full.py` | `run_online_perception.py` calls `rgbtrack_native_upright.main(--install-runtime-only)` | Uses training hook installer; verify scorer/refiner weights and exact phase B + causal occlusion recovery on real frames. |
| Mesh sampling | `openpi.object_geometry.production.load_mesh_surface_points` | `OnlineObjectGeometryBuilder.from_meshes` | **BLOCKED**: `src/openpi/policies/hand_scene_online.py` is absent from **Git tracked tree** at 3260d63. The current deployment script and its tests import it. This may exist only as an untracked local file on a deployment host; cannot certify or reimplement its original behavior without seeing that source. |
| RGB-D | `openpi.spatial.SpatialPreprocessor.preprocess` takes native role depth with calibrated depth intrinsics / T_color_depth | replay requires `front_depth_raw` to already have RGB color dimensions; RGBTrack treats it as color-aligned depth with K_color | **SEMANTIC SPLIT**: a depth array aligned to front color is NOT a native depth frame. Do not pass it to canonical native-depth unprojection without explicit conversion, matching intrinsics, and units. The replay packet key `front_depth_raw` is misleading and must be made explicit before parity is claimed. |
| Base visual cloud | `openpi.spatial.preprocess.SpatialPreprocessor` | `XHandSpatialOnlinePreprocess` | Same core implementation, but verify camera roles and native-depth contract at the runtime boundary. |
| Tactile | `SpatialPreprocessor.preprocess_tactile` and `CausalTactileEMA` | same in `XHandSpatialOnlinePreprocess`; geometry builder handles additional grasp fields | Check there is *one* EMA update per observation; avoid divergent duplicate filters. |
| Object geometry passed to policy | `spatial.object_geometry` sidecar join in training, consumed by `Observation.from_dict` | `run_policy_dry_run.py` puts geometry at **top level** while `XHandSpatialOnlinePreprocess` omitted it | **CONFIRMED BUG** fixed in this PR: strict canonical schema validation and forwarding to `spatial.object_geometry`, with fail-closed policy config for geometry checkpoints. |
| Actor | trained Pi0 policy output [50,18] | `deploy/ur7e_xhand_deploy_3d_tactile_multi.py` | Existing control path untouched. Current scripts/deploy dry-run only saves actions to disk. Do NOT claim hardware integration. |

## Status and remaining blockers

1. The current PR is draft and **must not be merged or deployed for actuation**.
2. Obtain / recover the implementation of `hand_scene_online.py` from the deployment host if locally present, or implement from the canonical offline functions with explicit same-pose equivalence tests. The class `OnlineObjectGeometryBuilder` is imported in `run_online_perception.py`, `compare_reference_npz.py`, and tests but cannot be found in the committed repository at the base commit.
3. Resolve native vs color-aligned front depth before running the core `SpatialPreprocessor`. The front color-aligned depth works for RGBTrack's K_color contract, but is not interchangeable with native raw depth for SpatialPreprocessor.
4. Run all new tests plus existing dry-run on the deployment machine. No real hardware and no dataset are available in this editing session.
5. Run Episode 23 with both segmentation backends and visually inspect the masks and 3-D viewer before any robot movement.
6. Verify each of the three checkpoint configs actually routes nonempty geometry tokens to the encoder (not only that action tensors have shape [50,18]).
