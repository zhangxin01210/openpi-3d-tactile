# Deployment-host uncommitted code audit — 2026-10-28

Inputs: uploaded `deploy_code_audit_20261028.tar.gz`; original host HEAD `3260d6328d229b10ccdbbfcce84aaf8897366e0a`, RGBTrack submodule `23e43f832b82f4813d6ab72773779bf8d2ee4d83`. Original working tree has ten modified tracked files and an untracked `hand_scene_online.py`. **Never discard or reset that host workspace.**

## Integrated on review branch

- `rgbtrack_upright_full.py`: `--install-runtime-only` for online causal replay; previously the deployment caller passed this flag but the training hook rejected it.
- `rgbtrack_native_upright.py`: runtime report state and `write_runtime_report()` for use after an external frame loop; the online script already calls this missing function.
- `rgbtrack_adapter.py`: shared scorer/refiner with bounded candidate batches from host; avoids independently constructing networks for each tracked object.
- `object_geometry/production.py`: `CausalHandSceneState.next_from_arrays()` and `reset()` are now shared between disk sidecar and current-frame online processing.
- `policies/hand_scene_online.py`: calls the shared next_from_arrays state rather than manually reproducing nearest-neighbour resizing and state reference logic; adds episode reset.
- `policies/policy_config.py`: preserves Orbax root-vs-params and normalization asset layout fallback from host.

## Important not-yet-integrated and not-yet-verified host changes

- `visualize_rgbtrack_full_episode.py` local modifications: online NPZ replay support, nearest-finger distances and training mesh seeds; leave out of active perception pipeline until separately reviewed.
- `policies/policy.py` local instrumentation: inference/preprocessing/host-sync latency timings; not essential for semantic parity, not merged.
- `check_rgbtrack_runtime.py` local import audit: useful, not yet merged.
- `config.json` local static_prior_start_frame=1: training exporter and deploy already use fallback 1; integrate with explicit migration review.
- The host's `hand_scene_online.py` differs in pose acceptance and exact status heuristics from the PR builder. The PR producer conservatively invalidates predicted or occluded dynamic geometry. This needs an offline fixture regression, not silent overwrite.
- The host has untracked `third_party/RGBTrack` build outputs and model symlinks; these are environment artifacts, not code to commit.

## Outstanding validation and risks

1. Runtime source imports, exact scorer/refiner API and the bounded-batch wrapper must be tested with the version-pinned RGBTrack source/weights.
2. Depth contract now expects separate `front_depth_native` and `front_depth_color_aligned`, from the **actual acquisition/registration pipeline**; older packets cannot be simply renamed.
3. The shared online geometry state must be tested against frame-by-frame training sidecar for the *same* pose, timestamp and tactile data. An online current-frame mask without calibrated hand offset remains invalid.
4. Episode 23 mask IoU, pose discontinuities, clouds, tactile, relation token and spatial encoder input tensor comparisons remain unverified.
5. This draft must not control UR/XHand hardware; a 50x18 action tensor is not parity proof.

All changes are GitHub source edits only. This report does not claim host pytest or physical deployment tests.
