# Online deployment validation

Generated on 2026-10-27 for the current checkout. Synthetic replay results are
kept under the ignored `output/deploy/` directory; they are not real-scene
accuracy evidence.

## Status

| Check | Result | Evidence / limit |
| --- | --- | --- |
| Perception package compatibility | PASS | The overlay pins NumPy 2.2.6, packaging 25.0, setuptools 80.10.2, and OmegaConf 2.3.0 to satisfy the shared LeRobot/SAM2 constraints; pip check reports no broken requirements. |
| Environment split | PASS | OpenPI `.venv` keeps JAX 0.5.3 GPU support. Perception `.venv-perception` reuses Torch 2.10.0+cu128 and passes an actual sm_120 Torch kernel. |
| SAM2 import and CUDA run | PASS | Official SAM2.1 Hiera Small processed two causal synthetic frames on the RTX 5090. No future frame API is used. |
| Segmentation quality on real objects | NOT TESTED | Input was a generated two-shape image, not a funnel/flask recording. The report makes no mask accuracy claim. |
| RGBTrack Phase A | PASS | Required imports and upstream `FoundationPose` import pass. TensorRT and BundleSDF are intentionally absent on the ordinary PyTorch path. |
| RGBTrack Phase B | BLOCKED | The scorer and refiner `model_best.pth` files are absent. No pose estimator was initialized and no object pose is claimed. |
| RGBTrack Phase C | PASS | NVDiffRast rendered a triangle on CUDA with Torch 2.10.0+cu128 on the RTX 5090. |
| `mycpp.cluster_poses` | PASS | Built against the perception Python 3.12 and imported from RGBTrack's local `mycpp/build`. |
| Geometry producer contract | PASS | Six online geometry tests pass; object XYZ is `[512,3]`, invalid dynamic poses use zero clouds/masks, and the fixed tactile contract remains 600 taxels. |
| Encoder token layout | PASS; one separate check remains flaky | The object-centric encoder layout test passed with 68 tokens (33 object-centric visual + 25 tactile + 10 relation). Two full module runs reported 7 passed and 1 failed: `test_structured_spatial_encoder_uses_finger_id_not_tactile_order` differed by `1.43e-5` against its `1e-5` tolerance in one `token_xyz_m` value. An isolated rerun passed. Model and test code were left unchanged. |
| Pi0 dry-run, M2 | PASS | Real trained checkpoint; `[50,18]`, finite; `robot_command_sent=false`. Also consumed an actual SAM2-produced packet in `--follow` mode. |
| Pi0 dry-run, DTCR | PASS | Real trained checkpoint; `[50,18]`, finite; no robot command. |
| Pi0 dry-run, DTCR-independent | PASS | Real trained checkpoint; `[50,18]`, finite; no robot command. |
| Pi0 dry-run, M2-OC | BLOCKED | Its checkpoint is not present locally. Its encoder layout unit test passed, but trained M2-OC inference was not run. |
| Offline/online Sidecar comparison | NOT TESTED | `data/1018_peg_all/object_geometry/v1_hand_scene_q2` is absent; machine-readable result is `docs/deploy/online_offline_comparison.json`. |
| Real camera / robot stream | NOT TESTED | No UR/XHand capture SDK or recorded task sequence is present. The file packet bridge is ready for a hardware-specific writer. No robot command path is implemented. |

## Synthetic causal replay

`output/deploy/synthetic_online_smoke/manifest.jsonl` records two ordered
frames. Frame 0 produced two non-overlapping masks with SAM2 scores `0.988`;
the masks were marked valid. On frame 1, both scores were about `1e-6`, so the
`0.05` minimum-score gate marked both masks `invalid_low_score`. The geometry
builder emitted fixed-size invalid object clouds instead of trusting these
masks. This checks causal propagation and failure gating only.

Measured perception time in the two-frame file-watch replay was 362 ms for the
first synthetic frame and 64 ms for the second. These values are not a real-
scene throughput result. In a two-frame M2 policy watcher run, the first call
took 26.3 s and the second took 94 ms. Across those two frames, spatial input
transforms averaged 25.7 ms, model-input preparation 185 ms, model sampling and
synchronization 13.0 s (dominated by first-call compilation), and output
transforms 0.17 ms. Two samples are not a meaningful P50/P95 estimate. Concurrent
SAM2 + RGBTrack + Pi0 memory use has not been measured. The environment audit
observed up to 24.6 GiB GPU memory in use during a policy run; the GPU is idle
after the run.

Artifacts:

- `output/deploy/synthetic_online_smoke/overlays/000000.jpg`
- `output/deploy/synthetic_online_smoke/overlays/000001.jpg`
- `output/deploy/synthetic_online_smoke/frames/`
- `output/deploy/synthetic_policy_m2/summary.json`
- `output/deploy/synthetic_policy_m2_twoframe_fixed/summary.json`
- `output/deploy/sam2_stream_smoke/report.json`

## Environment and source provenance

The OpenPI `.venv` resolves to the existing shared Python 3.11 environment. Its
Torch 2.7.1+cu126 wheel does not contain an sm_120 kernel, while JAX 0.5.3
executes a GPU smoke kernel. The project-owned `.venv-perception` is a Python
3.12 overlay on the existing `lerobot5090` environment; its Torch 2.10.0+cu128
sm_120 kernel passes. This overlay is where SAM2/RGBTrack dependencies live.

SAM2.1 Hiera Small was downloaded from its official checkpoint URL with direct
no-proxy access. The file is 184,416,285 bytes and has SHA-256
`6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38`. Source
revision: `2b90b9f5ceec907a1c18123530e92e794ad901a4`.

PyTorch3D 0.7.9 was built from source revision
`204fe5831b3a5bbe079f09d72afb4515178f8892`; NVDiffRast v0.3.5 uses revision
`20c4135fda78ae29d7658bea28b6c1d5b5e103e5`; the BOP toolkit source is pinned
to `cea62d651c7e395b2e1962b9749e4e89693c6ac4`. RGBTrack remains at the
registered submodule revision `23e43f832b82f4813d6ab72773779bf8d2ee4d83`.
Python packages were installed from the Tsinghua PyPI mirror. ModelScope was
checked without guessing a model ID; its queried public endpoint returned
HTTP 404, so the verified official SAM2 checkpoint was used. No proxy was
enabled.

The funnel and flask GLBs passed `scripts/deploy/check_assets.py`. Both explicit
scale entries are `1.0 m/unit`; each asset's bounds match its recorded CAD
millimetre dimensions to less than `0.000004 mm`. See
`data/pose_eval/hand_scene_metric_mesh_scales.json` for provenance and hashes.

## Files still required

Copy the two trained RGBTrack files from computer 4 into these exact paths:

```text
checkpoints/rgbtrack/2024-01-11-20-02-45/model_best.pth
checkpoints/rgbtrack/2023-10-28-18-33-37/model_best.pth
```

Then run `scripts/deploy/download_weights.py --verify-only` and the RGBTrack
Phase B command. The local `third_party/RGBTrack/weights` symlink already
points to `checkpoints/rgbtrack`.

The trained M2-OC Orbax checkpoint and its `1018_peg_all` normalization stats
must also be copied from computer 4 to:

```text
checkpoints/pi0_xhand_spatial_hand_scene_oo_object_centric_v1/m2_oc_b8_20k_1026/19999/
```

For parity testing, copy the matching training data and
`object_geometry/v1_hand_scene_q2` Sidecar. For real hardware, provide a packet
writer that atomically publishes the documented RGB, aligned depth, 1972-state
and optional 600-taxel NPZ arrays. The deployment code does not read hardware
or send actions to the robot.
