# XHand v1 three-task data-interface handoff

This module **does not change** upstream OpenPI's `src/openpi/training/data_loader.py`, `training/config.py`, model, or checkpoint. It gives Codex one reusable entry point for the approved three-sidecar training inputs.

## Files

- `src/openpi/training/xhand_multimodal_dataset.py` — read-only join + reusable `create_xhand_multimodal_dataset(data_config, model_config, action_horizon=50)`.
- `scripts/preprocess/verify_xhand_multimodal_v1.py` — read-only test using the real OpenPI LeRobot action-chunk and XHand transforms plus per-task stats.

Copy both files into the **existing** repository, preserving relative paths. Do not clone a new repository. The original `scripts/preprocess/smoke_three_task_xhand_loader_v1.py` can remain unchanged.

From the repository root:

```bash
PYTHONPATH=src .venv/bin/python scripts/preprocess/verify_xhand_multimodal_v1.py
```

Expected: three `[BATCH PASS]` lines and final `[DATALOADER DATA-ONLY SMOKE PASS]`. Default report is at `output/preprocess/training_readiness/dataloader_production_interface_v1/report.json`.

## Use from a future train config (Codex)

Each of the three tasks must supply its **own** data config: `repo_id=<dataset_root>`, `spatial=SpatialDataConfig(dataset_root=<same_root>, version='v1', camera_roles=('front',), front_calibration_profile=<approved extrinsic-only front_v1_match.json>)`, `assets=AssetsConfig(assets_dir='assets/xhand', asset_id='<task>_unified_v1')`. The dataset is then built from the created task-specific `data_config` and its Pi0 `model_config`:

```python
from openpi.training.xhand_multimodal_dataset import create_xhand_multimodal_dataset

training_dataset = create_xhand_multimodal_dataset(
    data_config, model_config, action_horizon=50,
)
# Compatible with openpi.training.data_loader.TorchDataLoader.
```

Do **not** pass a data_config from one task with another task's dataset_root or asset_id. Existing `DataConfigFactory.create()` loads stats with its asset_id; the returned `data_config.norm_stats` must contain 18D `state` and 18D `actions` for the current task. The helper checks basic dimensions, but the caller remains responsible for correct task-to-asset mapping.

The dataset samples retain normal transformed RGB, normalized 32D state and `[50,32]` actions plus unnormalized metric sidecars `spatial`, `object_geometry` and `hand_keypoints`. Object validity masks and pose source/age/confidence remain intact. Hand keypoints are FK anchors, **not measured skin contacts**. No relation-token computation or model changes happen here. The model must explicitly consume the sidecar fields; `Observation.from_dict()` will not automatically do so.

## Evidence / limitations

Earlier exhaustive QA passed **281 episodes / 93,791 frames, zero warnings**, and original data-only smoke passed 12 sampled frames and Batch Size 2 across all three tasks. This new factory/test is a refactor of that path and needs its own cloud-side pass. No full-model forward/backward, throughput/load test, pose ground-truth validation, or real robot validation is claimed. `1018_peg_all` Object Geometry reuses legacy tracked poses; RGBTrack metric accuracy remains qualified. The formal test is a **sample smoke**, not a new 93,791-frame scan.
