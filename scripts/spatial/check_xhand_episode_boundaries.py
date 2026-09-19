from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from openpi.training import config as _config
from openpi.training import data_loader


def _episode_table(dataset_root: Path) -> pd.DataFrame:
    data = pd.read_parquet(dataset_root / "data/chunk-000/file-000.parquet")
    return data


def _assert_close(name: str, actual, expected) -> None:
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    if not np.allclose(actual, expected, atol=1e-6, rtol=0.0):
        diff = float(np.max(np.abs(actual - expected)))
        raise AssertionError(f"{name} mismatch; max_abs_diff={diff}")


def _check_index(
    *,
    dataset,
    table: pd.DataFrame,
    global_index: int,
    action_horizon: int,
    fps: float,
) -> None:
    item = dataset[global_index]
    episode_index = int(np.asarray(item["episode_index"]).reshape(()))
    frame_index = int(np.asarray(item["frame_index"]).reshape(()))
    timestamp = float(np.asarray(item["timestamp"]).reshape(()))
    actions = np.asarray(item["action"], dtype=np.float32)

    episode = table[table["episode_index"] == episode_index].sort_values("frame_index")
    length = len(episode)
    if frame_index < 0 or frame_index >= length:
        raise AssertionError(f"frame_index out of range: ep={episode_index} frame={frame_index} length={length}")

    expected_timestamp = frame_index / fps
    if abs(timestamp - expected_timestamp) > 1e-3:
        raise AssertionError(
            f"timestamp mismatch at global={global_index}: got {timestamp}, expected {expected_timestamp}"
        )

    available = min(action_horizon, length - frame_index)
    expected_actions = np.stack(
        episode.iloc[frame_index : frame_index + available]["action"].to_numpy(),
        axis=0,
    ).astype(np.float32)
    _assert_close(
        f"global={global_index} available action chunk",
        actions[:available],
        expected_actions,
    )

    if available < action_horizon:
        final_action = np.asarray(episode.iloc[-1]["action"], dtype=np.float32)
        repeated_tail = np.repeat(
            final_action[None, :],
            action_horizon - available,
            axis=0,
        )
        _assert_close(
            f"global={global_index} repeated terminal action tail",
            actions[available:],
            repeated_tail,
        )

    if actions.shape != (action_horizon, 18):
        raise AssertionError(f"action shape mismatch at global={global_index}: {actions.shape}")
    if not np.isfinite(actions).all():
        raise AssertionError(f"non-finite action at global={global_index}")

    print(
        f"global={global_index} ep={episode_index} frame={frame_index} "
        f"available={available} tail_repeat={available < action_horizon} PASS"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="pi0_xhand_spatial_joint_pointnet_suffix")
    args = parser.parse_args()

    train_config = _config.get_config(args.config_name)
    data_config = train_config.data.create(
        train_config.assets_dirs,
        train_config.model,
    )
    if data_config.repo_id is None:
        raise AssertionError("repo_id is required")

    root = Path(data_config.repo_id)
    table = _episode_table(root)
    lengths = table.groupby("episode_index").size().to_dict()
    starts = {}
    offset = 0
    for episode_index, length in lengths.items():
        starts[int(episode_index)] = offset
        offset += int(length)

    dataset = data_loader.create_torch_dataset(
        data_config,
        train_config.model.action_horizon,
        train_config.model,
    )

    indices = []
    for episode_index, length in lengths.items():
        start = starts[int(episode_index)]
        length = int(length)
        indices.extend(
            [
                start,
                start + length // 2,
                start + max(0, length - train_config.model.action_horizon),
                start + max(0, length - train_config.model.action_horizon + 1),
                start + length - 1,
            ]
        )

    seen = set()
    for global_index in indices:
        if global_index in seen:
            continue
        seen.add(global_index)
        _check_index(
            dataset=dataset,
            table=table,
            global_index=global_index,
            action_horizon=train_config.model.action_horizon,
            fps=15.0,
        )

    print("XHAND_EPISODE_BOUNDARY_QA_PASS")


if __name__ == "__main__":
    main()
