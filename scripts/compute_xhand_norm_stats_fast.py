"""Fast XHand norm stats: read only state/action from Parquet."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from openpi.shared import normalize
from openpi.training import config as config_lib


CONFIG_NAME = "pi0_xhand_rgb_baseline"


def main():
    cfg = config_lib.get_config(CONFIG_NAME)
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)

    root = Path(dc.repo_id)
    info = json.loads((root / "meta/info.json").read_text())

    assert cfg.model.action_horizon == 50
    assert not cfg.data.use_delta_arm_actions
    assert not cfg.data.delta_hand_actions

    files = sorted((root / "data").rglob("*.parquet"))
    assert files, f"No Parquet files found: {root}"

    episodes = defaultdict(list)

    print(f"Dataset: {root}")
    print(f"Parquet files: {len(files)}")
    print("Reading only state/action (no RGB, depth or spatial)...")

    for path in tqdm(files, desc="Reading Parquet"):
        df = pd.read_parquet(
            path,
            columns=[
                "episode_index",
                "frame_index",
                "observation.state",
                "action",
            ],
        )

        for ep, group in df.groupby("episode_index", sort=False):
            frames = group["frame_index"].to_numpy(dtype=np.int64)

            raw_state = np.stack(
                group["observation.state"].to_numpy()
            ).astype(np.float32)

            raw_actions = np.stack(
                group["action"].to_numpy()
            ).astype(np.float32)

            assert raw_state.shape[1] == 1972, raw_state.shape
            assert raw_actions.shape[1] == 18, raw_actions.shape

            # 与 XHandInputs.extract_joint_state 完全一致：
            # UR7e 6 joints + XHand 12 joint positions
            state18 = np.concatenate(
                [
                    raw_state[:, :6],
                    raw_state[:, 28:52:2],
                ],
                axis=-1,
            )

            episodes[int(ep)].append(
                (frames, state18, raw_actions)
            )

    all_states = []
    all_actions = []
    all_episode_ends = []
    offset = 0

    for ep in sorted(episodes):
        parts = episodes[ep]

        frames = np.concatenate([p[0] for p in parts])
        states = np.concatenate([p[1] for p in parts])
        actions = np.concatenate([p[2] for p in parts])

        order = np.argsort(frames)
        frames = frames[order]
        states = states[order]
        actions = actions[order]

        assert np.array_equal(
            frames, np.arange(len(frames))
        ), f"Episode {ep}: invalid frame indices"

        n = len(frames)
        all_states.append(states)
        all_actions.append(actions)

        # 每个样本的 future actions 不允许跨越 episode。
        # episode 末尾不足 50 步时重复最后一个 action。
        all_episode_ends.append(
            np.full(n, offset + n - 1, dtype=np.int64)
        )
        offset += n

    states = np.concatenate(all_states)
    actions = np.concatenate(all_actions)
    episode_ends = np.concatenate(all_episode_ends)

    assert len(states) == info["total_frames"]
    assert len(episodes) == info["total_episodes"]
    assert np.isfinite(states).all()
    assert np.isfinite(actions).all()

    print(f"Episodes: {len(episodes)}")
    print(f"Frames: {len(states)}")
    print(f"State shape: {states.shape}")
    print(f"Action shape: {actions.shape}")

    batch_size = cfg.batch_size
    horizon = cfg.model.action_horizon

    # 与原脚本一致：只处理完整 batch。
    used = (len(states) // batch_size) * batch_size
    offsets = np.arange(horizon)[None, :]

    stats = {
        "state": normalize.RunningStats(),
        "actions": normalize.RunningStats(),
    }

    for i in tqdm(
        range(0, used, batch_size),
        desc="Computing stats",
    ):
        batch_indices = np.arange(i, i + batch_size)

        action_indices = np.minimum(
            batch_indices[:, None] + offsets,
            episode_ends[batch_indices, None],
        )

        stats["state"].update(states[batch_indices])
        stats["actions"].update(actions[action_indices])

    result = {
        key: value.get_statistics()
        for key, value in stats.items()
    }

    assets_dir = Path(cfg.data.assets.assets_dir or cfg.assets_dirs)
    output_dir = assets_dir / dc.asset_id

    normalize.save(output_dir, result)

    print(f"\nSaved: {output_dir / 'norm_stats.json'}")
    print(f"Frames used: {used}")
    print(f"Action horizon: {horizon}")
    print("DONE")


if __name__ == "__main__":
    main()