"""Fast XHand norm stats: read only state/action from Parquet."""

import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--raw-tactile-state", action="store_true", help="compute independent 1818-D state stats")
    args = parser.parse_args()

    cfg = config_lib.get_config(CONFIG_NAME)
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)

    root = args.dataset.expanduser().resolve()
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

            if args.raw_tactile_state:
                from openpi.policies.xhand_raw_tactile import extract_raw_tactile_state
                state18 = extract_raw_tactile_state(raw_state)
            else:
                state18 = np.concatenate([raw_state[:, :6], raw_state[:, 28:52:2]], axis=-1)

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
    if args.raw_tactile_state:
        print(f"Raw tactile zero-variance dimensions: {int(np.sum(np.std(states[:, 18:], axis=0) == 0))}")
        print(f"Raw tactile finite range: [{np.min(states[:, 18:]):.6g}, {np.max(states[:, 18:]):.6g}]")
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

    if args.raw_tactile_state:
        # Preserve the RGB baseline's exact proprio/action normalization. Only
        # compute the 1,800 newly added tactile coordinates.
        assets_dir = Path(cfg.data.assets.assets_dir or cfg.assets_dirs)
        baseline_dir = assets_dir / f"{root.name}_unified_v1"
        baseline = normalize.load(baseline_dir)
        for key in ("state", "actions"):
            item = baseline[key]
            if np.asarray(item.mean).shape != (18,) or np.asarray(item.std).shape != (18,):
                raise ValueError(f"{baseline_dir}: {key} must have 18-D RGB baseline stats")
            if item.q01 is None or item.q99 is None:
                raise ValueError(f"{baseline_dir}: {key} missing quantile stats required by pi05")
            if any(not np.isfinite(x).all() for x in (item.mean, item.std, item.q01, item.q99)):
                raise ValueError(f"{baseline_dir}: {key} has non-finite statistics")
        tactile_stats = normalize.RunningStats()
        tactile_stats.update(states[:used, 18:])
        tactile = tactile_stats.get_statistics()
        proprio = baseline["state"]
        result = {
            "state": normalize.NormStats(**{field: np.concatenate((getattr(proprio, field), getattr(tactile, field)))
                                            for field in ("mean", "std", "q01", "q99")}),
            "actions": baseline["actions"],
        }
        print(f"Reused exact proprio and action stats: {baseline_dir}")
    else:
        for i in tqdm(range(0, used, batch_size), desc="Computing stats"):
            batch_indices = np.arange(i, i + batch_size)
            action_indices = np.minimum(batch_indices[:, None] + offsets, episode_ends[batch_indices, None])
            stats["state"].update(states[batch_indices])
            stats["actions"].update(actions[action_indices])
        result = {key: value.get_statistics() for key, value in stats.items()}

    assets_dir = Path(cfg.data.assets.assets_dir or cfg.assets_dirs)
    output_dir = assets_dir / args.asset_id

    if (output_dir / "norm_stats.json").exists():
        raise FileExistsError(
            f"Refusing to overwrite existing stats: {output_dir}"
        )

    normalize.save(output_dir, result)

    print(f"\nSaved: {output_dir / 'norm_stats.json'}")
    print(f"Frames used: {used}")
    print(f"Action horizon: {horizon}")
    print("DONE")


if __name__ == "__main__":
    main()