from __future__ import annotations

import argparse

import numpy as np

from openpi.policies import xhand_policy
from openpi.training import config as _config
from openpi.training import data_loader
import openpi.transforms as _transforms


_CONFIG_NAMES = (
    "pi0_xhand_spatial_joint_pointnet_prefix",
    "pi0_xhand_spatial_joint_pointnet_suffix",
    "pi0_xhand_spatial_joint_pointnet_both",
)


def _walk_transforms(group) -> list[object]:
    transforms = []
    for sequence in (group.inputs, group.outputs):
        transforms.extend(sequence)
    return transforms


def _assert_close(name: str, actual, expected) -> None:
    if not np.allclose(np.asarray(actual), np.asarray(expected), atol=1e-6, rtol=0.0):
        diff = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
        raise AssertionError(f"{name} mismatch; max_abs_diff={diff}")


def run_config(config_name: str, sample_index: int) -> None:
    train_config = _config.get_config(config_name)
    data_config = train_config.data.create(
        train_config.assets_dirs,
        train_config.model,
    )

    transforms = [
        *_walk_transforms(data_config.data_transforms),
    ]
    if any(isinstance(transform, _transforms.DeltaActions) for transform in transforms):
        raise AssertionError(f"{config_name} unexpectedly contains DeltaActions")

    raw_dataset = data_loader.create_torch_dataset(
        data_config,
        train_config.model.action_horizon,
        train_config.model,
    )
    raw = raw_dataset[sample_index]
    transformed = data_loader.transform_dataset(
        raw_dataset,
        data_config,
        skip_norm_stats=True,
    )[sample_index]

    raw_actions = np.asarray(raw["action"], dtype=np.float32)
    if raw_actions.shape != (
        train_config.model.action_horizon,
        xhand_policy.ROBOT_ACTION_DIM,
    ):
        raise AssertionError(f"raw action chunk shape mismatch: {raw_actions.shape}")

    _assert_close(
        f"{config_name} absolute action",
        transformed["actions"][..., : xhand_policy.ROBOT_ACTION_DIM],
        raw_actions,
    )
    _assert_close(
        f"{config_name} action zero padding",
        transformed["actions"][..., xhand_policy.ROBOT_ACTION_DIM :],
        np.zeros(
            (
                train_config.model.action_horizon,
                train_config.model.action_dim - xhand_policy.ROBOT_ACTION_DIM,
            ),
            dtype=np.float32,
        ),
    )

    model_actions = np.arange(
        train_config.model.action_horizon * train_config.model.action_dim,
        dtype=np.float32,
    ).reshape(
        train_config.model.action_horizon,
        train_config.model.action_dim,
    )
    policy_actions = xhand_policy.XHandOutputs()({"actions": model_actions})["actions"]
    if policy_actions.shape != (
        train_config.model.action_horizon,
        xhand_policy.ROBOT_ACTION_DIM,
    ):
        raise AssertionError(f"inference action shape mismatch: {policy_actions.shape}")
    _assert_close(
        f"{config_name} inference first-18 slice",
        policy_actions,
        model_actions[..., : xhand_policy.ROBOT_ACTION_DIM],
    )

    print(
        f"{config_name}: raw absolute 18-D -> padded 32-D -> output first 18-D PASS"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-index", type=int, default=213)
    parser.add_argument("--configs", nargs="*", default=_CONFIG_NAMES)
    args = parser.parse_args()

    for config_name in args.configs:
        run_config(config_name, args.sample_index)

    print("XHAND_ACTION_CONTRACT_PASS")


if __name__ == "__main__":
    main()
