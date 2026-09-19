from __future__ import annotations

import argparse

import numpy as np

from openpi.training import config as _config


_CONFIG_NAMES = (
    "pi0_xhand_spatial_joint_pointnet_prefix",
    "pi0_xhand_spatial_joint_pointnet_suffix",
    "pi0_xhand_spatial_joint_pointnet_both",
)


def _assert_same_stats(reference, candidate, *, name: str) -> None:
    if set(reference) != set(candidate):
        raise AssertionError(
            f"{name} norm stats keys mismatch: {set(reference)!r} != {set(candidate)!r}"
        )

    for key in reference:
        for field in ("mean", "std", "q01", "q99"):
            ref = np.asarray(getattr(reference[key], field))
            got = np.asarray(getattr(candidate[key], field))
            if not np.allclose(ref, got, atol=0.0, rtol=0.0):
                raise AssertionError(f"{name} {key}.{field} differs from reference")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="*",
        default=_CONFIG_NAMES,
    )
    args = parser.parse_args()

    reference_stats = None
    reference_asset = None

    for config_name in args.configs:
        train_config = _config.get_config(config_name)
        data_config = train_config.data.create(
            train_config.assets_dirs,
            train_config.model,
        )

        if data_config.norm_stats is None:
            raise AssertionError(f"{config_name} did not load norm stats")

        keys = set(data_config.norm_stats)
        if keys != {"state", "actions"}:
            raise AssertionError(
                f"{config_name} should only have state/actions stats, got {sorted(keys)!r}"
            )

        if data_config.spatial is None:
            raise AssertionError(f"{config_name} unexpectedly has no spatial config")

        asset = (
            train_config.data.assets.assets_dir,
            data_config.asset_id,
        )

        print(
            f"{config_name}: asset={asset} keys={sorted(keys)} "
            f"state_mean_shape={data_config.norm_stats['state'].mean.shape} "
            f"actions_mean_shape={data_config.norm_stats['actions'].mean.shape}"
        )

        if reference_stats is None:
            reference_stats = data_config.norm_stats
            reference_asset = asset
            continue

        if asset != reference_asset:
            raise AssertionError(
                f"{config_name} loads a different norm stats asset: {asset!r} != {reference_asset!r}"
            )

        _assert_same_stats(
            reference_stats,
            data_config.norm_stats,
            name=config_name,
        )

    print("XHAND_NORM_STATS_LOAD_PASS")


if __name__ == "__main__":
    main()
