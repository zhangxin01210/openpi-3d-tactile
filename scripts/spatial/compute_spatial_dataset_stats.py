"""
OpenPI 3D + tactile：Derived Spatial Dataset 输入统计
（compute_spatial_dataset_stats.py）

作用
----
直接扫描已经生成的：

    <dataset>/spatial/<version>/

统计后续模型输入设计真正需要的数值分布。

本脚本不读取 raw RGB-D，不运行：

    calibration
    FK
    voxel
    sampling
    SpatialPreprocessor

数据路径：

    spatial/v1
        ↓ mmap
    SpatialDerivedDataset
        ↓
    streaming statistics

为什么现在需要这一步
--------------------
进入 point encoder 前，需要先冻结连续特征的 normalization contract。

其中：

    RGB
        通常可以稳定地映射到 [0,1]

    XYZ
        可以使用固定 workspace / metric scale

    tactile force
        当前单位仍是 dataset_native

因此不能在不知道真实分布的情况下随意写：

    force / 50
    force / 100

本脚本先回答：

    - visual / tactile XYZ 实际分布如何？
    - RGB-valid 比例是多少？
    - 有效 RGB 的均值 / 标准差如何？
    - tactile force component / norm 的分位数是多少？
    - force 有多稀疏？
    - 不同 finger 的 force 分布是否差很多？
    - 每帧最大 force 的分布如何？

输出
----
默认写：

    <dataset>/spatial/<version>/stats.json

主要字段：

    visual_xyz
    tactile_xyz
    visual_rgb_valid
    visual_rgb_valid_only
    tactile_force
    tactile_force_abs
    tactile_force_norm
    tactile_force_norm_by_finger
    tactile_activity
    frame_max_force_norm

基本使用
--------
    PYTHONPATH=src python scripts/spatial/compute_spatial_dataset_stats.py \
        --dataset data/press_0828_17 \
        --version v1

指定输出：

    ... --output output/spatial_v1_stats.json

注意
----
- 全部统计基于 derived spatial/v1，而不是 raw dense cloud。
- visual XYZ 因此统计的是模型真正会看到的固定 N 点。
- RGB 均值 / 标准差只统计 rgb_valid=True 的点。
- force 数值保持 dataset_native，不伪装成 Newton。
- percentile 目前对 tactile force 做精确统计；
  2289×600 的规模很小，不需要近似直方图。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from openpi.spatial_dataset.derived import SpatialDerivedDataset


# =============================================================================
# 1. CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute model-input statistics from "
            "a derived spatial dataset."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "data/press_0828_17"
        ),
    )

    parser.add_argument(
        "--version",
        type=str,
        default="v1",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    return parser.parse_args()


# =============================================================================
# 2. 通用数值 accumulator
# =============================================================================

class VectorMoments:
    """
    对最后一维为 feature dimension 的数据累计：

        count
        min
        max
        mean
        std

    使用 float64 sum / squared-sum。
    当前 XYZ / RGB / force 数值范围较小，精度足够。
    """

    def __init__(
        self,
        dimension: int,
    ) -> None:
        self.dimension = int(
            dimension
        )

        self.count = 0

        self.sum = np.zeros(
            self.dimension,
            dtype=np.float64,
        )

        self.sum_sq = np.zeros(
            self.dimension,
            dtype=np.float64,
        )

        self.minimum = np.full(
            self.dimension,
            np.inf,
            dtype=np.float64,
        )

        self.maximum = np.full(
            self.dimension,
            -np.inf,
            dtype=np.float64,
        )

    def update(
        self,
        values: np.ndarray,
    ) -> None:
        array = np.asarray(
            values
        )

        if array.size == 0:
            return

        reshaped = np.asarray(
            array,
            dtype=np.float64,
        ).reshape(
            -1,
            self.dimension,
        )

        if not np.all(
            np.isfinite(
                reshaped
            )
        ):
            raise ValueError(
                "Non-finite value encountered "
                "while computing statistics"
            )

        self.count += int(
            reshaped.shape[
                0
            ]
        )

        self.sum += np.sum(
            reshaped,
            axis=0,
            dtype=np.float64,
        )

        self.sum_sq += np.sum(
            reshaped
            * reshaped,
            axis=0,
            dtype=np.float64,
        )

        self.minimum = np.minimum(
            self.minimum,
            np.min(
                reshaped,
                axis=0,
            ),
        )

        self.maximum = np.maximum(
            self.maximum,
            np.max(
                reshaped,
                axis=0,
            ),
        )

    def report(
        self,
    ) -> dict[str, object]:
        if self.count == 0:
            return {
                "count": 0,
            }

        mean = (
            self.sum
            / self.count
        )

        variance = (
            self.sum_sq
            / self.count
            - mean
            * mean
        )

        variance = np.maximum(
            variance,
            0.0,
        )

        std = np.sqrt(
            variance
        )

        return {
            "count": int(
                self.count
            ),
            "min": self.minimum.tolist(),
            "max": self.maximum.tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
        }


# =============================================================================
# 3. Percentile helpers
# =============================================================================

_PERCENTILES = (
    0.0,
    1.0,
    5.0,
    10.0,
    25.0,
    50.0,
    75.0,
    90.0,
    95.0,
    99.0,
    99.5,
    99.9,
    100.0,
)


def percentile_report(
    values: np.ndarray,
) -> dict[str, float]:
    array = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(
        -1
    )

    if array.size == 0:
        return {}

    result = np.percentile(
        array,
        _PERCENTILES,
    )

    return {
        f"p{percentile:g}": float(
            value
        )
        for percentile, value
        in zip(
            _PERCENTILES,
            result,
            strict=True,
        )
    }


# =============================================================================
# 4. Main statistics
# =============================================================================

def compute_stats(
    dataset: SpatialDerivedDataset,
) -> dict[str, object]:
    visual_xyz = VectorMoments(
        3
    )

    tactile_xyz = VectorMoments(
        3
    )

    valid_rgb = VectorMoments(
        3
    )

    tactile_force = VectorMoments(
        3
    )

    # tactile 总量：
    # 2289 × 600 ~= 1.37M
    # float32 只约 5.5 MB，可以直接保留做精确 percentile。
    force_norm_chunks = []

    # force component absolute value：
    # 约 4.1M scalar，float32 约 16 MB。
    force_abs_chunks = []

    frame_max_force = []

    rgb_valid_count = 0
    rgb_total_count = 0

    per_frame_rgb_valid_ratio = []

    finger_force_chunks: dict[
        int,
        list[np.ndarray],
    ] = {
        int(
            finger
        ): []
        for finger in np.unique(
            np.asarray(
                dataset.finger_id
            )
        )
    }

    # activity threshold 完全保持 dataset_native。
    activity_thresholds = (
        0.0,
        0.1,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        20.0,
        50.0,
    )

    activity_counts = {
        float(
            threshold
        ): 0
        for threshold in activity_thresholds
    }

    total_tactile_points = 0

    print(
        "frames:",
        len(
            dataset
        ),
    )

    for global_index in range(
        len(
            dataset
        )
    ):
        sample = dataset[
            global_index
        ]

        visual_xyz.update(
            sample.visual_xyz_m
        )

        tactile_xyz.update(
            sample.tactile_xyz_m
        )

        mask = np.asarray(
            sample.visual_rgb_valid,
            dtype=np.bool_,
        )

        rgb = np.asarray(
            sample.visual_rgb
        )

        if mask.shape != (
            dataset.visual_points_per_frame,
        ):
            raise ValueError(
                f"Unexpected rgb_valid shape: "
                f"{mask.shape}"
            )

        valid_count = int(
            np.count_nonzero(
                mask
            )
        )

        total_count = int(
            mask.size
        )

        rgb_valid_count += (
            valid_count
        )

        rgb_total_count += (
            total_count
        )

        per_frame_rgb_valid_ratio.append(
            valid_count
            / total_count
        )

        if valid_count:
            valid_rgb.update(
                rgb[
                    mask
                ]
            )

        force = np.asarray(
            sample.tactile_force_base,
            dtype=np.float32,
        )

        force_norm = np.asarray(
            sample.tactile_force_norm,
            dtype=np.float32,
        )

        tactile_force.update(
            force
        )

        force_abs_chunks.append(
            np.abs(
                force
            ).reshape(
                -1
            )
        )

        force_norm_chunks.append(
            force_norm.reshape(
                -1
            )
        )

        frame_max_force.append(
            float(
                np.max(
                    force_norm
                )
            )
        )

        total_tactile_points += int(
            force_norm.size
        )

        for threshold in activity_thresholds:
            activity_counts[
                float(
                    threshold
                )
            ] += int(
                np.count_nonzero(
                    force_norm
                    > threshold
                )
            )

        finger_id = np.asarray(
            sample.finger_id
        )

        for finger in finger_force_chunks:
            finger_mask = (
                finger_id
                == finger
            )

            finger_force_chunks[
                finger
            ].append(
                force_norm[
                    finger_mask
                ]
            )

        if (
            global_index
            + 1
        ) % 250 == 0:
            print(
                "processed:",
                global_index
                + 1,
                "/",
                len(
                    dataset
                ),
            )

    all_force_norm = np.concatenate(
        force_norm_chunks,
        axis=0,
    )

    all_force_abs = np.concatenate(
        force_abs_chunks,
        axis=0,
    )

    frame_max_force_array = np.asarray(
        frame_max_force,
        dtype=np.float32,
    )

    rgb_valid_ratio_array = np.asarray(
        per_frame_rgb_valid_ratio,
        dtype=np.float64,
    )

    force_by_finger = {}

    for finger, chunks in finger_force_chunks.items():
        values = np.concatenate(
            chunks,
            axis=0,
        )

        force_by_finger[
            str(
                finger
            )
        ] = {
            "count": int(
                values.size
            ),
            "percentiles": (
                percentile_report(
                    values
                )
            ),
            "mean": float(
                np.mean(
                    values,
                    dtype=np.float64,
                )
            ),
            "std": float(
                np.std(
                    values,
                    dtype=np.float64,
                )
            ),
            "max": float(
                np.max(
                    values
                )
            ),
        }

    activity = {}

    for threshold in activity_thresholds:
        count = activity_counts[
            float(
                threshold
            )
        ]

        activity[
            f"gt_{threshold:g}"
        ] = {
            "count": int(
                count
            ),
            "ratio": float(
                count
                / total_tactile_points
            ),
        }

    report = {
        "dataset": {
            "frames": int(
                len(
                    dataset
                )
            ),
            "episodes": [
                int(
                    value
                )
                for value
                in dataset.episode_ids
            ],
            "camera_roles": list(
                dataset.camera_roles
            ),
            "visual_points_per_frame": int(
                dataset.visual_points_per_frame
            ),
            "tactile_points_per_frame": int(
                dataset.tactile_points_per_frame
            ),
            "coordinate_frame": (
                dataset.coordinate_frame
            ),
            "xyz_unit": (
                dataset.xyz_unit
            ),
            "force_unit": (
                dataset.force_unit
            ),
        },
        "visual_xyz": (
            visual_xyz.report()
        ),
        "tactile_xyz": (
            tactile_xyz.report()
        ),
        "visual_rgb_valid": {
            "valid_points": int(
                rgb_valid_count
            ),
            "total_points": int(
                rgb_total_count
            ),
            "global_ratio": float(
                rgb_valid_count
                / rgb_total_count
            ),
            "per_frame_ratio_percentiles": (
                percentile_report(
                    rgb_valid_ratio_array
                )
            ),
        },
        "visual_rgb_valid_only": (
            valid_rgb.report()
        ),
        "tactile_force": (
            tactile_force.report()
        ),
        "tactile_force_abs": {
            "count": int(
                all_force_abs.size
            ),
            "percentiles": (
                percentile_report(
                    all_force_abs
                )
            ),
            "mean": float(
                np.mean(
                    all_force_abs,
                    dtype=np.float64,
                )
            ),
            "std": float(
                np.std(
                    all_force_abs,
                    dtype=np.float64,
                )
            ),
        },
        "tactile_force_norm": {
            "count": int(
                all_force_norm.size
            ),
            "percentiles": (
                percentile_report(
                    all_force_norm
                )
            ),
            "mean": float(
                np.mean(
                    all_force_norm,
                    dtype=np.float64,
                )
            ),
            "std": float(
                np.std(
                    all_force_norm,
                    dtype=np.float64,
                )
            ),
        },
        "tactile_force_norm_by_finger": (
            force_by_finger
        ),
        "tactile_activity": (
            activity
        ),
        "frame_max_force_norm": {
            "count": int(
                frame_max_force_array.size
            ),
            "percentiles": (
                percentile_report(
                    frame_max_force_array
                )
            ),
            "mean": float(
                np.mean(
                    frame_max_force_array,
                    dtype=np.float64,
                )
            ),
            "std": float(
                np.std(
                    frame_max_force_array,
                    dtype=np.float64,
                )
            ),
        },
    }

    return report


# =============================================================================
# 5. Entry point
# =============================================================================

def main() -> None:
    args = parse_args()

    repo_root = Path(
        "."
    ).resolve()

    dataset_root = (
        args.dataset
        if args.dataset.is_absolute()
        else (
            repo_root
            / args.dataset
        )
    ).expanduser().resolve()

    dataset = SpatialDerivedDataset(
        dataset_root,
        version=args.version,
    )

    report = compute_stats(
        dataset
    )

    output_path = (
        args.output
        if args.output is not None
        else (
            dataset_root
            / "spatial"
            / args.version
            / "stats.json"
        )
    )

    if not output_path.is_absolute():
        output_path = (
            repo_root
            / output_path
        )

    output_path = (
        output_path
        .expanduser()
        .resolve()
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\n===== SUMMARY ====="
    )

    print(
        "RGB valid ratio:",
        f"{report['visual_rgb_valid']['global_ratio']:.6f}",
    )

    force_norm = report[
        "tactile_force_norm"
    ][
        "percentiles"
    ]

    print(
        "force norm p50/p90/p95/p99/p99.5/max:",
        force_norm[
            "p50"
        ],
        force_norm[
            "p90"
        ],
        force_norm[
            "p95"
        ],
        force_norm[
            "p99"
        ],
        force_norm[
            "p99.5"
        ],
        force_norm[
            "p100"
        ],
    )

    print(
        "frame max force p50/p90/p95/p99/max:",
        report[
            "frame_max_force_norm"
        ][
            "percentiles"
        ][
            "p50"
        ],
        report[
            "frame_max_force_norm"
        ][
            "percentiles"
        ][
            "p90"
        ],
        report[
            "frame_max_force_norm"
        ][
            "percentiles"
        ][
            "p95"
        ],
        report[
            "frame_max_force_norm"
        ][
            "percentiles"
        ][
            "p99"
        ],
        report[
            "frame_max_force_norm"
        ][
            "percentiles"
        ][
            "p100"
        ],
    )

    print(
        "JSON:",
        output_path,
    )


if __name__ == "__main__":
    main()
