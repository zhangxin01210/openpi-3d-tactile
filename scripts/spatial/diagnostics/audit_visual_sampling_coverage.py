"""
OpenPI 3D + tactile：视觉采样覆盖率审计（audit_visual_sampling_coverage.py）

作用
----
前面的网页已经说明一个直观事实：

    sampling 前的 dense / voxel point cloud 很密，
    最终 4096 点肉眼看起来明显稀疏。

但“看起来稀”不等于“对模型空间覆盖不足”。

本脚本专门量化：

    2048 / 4096 / 8192 个 Morton-stride visual points
    对 5 mm voxel candidate cloud 的空间覆盖到底是多少。

核心问题
--------
对每个 voxel candidate point，找最近的 sampled point：

    d(candidate -> sampled)

统计：
    median
    p90
    p95
    p99
    max

以及：
    多少 candidate 在 5 / 10 / 15 / 20 mm 内能找到 sampled point。

同时统计 sampled points 自身的最近邻距离：

    d(sample_i -> nearest other sample)

它反映 final point cloud 的典型稀疏程度。

为什么不是继续看 dense cloud
---------------------------
Dense cloud 是 QA 用的；
model 真正吃的是固定 N 点。

因此这一项回答的是：
    “4096 是否在当前 workspace / voxel size 下形成合理覆盖？”

而不是：
    “人眼 zoom 后是不是像连续表面？”

当前 baseline
-------------
    voxel = 5 mm
    sampler = morton_stride
    N = 4096

默认比较：
    2048,4096,8192

注意
----
- tactile 600 点不参与 visual sampling。
- 本脚本不修改 baseline config。
- 本脚本只做 coverage audit，不代表 N 越大一定越好；
  N 增大会直接增加 encoder compute / memory。
- candidate voxelization 在脚本内独立实现，并和
  build_visual_geometry().voxel_candidate_count 做 count parity，
  用来检查审计基准没有偏离正式 preprocessing。

基本使用
--------
    PYTHONPATH=src python scripts/audit_visual_sampling_coverage.py \
        --legacy-repo ../3D_tactile \
        --dataset data/press_0828_17 \
        --episode 0 \
        --frames 0,50,100,213 \
        --cameras front,left

单相机也可以：

    ... --cameras front
    ... --cameras left

输出默认：
    output/visual_sampling_coverage_ep0_front-left.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from openpi.spatial.calibration import load_camera_calibrations
from openpi.spatial.config import make_baseline_config
from openpi.spatial.geometry import build_camera_cache
from openpi.spatial.geometry import build_visual_geometry
from openpi.spatial.geometry import depth_to_base_roi


# =============================================================================
# 1. CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit spatial coverage of fixed-N visual sampling."
    )

    parser.add_argument(
        "--legacy-repo",
        type=Path,
        default=Path("../3D_tactile"),
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/press_0828_17"),
    )

    parser.add_argument(
        "--episode",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--frames",
        type=str,
        default="0,50,100,213",
    )

    parser.add_argument(
        "--cameras",
        type=str,
        default="front,left",
    )

    parser.add_argument(
        "--num-points",
        type=str,
        default="2048,4096,8192",
        help="逗号分隔要审计的 visual point 数。",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    return parser.parse_args()


# =============================================================================
# 2. Dataset adapter
# =============================================================================

def load_frames(
    *,
    legacy_repo: Path,
    dataset_relative: Path,
    episode: int,
    frames: list[int],
    camera_roles: tuple[str, ...],
) -> dict[
    int,
    tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
    ],
]:
    """
    读取多帧 depth / RGB。

    这里只负责旧 dataset adapter。
    """
    legacy_src = (
        legacy_repo
        / "pointcloud_delivery"
        / "src"
    ).resolve()

    sys.path.insert(
        0,
        str(
            legacy_src
        ),
    )

    try:
        from dataset import Dataset
        from dataset import video_frames
    finally:
        sys.path.pop(
            0
        )

    dataset = Dataset(
        (
            legacy_repo
            / dataset_relative
        ).resolve(),
        episode,
    )

    rows = {
        int(
            row[
                "frame_index"
            ]
        ): row
        for row in dataset.rows(
            frames
        )
    }

    missing = sorted(
        set(
            frames
        )
        - set(
            rows
        )
    )

    if missing:
        raise KeyError(
            f"Dataset missing frames: {missing}"
        )

    output = {}

    for frame in frames:
        row = rows[
            frame
        ]

        depth_by_role = {}
        rgb_by_role = {}

        for role in camera_roles:
            depth_by_role[
                role
            ] = np.asarray(
                row[
                    f"observation.depths.cam_{role}"
                ]
            )

            decoded = video_frames(
                dataset.video(
                    role
                ),
                [frame],
            )

            images = (
                decoded[0]
                if isinstance(
                    decoded,
                    tuple,
                )
                else decoded
            )

            rgb = (
                images[
                    frame
                ]
                if isinstance(
                    images,
                    dict,
                )
                else images[
                    0
                ]
            )

            rgb_by_role[
                role
            ] = np.asarray(
                rgb,
                dtype=np.uint8,
            )

        output[
            frame
        ] = (
            depth_by_role,
            rgb_by_role,
        )

    return output


# =============================================================================
# 3. sampling 前 dense ROI
# =============================================================================

def build_merged_dense_roi(
    *,
    depth_by_role: dict[str, np.ndarray],
    calibrations,
    camera_roles: tuple[str, ...],
    config,
) -> np.ndarray:
    """
    使用正式 geometry.py 的 depth->base ROI，
    合并所有 camera 的 sampling 前 xyz。
    """
    pieces = []

    for role in camera_roles:
        depth = depth_by_role[
            role
        ]

        calibration = calibrations[
            role
        ]

        cache = build_camera_cache(
            calibration,
            image_height=depth.shape[0],
            image_width=depth.shape[1],
        )

        xyz, _ = depth_to_base_roi(
            depth,
            cache,
            roi=config.roi,
            min_depth_m=config.visual.min_depth_m,
            max_depth_m=config.visual.max_depth_m,
        )

        pieces.append(
            np.asarray(
                xyz,
                dtype=np.float64,
            )
        )

    if not pieces:
        raise RuntimeError(
            "No camera point clouds"
        )

    return np.concatenate(
        pieces,
        axis=0,
    )


# =============================================================================
# 4. 独立 voxel representative
# =============================================================================

def independent_voxel_representatives(
    xyz_m: np.ndarray,
    *,
    voxel_size_m: float,
    roi,
) -> np.ndarray:
    """
    独立实现 baseline voxel representative。

    voxel index 以 frozen ROI 最小角为原点：

        floor((xyz - roi_min) / voxel_size)

    同一 voxel 保留 merged input 中第一个真实观测点，
    不使用 voxel center。

    这和正式设计 contract 一致，但实现独立于 geometry.py，
    因此可以对 candidate count 做 parity。
    """
    xyz = np.asarray(
        xyz_m,
        dtype=np.float64,
    )

    origin = np.array(
        [
            roi.x_min_m,
            roi.y_min_m,
            roi.z_min_m,
        ],
        dtype=np.float64,
    )

    voxel = np.floor(
        (
            xyz
            - origin[
                None,
                :
            ]
        )
        / float(
            voxel_size_m
        )
    ).astype(
        np.int64,
    )

    # np.unique 默认按 key 排序；
    # return_index 返回每个 key 在原数组第一次出现的位置。
    # 我们只需要 representative 集合，最后按原 input index 排序，
    # 保持“first observed point”的顺序语义。
    structured = np.ascontiguousarray(
        voxel
    ).view(
        [
            ("x", np.int64),
            ("y", np.int64),
            ("z", np.int64),
        ]
    ).reshape(
        -1
    )

    _, first = np.unique(
        structured,
        return_index=True,
    )

    first.sort()

    return xyz[
        first
    ]


# =============================================================================
# 5. Coverage metrics
# =============================================================================

def nearest_neighbor_metrics(
    *,
    candidates: np.ndarray,
    sampled: np.ndarray,
) -> dict[str, object]:
    """
    计算：
        candidate -> nearest sampled
        sampled -> nearest sampled

    需要 scipy cKDTree；这里只是 audit dependency，
    不进入 preprocessing core。
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError(
            "Sampling coverage audit requires scipy. "
            "Install with: pip install scipy"
        ) from exc

    candidate_tree = cKDTree(
        np.asarray(
            sampled,
            dtype=np.float64,
        )
    )

    candidate_distance_m, _ = (
        candidate_tree.query(
            np.asarray(
                candidates,
                dtype=np.float64,
            ),
            k=1,
            workers=-1,
        )
    )

    candidate_distance_mm = (
        candidate_distance_m
        * 1000.0
    )

    sample_tree = cKDTree(
        np.asarray(
            sampled,
            dtype=np.float64,
        )
    )

    # k=2：
    # 第一邻居是自己，第二邻居才是其它 sampled point。
    sample_nn_m, _ = sample_tree.query(
        np.asarray(
            sampled,
            dtype=np.float64,
        ),
        k=2,
        workers=-1,
    )

    sample_nn_mm = (
        sample_nn_m[
            :,
            1
        ]
        * 1000.0
    )

    def summary(
        values: np.ndarray,
    ) -> dict[str, float]:
        return {
            "median_mm": float(
                np.median(
                    values
                )
            ),
            "p90_mm": float(
                np.percentile(
                    values,
                    90,
                )
            ),
            "p95_mm": float(
                np.percentile(
                    values,
                    95,
                )
            ),
            "p99_mm": float(
                np.percentile(
                    values,
                    99,
                )
            ),
            "max_mm": float(
                np.max(
                    values
                )
            ),
        }

    return {
        "candidate_to_sample": summary(
            candidate_distance_mm
        ),
        "candidate_coverage_fraction": {
            "within_5mm": float(
                np.mean(
                    candidate_distance_mm
                    <= 5.0
                )
            ),
            "within_10mm": float(
                np.mean(
                    candidate_distance_mm
                    <= 10.0
                )
            ),
            "within_15mm": float(
                np.mean(
                    candidate_distance_mm
                    <= 15.0
                )
            ),
            "within_20mm": float(
                np.mean(
                    candidate_distance_mm
                    <= 20.0
                )
            ),
        },
        "sample_nearest_other": summary(
            sample_nn_mm
        ),
    }


# =============================================================================
# 6. Main
# =============================================================================

def main() -> None:
    args = parse_args()

    repo_root = Path(
        "."
    ).resolve()

    legacy_repo = (
        args.legacy_repo
        if args.legacy_repo.is_absolute()
        else (
            repo_root
            / args.legacy_repo
        )
    ).resolve()

    camera_roles = tuple(
        role.strip()
        for role in args.cameras.split(
            ","
        )
        if role.strip()
    )

    frames = [
        int(
            item.strip()
        )
        for item in args.frames.split(
            ","
        )
        if item.strip()
    ]

    point_counts = [
        int(
            item.strip()
        )
        for item in args.num_points.split(
            ","
        )
        if item.strip()
    ]

    if not camera_roles:
        raise ValueError(
            "--cameras cannot be empty"
        )

    if not frames:
        raise ValueError(
            "--frames cannot be empty"
        )

    if not point_counts:
        raise ValueError(
            "--num-points cannot be empty"
        )

    cfg = (
        make_baseline_config()
        .with_camera_roles(
            *camera_roles
        )
    )

    calibration_bundle = (
        load_camera_calibrations(
            repo_root=repo_root,
            calibration_config=cfg.calibration,
            camera_roles=camera_roles,
        )
    )

    frame_data = load_frames(
        legacy_repo=legacy_repo,
        dataset_relative=args.dataset,
        episode=args.episode,
        frames=frames,
        camera_roles=camera_roles,
    )

    report = {
        "episode": int(
            args.episode
        ),
        "frames": frames,
        "camera_roles": list(
            camera_roles
        ),
        "voxel_size_m": float(
            cfg.visual.voxel_size_m
        ),
        "sampler": str(
            cfg.visual.sampler
        ),
        "point_counts": point_counts,
        "per_frame": {},
    }

    for frame in frames:
        (
            depth_by_role,
            rgb_by_role,
        ) = frame_data[
            frame
        ]

        dense = build_merged_dense_roi(
            depth_by_role=depth_by_role,
            calibrations=(
                calibration_bundle.cameras
            ),
            camera_roles=camera_roles,
            config=cfg,
        )

        candidates = (
            independent_voxel_representatives(
                dense,
                voxel_size_m=(
                    cfg.visual.voxel_size_m
                ),
                roi=cfg.roi,
            )
        )

        frame_report = {
            "dense_roi_count": int(
                len(
                    dense
                )
            ),
            "independent_voxel_candidate_count": int(
                len(
                    candidates
                )
            ),
            "sampling": {},
        }

        print(
            f"\n===== FRAME {frame} ====="
        )
        print(
            "dense ROI:",
            len(
                dense
            ),
        )
        print(
            "independent voxel candidates:",
            len(
                candidates
            ),
        )

        for count in point_counts:
            sample_cfg = (
                cfg.with_visual_sampling(
                    num_points=count,
                )
            )

            visual = build_visual_geometry(
                depth_by_role=depth_by_role,
                rgb_by_role=rgb_by_role,
                calibrations=(
                    calibration_bundle.cameras
                ),
                config=sample_cfg.visual,
                roi=sample_cfg.roi,
                diagnostics=(
                    sample_cfg.diagnostics
                ),
            )

            candidate_count_match = (
                int(
                    visual.voxel_candidate_count
                )
                == len(
                    candidates
                )
            )

            metrics = (
                nearest_neighbor_metrics(
                    candidates=candidates,
                    sampled=(
                        visual.xyz_m
                    ),
                )
            )

            item = {
                "final_count": int(
                    len(
                        visual.xyz_m
                    )
                ),
                "official_voxel_candidate_count": int(
                    visual.voxel_candidate_count
                ),
                "candidate_count_match": bool(
                    candidate_count_match
                ),
                "rgb_valid_ratio": float(
                    visual.rgb_valid.mean()
                ),
                **metrics,
            }

            frame_report[
                "sampling"
            ][
                str(
                    count
                )
            ] = item

            c2s = item[
                "candidate_to_sample"
            ]

            coverage = item[
                "candidate_coverage_fraction"
            ]

            spacing = item[
                "sample_nearest_other"
            ]

            print(
                f"N={count}",
                "| candidate parity=",
                candidate_count_match,
                "| coverage p95(mm)=",
                f"{c2s['p95_mm']:.3f}",
                "| p99=",
                f"{c2s['p99_mm']:.3f}",
                "| <=10mm=",
                f"{coverage['within_10mm']:.3f}",
                "| <=15mm=",
                f"{coverage['within_15mm']:.3f}",
                "| sample NN median(mm)=",
                f"{spacing['median_mm']:.3f}",
            )

        report[
            "per_frame"
        ][
            str(
                frame
            )
        ] = frame_report

    # -------------------------------------------------------------------------
    # Global summary
    # -------------------------------------------------------------------------
    print(
        "\n===== GLOBAL SUMMARY ====="
    )

    report[
        "global"
    ] = {}

    for count in point_counts:
        key = str(
            count
        )

        p95_values = [
            report[
                "per_frame"
            ][
                str(
                    frame
                )
            ][
                "sampling"
            ][
                key
            ][
                "candidate_to_sample"
            ][
                "p95_mm"
            ]
            for frame in frames
        ]

        within10 = [
            report[
                "per_frame"
            ][
                str(
                    frame
                )
            ][
                "sampling"
            ][
                key
            ][
                "candidate_coverage_fraction"
            ][
                "within_10mm"
            ]
            for frame in frames
        ]

        within15 = [
            report[
                "per_frame"
            ][
                str(
                    frame
                )
            ][
                "sampling"
            ][
                key
            ][
                "candidate_coverage_fraction"
            ][
                "within_15mm"
            ]
            for frame in frames
        ]

        global_item = {
            "coverage_p95_mm_median_across_frames": float(
                np.median(
                    p95_values
                )
            ),
            "coverage_p95_mm_max_across_frames": float(
                np.max(
                    p95_values
                )
            ),
            "within_10mm_mean_across_frames": float(
                np.mean(
                    within10
                )
            ),
            "within_15mm_mean_across_frames": float(
                np.mean(
                    within15
                )
            ),
        }

        report[
            "global"
        ][
            key
        ] = global_item

        print(
            f"N={count}",
            "| p95 median/max(mm)=",
            f"{global_item['coverage_p95_mm_median_across_frames']:.3f}/"
            f"{global_item['coverage_p95_mm_max_across_frames']:.3f}",
            "| <=10mm mean=",
            f"{global_item['within_10mm_mean_across_frames']:.3f}",
            "| <=15mm mean=",
            f"{global_item['within_15mm_mean_across_frames']:.3f}",
        )

    all_candidate_parity = all(
        report[
            "per_frame"
        ][
            str(
                frame
            )
        ][
            "sampling"
        ][
            str(
                count
            )
        ][
            "candidate_count_match"
        ]
        for frame in frames
        for count in point_counts
    )

    report[
        "all_candidate_count_parity"
    ] = bool(
        all_candidate_parity
    )

    print(
        "candidate count parity:",
        all_candidate_parity,
    )

    # -------------------------------------------------------------------------
    # JSON output
    # -------------------------------------------------------------------------
    camera_tag = "-".join(
        camera_roles
    )

    output_path = (
        args.output
        if args.output is not None
        else Path(
            "output"
        )
        / (
            f"visual_sampling_coverage_ep{args.episode}_"
            f"{camera_tag}.json"
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
        "JSON:",
        output_path,
    )


if __name__ == "__main__":
    main()
