"""
OpenPI 3D + tactile：canonical SpatialPreprocessor 延迟基准
（benchmark_spatial_preprocess_latency.py）

作用
----
这个脚本专门比较：

    4096 visual points
    vs
    8192 visual points

在“完整 canonical preprocessing”上的运行时间差异。

计时范围
--------
计入：
    - 两相机 depth -> base_link
    - ROI
    - voxelization
    - Morton sampling
    - RGB association
    - state -> FK
    - 600 tactile xyz / force
    - SpatialObservation assembly + validate

不计入：
    - 从磁盘读取 parquet / video
    - H.264 / image decode
    - SpatialPreprocessor 初始化
    - URDF / YAML / JSON / CSV 静态资产加载
    - 后续 point encoder / policy forward

原因：
    train / deployment 中这些静态资产应该只初始化一次；
    我们当前要测的是“每帧 canonical preprocessing”的稳定成本。

为什么还不能只靠这个决定 4096 / 8192
------------------------------------
8192 的 preprocessing 成本可能只比 4096 多一点，
但它会把后续 point encoder 的输入数量翻倍。

因此本脚本回答的是：

    “8192 在 preprocessing 阶段贵多少？”

而不是最终回答：

    “policy 应该用 4096 还是 8192？”

完整决策还需要后续 point encoder 的 latency / memory / task performance。

Benchmark 方法
--------------
1. 先把真实 frame 的 depth / RGB / state 全部读入内存。
2. 分别构造 4096 / 8192 preprocessor。
3. warmup 若干轮。
4. 关闭 Python GC 后重复多轮计时。
5. 统计：
       median
       p90
       p95
       p99
       max
       mean
       FPS = 1000 / median_ms
6. 同时检查两种 N 下 tactile 输出完全一致，
   避免 benchmark 时偷偷改变了不相关分支。

默认：
    frames = 0,50,100,213
    repeats = 25
    warmup = 5
    point counts = 4096,8192

这样每个配置总计：
    4 frames × 25 = 100 次真实 preprocess

基本使用
--------
    PYTHONPATH=src python scripts/spatial/diagnostics/benchmark_spatial_preprocess_latency.py \
        --dataset data/press_0828_17 \
        --episode 0 \
        --frames 0,50,100,213 \
        --cameras front,left \
        --repeats 25 \
        --warmup 5

输出默认：
    output/spatial_preprocess_latency_ep0_front-left.json

注意
----
- 尽量在机器空闲时运行。
- 第一次运行前建议不要同时开大规模训练 / 编译任务。
- 这是 CPU preprocessing benchmark；GPU policy 不在本脚本范围。
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np

from openpi.spatial.config import make_baseline_config
from openpi.spatial.preprocess import SpatialPreprocessor
from openpi.spatial_dataset.source import RawSpatialDataset


# =============================================================================
# 1. CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark canonical spatial preprocessing latency."
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
        default="4096,8192",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    return parser.parse_args()


# =============================================================================
# 2. 旧 Dataset adapter：只在计时前读取一次
# =============================================================================

def load_frames(
    *,
    dataset_root: Path,
    episode: int,
    frames: list[int],
    camera_roles: tuple[str, ...],
) -> dict[
    int,
    dict[str, object],
]:
    """
    把 benchmark 需要的真实 frame 全部读入内存。

    返回：
        {
            frame: {
                "state": ...,
                "depth_by_role": ...,
                "rgb_by_role": ...,
            }
        }

    dataset / video decode 不进入 latency 计时。

    每个 camera video 对这一批 selected frames 只顺序 decode 一次。
    """
    dataset = RawSpatialDataset(
        dataset_root,
        episode=episode,
    )

    rows = {
        int(
            row[
                "frame_index"
            ]
        ): row
        for row in dataset.iter_rows(
            selected=frames,
            depth_roles=camera_roles,
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

    rgb_frames_by_role = {
        role: dataset.load_video_frames(
            role,
            selected=frames,
        )
        for role in camera_roles
    }

    output = {}

    for frame in frames:
        row = rows[
            frame
        ]

        state = np.asarray(
            row[
                "observation.state"
            ]
        )

        depth_by_role = {
            role: np.asarray(
                row[
                    f"observation.depths.cam_{role}"
                ]
            )
            for role in camera_roles
        }

        rgb_by_role = {
            role: np.asarray(
                rgb_frames_by_role[
                    role
                ][
                    frame
                ],
                dtype=np.uint8,
            )
            for role in camera_roles
        }

        output[
            frame
        ] = {
            "state": state,
            "depth_by_role": depth_by_role,
            "rgb_by_role": rgb_by_role,
        }

    return output


# =============================================================================
# 3. 单次 preprocess
# =============================================================================

def run_one(
    *,
    preprocessor: SpatialPreprocessor,
    frame: int,
    frame_data: dict[str, object],
):
    """
    执行一次完整 canonical preprocess。

    独立封装的目的：
        benchmark 主循环只负责计时，
        避免把其它逻辑混进 timed region。
    """
    return preprocessor.preprocess(
        frame_index=frame,
        timestamp_s=0.0,
        state=frame_data[
            "state"
        ],
        depth_by_role=frame_data[
            "depth_by_role"
        ],
        rgb_by_role=frame_data[
            "rgb_by_role"
        ],
    )


# =============================================================================
# 4. Latency summary
# =============================================================================

def summarize_ms(
    values_ms: np.ndarray,
) -> dict[str, float | int]:
    """常用 latency 统计。"""
    values = np.asarray(
        values_ms,
        dtype=np.float64,
    )

    median = float(
        np.median(
            values
        )
    )

    return {
        "count": int(
            values.size
        ),
        "mean_ms": float(
            np.mean(
                values
            )
        ),
        "median_ms": median,
        "p90_ms": float(
            np.percentile(
                values,
                90,
            )
        ),
        "p95_ms": float(
            np.percentile(
                values,
                95,
            )
        ),
        "p99_ms": float(
            np.percentile(
                values,
                99,
            )
        ),
        "min_ms": float(
            np.min(
                values
            )
        ),
        "max_ms": float(
            np.max(
                values
            )
        ),
        "fps_from_median": (
            1000.0
            / median
            if median > 0
            else float(
                "inf"
            )
        ),
    }


# =============================================================================
# 5. 单配置 benchmark
# =============================================================================

def benchmark_config(
    *,
    preprocessor: SpatialPreprocessor,
    frames: list[int],
    loaded_frames: dict[
        int,
        dict[str, object],
    ],
    warmup: int,
    repeats: int,
) -> tuple[
    dict[str, object],
    dict[int, object],
]:
    """
    benchmark 某一个 fixed visual N。

    返回：
        latency report
        每个 frame 最后一次 SpatialObservation
            -> 用于跨 N 的 tactile parity 检查
    """

    # -------------------------------------------------------------------------
    # 5.1 Warmup
    # -------------------------------------------------------------------------
    for _ in range(
        warmup
    ):
        for frame in frames:
            run_one(
                preprocessor=preprocessor,
                frame=frame,
                frame_data=loaded_frames[
                    frame
                ],
            )

    # -------------------------------------------------------------------------
    # 5.2 Timed runs
    # -------------------------------------------------------------------------
    all_times_ms = []
    per_frame_times = {
        frame: []
        for frame in frames
    }

    last_observation = {}

    gc_was_enabled = gc.isenabled()

    try:
        gc.disable()

        for _ in range(
            repeats
        ):
            for frame in frames:
                start_ns = (
                    time.perf_counter_ns()
                )

                observation = run_one(
                    preprocessor=preprocessor,
                    frame=frame,
                    frame_data=loaded_frames[
                        frame
                    ],
                )

                end_ns = (
                    time.perf_counter_ns()
                )

                elapsed_ms = (
                    end_ns
                    - start_ns
                ) / 1e6

                all_times_ms.append(
                    elapsed_ms
                )

                per_frame_times[
                    frame
                ].append(
                    elapsed_ms
                )

                last_observation[
                    frame
                ] = observation

    finally:
        if gc_was_enabled:
            gc.enable()

    # -------------------------------------------------------------------------
    # 5.3 Summary
    # -------------------------------------------------------------------------
    report = {
        "overall": summarize_ms(
            np.asarray(
                all_times_ms
            )
        ),
        "per_frame": {
            str(
                frame
            ): summarize_ms(
                np.asarray(
                    per_frame_times[
                        frame
                    ]
                )
            )
            for frame in frames
        },
    }

    return (
        report,
        last_observation,
    )


# =============================================================================
# 6. 跨 N 检查 tactile 不应变化
# =============================================================================

def tactile_parity(
    observations_by_n: dict[
        int,
        dict[int, object],
    ],
    *,
    frames: list[int],
    point_counts: list[int],
) -> dict[str, object]:
    """
    visual num_points 改变不应该影响 tactile branch。

    以最小 N 作为 reference。
    """
    reference_n = point_counts[
        0
    ]

    report = {
        "reference_num_points": int(
            reference_n
        ),
        "comparisons": {},
    }

    for count in point_counts[
        1:
    ]:
        xyz_max = 0.0
        force_max = 0.0
        norm_max = 0.0
        ids_match = True

        for frame in frames:
            reference = observations_by_n[
                reference_n
            ][
                frame
            ]

            current = observations_by_n[
                count
            ][
                frame
            ]

            xyz_max = max(
                xyz_max,
                float(
                    np.max(
                        np.abs(
                            reference.tactile_xyz_m
                            - current.tactile_xyz_m
                        )
                    )
                ),
            )

            force_max = max(
                force_max,
                float(
                    np.max(
                        np.abs(
                            reference.tactile_force_base
                            - current.tactile_force_base
                        )
                    )
                ),
            )

            norm_max = max(
                norm_max,
                float(
                    np.max(
                        np.abs(
                            reference.tactile_force_norm
                            - current.tactile_force_norm
                        )
                    )
                ),
            )

            ids_match = (
                ids_match
                and np.array_equal(
                    reference.finger_id,
                    current.finger_id,
                )
                and np.array_equal(
                    reference.taxel_id,
                    current.taxel_id,
                )
            )

        report[
            "comparisons"
        ][
            str(
                count
            )
        ] = {
            "tactile_xyz_max_abs_m": xyz_max,
            "tactile_force_max_abs": force_max,
            "tactile_norm_max_abs": norm_max,
            "ids_match": bool(
                ids_match
            ),
        }

    return report


# =============================================================================
# 7. Main
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

    if not dataset_root.is_dir():
        raise FileNotFoundError(
            dataset_root
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

    camera_roles = tuple(
        item.strip()
        for item in args.cameras.split(
            ","
        )
        if item.strip()
    )

    point_counts = [
        int(
            item.strip()
        )
        for item in args.num_points.split(
            ","
        )
        if item.strip()
    ]

    if not frames:
        raise ValueError(
            "--frames cannot be empty"
        )

    if not camera_roles:
        raise ValueError(
            "--cameras cannot be empty"
        )

    if not point_counts:
        raise ValueError(
            "--num-points cannot be empty"
        )

    if args.repeats <= 0:
        raise ValueError(
            "--repeats must be > 0"
        )

    if args.warmup < 0:
        raise ValueError(
            "--warmup must be >= 0"
        )

    # -------------------------------------------------------------------------
    # 7.1 先把所有输入读入 RAM
    # -------------------------------------------------------------------------
    print(
        "Loading benchmark frames into memory..."
    )

    loaded_frames = load_frames(
        dataset_root=dataset_root,
        episode=args.episode,
        frames=frames,
        camera_roles=camera_roles,
    )

    print(
        "Loaded:",
        frames,
    )

    # -------------------------------------------------------------------------
    # 7.2 初始化不同 visual-N 的 canonical preprocessors
    # -------------------------------------------------------------------------
    preprocessors = {}

    for count in point_counts:
        cfg = (
            make_baseline_config()
            .with_camera_roles(
                *camera_roles
            )
            .with_visual_sampling(
                num_points=count,
            )
        )

        preprocessors[
            count
        ] = (
            SpatialPreprocessor.from_repo_root(
                repo_root=repo_root,
                config=cfg,
            )
        )

    # -------------------------------------------------------------------------
    # 7.3 benchmark
    # -------------------------------------------------------------------------
    report = {
        "episode": int(
            args.episode
        ),
        "frames": frames,
        "camera_roles": list(
            camera_roles
        ),
        "warmup_rounds": int(
            args.warmup
        ),
        "repeat_rounds": int(
            args.repeats
        ),
        "point_counts": point_counts,
        "timing_scope": (
            "SpatialPreprocessor.preprocess only; "
            "dataset/video decode and static initialization excluded"
        ),
        "results": {},
    }

    observations_by_n = {}

    for count in point_counts:
        print(
            f"\n===== N={count} ====="
        )

        result, observations = (
            benchmark_config(
                preprocessor=preprocessors[
                    count
                ],
                frames=frames,
                loaded_frames=loaded_frames,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        )

        report[
            "results"
        ][
            str(
                count
            )
        ] = result

        observations_by_n[
            count
        ] = observations

        overall = result[
            "overall"
        ]

        print(
            "count:",
            overall[
                "count"
            ],
        )
        print(
            "mean/median [ms]:",
            f"{overall['mean_ms']:.3f}/"
            f"{overall['median_ms']:.3f}",
        )
        print(
            "p90/p95/p99 [ms]:",
            f"{overall['p90_ms']:.3f}/"
            f"{overall['p95_ms']:.3f}/"
            f"{overall['p99_ms']:.3f}",
        )
        print(
            "min/max [ms]:",
            f"{overall['min_ms']:.3f}/"
            f"{overall['max_ms']:.3f}",
        )
        print(
            "FPS from median:",
            f"{overall['fps_from_median']:.2f}",
        )

    # -------------------------------------------------------------------------
    # 7.4 相对开销
    # -------------------------------------------------------------------------
    baseline_n = point_counts[
        0
    ]

    baseline_median = (
        report[
            "results"
        ][
            str(
                baseline_n
            )
        ][
            "overall"
        ][
            "median_ms"
        ]
    )

    comparison = {}

    print(
        "\n===== RELATIVE TO BASELINE ====="
    )

    for count in point_counts:
        median = (
            report[
                "results"
            ][
                str(
                    count
                )
            ][
                "overall"
            ][
                "median_ms"
            ]
        )

        delta_ms = (
            median
            - baseline_median
        )

        ratio = (
            median
            / baseline_median
            if baseline_median > 0
            else float(
                "nan"
            )
        )

        comparison[
            str(
                count
            )
        ] = {
            "median_delta_ms": float(
                delta_ms
            ),
            "median_ratio": float(
                ratio
            ),
            "median_percent_change": float(
                (
                    ratio
                    - 1.0
                )
                * 100.0
            ),
        }

        print(
            f"N={count}",
            "| Δmedian(ms)=",
            f"{delta_ms:+.3f}",
            "| ratio=",
            f"{ratio:.3f}",
            "| change=",
            f"{(ratio - 1.0) * 100.0:+.2f}%",
        )

    report[
        "relative_to_baseline"
    ] = comparison

    # -------------------------------------------------------------------------
    # 7.5 tactile branch parity
    # -------------------------------------------------------------------------
    tactile_report = tactile_parity(
        observations_by_n,
        frames=frames,
        point_counts=point_counts,
    )

    report[
        "tactile_parity"
    ] = tactile_report

    print(
        "\n===== TACTILE PARITY ACROSS N ====="
    )

    for count, item in tactile_report[
        "comparisons"
    ].items():
        print(
            f"N={count}",
            "| xyz max abs(m)=",
            item[
                "tactile_xyz_max_abs_m"
            ],
            "| force max abs=",
            item[
                "tactile_force_max_abs"
            ],
            "| norm max abs=",
            item[
                "tactile_norm_max_abs"
            ],
            "| ids=",
            item[
                "ids_match"
            ],
        )

    # -------------------------------------------------------------------------
    # 7.6 输出 JSON
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
            f"spatial_preprocess_latency_ep{args.episode}_"
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
        "\nJSON:",
        output_path,
    )


if __name__ == "__main__":
    main()
