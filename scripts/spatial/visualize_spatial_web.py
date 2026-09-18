"""
OpenPI 3D + tactile：交互式网页空间观测查看器（visualize_spatial_web.py）

作用
----
上一版 matplotlib 查看器直接画最终模型输入：

    4096 visual + 600 tactile

它适合检查“最终 schema 能否生成”，但不适合人工检查局部几何，因为：
    1. 4096 本来就是经过 voxel + Morton sampling 的稀疏模型输入；
    2. matplotlib 3D 的交互缩放与局部查看能力有限。

本脚本专门用于人工 QA，输出一个可交互 HTML，并同时保留两种 visual 层：

    Dense ROI
        depth -> base_link -> ROI 后、sampling 前的稠密点云
        用于检查：
            局部边缘
            深度洞
            飞点
            多相机 ghost / gap
            接触区域

    Final 4096
        canonical SpatialPreprocessor 真正输出给模型的 visual points
        用于检查：
            模型实际看到的覆盖
            sampling 是否过稀
            RGB-valid 分布

再叠加：
    600 tactile points
    strongest tactile force vectors

为什么这个脚本不破坏“单一 preprocessing”原则
------------------------------------------
Dense ROI 不是另一套正式 preprocessing。

它直接调用 geometry.py 已冻结的低层函数：
    build_camera_cache()
    depth_to_base_roi()
    colorize_from_source_cameras()

Final 4096 则直接调用：
    SpatialPreprocessor.preprocess()

因此：
    正式模型输入只有一套；
    Dense ROI 只是 QA 时查看同一条 pipeline 在 sampling 前的中间状态。

网页能力
--------
- 鼠标旋转
- 滚轮连续缩放
- 拖拽平移
- hover 查看 xyz / camera / RGB-valid
- legend 点击隐藏 / 显示某一层
- 双击 legend 项可只看该层
- HTML 完全自包含，不依赖浏览器联网

默认输出
--------
    output/spatial_debug_ep0_f213_front-left.html

示例
----
双相机：

    PYTHONPATH=src python scripts/visualize_spatial_web.py \
        --legacy-repo ../3D_tactile \
        --dataset data/press_0828_17 \
        --episode 0 \
        --frame 213 \
        --cameras front,left \
        --open

单相机：

    ... --cameras front --open
    ... --cameras left --open

如果原始 ROI 点太多、浏览器较卡：

    ... --max-dense-points-per-camera 60000 --open

如果要完整保留所有 dense ROI 点：

    ... --max-dense-points-per-camera 0 --open

注意
----
1. Dense ROI 可能有 10~20 万点，所以 HTML 会明显大于普通网页。
2. Dense trace 默认 point size 很小；Final 4096 故意画得更大。
3. force arrow 的显示长度经过归一化，只表达方向和相对强弱，
   不是“米 / 牛顿”的绝对物理长度。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import webbrowser

import numpy as np

from openpi.spatial.config import make_baseline_config
from openpi.spatial.geometry import build_camera_cache
from openpi.spatial.geometry import colorize_from_source_cameras
from openpi.spatial.geometry import depth_to_base_roi
from openpi.spatial.preprocess import SpatialPreprocessor


# =============================================================================
# 1. CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export interactive HTML for dense + final spatial observation QA."
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
        "--frame",
        type=int,
        default=213,
    )

    parser.add_argument(
        "--cameras",
        type=str,
        default="front,left",
        help="例如 front / left / front,left。",
    )

    parser.add_argument(
        "--max-dense-points-per-camera",
        type=int,
        default=100000,
        help=(
            "每个相机最多写入 HTML 的 dense ROI 点数；"
            "0 表示全部保留。只影响 QA 网页，不影响正式 preprocessing。"
        ),
    )

    parser.add_argument(
        "--dense-point-size",
        type=float,
        default=1.3,
    )

    parser.add_argument(
        "--final-point-size",
        type=float,
        default=3.5,
    )

    parser.add_argument(
        "--tactile-point-size",
        type=float,
        default=4.5,
    )

    parser.add_argument(
        "--force-topk",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--force-arrow-max-m",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--open",
        action="store_true",
        help="生成后用默认浏览器打开 HTML。",
    )

    return parser.parse_args()


# =============================================================================
# 2. Dataset adapter
# =============================================================================

def load_real_frame(
    *,
    legacy_repo: Path,
    dataset_relative: Path,
    episode: int,
    frame: int,
    camera_roles: tuple[str, ...],
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """
    只负责旧 dataset -> raw arrays。

    正式 spatial preprocessing 不放在 adapter 中。
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

    dataset_path = (
        legacy_repo
        / dataset_relative
    ).resolve()

    dataset = Dataset(
        dataset_path,
        episode,
    )

    row = next(
        dataset.rows(
            [frame]
        )
    )

    state = np.asarray(
        row[
            "observation.state"
        ]
    )

    depth_by_role: dict[
        str,
        np.ndarray,
    ] = {}

    rgb_by_role: dict[
        str,
        np.ndarray,
    ] = {}

    for role in camera_roles:
        depth_key = (
            f"observation.depths.cam_{role}"
        )

        if depth_key not in row:
            raise KeyError(
                f"Dataset row missing {depth_key!r}"
            )

        depth_by_role[
            role
        ] = np.asarray(
            row[
                depth_key
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

        if isinstance(
            images,
            dict,
        ):
            rgb = images[
                frame
            ]
        else:
            rgb = images[
                0
            ]

        rgb_by_role[
            role
        ] = np.asarray(
            rgb,
            dtype=np.uint8,
        )

    return (
        state,
        depth_by_role,
        rgb_by_role,
    )


# =============================================================================
# 3. Dense ROI：使用正式 geometry.py 的 sampling 前中间层
# =============================================================================

def build_dense_roi_by_camera(
    *,
    preprocessor: SpatialPreprocessor,
    depth_by_role: dict[str, np.ndarray],
    rgb_by_role: dict[str, np.ndarray],
    camera_roles: tuple[str, ...],
) -> dict[
    str,
    dict[str, np.ndarray],
]:
    """
    对每个相机生成 ROI 内 sampling 前的稠密点云。

    返回：
        {
            role: {
                "xyz": [N,3],
                "rgb": [N,3],
                "rgb_valid": [N],
            }
        }

    注意：
        这里只作为 QA view。
        模型输入仍由 SpatialPreprocessor 生成。
    """
    result: dict[
        str,
        dict[str, np.ndarray],
    ] = {}

    cfg = preprocessor.config

    for role in camera_roles:
        calibration = (
            preprocessor
            .calibration_bundle
            .cameras[
                role
            ]
        )

        depth = np.asarray(
            depth_by_role[
                role
            ]
        )

        cache = build_camera_cache(
            calibration,
            image_height=depth.shape[0],
            image_width=depth.shape[1],
        )

        xyz, _flat_pixel = (
            depth_to_base_roi(
                depth,
                cache,
                roi=cfg.roi,
                min_depth_m=(
                    cfg.visual.min_depth_m
                ),
                max_depth_m=(
                    cfg.visual.max_depth_m
                ),
            )
        )

        source_camera_index = np.zeros(
            len(
                xyz
            ),
            dtype=np.int16,
        )

        rgb, rgb_valid = (
            colorize_from_source_cameras(
                xyz,
                source_camera_index,
                camera_roles=(
                    role,
                ),
                rgb_by_role={
                    role: rgb_by_role[
                        role
                    ],
                },
                calibrations={
                    role: calibration,
                },
            )
        )

        result[
            role
        ] = {
            "xyz": np.asarray(
                xyz,
                dtype=np.float32,
            ),
            "rgb": np.asarray(
                rgb,
                dtype=np.uint8,
            ),
            "rgb_valid": np.asarray(
                rgb_valid,
                dtype=np.bool_,
            ),
        }

    return result


# =============================================================================
# 4. 只为网页显示做 deterministic thinning
# =============================================================================

def thin_for_display(
    xyz: np.ndarray,
    rgb: np.ndarray,
    rgb_valid: np.ndarray,
    *,
    maximum: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    当 dense 点过多时，只为 HTML 大小 / 浏览器性能做等步长抽样。

    这个抽样不会回写正式 preprocessing，也不会进入模型。
    """
    count = len(
        xyz
    )

    if (
        maximum <= 0
        or count <= maximum
    ):
        return (
            xyz,
            rgb,
            rgb_valid,
        )

    indices = np.linspace(
        0,
        count - 1,
        num=maximum,
        dtype=np.int64,
    )

    return (
        xyz[
            indices
        ],
        rgb[
            indices
        ],
        rgb_valid[
            indices
        ],
    )


# =============================================================================
# 5. Plotly helpers
# =============================================================================

def rgb_to_plotly(
    rgb: np.ndarray,
    rgb_valid: np.ndarray,
) -> list[str]:
    """
    uint8 RGB -> Plotly rgb(r,g,b) 字符串。

    invalid RGB 用灰色，便于观察投影无效区域。
    """
    color = np.asarray(
        rgb,
        dtype=np.uint8,
    ).copy()

    invalid = ~np.asarray(
        rgb_valid,
        dtype=np.bool_,
    )

    color[
        invalid
    ] = np.array(
        [
            135,
            135,
            135,
        ],
        dtype=np.uint8,
    )

    return [
        f"rgb({int(r)},{int(g)},{int(b)})"
        for r, g, b in color
    ]


def tactile_color_strings(
    finger_id: np.ndarray,
) -> list[str]:
    """五根手指仅用于 QA 的固定显示颜色。"""
    palette = (
        "rgb(230,60,60)",    # thumb
        "rgb(55,105,230)",   # index
        "rgb(50,180,85)",    # middle
        "rgb(175,80,210)",   # ring
        "rgb(235,155,45)",   # pinky
    )

    return [
        palette[
            int(
                finger
            )
        ]
        for finger in finger_id
    ]


def force_line_trace(
    *,
    go,
    xyz_m: np.ndarray,
    force_base: np.ndarray,
    force_norm: np.ndarray,
    topk: int,
    arrow_max_m: float,
):
    """
    用一组 line segments 表示 strongest force direction。

    WebGL Scatter3d line 比 3D annotation 更轻量。
    """
    norm = np.asarray(
        force_norm,
        dtype=np.float64,
    )

    nonzero = np.flatnonzero(
        norm > 0
    )

    if len(
        nonzero
    ) == 0:
        return None

    count = min(
        int(
            topk
        ),
        len(
            nonzero
        ),
    )

    selected = nonzero[
        np.argsort(
            norm[
                nonzero
            ]
        )[
            -count:
        ]
    ]

    maximum = float(
        norm[
            selected
        ].max()
    )

    x: list[
        float | None
    ] = []
    y: list[
        float | None
    ] = []
    z: list[
        float | None
    ] = []

    for index in selected:
        force = np.asarray(
            force_base[
                index
            ],
            dtype=np.float64,
        )

        magnitude = float(
            np.linalg.norm(
                force
            )
        )

        if magnitude <= 0:
            continue

        direction = (
            force
            / magnitude
        )

        display_length = (
            norm[
                index
            ]
            / maximum
            * float(
                arrow_max_m
            )
        )

        start = np.asarray(
            xyz_m[
                index
            ],
            dtype=np.float64,
        )

        end = (
            start
            + direction
            * display_length
        )

        x.extend(
            [
                float(
                    start[0]
                ),
                float(
                    end[0]
                ),
                None,
            ]
        )
        y.extend(
            [
                float(
                    start[1]
                ),
                float(
                    end[1]
                ),
                None,
            ]
        )
        z.extend(
            [
                float(
                    start[2]
                ),
                float(
                    end[2]
                ),
                None,
            ]
        )

    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        name="tactile force direction",
        line={
            "width": 5,
            "color": "rgb(20,20,20)",
        },
        hoverinfo="skip",
        visible=True,
    )


# =============================================================================
# 6. Main
# =============================================================================

def main() -> None:
    args = parse_args()

    # Plotly 只属于这个 QA 脚本，不进入 openpi.spatial core dependency。
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError(
            "This QA viewer requires plotly. "
            "Install it in the current environment with: pip install plotly"
        ) from exc

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

    if not camera_roles:
        raise ValueError(
            "--cameras cannot be empty"
        )

    (
        state,
        depth_by_role,
        rgb_by_role,
    ) = load_real_frame(
        legacy_repo=legacy_repo,
        dataset_relative=args.dataset,
        episode=args.episode,
        frame=args.frame,
        camera_roles=camera_roles,
    )

    cfg = (
        make_baseline_config()
        .with_camera_roles(
            *camera_roles
        )
    )

    preprocessor = (
        SpatialPreprocessor.from_repo_root(
            repo_root=repo_root,
            config=cfg,
        )
    )

    # -------------------------------------------------------------------------
    # 6.1 Final canonical SpatialObservation
    # -------------------------------------------------------------------------
    observation = (
        preprocessor.preprocess(
            frame_index=args.frame,
            timestamp_s=0.0,
            state=state,
            depth_by_role=depth_by_role,
            rgb_by_role=rgb_by_role,
        )
    )

    # -------------------------------------------------------------------------
    # 6.2 Dense sampling-before intermediate
    # -------------------------------------------------------------------------
    dense_by_role = (
        build_dense_roi_by_camera(
            preprocessor=preprocessor,
            depth_by_role=depth_by_role,
            rgb_by_role=rgb_by_role,
            camera_roles=camera_roles,
        )
    )

    figure = go.Figure()

    # -------------------------------------------------------------------------
    # 6.3 每个 camera 单独一层 dense cloud
    #
    # legend 可独立开关，因此可以方便观察：
    #   front only
    #   left only
    #   front + left
    # -------------------------------------------------------------------------
    for role in camera_roles:
        dense = dense_by_role[
            role
        ]

        original_count = len(
            dense[
                "xyz"
            ]
        )

        (
            xyz,
            rgb,
            rgb_valid,
        ) = thin_for_display(
            dense[
                "xyz"
            ],
            dense[
                "rgb"
            ],
            dense[
                "rgb_valid"
            ],
            maximum=(
                args
                .max_dense_points_per_camera
            ),
        )

        customdata = np.column_stack(
            [
                rgb_valid.astype(
                    np.int8
                ),
            ]
        )

        figure.add_trace(
            go.Scatter3d(
                x=xyz[
                    :,
                    0
                ],
                y=xyz[
                    :,
                    1
                ],
                z=xyz[
                    :,
                    2
                ],
                mode="markers",
                name=(
                    f"dense {role} "
                    f"({len(xyz)}/{original_count})"
                ),
                marker={
                    "size": float(
                        args.dense_point_size
                    ),
                    "color": rgb_to_plotly(
                        rgb,
                        rgb_valid,
                    ),
                    "opacity": 0.88,
                },
                customdata=customdata,
                hovertemplate=(
                    f"camera={role}<br>"
                    "x=%{x:.5f} m<br>"
                    "y=%{y:.5f} m<br>"
                    "z=%{z:.5f} m<br>"
                    "rgb_valid=%{customdata[0]}"
                    "<extra></extra>"
                ),
            )
        )

    # -------------------------------------------------------------------------
    # 6.4 Final 4096：默认先隐藏，legend 点一下即可打开
    # -------------------------------------------------------------------------
    figure.add_trace(
        go.Scatter3d(
            x=observation.visual_xyz_m[
                :,
                0
            ],
            y=observation.visual_xyz_m[
                :,
                1
            ],
            z=observation.visual_xyz_m[
                :,
                2
            ],
            mode="markers",
            name="FINAL model visual 4096",
            marker={
                "size": float(
                    args.final_point_size
                ),
                "color": rgb_to_plotly(
                    observation.visual_rgb,
                    observation.visual_rgb_valid,
                ),
                "opacity": 1.0,
                "symbol": "diamond",
            },
            visible="legendonly",
            customdata=np.column_stack(
                [
                    observation
                    .visual_rgb_valid
                    .astype(
                        np.int8
                    )
                ]
            ),
            hovertemplate=(
                "FINAL 4096<br>"
                "x=%{x:.5f} m<br>"
                "y=%{y:.5f} m<br>"
                "z=%{z:.5f} m<br>"
                "rgb_valid=%{customdata[0]}"
                "<extra></extra>"
            ),
        )
    )

    # -------------------------------------------------------------------------
    # 6.5 600 tactile
    # -------------------------------------------------------------------------
    tactile_custom = np.column_stack(
        [
            observation.finger_id,
            observation.taxel_id,
            observation.tactile_force_norm,
        ]
    )

    figure.add_trace(
        go.Scatter3d(
            x=observation.tactile_xyz_m[
                :,
                0
            ],
            y=observation.tactile_xyz_m[
                :,
                1
            ],
            z=observation.tactile_xyz_m[
                :,
                2
            ],
            mode="markers",
            name="tactile 600",
            marker={
                "size": float(
                    args.tactile_point_size
                ),
                "color": tactile_color_strings(
                    observation.finger_id
                ),
                "opacity": 1.0,
            },
            customdata=tactile_custom,
            hovertemplate=(
                "tactile<br>"
                "x=%{x:.5f} m<br>"
                "y=%{y:.5f} m<br>"
                "z=%{z:.5f} m<br>"
                "finger_id=%{customdata[0]:.0f}<br>"
                "taxel_id=%{customdata[1]:.0f}<br>"
                "force_norm=%{customdata[2]:.4f}"
                "<extra></extra>"
            ),
        )
    )

    force_trace = force_line_trace(
        go=go,
        xyz_m=observation.tactile_xyz_m,
        force_base=observation.tactile_force_base,
        force_norm=observation.tactile_force_norm,
        topk=args.force_topk,
        arrow_max_m=args.force_arrow_max_m,
    )

    if force_trace is not None:
        figure.add_trace(
            force_trace
        )

    # -------------------------------------------------------------------------
    # 6.6 Layout
    # -------------------------------------------------------------------------
    dense_counts_text = ", ".join(
        (
            f"{role}={len(dense_by_role[role]['xyz'])}"
        )
        for role in camera_roles
    )

    figure.update_layout(
        title=(
            "Spatial QA | "
            f"episode={args.episode}, "
            f"frame={args.frame}, "
            f"cameras={','.join(camera_roles)}"
            "<br>"
            f"<sup>Dense ROI: {dense_counts_text}; "
            f"Final=4096; tactile=600; "
            f"final RGB-valid={float(observation.visual_rgb_valid.mean()):.3f}</sup>"
        ),
        scene={
            "xaxis_title": "base_link X [m]",
            "yaxis_title": "base_link Y [m]",
            "zaxis_title": "base_link Z [m]",
            "aspectmode": "data",
            "dragmode": "orbit",
        },
        legend={
            "itemsizing": "constant",
        },
        margin={
            "l": 0,
            "r": 0,
            "b": 0,
            "t": 85,
        },
        width=1400,
        height=900,
    )

    # -------------------------------------------------------------------------
    # 6.7 Output
    # -------------------------------------------------------------------------
    output_path = args.output

    if output_path is None:
        camera_tag = "-".join(
            camera_roles
        )

        output_path = Path(
            "output"
        ) / (
            f"spatial_debug_ep{args.episode}_"
            f"f{args.frame}_{camera_tag}.html"
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

    figure.write_html(
        str(
            output_path
        ),
        include_plotlyjs="inline",
        full_html=True,
        auto_open=False,
    )

    print(
        "frame:",
        args.frame,
    )
    print(
        "cameras:",
        camera_roles,
    )

    for role in camera_roles:
        dense = dense_by_role[
            role
        ]

        print(
            f"dense {role}:",
            len(
                dense[
                    "xyz"
                ]
            ),
            "RGB-valid=",
            float(
                dense[
                    "rgb_valid"
                ].mean()
            ),
        )

    print(
        "final visual:",
        observation.visual_xyz_m.shape,
        "RGB-valid=",
        float(
            observation
            .visual_rgb_valid
            .mean()
        ),
    )

    print(
        "tactile:",
        observation.tactile_xyz_m.shape,
    )

    strongest = int(
        np.argmax(
            observation
            .tactile_force_norm
        )
    )

    print(
        "strongest tactile:",
        "finger_id=",
        int(
            observation
            .finger_id[
                strongest
            ]
        ),
        "taxel_id=",
        int(
            observation
            .taxel_id[
                strongest
            ]
        ),
        "norm=",
        float(
            observation
            .tactile_force_norm[
                strongest
            ]
        ),
    )

    print(
        "HTML:",
        output_path,
    )

    if args.open:
        webbrowser.open(
            output_path.as_uri()
        )


if __name__ == "__main__":
    main()
