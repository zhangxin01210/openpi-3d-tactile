#!/usr/bin/env python3
"""Compare corrected front-only clouds before/after conservative depth-edge invalidation."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import cv2
import numpy as np

from audit_front_depth_cad import evaluate, mesh_groups, render_cad_depth, summarize
from depth_edge_filter import filter_depth_edges
from openpi.spatial.calibration_profile import apply_front_calibration_profile
from openpi.spatial.config import make_baseline_config
from openpi.spatial.geometry import apply_diagnostic_overrides, build_camera_cache, depth_to_base_roi
from openpi.spatial.preprocess import SpatialPreprocessor
from rgb_depth_joint_diagnostic import align_depth_to_color, colorize_depth, depth_range
from root_to_tip_mesh import load_visual_meshes, resolve_mesh_urdf
from visualize_spatial_web import (
    build_dense_roi_by_camera, cad_root_to_tip_traces, load_real_frame,
    rgb_to_plotly, thin_for_display,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def depth_mask_image(depth_m: np.ndarray, removed: np.ndarray) -> np.ndarray:
    image = colorize_depth(depth_m, depth_range(depth_m))
    image[removed] = (0, 0, 255)
    return image


def rgb_mask_image(rgb: np.ndarray, removed_depth_m: np.ndarray, camera) -> np.ndarray:
    projected = align_depth_to_color(removed_depth_m, camera, rgb.shape, splat="nearest")
    mask = np.isfinite(projected)
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    image[mask] = (0.4 * image[mask] + 0.6 * np.array((0, 0, 255))).astype(np.uint8)
    return image


def registered_depth_overlay(rgb: np.ndarray, depth_m: np.ndarray, camera,
                             limits: tuple[float, float]) -> np.ndarray:
    projected = align_depth_to_color(depth_m, camera, rgb.shape, splat="four")
    valid = np.isfinite(projected)
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    pseudo = colorize_depth(projected, limits)
    image[valid] = (0.5 * image[valid] + 0.5 * pseudo[valid]).astype(np.uint8)
    return image


def scatter(go, name: str, xyz: np.ndarray, rgb: np.ndarray | None = None,
            rgb_valid: np.ndarray | None = None, *, visible=True, size=1.5,
            fixed_color="rgb(235,60,60)"):
    colors = fixed_color if rgb is None else rgb_to_plotly(rgb, rgb_valid)
    return go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers", name=name,
        marker={"size": size, "color": colors, "opacity": 0.85}, visible=visible,
        hovertemplate="x=%{x:.4f} m<br>y=%{y:.4f} m<br>z=%{z:.4f} m<extra></extra>",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--profile", type=Path, required=True,
                        help="Front candidate containing extrinsic translation and color intrinsics")
    parser.add_argument("--jump-mm", type=float, default=25.0)
    parser.add_argument("--radius-px", type=int, default=1)
    parser.add_argument("--include-hole-boundaries", action="store_true")
    parser.add_argument("--max-dense-points", type=int, default=50000,
                        help="Per cloud; 0 keeps all dense points")
    parser.add_argument("--max-cad-edges-per-group", type=int, default=1500)
    parser.add_argument("--mesh-urdf", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    if args.frame < 0 or args.max_dense_points < 0 or args.max_cad_edges_per_group < 0:
        raise ValueError("Frame and display limits must be nonnegative")
    profile = args.profile.expanduser().resolve()
    profile_data = json.loads(profile.read_text(encoding="utf-8"))
    if not {"front_extrinsic_translation_base_m", "front_color_intrinsics"} <= profile_data.keys():
        raise ValueError("Profile must contain both front extrinsic translation and color intrinsics")
    dataset = args.dataset.expanduser().resolve()
    cfg = apply_front_calibration_profile(make_baseline_config().with_camera_roles("front"), profile)
    pre = SpatialPreprocessor.from_repo_root(repo_root=REPO_ROOT, config=cfg)
    camera = apply_diagnostic_overrides(pre.calibration_bundle.cameras, cfg.diagnostics)["front"]
    mesh_urdf = resolve_mesh_urdf(REPO_ROOT, args.mesh_urdf)

    timestamp, state, depths, rgbs = load_real_frame(
        dataset_root=dataset, episode=args.episode, frame=args.frame, camera_roles=("front",)
    )
    raw = np.asarray(depths["front"])
    filtered, removed, seeds = filter_depth_edges(
        raw, scale_m_per_unit=camera.depth_scale_m_per_unit,
        jump_mm=args.jump_mm, radius_px=args.radius_px,
        min_depth_m=cfg.visual.min_depth_m, max_depth_m=cfg.visual.max_depth_m,
        include_hole_boundaries=args.include_hole_boundaries,
    )
    clean_depths = {"front": filtered}
    raw_observation = pre.preprocess(
        frame_index=args.frame, timestamp_s=timestamp, state=state,
        depth_by_role=depths, rgb_by_role=rgbs,
    )
    clean_observation = pre.preprocess(
        frame_index=args.frame, timestamp_s=timestamp, state=state,
        depth_by_role=clean_depths, rgb_by_role=rgbs,
    )
    raw_dense = build_dense_roi_by_camera(
        preprocessor=pre, depth_by_role=depths, rgb_by_role=rgbs, camera_roles=("front",)
    )["front"]
    clean_dense = build_dense_roi_by_camera(
        preprocessor=pre, depth_by_role=clean_depths, rgb_by_role=rgbs, camera_roles=("front",)
    )["front"]
    removed_only = np.zeros_like(raw)
    removed_only[removed] = raw[removed]
    cache = build_camera_cache(camera, image_height=raw.shape[0], image_width=raw.shape[1])
    removed_xyz, _ = depth_to_base_roi(
        removed_only, cache, roi=cfg.roi,
        min_depth_m=cfg.visual.min_depth_m, max_depth_m=cfg.visual.max_depth_m,
    )

    meshes = load_visual_meshes(mesh_urdf, include_fingers=True)
    groups = mesh_groups(meshes)
    q = {name: float(state[index]) for name, index in pre.state_mapping.items()
         if name in pre.kinematics.measured_movable}
    transforms = pre.kinematics.compute(q, root="base_link")
    raw_m = raw.astype(np.float32) * camera.depth_scale_m_per_unit
    filtered_m = filtered.astype(np.float32) * camera.depth_scale_m_per_unit
    cad_depth, cad_groups = render_cad_depth(meshes, transforms, camera, raw.shape, groups)
    raw_stats, residual, raw_valid, _ = evaluate(raw_m, cad_depth, cad_groups, 3, 0.05, groups)
    clean_stats, _, clean_valid, _ = evaluate(filtered_m, cad_depth, cad_groups, 3, 0.05, groups)

    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError("The offline point-cloud comparison requires plotly") from exc
    figure = go.Figure()
    for name, dense, visible in (("raw dense ROI", raw_dense, "legendonly"),
                                 ("filtered dense ROI", clean_dense, True)):
        xyz, rgb, rgb_valid = thin_for_display(
            dense["xyz"], dense["rgb"], dense["rgb_valid"], maximum=args.max_dense_points
        )
        figure.add_trace(scatter(go, f"{name} ({len(dense['xyz'])})", xyz, rgb, rgb_valid,
                                 visible=visible))
    xyz = removed_xyz
    if args.max_dense_points and len(xyz) > args.max_dense_points:
        xyz = xyz[np.linspace(0, len(xyz) - 1, args.max_dense_points, dtype=np.int64)]
    figure.add_trace(scatter(go, f"removed ROI ({len(removed_xyz)})", xyz, visible="legendonly", size=2))
    for name, observation in (("raw FINAL 4096", raw_observation),
                              ("filtered FINAL 4096", clean_observation)):
        figure.add_trace(scatter(
            go, name, observation.visual_xyz_m, observation.visual_rgb,
            observation.visual_rgb_valid, visible="legendonly", size=3.5,
        ))
    figure.add_trace(scatter(
        go, "tactile 600", clean_observation.tactile_xyz_m,
        visible=True, size=3.5, fixed_color="rgb(30,140,190)",
    ))
    for trace in cad_root_to_tip_traces(
        go=go, preprocessor=pre, state=state, camera=camera,
        mesh_urdf=mesh_urdf, max_edges=args.max_cad_edges_per_group,
    ):
        figure.add_trace(trace)
    figure.update_layout(
        title=(f"Front depth edge comparison | frame {args.frame} | "
               f"jump {args.jump_mm:g} mm, radius {args.radius_px} px"),
        scene={"xaxis_title": "base_link X [m]", "yaxis_title": "base_link Y [m]",
               "zaxis_title": "base_link Z [m]", "aspectmode": "data", "dragmode": "orbit"},
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 220, "b": 0, "t": 70}, height=850,
    )

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    figure.write_html(str(output / "cloud.html"), include_plotlyjs="inline", full_html=True)
    images = {
        "raw_depth_removed_red.png": depth_mask_image(raw_m, removed),
        "rgb_removed_red.png": rgb_mask_image(
            rgbs["front"], np.where(removed, raw_m, 0), camera,
        ),
        "rgb_registered_raw.png": registered_depth_overlay(
            rgbs["front"], raw_m, camera, depth_range(raw_m),
        ),
        "rgb_registered_filtered.png": registered_depth_overlay(
            rgbs["front"], filtered_m, camera, depth_range(raw_m),
        ),
    }
    for name, image in images.items():
        if not cv2.imwrite(str(output / name), image):
            raise RuntimeError(f"Could not write {output / name}")
    valid_depth = np.isfinite(raw_m) & (raw_m >= cfg.visual.min_depth_m) & (raw_m <= cfg.visual.max_depth_m)
    report = {
        "dataset": str(dataset), "episode": args.episode, "frame": args.frame,
        "profile": str(profile), "profile_extrinsic_translation_base_m": profile_data["front_extrinsic_translation_base_m"],
        "profile_color_intrinsics": profile_data["front_color_intrinsics"],
        "applied_T_base_color": camera.T_base_color.tolist(),
        "applied_T_color_depth": camera.T_color_depth.tolist(),
        "applied_color_intrinsics": {key: float(getattr(camera.color_intrinsics, key))
                                     for key in ("fx", "fy", "cx", "cy")},
        "depth_intrinsics": {key: float(getattr(camera.depth_intrinsics, key))
                             for key in ("fx", "fy", "cx", "cy")},
        "filter": {"jump_mm": args.jump_mm, "radius_px": args.radius_px,
                   "include_hole_boundaries": args.include_hole_boundaries,
                   "raw_depth_valid_pixels": int(valid_depth.sum()),
                   "jump_or_hole_seed_pixels": int(seeds.sum()),
                   "removed_valid_pixels": int(removed.sum()),
                   "removed_fraction": float(removed.sum() / valid_depth.sum()) if np.any(valid_depth) else 0.0},
        "dense_roi": {"raw_points": len(raw_dense["xyz"]), "filtered_points": len(clean_dense["xyz"]),
                      "removed_points": len(removed_xyz)},
        "final_4096": {"raw_rgb_valid_fraction": float(raw_observation.visual_rgb_valid.mean()),
                       "filtered_rgb_valid_fraction": float(clean_observation.visual_rgb_valid.mean())},
        "cad_depth_residual": {
            "raw_near_surface": raw_stats["near_surface"],
            "filtered_near_surface": clean_stats["near_surface"],
            "removed_interior": summarize(residual[raw_valid & ~clean_valid]),
            "comparison_note": "Filtered values are unchanged; residual shifts reflect a changed pixel subset, not corrected XYZ.",
        },
        "interpretation": "Color intrinsics affect RGB projection/point colors, not depth-derived XYZ. Edge removal cannot fix depth K, scale, T_color_depth, T_base_color, CAD/FK or timing errors.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    page = ("<!doctype html><meta charset='utf-8'><title>Front depth edge comparison</title>"
            "<style>body{font:15px system-ui;margin:20px}img{max-width:100%}"
            ".grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}"
            "@media(max-width:800px){.grid{grid-template-columns:1fr}}</style>"
            f"<h1>Front depth edge comparison: frame {args.frame}</h1>"
            f"<p>{html.escape(str(dataset))} | removed {removed.sum()}/{valid_depth.sum()} "
            f"({100 * report['filter']['removed_fraction']:.1f}%) valid depth pixels. "
            "Red is invalidated, not moved or filled.</p>"
            "<p><a href='cloud.html'>Open interactive raw/filtered cloud + CAD</a> | "
            "<a href='report.json'>Open metrics</a></p>"
            "<div class='grid'><div><h2>Raw depth grid</h2><img src='raw_depth_removed_red.png'></div>"
            "<div><h2>Removed samples projected into RGB</h2><img src='rgb_removed_red.png'></div>"
            "<div><h2>RGB + raw depth registration</h2><img src='rgb_registered_raw.png'></div>"
            "<div><h2>RGB + filtered depth registration</h2><img src='rgb_registered_filtered.png'></div></div>"
            "<p>Registered depth uses four-pixel display coverage; those added pixels are not new measurements.</p>"
            "<iframe src='cloud.html' style='width:100%;height:900px;border:0' title='3D cloud'></iframe>")
    (output / "index.html").write_text(page, encoding="utf-8")
    print(f"Removed {removed.sum()}/{valid_depth.sum()} valid depth pixels; "
          f"dense ROI {len(raw_dense['xyz'])} -> {len(clean_dense['xyz'])}")
    print(f"Open {output / 'index.html'}")


if __name__ == "__main__":
    main()
