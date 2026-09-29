#!/usr/bin/env python3
"""Offline depth/CAD residual and state-lag probes on fixed front depth frames."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from audit_front_depth_cad import evaluate, mesh_groups, render_cad_depth, summarize
from openpi.spatial.calibration_profile import apply_front_calibration_profile
from openpi.spatial.config import make_baseline_config
from openpi.spatial.geometry import apply_diagnostic_overrides
from openpi.spatial.preprocess import SpatialPreprocessor
from openpi.spatial_dataset.source import RawSpatialDataset
from root_to_tip_mesh import load_visual_meshes, resolve_mesh_urdf


REPO_ROOT = Path(__file__).resolve().parents[2]


def fixed_pixel_lag_scores(candidates: dict[int, tuple[np.ndarray, np.ndarray]],
                           *, max_abs_m: float, pixel_mask: np.ndarray | None = None) -> dict:
    """Score all poses on one pixel set to avoid coverage-driven lag winners."""
    offsets = sorted(candidates)
    valid = np.logical_and.reduce([candidates[offset][1] for offset in offsets])
    near = np.logical_and.reduce([
        np.abs(candidates[offset][0]) <= max_abs_m for offset in offsets
    ])
    common = valid & near
    if pixel_mask is not None:
        common &= pixel_mask
    scores = {str(offset): summarize(candidates[offset][0][common]) for offset in offsets}
    best = (min(offsets, key=lambda offset: scores[str(offset)]["abs_median_mm"])
            if np.any(common) else None)
    return {"common_pixels": int(common.sum()), "scores": scores, "best_offset": best}, common


def centered_bias_probe(candidates: dict[int, tuple[np.ndarray, np.ndarray]],
                        common: np.ndarray) -> dict:
    """Show how an additive frame-wise residual bias can alter lag ranking."""
    if not np.any(common):
        return {"offset_mm": None, "best_offset": None, "scores": {}}
    bias = -float(np.median(candidates[0][0][common]))
    scores = {str(offset): summarize(residual[common] + bias)
              for offset, (residual, _) in sorted(candidates.items())}
    best = min(candidates, key=lambda offset: scores[str(offset)]["abs_median_mm"])
    return {"offset_mm": bias * 1000, "best_offset": best, "scores": scores,
            "warning": "This is an algebraic sensitivity probe, not a depth correction or time calibration."}


def stratified_residual(obs_m: np.ndarray, residual: np.ndarray, valid: np.ndarray,
                        groups: np.ndarray, group_defs: tuple, *, max_abs_m: float) -> list[dict]:
    """Descriptive residuals only; bins and CAD occlusion can confound calibration."""
    h, w = obs_m.shape
    near = valid & (np.abs(residual) <= max_abs_m)
    layers = []
    for index, (name, _) in enumerate(group_defs):
        layers.append(("link", name, groups == index))
    # Fixed metric bins make different frames directly comparable.
    for lo, hi in ((0.1, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.1),
                   (1.1, 1.4), (1.4, 2.0), (2.0, 3.0)):
        layers.append(("depth_m", f"[{lo:g},{hi:g})", (obs_m >= lo) & (obs_m < hi)))
    yy, xx = np.indices(obs_m.shape)
    for row in range(3):
        for col in range(3):
            region = ((xx >= col * w // 3) & (xx < (col + 1) * w // 3)
                      & (yy >= row * h // 3) & (yy < (row + 1) * h // 3))
            layers.append(("image_cell", f"r{row}c{col}", region))
    return [
        {"kind": kind, "bin": name, **summarize(residual[near & mask])}
        for kind, name, mask in layers
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frames", required=True, help="Comma-separated depth frame indices")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--lag-radius", type=int, default=2, help="Evaluate state offsets -N..+N")
    parser.add_argument("--near-surface-mm", type=float, default=50)
    parser.add_argument("--edge-px", type=int, default=3)
    parser.add_argument("--mesh-urdf", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    frames = [int(value) for value in args.frames.split(",")]
    if (not frames or min(frames) < 0 or len(frames) != len(set(frames))
            or args.lag_radius < 1 or args.lag_radius > 4
            or args.near_surface_mm <= 0 or args.edge_px < 0):
        raise ValueError("Invalid frames, lag radius, or residual thresholds")
    offsets = range(-args.lag_radius, args.lag_radius + 1)
    if min(frames) < args.lag_radius:
        raise ValueError("Selected frames must be at least --lag-radius from episode start")
    required = {frame + offset for frame in frames for offset in offsets}
    dataset = RawSpatialDataset(args.dataset.expanduser().resolve(), episode=args.episode)
    rows = {int(row["frame_index"]): row for row in dataset.iter_rows(
        selected=required, depth_roles=("front",)
    )}
    if required - rows.keys():
        raise ValueError(f"Missing frames: {sorted(required - rows.keys())}")
    profile = args.profile.expanduser().resolve()
    config = apply_front_calibration_profile(
        make_baseline_config().with_camera_roles("front"), profile,
    )
    pre = SpatialPreprocessor.from_repo_root(repo_root=REPO_ROOT, config=config)
    camera = apply_diagnostic_overrides(pre.calibration_bundle.cameras, config.diagnostics)["front"]
    meshes = load_visual_meshes(resolve_mesh_urdf(REPO_ROOT, args.mesh_urdf), include_fingers=True)
    group_defs = mesh_groups(meshes)
    joint_positions = {
        frame: {name: float(rows[frame]["observation.state"][index])
                for name, index in pre.state_mapping.items()
                if name in pre.kinematics.measured_movable}
        for frame in required
    }
    per_frame = []
    table = []
    for frame in frames:
        obs = np.asarray(rows[frame]["observation.depths.cam_front"], np.float32)
        obs *= camera.depth_scale_m_per_unit
        candidates = {}
        center_detail = None
        for offset in offsets:
            transforms = pre.kinematics.compute(joint_positions[frame + offset], root="base_link")
            pred, groups = render_cad_depth(meshes, transforms, camera, obs.shape, group_defs)
            report, residual, valid, _ = evaluate(
                obs, pred, groups, args.edge_px, args.near_surface_mm / 1000, group_defs
            )
            candidates[offset] = (residual, valid)
            if offset == 0:
                center_detail = (groups, report, residual, valid)
        lag, common = fixed_pixel_lag_scores(candidates, max_abs_m=args.near_surface_mm / 1000)
        bias_probe = centered_bias_probe(candidates, common)
        groups, report, residual, valid = center_detail
        lag_by_link = {
            name: fixed_pixel_lag_scores(
                candidates, max_abs_m=args.near_surface_mm / 1000,
                pixel_mask=groups == index,
            )[0]
            for index, (name, _) in enumerate(group_defs)
        }
        bins = stratified_residual(
            obs, residual, valid, groups, group_defs,
            max_abs_m=args.near_surface_mm / 1000,
        )
        for item in bins:
            table.append({"frame": frame, **item})
        q_before = np.array(list(joint_positions[frame - 1].values()))
        q_after = np.array(list(joint_positions[frame + 1].values()))
        arm_names = [name for name in joint_positions[frame] if pre.state_mapping[name] < 6]
        hand_names = [name for name in joint_positions[frame] if pre.state_mapping[name] >= 6]
        entry = {
            "frame": frame, "timestamp_s": float(rows[frame]["timestamp"]),
            "joint_step_two_frames_rad": float(np.linalg.norm(q_after - q_before)),
            "arm_joint_step_two_frames_rad": float(np.linalg.norm([
                joint_positions[frame + 1][name] - joint_positions[frame - 1][name]
                for name in arm_names
            ])),
            "hand_joint_step_two_frames_rad": float(np.linalg.norm([
                joint_positions[frame + 1][name] - joint_positions[frame - 1][name]
                for name in hand_names
            ])),
            "center_near_surface": report["near_surface"],
            "state_lag_probe": lag,
            "centered_bias_probe": bias_probe,
            "state_lag_by_center_link": lag_by_link,
        }
        per_frame.append(entry)
        center_median = report["near_surface"]["signed_median_mm"]
        label = f"{center_median:.2f} mm" if center_median is not None else "no near pixels"
        print(f"frame {frame}: center median {label}; "
              f"lag best {lag['best_offset']} on {lag['common_pixels']} common pixels", flush=True)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps({
        "dataset": str(dataset.root), "episode": args.episode, "frames": frames,
        "profile": str(profile), "lag_radius": args.lag_radius,
        "near_surface_mm": args.near_surface_mm, "edge_px": args.edge_px,
        "calibration": {
            "depth_scale_m_per_unit": camera.depth_scale_m_per_unit,
            "depth_intrinsics": {key: float(getattr(camera.depth_intrinsics, key))
                                 for key in ("fx", "fy", "cx", "cy")},
            "color_intrinsics": {key: float(getattr(camera.color_intrinsics, key))
                                 for key in ("fx", "fy", "cx", "cy")},
            "T_base_color": camera.T_base_color.tolist(),
            "T_color_depth": camera.T_color_depth.tolist(),
        },
        "timing_warning": "Dataset timestamp is frame_index/fps, not independent device time. Lag scores are geometric proxies; a nonzero winner does not prove clock offset.",
        "identifiability_warning": "Depth-CAD residual constrains T_base_color @ T_color_depth, not either transform alone. CAD/FK, occlusion and depth scale can mimic calibration errors.",
        "results": per_frame,
    }, indent=2), encoding="utf-8")
    with (output / "stratified_residual.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"Wrote {output / 'report.json'} and {output / 'stratified_residual.csv'}")


if __name__ == "__main__":
    main()
