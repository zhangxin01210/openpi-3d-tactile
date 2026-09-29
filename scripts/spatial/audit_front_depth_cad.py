#!/usr/bin/env python3
"""Compare raw front depth to FK CAD surfaces in the depth optical image."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from openpi.spatial.calibration_profile import apply_front_calibration_profile
from openpi.spatial.config import make_baseline_config
from openpi.spatial.geometry import apply_diagnostic_overrides
from openpi.spatial.preprocess import SpatialPreprocessor
from openpi.spatial_dataset.source import RawSpatialDataset
from root_to_tip_mesh import (
    CHAIN_GROUPS, COLORS, FINGER_COLORS, FINGER_PREFIXES,
    label_panel, load_visual_meshes, make_sheet as make_root_to_tip_sheet, resolve_mesh_urdf,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def mesh_groups(meshes):
    return CHAIN_GROUPS + tuple(
        (name, tuple(link for link in meshes if link.startswith(prefix)))
        for name, prefix in FINGER_PREFIXES
    )


def array_fingerprint(value):
    data = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(data.dtype).encode("ascii"))
    digest.update(str(data.shape).encode("ascii"))
    digest.update(data.tobytes())
    return digest.hexdigest()


def render_cad_depth(meshes, transforms, camera, shape, group_defs=CHAIN_GROUPS):
    """Z-buffer triangular visual meshes; return frontmost depth and chain ID."""
    h, w = shape
    kd = camera.depth_intrinsics
    T_base_depth = camera.T_base_color @ camera.T_color_depth
    R_depth_base = T_base_depth[:3, :3].T
    t_base_depth = T_base_depth[:3, 3]
    zbuf = np.full((h, w), np.inf, np.float32)
    groups = np.full((h, w), -1, np.int16)
    for group_index, (_, links) in enumerate(group_defs):
        for link in links:
            mesh = meshes[link]
            T = transforms[link]
            points_base = mesh.vertices @ T[:3, :3].T + T[:3, 3]
            xyz = (points_base - t_base_depth) @ R_depth_base.T
            z = xyz[:, 2]
            uv = np.empty((len(z), 2), np.float64)
            good = np.isfinite(xyz).all(axis=1) & (z > 0.05)
            uv[good, 0] = kd.fx * xyz[good, 0] / z[good] + kd.cx
            uv[good, 1] = kd.fy * xyz[good, 1] / z[good] + kd.cy
            faces = np.asarray(mesh.faces, np.int32)
            for face in faces[np.all(good[faces], axis=1)]:
                tri = uv[face]
                x0 = max(0, int(np.floor(tri[:, 0].min())) - 1)
                x1 = min(w, int(np.ceil(tri[:, 0].max())) + 2)
                y0 = max(0, int(np.floor(tri[:, 1].min())) - 1)
                y1 = min(h, int(np.ceil(tri[:, 1].max())) + 2)
                if x0 >= x1 or y0 >= y1:
                    continue
                p = np.rint((tri - [x0, y0]) * 16).astype(np.int32)
                mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
                cv2.fillConvexPoly(mask, p, 1, lineType=cv2.LINE_8, shift=4)
                yy, xx = np.nonzero(mask)
                if not len(xx):
                    continue
                # 1/z is affine under perspective projection; interpolate it in image space.
                a, b, c = tri
                denom = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
                if abs(denom) < 1e-10:
                    continue
                px, py = xx + x0, yy + y0
                wb = ((px - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (py - a[1])) / denom
                wc = ((b[0] - a[0]) * (py - a[1]) - (px - a[0]) * (b[1] - a[1])) / denom
                wa = 1 - wb - wc
                inv_z = wa / z[face[0]] + wb / z[face[1]] + wc / z[face[2]]
                valid = inv_z > 0
                if not np.any(valid):
                    continue
                px, py, inv_z = px[valid], py[valid], inv_z[valid]
                pred = (1 / inv_z).astype(np.float32)
                update = pred < zbuf[py, px]
                zbuf[py[update], px[update]] = pred[update]
                groups[py[update], px[update]] = group_index
    return zbuf, groups


def summarize(values):
    mm = np.asarray(values, np.float64) * 1000
    if not len(mm):
        return {"count": 0, "signed_median_mm": None, "abs_median_mm": None, "abs_p90_mm": None}
    return {
        "count": int(len(mm)),
        "signed_median_mm": float(np.median(mm)),
        "abs_median_mm": float(np.median(np.abs(mm))),
        "abs_p90_mm": float(np.percentile(np.abs(mm), 90)),
    }


def evaluate(obs, pred, groups, edge_px, max_abs_m, group_defs=CHAIN_GROUPS):
    robot = np.isfinite(pred)
    interior = cv2.erode(robot.astype(np.uint8), np.ones((2 * edge_px + 1,) * 2, np.uint8)) != 0
    observed = np.isfinite(obs) & (obs > 0)
    valid = interior & observed
    residual = np.full_like(obs, np.nan, dtype=np.float32)
    residual[valid] = obs[valid] - pred[valid]
    matched = valid & (np.abs(residual) <= max_abs_m)
    report = {
        "predicted_pixels": int(robot.sum()),
        "interior_pixels": int(interior.sum()),
        "observed_interior_pixels": int(valid.sum()),
        "near_surface_pixels": int(matched.sum()),
        "all_overlap": summarize(residual[valid]),
        "near_surface": summarize(residual[matched]),
        "per_group": {},
    }
    for index, (name, _) in enumerate(group_defs):
        mask = valid & (groups == index)
        near = matched & (groups == index)
        report["per_group"][name] = {
            "predicted_visible_pixels": int((robot & (groups == index)).sum()),
            "interior_pixels": int((interior & (groups == index)).sum()),
            "interior_overlap_pixels": int(mask.sum()),
            "all_overlap": summarize(residual[mask]),
            "near_surface": summarize(residual[near]),
        }
    return report, residual, valid, matched


def make_sheet(obs, pred, residual, valid, matched, clip_mm):
    h, w = obs.shape
    def depth_vis(depth):
        ok = np.isfinite(depth) & (depth > 0)
        out = np.zeros((h, w, 3), np.uint8)
        if np.any(ok):
            lo, hi = np.percentile(depth[ok], [2, 98])
            scaled = np.zeros((h, w), np.uint8)
            scaled[ok] = np.clip(255 * (1 - (depth[ok] - lo) / max(hi - lo, 1e-6)), 0, 255).astype(np.uint8)
            colored = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
            out[ok] = colored[ok]
        return out

    signed = np.zeros((h, w, 3), np.uint8)
    scale = np.zeros((h, w), np.uint8)
    scale[valid] = np.clip(127.5 + 127.5 * residual[valid] * 1000 / clip_mm, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(scale, cv2.COLORMAP_JET)
    signed[valid] = colored[valid]
    near = signed.copy()
    near[valid & ~matched] = (80, 80, 80)
    names = (
        "Raw depth (each auto-scaled)", "Predicted CAD depth",
        f"Observed - CAD [blue - / red +; +/-{clip_mm:g} mm]",
        "Near surface; gray = excluded",
    )
    panels = (depth_vis(obs), depth_vis(pred), signed, near)
    sheet = np.zeros((2 * (h + 32), 2 * w, 3), np.uint8)
    for index, (name, panel) in enumerate(zip(names, panels, strict=True)):
        y, x = divmod(index, 2)
        sheet[y * (h + 32) + 32:(y + 1) * (h + 32), x * w:(x + 1) * w] = panel
        cv2.putText(sheet, name, (x * w + 8, y * (h + 32) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return sheet


def make_depth_root_to_tip(obs, groups, group_defs):
    """Show frontmost CAD link contours on the raw depth pixel grid."""
    h, w = obs.shape
    ok = np.isfinite(obs) & (obs > 0)
    gray = np.zeros((h, w), np.uint8)
    if np.any(ok):
        lo, hi = np.percentile(obs[ok], [2, 98])
        gray[ok] = np.clip(225 - 175 * (obs[ok] - lo) / max(hi - lo, 1e-6), 40, 225).astype(np.uint8)
    background = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    palette = tuple(tuple(reversed(color)) for color in COLORS + FINGER_COLORS)
    panels = [label_panel(background, "raw front depth / grayscale")]

    def add_group(canvas, index):
        mask = (groups == index).astype(np.uint8)
        if not np.any(mask):
            return canvas
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        out = canvas.copy()
        cv2.drawContours(out, contours, -1, palette[index], 2, cv2.LINE_AA)
        return out

    for index, (name, _) in enumerate(group_defs):
        panels.append(label_panel(add_group(background, index), f"{name} ONLY / visible CAD"))
    cumulative = background.copy()
    for index, (name, _) in enumerate(group_defs):
        cumulative = add_group(cumulative, index)
        panels.append(label_panel(cumulative, f"cumulative through {name}"))
    return make_root_to_tip_sheet(panels, width=380, cols=4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--mesh-urdf", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--edge-px", type=int, default=3)
    parser.add_argument("--near-surface-mm", type=float, default=50)
    parser.add_argument("--clip-mm", type=float, default=50)
    parser.add_argument("--save-front-rgb", action="store_true", help="Also export selected raw front RGB frames")
    args = parser.parse_args()
    frames = [int(x.strip()) for x in args.frames.split(",") if x.strip()]
    if not frames or min(frames) < 0 or len(frames) != len(set(frames)):
        raise ValueError("--frames requires unique nonnegative indices")
    if args.edge_px < 0 or args.near_surface_mm <= 0 or args.clip_mm <= 0:
        raise ValueError("Invalid edge or distance threshold")
    if args.profile is not None and not args.profile.expanduser().is_file():
        raise FileNotFoundError(args.profile)
    dataset = RawSpatialDataset(args.dataset.expanduser().resolve(), episode=args.episode)
    meshes = load_visual_meshes(resolve_mesh_urdf(REPO_ROOT, args.mesh_urdf), include_fingers=True)
    group_defs = mesh_groups(meshes)
    rows = {int(row["frame_index"]): row for row in dataset.iter_rows(selected=frames, depth_roles=("front",))}
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    fingerprints = {
        str(frame): {
            "state_sha256": array_fingerprint(np.asarray(rows[frame]["observation.state"], dtype=np.float32)),
            "depth_sha256": array_fingerprint(rows[frame]["observation.depths.cam_front"]),
        }
        for frame in frames
    }
    if args.save_front_rgb:
        rgb_dir = output / "raw_front_rgb"
        rgb_dir.mkdir()
        for frame, rgb in dataset.load_video_frames("front", selected=frames).items():
            fingerprints[str(frame)]["rgb_sha256"] = array_fingerprint(rgb)
            path = rgb_dir / f"frame_{frame:06d}_front_rgb.png"
            if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"Could not write {path}")
    results = []
    calibrations = {}
    baseline_maps = {}
    paired_comparisons = {}
    for mode, profile in (("baseline", None), ("corrected", args.profile)):
        if mode == "corrected" and profile is None:
            continue
        config = make_baseline_config().with_camera_roles("front")
        if profile is not None:
            config = apply_front_calibration_profile(config, profile.expanduser().resolve())
        pre = SpatialPreprocessor.from_repo_root(repo_root=REPO_ROOT, config=config)
        camera = apply_diagnostic_overrides(pre.calibration_bundle.cameras, config.diagnostics)["front"]
        calibrations[mode] = {
            "depth_intrinsics": {key: float(getattr(camera.depth_intrinsics, key)) for key in ("fx", "fy", "cx", "cy")},
            "depth_scale_m_per_unit": camera.depth_scale_m_per_unit,
            "T_base_color": camera.T_base_color.tolist(),
            "T_color_depth": camera.T_color_depth.tolist(),
        }
        folder = output / mode
        folder.mkdir()
        for frame in frames:
            row = rows[frame]
            obs = np.asarray(row["observation.depths.cam_front"], np.float32) * camera.depth_scale_m_per_unit
            state = np.asarray(row["observation.state"], np.float64)
            q = {name: float(state[i]) for name, i in pre.state_mapping.items()
                 if name in pre.kinematics.measured_movable}
            transforms = pre.kinematics.compute(q, root="base_link")
            pred, groups = render_cad_depth(meshes, transforms, camera, obs.shape, group_defs)
            report, residual, valid, matched = evaluate(
                obs, pred, groups, args.edge_px, args.near_surface_mm / 1000, group_defs
            )
            if mode == "baseline":
                baseline_maps[frame] = (residual, valid)
            else:
                baseline_residual, baseline_valid = baseline_maps[frame]
                common = baseline_valid & valid
                common &= (np.abs(baseline_residual) <= args.near_surface_mm / 1000)
                common &= (np.abs(residual) <= args.near_surface_mm / 1000)
                paired_comparisons[str(frame)] = {
                    "common_pixels": int(common.sum()),
                    "baseline": summarize(baseline_residual[common]),
                    "corrected": summarize(residual[common]),
                    "improved_fraction": (
                        float(np.mean(np.abs(residual[common]) < np.abs(baseline_residual[common])))
                        if np.any(common) else None
                    ),
                }
            report.update({"mode": mode, "frame": frame})
            results.append(report)
            path = folder / f"frame_{frame:06d}_depth_cad.png"
            sheet = make_sheet(obs, pred, residual, valid, matched, args.clip_mm)
            if not cv2.imwrite(str(path), sheet):
                raise RuntimeError(f"Could not write {path}")
            overlay_dir = folder / "root_to_tip_depth"
            overlay_dir.mkdir(exist_ok=True)
            overlay_path = overlay_dir / f"frame_{frame:06d}_front_depth_root_to_tip.png"
            if not cv2.imwrite(str(overlay_path), make_depth_root_to_tip(obs, groups, group_defs)):
                raise RuntimeError(f"Could not write {overlay_path}")
            print(f"{mode} frame {frame}: {report['near_surface']}; {path}", flush=True)
    (output / "report.json").write_text(json.dumps({
        "dataset": str(args.dataset), "episode": args.episode, "frames": frames,
        "profile": str(args.profile) if args.profile else None,
        "depth_coordinate_system": "raw front depth optical", "cad": "URDF visual meshes C0-C8 plus five fingers",
        "residual_sign": "observed minus predicted, depth optical z, mm",
        "edge_px": args.edge_px, "near_surface_mm": args.near_surface_mm, "clip_mm": args.clip_mm,
        "calibrations": calibrations,
        "source_fingerprints": fingerprints,
        "paired_common_pixels": paired_comparisons,
        "note": "Predicted CAD pixels are not a robot segmentation; occluders and depth holes can bias overlap statistics.",
        "results": results,
    }, indent=2), encoding="utf-8")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "mode", "frame", "group", "predicted_visible_pixels", "interior_pixels",
            "overlap_pixels", "near_pixels", "signed_median_mm", "abs_median_mm", "abs_p90_mm",
        ])
        writer.writeheader()
        for result in results:
            for name, group in result["per_group"].items():
                writer.writerow({"mode": result["mode"], "frame": result["frame"], "group": name,
                                 "predicted_visible_pixels": group["predicted_visible_pixels"],
                                 "interior_pixels": group["interior_pixels"],
                                 "overlap_pixels": group["interior_overlap_pixels"],
                                 "near_pixels": group["near_surface"]["count"],
                                 **{key: group["near_surface"][key] for key in ("signed_median_mm", "abs_median_mm", "abs_p90_mm")}})
    print(f"Wrote {output / 'report.json'} and {output / 'summary.csv'}")
    if args.profile is not None:
        from render_depth_cad_comparison import render

        render(output, output / "comparison")


if __name__ == "__main__":
    main()
