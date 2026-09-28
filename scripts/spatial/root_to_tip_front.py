#!/usr/bin/env python3
"""Render front RGB root-to-tip CAD contours on an expanded canvas."""

from __future__ import annotations

import argparse
import dataclasses
import re
from pathlib import Path

import cv2
import numpy as np

from openpi.spatial.calibration_profile import apply_front_calibration_profile
from openpi.spatial.config import make_baseline_config
from openpi.spatial.geometry import apply_diagnostic_overrides
from openpi.spatial.preprocess import SpatialPreprocessor
from openpi.spatial_dataset.source import RawSpatialDataset
import root_to_tip_mesh as overlay


REPO_ROOT = Path(__file__).resolve().parents[2]
def _frames(text: str, report: dict | None) -> list[int]:
    if text == "report-qc":
        if report is None:
            raise ValueError("--frames report-qc requires --marker-camera-report")
        frames = sorted({int(match.group(1)) for path in report.get("qc_images", [])
                         if (match := re.search(r"holdout_(\d+)_", str(path)))})
    else:
        frames = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not frames or min(frames) < 0 or len(set(frames)) != len(frames):
        raise ValueError("--frames must list unique nonnegative frame indices")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--profile", type=Path, default=None)
    parser.add_argument("--marker-camera-report", type=Path, default=None,
                        help="Report JSON containing a marker-anchored T_base_color; overrides profile extrinsic")
    parser.add_argument("--camera-report-key", default="free_camera_T_base_color")
    parser.add_argument("--mesh-urdf", type=Path, default=None,
                        help="Verified URDF with relative visual meshes installed alongside it")
    parser.add_argument("--alpha", type=float, default=0.42)
    parser.add_argument("--pad-px", type=int, default=55)
    parser.add_argument("--max-extra-px", type=int, default=1100)
    parser.add_argument("--panel-width", type=int, default=460)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (0 < args.alpha <= 1) or args.pad_px < 0 or args.max_extra_px < args.pad_px or args.panel_width < 1:
        raise ValueError("Invalid rendering parameters")
    marker_T, report = (None, None)
    if args.marker_camera_report is not None:
        marker_T, report = overlay.load_marker_camera_report(
            args.marker_camera_report.expanduser().resolve(), args.camera_report_key
        )
    frames = _frames(args.frames, report)
    urdf = overlay.resolve_mesh_urdf(REPO_ROOT, args.mesh_urdf)
    meshes = overlay.load_visual_meshes(urdf)
    dataset = RawSpatialDataset(args.dataset.expanduser().resolve(), episode=args.episode)
    config = make_baseline_config().with_camera_roles("front")
    if args.profile is not None:
        config = apply_front_calibration_profile(config, args.profile.expanduser().resolve())
    pre = SpatialPreprocessor.from_repo_root(repo_root=REPO_ROOT, config=config)
    camera = apply_diagnostic_overrides(pre.calibration_bundle.cameras, config.diagnostics)["front"]
    if marker_T is not None:
        camera = dataclasses.replace(camera, T_base_color=marker_T)
    states = dataset.states()
    state_by_frame = {
        int(frame): np.asarray(state, dtype=float)
        for frame, state in zip(states["frame_index"], states["observation.state"], strict=True)
        if int(frame) in frames
    }
    if set(state_by_frame) != set(frames):
        raise ValueError(f"Missing frames: {sorted(set(frames) - set(state_by_frame))}")
    images = dataset.load_video_frames("front", selected=frames)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    for frame in frames:
        state = state_by_frame[frame]
        q = {
            name: float(state[index]) for name, index in pre.state_mapping.items()
            if name in pre.kinematics.measured_movable
        }
        transforms = pre.kinematics.compute(q, root="base_link")
        rgb = images[frame]
        bounds = overlay.compute_expanded_bounds(
            rgb.shape, meshes, transforms, camera, args.pad_px, args.max_extra_px
        )
        canvas = overlay.make_base_canvas(rgb, bounds)
        label = "marker camera" if marker_T is not None else "profile camera" if args.profile else "baseline camera"
        panels = [overlay.label_panel(canvas, f"raw RGB + black outside FOV / frame {frame} / {label}")]
        for index, (name, links) in enumerate(overlay.CHAIN_GROUPS):
            only = overlay.render_group(
                canvas, links, meshes, transforms, camera, bounds, overlay.COLORS[index], args.alpha
            )
            panels.append(overlay.label_panel(only, f"{name} ONLY / {label}"))
        for index, (name, _) in enumerate(overlay.CHAIN_GROUPS):
            cumulative = canvas.copy()
            for group_index, (_, group) in enumerate(overlay.CHAIN_GROUPS[:index+1]):
                cumulative = overlay.render_group(
                    cumulative, group, meshes, transforms, camera, bounds,
                    overlay.COLORS[group_index], args.alpha
                )
            panels.append(overlay.label_panel(cumulative, f"cumulative through {name}"))
        image = overlay.make_sheet(panels, width=args.panel_width)
        path = output / f"frame_{frame:06d}_front_root_to_tip.png"
        if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Could not write {path}")
        print(f"Wrote {path} (CAD mesh contours; {label})")


if __name__ == "__main__":
    main()
