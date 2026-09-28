#!/usr/bin/env python3
"""Render front RGB root-to-tip projections with the canonical camera calibration."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import cv2
import numpy as np

from openpi.spatial.calibration_profile import apply_front_calibration_profile
from openpi.spatial.config import make_baseline_config
from openpi.spatial.geometry import apply_diagnostic_overrides, project_pinhole
from openpi.spatial.preprocess import SpatialPreprocessor
from openpi.spatial_dataset.source import RawSpatialDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_SOURCE = REPO_ROOT / "3D_tactile/pointcloud_delivery/src/root_to_tip_expanded_contours.py"
LEGACY_URDF = REPO_ROOT / "3D_tactile/pointcloud_delivery/configs/ur7e_xhand_verified.urdf"
CHAIN_LINKS = (
    "base_link_inertia", "shoulder_link", "upper_arm_link", "forearm_link",
    "wrist_1_link", "wrist_2_link", "wrist_3_link", "Flange_base_link", "right_hand_link",
)
COLORS = (
    (210, 87, 82), (228, 163, 55), (209, 201, 78), (67, 173, 104),
    (58, 171, 182), (75, 135, 230), (147, 110, 221), (222, 118, 183), (255, 255, 255),
)


def _legacy_renderer():
    if not LEGACY_SOURCE.is_file() or not LEGACY_URDF.is_file():
        return None
    spec = importlib.util.spec_from_file_location("legacy_root_to_tip", LEGACY_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError:
        return None
    return module


def _camera_for_legacy(camera):
    from scipy.spatial.transform import Rotation

    T = camera.T_base_color
    return {
        "T_base_color": {
            "translation_m": T[:3, 3].tolist(),
            "quaternion_xyzw": Rotation.from_matrix(T[:3, :3]).as_quat().tolist(),
        },
        "color_intrinsics": {
            key: float(getattr(camera.color_intrinsics, key)) for key in ("fx", "fy", "cx", "cy")
        },
    }


def _fallback_skeleton(rgb: np.ndarray, transforms: dict, camera) -> np.ndarray:
    h, w = rgb.shape[:2]
    image = np.zeros((h + 160, w + 160, 3), dtype=np.uint8)
    image[80:80+h, 80:80+w] = rgb
    cv2.rectangle(image, (80, 80), (80+w-1, 80+h-1), (180, 180, 180), 1)
    points = []
    for link in CHAIN_LINKS:
        if link not in transforms:
            continue
        point = transforms[link][:3, 3]
        pc = (point - camera.T_base_color[:3, 3]) @ camera.T_base_color[:3, :3]
        uv = project_pinhole(pc[None], camera.color_intrinsics)[0]
        if np.all(np.isfinite(uv)) and -80 < uv[0] < w + 80 and -80 < uv[1] < h + 80:
            points.append((link, tuple(np.rint(uv + 80).astype(int))))
    for index, (link, uv) in enumerate(points):
        color = COLORS[min(index, len(COLORS) - 1)]
        if index:
            cv2.line(image, points[index - 1][1], uv, color, 2, cv2.LINE_AA)
        cv2.circle(image, uv, 5, color, -1, cv2.LINE_AA)
        cv2.putText(image, link, (uv[0] + 7, uv[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--profile", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frames = [int(part.strip()) for part in args.frames.split(",") if part.strip()]
    if not frames or min(frames) < 0 or len(set(frames)) != len(frames):
        raise ValueError("--frames must list unique nonnegative frame indices")
    dataset = RawSpatialDataset(args.dataset.expanduser().resolve(), episode=args.episode)
    config = make_baseline_config().with_camera_roles("front")
    if args.profile is not None:
        config = apply_front_calibration_profile(config, args.profile.expanduser().resolve())
    pre = SpatialPreprocessor.from_repo_root(repo_root=REPO_ROOT, config=config)
    camera = apply_diagnostic_overrides(pre.calibration_bundle.cameras, config.diagnostics)["front"]
    states = dataset.states()
    state_by_frame = {
        int(frame): np.asarray(state, dtype=float)
        for frame, state in zip(states["frame_index"], states["observation.state"], strict=True)
        if int(frame) in frames
    }
    if set(state_by_frame) != set(frames):
        raise ValueError(f"Missing frames: {sorted(set(frames) - set(state_by_frame))}")
    images = dataset.load_video_frames("front", selected=frames)
    legacy = _legacy_renderer()
    meshes = legacy.load_visual_meshes(LEGACY_URDF) if legacy is not None else None
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
        if meshes:
            legacy_camera = _camera_for_legacy(camera)
            bounds = legacy.compute_expanded_bounds(rgb.shape, meshes, transforms, legacy_camera, 55, 1100)
            canvas = legacy.make_base_canvas(rgb, bounds)
            panels = [legacy.label_panel(canvas, f"raw RGB / frame {frame}")]
            for index, (name, links) in enumerate(legacy.CHAIN_GROUPS):
                only, _ = legacy.render_group(
                    canvas, links, meshes, transforms, legacy_camera, bounds, legacy.COLORS[index], 0.42
                )
                panels.append(legacy.label_panel(only, f"{name} ONLY"))
            for index, (name, _) in enumerate(legacy.CHAIN_GROUPS):
                cumulative = legacy.render_cumulative(canvas, index, meshes, transforms, legacy_camera, bounds, 0.42)
                panels.append(legacy.label_panel(cumulative, f"through {name}"))
            image = legacy.make_sheet(panels, width=460, cols=3)
        else:
            image = _fallback_skeleton(rgb, transforms, camera)
        path = output / f"frame_{frame:06d}_front_root_to_tip.png"
        if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Could not write {path}")
        print(f"Wrote {path} ({'mesh contours' if meshes else 'FK skeleton'})")


if __name__ == "__main__":
    main()
