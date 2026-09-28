#!/usr/bin/env python3
"""Create paired root-to-tip overlays and canonical 3D point-cloud pages for a dataset."""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from pathlib import Path

from root_to_tip_mesh import load_visual_meshes, resolve_mesh_urdf


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frames", required=True, help="Comma-separated frame indices")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--mesh-urdf", type=Path, default=None)
    parser.add_argument("--marker-camera-report", type=Path, default=None)
    parser.add_argument("--camera-report-key", default="free_camera_T_base_color")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-dense-points-per-camera", type=int, default=30000)
    args = parser.parse_args()
    dataset = args.dataset.expanduser().resolve()
    profile = args.profile.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not dataset.is_dir() or not profile.is_file():
        raise FileNotFoundError("Dataset or calibration profile is missing")
    marker_report = args.marker_camera_report.expanduser().resolve() if args.marker_camera_report else None
    if marker_report is not None:
        from root_to_tip_mesh import load_marker_camera_report

        load_marker_camera_report(marker_report, args.camera_report_key)
    mesh_urdf = resolve_mesh_urdf(REPO_ROOT, args.mesh_urdf)
    load_visual_meshes(mesh_urdf)
    frames = [int(part.strip()) for part in args.frames.split(",") if part.strip()]
    if not frames or min(frames) < 0 or len(set(frames)) != len(frames):
        raise ValueError("--frames must list unique nonnegative indices")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    page_rows = []
    modes = [("baseline", False, False), ("corrected", True, False)]
    if marker_report is not None:
        modes.append(("marker", False, True))
    for mode, use_profile, use_marker in modes:
        overlays = output / mode / "root_to_tip"
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts/spatial/root_to_tip_front.py"),
            "--dataset", str(dataset),
            "--episode", str(args.episode), "--frames", args.frames,
            "--output", str(overlays),
        ]
        if use_profile:
            command += ["--profile", str(profile)]
        if use_marker:
            command += ["--marker-camera-report", str(marker_report),
                        "--camera-report-key", args.camera_report_key]
        command += ["--mesh-urdf", str(mesh_urdf)]
        subprocess.run(command, check=True, cwd=REPO_ROOT)

        for frame in frames:
            cloud = output / mode / f"frame_{frame:06d}_cloud.html"
            command = [
                sys.executable, str(REPO_ROOT / "scripts/spatial/visualize_spatial_web.py"),
                "--dataset", str(dataset), "--episode", str(args.episode),
                "--frame", str(frame), "--cameras", "front", "--output", str(cloud),
                "--max-dense-points-per-camera", str(args.max_dense_points_per_camera),
                "--cad-root-to-tip", "--mesh-urdf", str(mesh_urdf),
            ]
            if use_profile:
                command += ["--front-calibration-profile", str(profile)]
            if use_marker:
                command += ["--marker-camera-report", str(marker_report),
                            "--camera-report-key", args.camera_report_key]
            subprocess.run(command, check=True, cwd=REPO_ROOT)
            overlay = overlays / f"frame_{frame:06d}_front_root_to_tip.png"
            page_rows.append((mode, frame, overlay.relative_to(output), cloud.relative_to(output)))

    rows = "\n".join(
        f"<tr><th>{html.escape(mode)} frame {frame}</th>"
        f"<td><a href='{overlay.as_posix()}'><img src='{overlay.as_posix()}' width='700'></a></td>"
        f"<td><a href='{cloud.as_posix()}'>Open interactive 3D cloud + tactile points</a></td></tr>"
        for mode, frame, overlay, cloud in page_rows
    )
    index = (
        "<!doctype html><html><meta charset='utf-8'><title>Front calibration comparison</title>"
        "<style>body{font:15px system-ui;margin:24px}table{border-collapse:collapse}"
        "td,th{padding:12px;border:1px solid #bbb;vertical-align:top}img{max-width:65vw}</style>"
        f"<h1>Front camera calibration comparison</h1><p>Dataset: {html.escape(str(dataset))}</p>"
        "<p>Each mode uses the same raw RGB-D and SpatialPreprocessor with its selected front "
        "calibration. Cloud pages show dense ROI, actual 4096 sampled points, tactile points, "
        "and FK CAD contours. Marker mode is offline-only and is not a training profile.</p>"
        f"<table>{rows}</table></html>"
    )
    (output / "index.html").write_text(index, encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({
        "dataset": str(dataset), "episode": args.episode, "frames": frames,
        "profile": str(profile), "marker_camera_report": str(marker_report) if marker_report else None,
        "camera_report_key": args.camera_report_key if marker_report else None,
        "mesh_urdf": str(mesh_urdf), "output": str(output),
    }, indent=2), encoding="utf-8")
    print(f"Open {output / 'index.html'}")


if __name__ == "__main__":
    main()
