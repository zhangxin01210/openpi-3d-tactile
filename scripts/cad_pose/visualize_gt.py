#!/usr/bin/env python3
"""Review existing Geometry v1 against raw RGB-D; no prediction/checkpoint involved."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import cv2
import numpy as np

from openpi.cad_pose.preprocess import FrontRGBDPreprocessor
from openpi.object_geometry_dataset.derived import ObjectGeometryDerivedDataset
from openpi.spatial_dataset.source import RawSpatialDataset
from scripts.cad_pose.preflight import ROLES


def fit_review_cad(sample, ordered_cad):
    """Project only fixed-order rigid fits; a low residual is not independent GT proof."""
    from openpi.cad_pose.geometry import fit_corresponding_rigid_pose

    output, errors = {}, []
    for i, role in enumerate(("manipulated", "target")):
        points = np.asarray(getattr(sample, f"{role}_xyz_m"))
        mask = np.asarray(getattr(sample, f"{role}_mask"), dtype=bool)
        if mask.shape != (512,) or not mask.all():
            errors.append(f"{role}: no complete fixed-point geometry")
            continue
        try:
            pose, rmse = fit_corresponding_rigid_pose(ordered_cad[i], points, mask, max_rmse_m=1e-4)
            output[role] = ordered_cad[i] @ pose[:3, :3].T + pose[:3, 3]
        except ValueError as exc:
            errors.append(f"{role}: {exc}")
    return output, errors


def render_gt(rgb, intrinsic, base_color, scene, objects, output, label, cad_objects=None):
    """Project base_link depth cloud and existing geometry into the original RGB."""
    image = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    original, gt, depth, cad = (image.copy() for _ in range(4))
    h, w = image.shape[:2]
    color_base = np.linalg.inv(base_color)

    def draw(panel, points, color, radius):
        xyz = np.asarray(points) @ color_base[:3, :3].T + color_base[:3, 3]
        xyz = xyz[np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 1e-6)]
        q = xyz @ intrinsic.T
        uv = np.rint(q[:, :2] / q[:, 2:3]).astype(np.int64)
        uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)]
        for u, v in uv:
            cv2.circle(panel, (int(u), int(v)), radius, color, -1)

    draw(depth, scene[:, :3], (255, 200, 0), 1)
    for role, color in (("manipulated", (0, 255, 0)), ("target", (0, 0, 255))):
        draw(gt, objects[role], color, 2)
        draw(depth, objects[role], color, 2)
        if cad_objects and role in cad_objects:
            draw(cad, cad_objects[role], color, 2)
    for panel, title in ((original, "RGB"), (gt, "Geometry v1 candidate"), (depth, "Depth cyan + candidate")):
        cv2.putText(panel, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    if cad_objects is not None:
        cv2.putText(cad, "CAD rigid fit (candidate)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    result = np.concatenate((original, cad, gt, depth) if cad_objects is not None else (original, gt, depth), axis=1)
    cv2.putText(result, label, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    if not cv2.imwrite(str(output), result):
        raise OSError(f"failed to write {output}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path.cwd())
    p.add_argument("--data-root", type=Path)
    p.add_argument("--profile", default="configs/ur7e_xhand/front_calibration_candidate.json")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--tasks", nargs="+", choices=tuple(ROLES), default=list(ROLES))
    p.add_argument("--episodes", action="append", default=[], metavar="DATASET=EP,EP")
    a = p.parse_args()
    repo = a.repo.resolve()
    root = (a.data_root or repo / "data").resolve()
    overrides = {}
    for item in a.episodes:
        name, values = item.split("=", 1)
        if name not in ROLES:
            raise ValueError(f"unknown dataset {name}")
        overrides[name] = tuple(int(x) for x in values.split(","))
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if (a.output_dir / "index.html").exists():
        raise FileExistsError("refusing to overwrite existing review")
    profile = Path(a.profile)
    profile_path = profile if profile.is_absolute() else repo / profile
    profile_hash = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    preprocess = FrontRGBDPreprocessor(repo, profile)
    camera = preprocess.color_camera()
    k = camera.color_intrinsics
    intrinsic = np.array([[k.fx, 0, k.cx], [0, k.fy, k.cy], [0, 0, 1]])
    reports = {}
    sections = []
    for dataset in a.tasks:
        links = []
        fit_failures = []
        try:
            data = root / dataset
            geometry = ObjectGeometryDerivedDataset(data, version="v1")
            from openpi.cad_pose.policy import load_ordered_cad

            cad_load_error = None
            try:
                ordered_cad = load_ordered_cad(repo, dataset)
            except (OSError, ValueError, KeyError, ImportError, RuntimeError) as exc:
                ordered_cad = None
                cad_load_error = f"{type(exc).__name__}: {exc}"
            available = sorted(set(geometry.episode_ids) & set(RawSpatialDataset.discover_episode_ids(data)))
            episodes = overrides.get(dataset, (23, 65) if dataset == "1018_peg_all" else (available[0], available[-1]))
            if len(set(episodes)) < 2 or any(e not in available for e in episodes):
                raise ValueError(f"two distinct requested episodes required; available={available}")
            manifest_path = data / "object_geometry/v1/manifest.json"
            manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            for ep in episodes:
                raw = RawSpatialDataset(data, episode=ep)
                frames = geometry.frame_ids(ep)
                if len(frames) < 16:
                    raise ValueError(f"episode {ep}: fewer than 16 geometry frames")
                selected = [frames[i] for i in np.linspace(0, len(frames) - 1, 16, dtype=int)]
                count = 0
                for row, video in zip(
                    raw.iter_rows(selected=selected, depth_roles=("front",)),
                    raw.iter_video_frames("front", selected=selected),
                    strict=True,
                ):
                    frame = int(row["frame_index"])
                    if video.frame_index != frame or frame != selected[count]:
                        raise ValueError("RGB/depth/geometry frame mismatch")
                    scene, mask = preprocess(video.rgb, row["observation.depths.cam_front"])
                    sample = geometry.get(episode_index=ep, frame_index=frame)
                    objects = {
                        role: np.asarray(getattr(sample, f"{role}_xyz_m"))[
                            np.asarray(getattr(sample, f"{role}_mask"), dtype=bool)
                        ]
                        for role in ("manipulated", "target")
                    }
                    cad_objects, failures = (
                        ({}, [f"CAD unavailable: {cad_load_error}"])
                        if ordered_cad is None
                        else fit_review_cad(sample, ordered_cad)
                    )
                    fit_failures.extend(f"e{ep} f{frame} {err}" for err in failures)
                    path = a.output_dir / dataset / f"e{ep:05d}_f{frame:06d}.png"
                    render_gt(
                        video.rgb,
                        intrinsic,
                        camera.T_base_color,
                        scene[mask],
                        objects,
                        path,
                        f"{dataset} ep={ep} frame={frame}; green=manipulated red=target",
                        cad_objects=cad_objects,
                    )
                    links.append(path.relative_to(a.output_dir).as_posix())
                    count += 1
                if count != 16:
                    raise ValueError(f"episode {ep}: decoded {count}/16 requested frames")
            reports[dataset] = {
                "status": "rendered_requires_human_review",
                "episodes": episodes,
                "frames": len(links),
                "geometry_manifest_sha256": manifest_hash,
                "geometry_source": geometry.manifest.get("source"),
                "geometry_calibration": geometry.manifest.get("calibration"),
                "profile": str(profile),
                "profile_sha256": profile_hash,
                "cad_fit_failures_count": len(fit_failures),
                "cad_fit_failures_first": fit_failures[:10],
                "cad_fit_note": "sampled fixed-order rigid fits; not independent verification or model predictions",
            }
        except (OSError, ValueError, KeyError, IndexError, TypeError, ImportError) as exc:
            reports[dataset] = {"status": "blocked", "error": f"{type(exc).__name__}: {exc}", "frames": len(links)}
        sections.append(
            f"<h2>{html.escape(dataset)}</h2><pre>{html.escape(json.dumps(reports[dataset], ensure_ascii=False, indent=2))}</pre>"
            + "".join(
                f'<p><a href="{html.escape(x)}">{html.escape(x)}</a><br><img width="1200" src="{html.escape(x)}"></p>'
                for x in links
            )
        )
    (a.output_dir / "report.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n")
    (a.output_dir / "index.html").write_text(
        "<meta charset='utf-8'><h1>Geometry v1 GT candidates, not model predictions</h1>"
        "<p>RGB; CAD rigid-fit candidate; geometry projection; RGB-D workspace cloud with geometry. Green=manipulated, red=target. "
        "Human review required; calibration and provenance are recorded below.</p>" + "".join(sections),
        encoding="utf-8",
    )
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return int(any(x["status"] == "blocked" for x in reports.values()))


if __name__ == "__main__":
    raise SystemExit(main())
