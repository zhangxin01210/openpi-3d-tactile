#!/usr/bin/env python3
"""Test stored point-cloud frame claims against fixed upstream camera configuration.

This is an offline consistency test, not a simulator calibration certificate.
Candidate transforms are derived from source conventions, never fitted to data.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def rotation(q):
    q = np.asarray(q, float)
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def project(world, camera_position, camera_rotation, focal):
    local = (world-camera_position) @ camera_rotation
    optical = local * [1, -1, -1]
    uv = optical[:, :2] / np.maximum(optical[:, 2:3], 1e-12) * focal + 128
    valid = (optical[:, 2] > 0) & (uv >= 0).all(1) & (uv < 255.5).all(1)
    return uv, valid


def main():
    import cv2
    import zarr

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tasks", nargs="+", default=["insertion_usb", "insertion_peg"])
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    camera_position = np.array([.68, 0, .15])
    body_rotation = rotation([-.258819045, 0, .965925826, 0])
    # Gym camera Transform uses local +X forward, +Z up. View coordinates use
    # +X right, +Y up, -Z forward. Keep this explicit (not the same frame).
    view_to_body = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    r = body_rotation @ view_to_body
    focal = 128/np.tan(np.deg2rad(75)/2)
    # Source uses CV-like positive depth with an inverse-view transpose on row
    # points. Undo that rotation, change optical signs, then restore translation.
    correction = r @ np.diag([1, -1, -1]) @ r
    transforms = {
        "as_declared_base": (np.eye(3), np.zeros(3)),
        "translation_only": (np.eye(3), camera_position),
        "source_convention_candidate": (correction, camera_position),
    }
    result = {"camera_parameters": {"position": camera_position.tolist(), "quaternion_xyzw": [-.258819045, 0, .965925826, 0],
                                   "K": [[focal, 0, 128], [0, focal, 128], [0, 0, 1]], "resolution": [256, 256],
                                   "view_to_camera_body": view_to_body.tolist()},
              "candidate_rotation": correction.tolist(), "candidate_translation": camera_position.tolist(),
              "caveat": "Config-derived parameters, not calibration saved with demos. Color reprojection alone does not prove metric accuracy.",
              "tasks": {}}
    for task in args.tasks:
        root = zarr.open_group(str(args.data / task), mode="r")
        d = root["data"]
        ends = root["meta/episode_ends"][:]
        starts = np.r_[0, ends[:-1]]
        chosen = np.linspace(0, len(ends)-1, 10, dtype=int)
        scores = {name: [] for name in transforms}
        for ep in chosen:
            for label, row in (("first", int(starts[ep])), ("middle", int((starts[ep]+ends[ep]-1)//2)), ("last", int(ends[ep]-1))):
                pc = np.asarray(d["pointcloud"][row])
                rgb = np.asarray(d["front"][row])
                if rgb.shape[0] == 3:
                    rgb = np.moveaxis(rgb, 0, -1)
                if rgb.dtype == np.uint8:
                    rgb = rgb/255.
                panels = []
                for name, (rot, translation) in transforms.items():
                    points = pc[:, :3] @ rot.T + translation
                    uv, valid = project(points, camera_position, r, focal)
                    pixel = np.round(uv[valid]).astype(int)
                    pixel = np.clip(pixel, 0, 255)
                    error = np.abs(rgb[pixel[:, 1], pixel[:, 0]]-pc[valid, 3:6]).mean(-1)
                    # Integer pixel coincidence is a stronger check than color alone.
                    pixel_fraction = np.linalg.norm(uv[valid]-pixel, axis=-1)
                    scores[name].append({"episode": int(ep), "frame": row-int(starts[ep]), "global_row": row,
                        "in_image_fraction": float(valid.mean()), "rgb_mae": float(error.mean()) if len(error) else None,
                        "pixel_integer_residual_mean": float(pixel_fraction.mean()) if len(pixel_fraction) else None,
                        "rgb_exact_fraction": float((error < 1e-5).mean()) if len(error) else None})
                    panel = np.zeros((290, 256, 3), np.uint8)
                    panel[:256] = np.uint8(np.clip(rgb[..., ::-1]*255, 0, 255)*.45)
                    for (u, v), color in zip(pixel, pc[valid, 3:6]):
                        cv2.circle(panel, (int(u), int(v)), 1, (np.clip(color[::-1]*255, 0, 255)).tolist(), -1)
                    cv2.putText(panel, name, (3, 276), cv2.FONT_HERSHEY_SIMPLEX, .34, (255, 255, 255), 1)
                    panels.append(panel)
                # Overlay pose origins on original RGB using config calibration.
                original = np.zeros((290, 256, 3), np.uint8)
                original[:256] = np.uint8(np.clip(rgb[..., ::-1]*255, 0, 255))
                for key, color in (("socket_pos_gt", (0, 255, 0)), ("plug_pos", (0, 0, 255)), ("ee_pos", (255, 180, 0))):
                    if key in d:
                        uv, valid = project(np.asarray(d[key][row])[None], camera_position, r, focal)
                        if valid[0]:
                            xy = tuple(np.round(uv[0]).astype(int))
                            cv2.drawMarker(original, xy, color, cv2.MARKER_CROSS, 9, 1)
                cv2.putText(original, f"{task} ep{ep} {label}", (3, 276), cv2.FONT_HERSHEY_SIMPLEX, .36, (255,255,255), 1)
                cv2.imwrite(str(args.output / f"{task}_{ep:04d}_{label}.jpg"), np.hstack([original, *panels]))
        summary = {}
        for name, entries in scores.items():
            summary[name] = {"mean_in_image_fraction": float(np.mean([x["in_image_fraction"] for x in entries])),
                             "mean_rgb_mae": float(np.mean([x["rgb_mae"] for x in entries if x["rgb_mae"] is not None])) if any(x["rgb_mae"] is not None for x in entries) else None,
                             "frames": entries}
        result["tasks"][task] = summary
    (args.output / "geometry.json").write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({t: {k: {a: b for a, b in v.items() if a != "frames"} for k, v in scores.items()}
                      for t, scores in result["tasks"].items()}, indent=2))


if __name__ == "__main__":
    main()
