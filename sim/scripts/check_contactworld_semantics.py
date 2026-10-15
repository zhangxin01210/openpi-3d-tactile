#!/usr/bin/env python3
"""Offline endpoint and metric-geometry checks; never infer ground-truth contact labels."""
import argparse
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import zarr

from check_contactworld_geometry import rotation


def scalar(config, key):
    return float(re.search(rf"^\s*{key}:\s*([\d.eE+-]+)", config, re.M)[1])


def mesh_bounds(urdf):
    mesh = ET.parse(urdf).find(".//visual/geometry/mesh")
    path = (urdf.parent / mesh.attrib["filename"]).resolve()
    vertices = np.array([[float(x) for x in line.split()[1:4]]
                         for line in path.read_text().splitlines() if line.startswith("v ")])
    return path, vertices.min(0), vertices.max(0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--assets", required=True, type=Path, help="Verified released industreal directory")
    p.add_argument("--geometry", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    geometry = json.loads(args.geometry.read_text())
    correction, translation = np.array(geometry["candidate_rotation"]), np.array(geometry["candidate_translation"])
    result = {"caveat": "Current-source endpoint pose criterion recomputed offline, NOT paper success rates, collision labels or runtime replay.", "tasks": {}}
    for task, config_name in (("insertion_usb", "TacSLTaskUSB.yaml"), ("insertion_peg", "TacSLTaskPeg.yaml")):
        config = (args.source / "data/replace_part/manifeel/config/task" / config_name).read_text()
        threshold, scale, count = (scalar(config, k) for k in ("close_error_thresh", "keypoint_scale", "num_keypoints"))
        assert scalar(config, "insertion_frac") == 1
        offsets = np.zeros((int(count), 3))
        offsets[:, 2] = (np.linspace(0, 1, int(count))-.5)*scale
        root = zarr.open_group(str(args.data / task), mode="r")
        d = root["data"]
        ends = root["meta/episode_ends"][:]
        starts = np.r_[0, ends[:-1]]
        endpoints = []
        step_indices_ok = True
        for ep, (s, e) in enumerate(zip(starts, ends)):
            step_indices_ok &= np.array_equal(d["step_idx"][s:e], np.arange(e-s))
            row = int(e-1)
            plug = offsets @ rotation(d["plug_quat"][row]).T + d["plug_pos"][row]
            socket = offsets @ rotation(d["socket_quat"][row]).T + d["socket_pos_gt"][row]
            error = float(np.linalg.norm(plug-socket, axis=1).mean())
            endpoints.append({"episode": ep, "final_keypoint_error_mm": error*1000, "passes_current_source_pose_criterion": error < threshold})
        entry = {"threshold_m": threshold, "keypoint_scale_m": scale, "num_keypoints": int(count),
                 "endpoint_pass_count": sum(x["passes_current_source_pose_criterion"] for x in endpoints),
                 "endpoint_fail_count": sum(not x["passes_current_source_pose_criterion"] for x in endpoints),
                 "endpoints": endpoints, "step_idx_contiguous_from_zero": bool(step_indices_ok),
                 "step_idx_all_zero": bool(np.all(d["step_idx"][:] == 0))}
        if task == "insertion_usb":
            mesh_path, lo, hi = mesh_bounds(args.assets / "urdf/USB_socket.urdf")
            red_points, red_counts, blue_counts, white_z = [], [], [], []
            # Entire 10 selected demos, not only the geometry reprojection's 30 frames.
            for ep in np.linspace(0, len(ends)-1, 10, dtype=int):
                for row in range(starts[ep], ends[ep]):
                    pc = d["pointcloud"][row]
                    rgb = pc[:, 3:]
                    xyz = pc[:, :3] @ correction.T + translation
                    red = (rgb[:, 0] > 1.5*rgb[:, 1]) & (rgb[:, 0] > 1.5*rgb[:, 2])
                    blue = (rgb[:, 2] > 1.5*rgb[:, 0]) & (rgb[:, 2] > 1.5*rgb[:, 1])
                    white = (rgb.min(1) > .8) & (np.ptp(rgb, axis=1) < .03)
                    red_points.append((xyz[red]-d["socket_pos_gt"][row]) @ rotation(d["socket_quat"][row]))
                    red_counts.append(int(red.sum()))
                    blue_counts.append(int(blue.sum()))
                    white_z.extend(xyz[white, 2].tolist())
            points = np.concatenate(red_points)
            outside = np.linalg.norm(np.maximum(np.maximum(lo-points, points-hi), 0), axis=1)
            entry["released_socket_geometry_check"] = {
                "mesh": str(mesh_path), "mesh_bounds_m": [lo.tolist(), hi.tolist()],
                "corrected_red_point_bounds_relative_socket_m": [points.min(0).tolist(), points.max(0).tolist()],
                "red_points": len(points), "fraction_inside_mesh_bbox_with_1mm_tolerance": float((outside <= .001).mean()),
                "bbox_outside_distance_max_mm": float(outside.max()*1000),
                "red_points_per_frame_min_median_max": [float(x) for x in np.quantile(red_counts, [0,.5,1])],
                "blue_points_per_frame_min_median_max": [float(x) for x in np.quantile(blue_counts, [0,.5,1])],
                "white_point_z_median_mm": float(np.median(white_z)*1000),
                "caveat": "Color masks approximate socket/plug. Bounding-box agreement is weaker than a surface-distance or runtime calibration test.",
            }
        result["tasks"][task] = entry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({task: {k:v for k,v in entry.items() if k not in ("endpoints",)} for task,entry in result["tasks"].items()}, indent=2))


if __name__ == "__main__":
    main()
