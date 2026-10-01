#!/usr/bin/env python3
"""Audit all released USB/Peg trajectories using poses and source task thresholds.

Pose criteria and motion heuristics are diagnostics, not contact or success labels.
"""
import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation
import zarr

from openpi.spatial_dataset.contactworld import TASKS, episode_split


def config_scalar(config: str, name: str) -> float:
    match = re.search(r"^\s*" + re.escape(name) + r":\s*([\d.eE+-]+)", config, re.M)
    if match is None:
        raise ValueError(f"Missing task parameter {name}")
    return float(match.group(1))


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    return {"min": float(values.min()), "median": float(np.median(values)),
            "p95": float(np.quantile(values, .95)), "max": float(values.max())}


def keypoint_error(plug_pos, plug_quat, socket_pos, socket_quat, offsets):
    plug_rot = Rotation.from_quat(plug_quat)
    socket_rot = Rotation.from_quat(socket_quat)
    distances = []
    for offset in offsets:
        plug = plug_pos + plug_rot.apply(np.broadcast_to(offset, plug_pos.shape))
        socket = socket_pos + socket_rot.apply(np.broadcast_to(offset, socket_pos.shape))
        distances.append(np.linalg.norm(plug - socket, axis=1))
    return np.mean(distances, axis=0)


def retraction_candidate(xy, z, endpoint_pass):
    """Kinematic lead for manual review: mouth approach, >4mm lift, later descent."""
    if not endpoint_pass:
        return False
    for i in np.flatnonzero((xy < .015) & (z > -.005) & (z < .030)):
        stop = min(len(z), i + 26)
        if stop - i < 5:
            continue
        j = i + int(np.argmax(z[i:stop]))
        if z[j] - z[i] >= .004 and np.min(z[j:]) <= z[i] - .003:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    report = {"scope": "All released USB/Peg episodes; offline recorded poses only",
              "success_caveat": "Source keypoint proximity is a pose criterion, not an independent physical insertion label",
              "strict_proxy_caveat": "The added 2mm XYZ / 5deg unsigned axial / 3-frame proxy is exploratory and not a physical insertion label",
              "retraction_caveat": "Heuristic kinematic candidates require video/contact review", "tasks": {}}
    for task, filename in (("insertion_usb", "TacSLTaskUSB.yaml"),
                           ("insertion_peg", "TacSLTaskPeg.yaml")):
        config = (args.source / "thirdparty/manifeel/manifeel/config/task" / filename).read_text()
        threshold = config_scalar(config, "close_error_thresh")
        count = int(config_scalar(config, "num_keypoints"))
        scale = config_scalar(config, "keypoint_scale")
        if config_scalar(config, "insertion_frac") != 1:
            raise ValueError("Expected source insertion_frac=1")
        offsets = np.zeros((count, 3), dtype=np.float64)
        offsets[:, 2] = (np.linspace(0, 1, count) - .5) * scale
        group = zarr.open_group(str(args.data / task), mode="r")
        data = group["data"]
        ends = np.asarray(group["meta/episode_ends"][:], dtype=int)
        starts = np.r_[0, ends[:-1]]
        arrays = {name: np.asarray(data[name][:]) for name in
                  ("plug_pos", "plug_quat", "socket_pos_gt", "socket_quat", "ee_pos")}
        errors = keypoint_error(arrays["plug_pos"], arrays["plug_quat"],
                                arrays["socket_pos_gt"], arrays["socket_quat"], offsets)
        delta = arrays["plug_pos"] - arrays["socket_pos_gt"]
        xyz_error = np.linalg.norm(delta, axis=1)
        xy_error = np.linalg.norm(delta[:, :2], axis=1)
        plug_axis = Rotation.from_quat(arrays["plug_quat"]).apply(np.broadcast_to([0., 0., 1.], delta.shape))
        socket_axis = Rotation.from_quat(arrays["socket_quat"]).apply(np.broadcast_to([0., 0., 1.], delta.shape))
        # Source plug/socket frames have opposing local z axes at insertion.
        axis_angle_deg = np.rad2deg(np.arccos(np.clip(np.abs(np.sum(plug_axis * socket_axis, axis=1)), 0, 1)))
        for episode, (start, end) in enumerate(zip(starts, ends)):
            start, end = int(start), int(end)
            endpoint_pass = bool(errors[end-1] < threshold)
            strict = bool(endpoint_pass and end-start >= 3 and np.all(xyz_error[end-3:end] < .002)
                          and np.all(axis_angle_deg[end-3:end] < 5))
            retract = retraction_candidate(xy_error[start:end], delta[start:end, 2], endpoint_pass)
            row = {"task": task, "episode": episode, "split": episode_split(task, episode),
                   "frames": end-start, "source_endpoint_pass": endpoint_pass,
                   "source_ever_pass": bool(np.any(errors[start:end] < threshold)),
                   "strict_pose_proxy": strict, "retraction_candidate": retract,
                   "initial_socket_x_m": float(arrays["socket_pos_gt"][start, 0]),
                   "initial_socket_y_m": float(arrays["socket_pos_gt"][start, 1]),
                   "initial_plug_to_socket_xy_m": float(xy_error[start]),
                   "initial_plug_to_socket_z_m": float(delta[start, 2]),
                   "endpoint_keypoint_error_mm": float(errors[end-1] * 1000),
                   "endpoint_xyz_error_mm": float(xyz_error[end-1] * 1000),
                   "endpoint_unsigned_axis_error_deg": float(axis_angle_deg[end-1]),
                   "max_ee_motion_m": float(np.linalg.norm(np.diff(arrays["ee_pos"][start:end], axis=0), axis=1).max())
                        if end-start > 1 else 0.0}
            rows.append(row)
        task_rows = [r for r in rows if r["task"] == task]
        report["tasks"][task] = {
            "episodes": len(task_rows), "frames": int(ends[-1]),
            "source_threshold_mm": threshold * 1000,
            "source_endpoint_pass": sum(r["source_endpoint_pass"] for r in task_rows),
            "source_ever_pass": sum(r["source_ever_pass"] for r in task_rows),
            "strict_pose_proxy": sum(r["strict_pose_proxy"] for r in task_rows),
            "retraction_candidates": [r["episode"] for r in task_rows if r["retraction_candidate"]],
            "endpoint_pass_by_split": {split: sum(r["source_endpoint_pass"] for r in task_rows if r["split"] == split)
                                       for split in ("train", "val", "test")},
            "episode_count_by_split": {split: sum(r["split"] == split for r in task_rows)
                                       for split in ("train", "val", "test")},
            "initial_socket_x_m": describe([r["initial_socket_x_m"] for r in task_rows]),
            "initial_socket_y_m": describe([r["initial_socket_y_m"] for r in task_rows]),
            "initial_plug_to_socket_xy_m": describe([r["initial_plug_to_socket_xy_m"] for r in task_rows]),
            "initial_plug_to_socket_z_m": describe([r["initial_plug_to_socket_z_m"] for r in task_rows]),
            "endpoint_keypoint_error_mm": describe([r["endpoint_keypoint_error_mm"] for r in task_rows]),
        }
    with (args.output / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["tasks"], indent=2))


if __name__ == "__main__":
    main()
