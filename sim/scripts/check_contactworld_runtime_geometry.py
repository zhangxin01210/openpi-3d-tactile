#!/usr/bin/env python3
"""Validate released fixed-front clouds using measured legacy Gym matrices.

Requires the native probe's --contactworld-camera capture. Does not establish
the provenance of collection-time calibration or calibrate the wrist camera.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import zarr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capture = np.load(args.camera)
    view, projection = capture["view"], capture["projection"]
    height, width = capture["depth"].shape
    inverse = np.linalg.inv(view)
    r = inverse[:3, :3].T
    t = inverse[3, :3]
    focal = np.array([projection[0, 0]*width/2, projection[1, 1]*height/2])
    signs = np.diag([1., -1., -1.])
    pixel_center = np.eye(3)
    pixel_center[0, 2], pixel_center[1, 2] = .5/focal
    correction = r @ signs @ pixel_center @ r
    old_correction = r @ signs @ r
    report = {
        "status": "passed", "runtime_capture": str(args.camera.resolve()),
        "scope": "Fixed front camera, base at world origin with identity rotation; released source conversion only",
        "limitations": ["Collection-time matrices absent from released data", "Wrist extrinsics unverified",
                        "No dynamic full-task state/action/contact replay in this check"],
        "formula_column_vectors": "world = A @ stored_xyz + t",
        "A": correction.tolist(), "t": t.tolist(),
        "K": [[float(focal[0]), 0, width/2], [0, float(focal[1]), height/2], [0, 0, 1]],
        "pixel_centers": "Coordinates u+0.5, v+0.5; array lookup subtracts 0.5 before rounding",
        "tasks": {},
    }
    for task in ("insertion_usb", "insertion_peg"):
        root = zarr.open_group(str(args.data/task), mode="r")
        ends = root["meta/episode_ends"][:]
        starts = np.r_[0, ends[:-1]]
        records, biases = [], []
        for ep in np.linspace(0, len(ends)-1, 10, dtype=int):
            for row in (int(starts[ep]), int((starts[ep]+ends[ep]-1)//2), int(ends[ep]-1)):
                pc = np.asarray(root["data/pointcloud"][row])
                rgb = np.asarray(root["data/front"][row])
                world = pc[:, :3] @ correction.T+t
                previous = pc[:, :3] @ old_correction.T+t
                biases.extend(np.linalg.norm(world-previous, axis=1).tolist())
                optical = (world-t) @ r @ signs
                uv = optical[:, :2]/optical[:, 2:3]*focal+[width/2, height/2]-.5
                pixels = np.round(uv).astype(int)
                valid = (optical[:, 2] > 0) & (pixels >= 0).all(1) & (pixels[:,0] < width) & (pixels[:,1] < height)
                p = pixels[valid]
                err = np.abs(rgb[p[:,1], p[:,0]]-pc[valid,3:6])
                pixel_err = np.abs(uv[valid]-p)
                records.append(dict(episode=int(ep), global_row=row, valid_points=int(valid.sum()),
                                    rgb_mae=float(err.mean()), pixel_integer_max_error=float(pixel_err.max())))
        mae = max(x["rgb_mae"] for x in records)
        residual = max(x["pixel_integer_max_error"] for x in records)
        passed = mae < 1e-5 and residual < 1e-3
        if not passed:
            report["status"] = "failed"
        report["tasks"][task] = dict(passed=passed, frames=len(records), rgb_mae_max=mae,
                                    pixel_integer_max_error=residual,
                                    half_pixel_world_shift_m_median=float(np.median(biases)),
                                    half_pixel_world_shift_m_p99=float(np.quantile(biases,.99)),
                                    records=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({t: {k:v for k,v in s.items() if k != "records"} for t,s in report["tasks"].items()}, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
