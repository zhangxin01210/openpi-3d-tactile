#!/usr/bin/env python3
"""Read-only QA of XHand's *actual* 18-point URDF kinematic skeleton.

No synthetic 20/21-point topology, no new FK, no dataset preprocessing.
Requires the repo's verified URDF, state_mapping.csv, and existing FK.

Usage (from openpi-3d-tactile repo root):
  PYTHONPATH=src .venv/bin/python scripts/spatial/audit_xhand_keypoints.py \
      --dataset data/1028_press_button --episode 0 --frame 0

Creates an interactive HTML overlay (18 skeleton points + 600 taxels +
optional object/scene points), JSON/CSV and NPZ. No dataset is modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from openpi.spatial.kinematics import (
    RobotKinematics,
    joint_positions_from_state,
    load_state_mapping_csv,
)

from openpi.spatial.hand_keypoints import FINGERS, PALM_LINK, EXTRA_PALM_REFERENCE_LINKS, select_keypoints

COLORS = ("#dc644c", "#e4ae4d", "#479cc9", "#7bba83", "#ac86e9")


def sample_points(arr, n=1600):
    arr = np.asarray(arr)
    if len(arr) <= n:
        return arr
    return arr[np.linspace(0, len(arr)-1, n, dtype=int)]


def make_figure(out: Path, xyz, names, groups, kinds, edges, spatial, objects, extras):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[HTML SKIPPED] plotly not installed; JSON, CSV and NPZ are available")
        return
    fig = go.Figure()
    if spatial is not None:
        scene = sample_points(spatial.visual_xyz_m, 1800)
        fig.add_trace(go.Scatter3d(x=scene[:, 0], y=scene[:, 1], z=scene[:, 2],
                                   mode="markers", name="scene (subsample)",
                                   marker=dict(size=1.1, color="#b1b1b1", opacity=.14)))
        points = np.asarray(spatial.tactile_xyz_m)
        ids = np.asarray(spatial.finger_id)
        for fid, (finger, _, _) in enumerate(FINGERS):
            p = points[ids == fid]
            fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2],
                                       mode="markers", name=f"{finger} tactile surface",
                                       marker=dict(size=2.6, opacity=.65, color=COLORS[fid])))
    if objects is not None:
        for label, arr, mask, color in (
            ("manipulated", objects.manipulated_xyz_m, objects.manipulated_mask, "#4c7aee"),
            ("target", objects.target_xyz_m, objects.target_mask, "#e9814d"),
        ):
            p = np.asarray(arr)[np.asarray(mask).astype(bool)]
            if len(p):
                fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2],
                                           mode="markers", name=f"{label} {len(p)} points",
                                           marker=dict(size=2.5, color=color, opacity=.42)))
    for fid, (finger, _, _) in enumerate(FINGERS):
        inds = [0] + [i for i, g in enumerate(groups) if g == finger]
        p = xyz[inds]
        fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines+markers+text",
                                   name=f"{finger} skeleton", text=["" if i == 0 else str(i) for i in inds],
                                   textposition="top center", line=dict(width=7, color=COLORS[fid]),
                                   marker=dict(size=5, color=COLORS[fid]),
                                   customdata=[names[i] for i in inds],
                                   hovertemplate="%{customdata}<br>xyz=(%{x:.4f}, %{y:.4f}, %{z:.4f}) m<extra></extra>"))
    fig.add_trace(go.Scatter3d(x=[xyz[0, 0]], y=[xyz[0, 1]], z=[xyz[0, 2]],
                               mode="markers+text", text=["Palm frame?"], name="provisional palm",
                               marker=dict(size=8, color="#ffffff", line=dict(color="#222222", width=2))))
    for name, pos in extras.items():
        fig.add_trace(go.Scatter3d(x=[pos[0]], y=[pos[1]], z=[pos[2]], mode="markers",
                                   name=f"extra ref: {name}", marker=dict(size=6, symbol="diamond", color="#000000")))
    fig.update_layout(title="XHand FK: 12 joint axes + 5 link tips + 1 provisional palm", scene=dict(
        xaxis_title="base_link X (m)", yaxis_title="base_link Y (m)", zaxis_title="base_link Z (m)",
        aspectmode="data"), margin=dict(l=0, r=0, t=50, b=0), height=850)
    fig.write_html(out, include_plotlyjs=True, full_html=True)
    print("[HTML]", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=Path("data/1028_press_button"))
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--urdf", type=Path, default=None)
    ap.add_argument("--mapping", type=Path, default=None)
    ap.add_argument("--geometry-version", default="v1")
    ap.add_argument("--spatial-version", default="v1")
    ap.add_argument("--list-only", action="store_true", help="Validate URDF anchor definitions; no dataset needed")
    args = ap.parse_args()
    repo = args.repo.expanduser().resolve()
    dataset = args.dataset if args.dataset.is_absolute() else repo / args.dataset
    urdf = args.urdf or repo / "configs/ur7e_xhand/ur7e_xhand_verified.urdf"
    mapping_path = args.mapping or repo / "configs/ur7e_xhand/state_mapping.csv"
    kin = RobotKinematics(urdf)
    # These link origins have a clear URDF meaning; tip/palm contact location is NOT validated by this script.
    if args.list_only:
        anchors = [PALM_LINK] + [j for _, js, tip in FINGERS for j in (*js, tip)]
        for name in anchors:
            if name in kin.joint_by_name:
                print("joint", name, "->", kin.joint_by_name[name].child_link)
            elif name in kin.links:
                print("link ", name)
            else:
                raise KeyError(name)
        print("[TOPOLOGY] 18 candidate anchors")
        return
    from openpi.spatial_dataset.source import RawSpatialDataset
    from openpi.spatial_dataset.derived import SpatialDerivedDataset
    from openpi.object_geometry_dataset.derived import ObjectGeometryDerivedDataset

    row = next(RawSpatialDataset(dataset, episode=args.episode).iter_rows(
        selected=[args.frame], depth_roles=()), None)
    if row is None:
        raise IndexError(f"Episode {args.episode} frame {args.frame} not found")
    state = np.asarray(row["observation.state"], dtype=np.float64)
    if state.ndim != 1:
        raise ValueError(f"Expected 1D state, got {state.shape}")
    mapping = load_state_mapping_csv(mapping_path)
    q = joint_positions_from_state(state, mapping=mapping, kinematics=kin, root="base_link")
    tf = kin.compute(q, root="base_link")
    names, groups, kinds, xyz, edges = select_keypoints(kin, tf)
    spatial = SpatialDerivedDataset(dataset, version=args.spatial_version).get(
        episode_index=args.episode, frame_index=args.frame)
    objects = ObjectGeometryDerivedDataset(dataset, version=args.geometry_version).get(
        episode_index=args.episode, frame_index=args.frame)
    if getattr(spatial, "frame_index", args.frame) != args.frame:
        raise AssertionError("Spatial frame mismatch")
    if getattr(objects, "frame_index", args.frame) != args.frame:
        raise AssertionError("Object Geometry frame mismatch")
    extras = {link: np.asarray(tf[link][:3, 3], dtype=float) for link in EXTRA_PALM_REFERENCE_LINKS if link in tf}
    target = args.out or repo / "output/preprocess" / dataset.name / "keypoint_qa" / f"e{args.episode:05d}_f{args.frame:05d}"
    target = target.expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    rows = [{"index": i, "name": name, "group": groups[i], "kind": kinds[i],
             "x_m": float(p[0]), "y_m": float(p[1]), "z_m": float(p[2])}
            for i, (name, p) in enumerate(zip(names, xyz))]
    with (target / "keypoints.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(target / "keypoints.npz", xyz_m=xyz, names=np.array(names),
                        groups=np.array(groups), edges=np.array(edges, dtype=np.int32))
    tip_distances = {}
    for fid, (finger, _, _) in enumerate(FINGERS):
        i = next(i for i, (group, kind) in enumerate(zip(groups, kinds)) if group == finger and kind == "tip_link_origin")
        pts = np.asarray(spatial.tactile_xyz_m)[np.asarray(spatial.finger_id) == fid]
        if pts.size:
            tip_distances[finger] = float(np.linalg.norm(pts - xyz[i], axis=1).min())
    report = {"status": "computed_not_human_approved", "dataset": dataset.name,
              "episode": args.episode, "frame": args.frame, "num_points": 18,
              "coordinate_frame": "base_link", "unit": "m",
              "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
              "mapping_sha256": hashlib.sha256(mapping_path.read_bytes()).hexdigest(),
              "note": "Joint origins and tip link origins are not guaranteed to be contact-surface points; palm origin provisional.",
              "nearest_same_finger_taxel_to_tip_m": tip_distances,
              "extra_palm_references_xyz_m": {k: v.tolist() for k, v in extras.items()},
              "keypoints": rows}
    (target / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    make_figure(target / "review.html", xyz, names, groups, kinds, edges, spatial, objects, extras)
    print(f"[DONE] 18 XHand skeleton anchors, episode={args.episode}, frame={args.frame}, out={target}")
    print("[TIP-TO-TAXEL MIN DISTANCE (m)]", tip_distances)
    print("[NEXT] Open review.html and visually accept/reject palm and fingertip locations; do NOT batch export yet.")


if __name__ == "__main__":
    main()
