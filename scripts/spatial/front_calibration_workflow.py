#!/usr/bin/env python3
"""Dataset-independent front-camera annotation and two-stage residual fitting."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

from openpi.spatial.config import make_baseline_config
from openpi.spatial.preprocess import SpatialPreprocessor
from openpi.spatial_dataset.source import RawSpatialDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LANDMARKS = (
    REPO_ROOT / "configs/ur7e_xhand/front_landmarks_example.json"
)


def _frame_ids(value: str) -> list[int]:
    ids = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not ids or min(ids) < 0 or len(set(ids)) != len(ids):
        raise ValueError("--frames requires unique nonnegative comma-separated indices")
    return ids


def _landmarks(path: Path) -> list[dict]:
    entries = json.loads(path.read_text(encoding="utf-8"))["landmarks"]
    result = []
    for entry in entries:
        name = entry.get("name", entry.get("landmark_name"))
        link = entry.get("link", entry.get("corresponding_link"))
        xyz = entry.get("local_xyz_m", entry.get("local_xyz"))
        group = entry.get("group")
        if group is None:
            group = "tip" if name.startswith("finger_") and name.endswith("_tip") else "base"
        if (group not in ("base", "tip") or not name or not link
                or np.asarray(xyz).shape != (3,) or not np.all(np.isfinite(xyz))):
            raise ValueError(f"Invalid landmark {entry}")
        result.append({
            "name": name, "link": link, "local_xyz_m": xyz, "group": group,
            "description": entry.get("description", "Click the physically identifiable landmark center"),
        })
    return result


def _fk_points(preprocessor: SpatialPreprocessor, state: np.ndarray, landmarks: list[dict]) -> list[np.ndarray]:
    q = {
        name: float(state[index])
        for name, index in preprocessor.state_mapping.items()
        if name in preprocessor.kinematics.measured_movable
    }
    transforms = preprocessor.kinematics.compute(q, root="base_link")
    points = []
    for landmark in landmarks:
        T = transforms[landmark["link"]]
        points.append(T[:3, :3] @ np.asarray(landmark["local_xyz_m"], float) + T[:3, 3])
    return points


ANNOTATOR_HTML = """<!doctype html><html lang='zh'><meta charset='utf-8'>
<title>Front calibration annotation</title>
<style>body{font:15px system-ui;background:#15191e;color:#eee;margin:18px}button{padding:7px 12px;margin:4px;background:#334454;color:#fff;border:1px solid #607080;cursor:pointer}canvas{max-width:100%;height:auto;border:1px solid #555;cursor:crosshair}#wrap{display:grid;grid-template-columns:minmax(0,800px) 270px;gap:18px}p{line-height:1.5}</style>
<h2>Front calibration annotation</h2><p>Mark physical landmark centers on raw RGB. No predicted coordinates are shown. Skip occluded or ambiguous points.</p>
<div id='wrap'><canvas id='canvas'></canvas><div><p id='label'></p><p id='status'></p><button onclick='prev()'>Previous</button><button onclick='next()'>Next</button><button onclick='skip()'>Skip</button><button onclick='save()'>Export JSON</button><p>Click image to mark the current point. Export after finishing; fitting uses only marked points.</p></div></div>
<script>
const doc=__DOCUMENT__, images=__IMAGES__; let index=0;
const cv=document.getElementById('canvas'), ctx=cv.getContext('2d');let drawSerial=0,loadedSerial=0;
function draw(){const item=doc.annotations[index], img=new Image(), serial=++drawSerial; img.onload=()=>{if(serial!==drawSerial)return;cv.width=img.width;cv.height=img.height;ctx.drawImage(img,0,0);loadedSerial=serial;if(item.observed_uv){ctx.strokeStyle='#19e5a2';ctx.lineWidth=3;ctx.beginPath();ctx.arc(...item.observed_uv,8,0,Math.PI*2);ctx.stroke();}};img.src=images[item.frame_index];document.getElementById('label').textContent=`${index+1}/${doc.annotations.length} | frame ${item.frame_index} | ${item.name} | ${item.link} | ${item.group} | ${item.description}`;document.getElementById('status').textContent=item.observed_uv?`marked: ${item.observed_uv.map(x=>x.toFixed(1)).join(', ')}`:'unmarked';}
function next(){index=Math.min(index+1,doc.annotations.length-1);draw()}function prev(){index=Math.max(index-1,0);draw()}function skip(){doc.annotations[index].observed_uv=null;next()}
cv.onclick=e=>{if(loadedSerial!==drawSerial)return;const r=cv.getBoundingClientRect();doc.annotations[index].observed_uv=[(e.clientX-r.left)*cv.width/r.width,(e.clientY-r.top)*cv.height/r.height];draw()};
function save(){const blob=new Blob([JSON.stringify(doc,null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='front_annotations.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}draw();
</script></html>"""


def generate(args: argparse.Namespace) -> None:
    dataset = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    landmarks = _landmarks(args.landmarks)
    frames = _frame_ids(args.frames)
    reader = RawSpatialDataset(dataset, episode=args.episode)
    pre = SpatialPreprocessor.from_repo_root(
        repo_root=REPO_ROOT, config=make_baseline_config().with_camera_roles("front")
    )
    states = reader.states()
    state_by_frame = {
        int(frame): np.asarray(state, dtype=float)
        for frame, state in zip(states["frame_index"], states["observation.state"], strict=True)
        if int(frame) in frames
    }
    if set(state_by_frame) != set(frames):
        raise ValueError(f"Missing frame states: {sorted(set(frames) - set(state_by_frame))}")
    images = reader.load_video_frames("front", selected=frames)
    import cv2

    encoded_images = {}
    for frame in frames:
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(images[frame], cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError(f"Could not encode frame {frame}")
        encoded_images[frame] = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
    annotations = []
    for frame in frames:
        for landmark, point in zip(landmarks, _fk_points(pre, state_by_frame[frame], landmarks), strict=True):
            annotations.append({
                **landmark, "frame_index": frame, "base_xyz_m": point.tolist(), "observed_uv": None
            })
    doc = {"schema_version": 1, "dataset": str(dataset), "episode": args.episode, "annotations": annotations}
    (output / "front_annotations_template.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    html = ANNOTATOR_HTML.replace("__DOCUMENT__", json.dumps(doc)).replace("__IMAGES__", json.dumps(encoded_images))
    (output / "annotate.html").write_text(html, encoding="utf-8")
    print(f"Open {output / 'annotate.html'}; export front_annotations.json into {output}")


def _project(points_base: np.ndarray, T_base_color: np.ndarray, intrinsics: np.ndarray, delta: np.ndarray) -> np.ndarray:
    p_color = (points_base - T_base_color[:3, 3] - delta) @ T_base_color[:3, :3]
    if np.any(p_color[:, 2] <= 0):
        raise ValueError("Annotated 3D point lies behind front camera")
    return p_color[:, :2] / p_color[:, 2, None] * intrinsics[:2] + intrinsics[2:]


def fit(args: argparse.Namespace) -> None:
    from scipy.optimize import least_squares

    doc = json.loads(args.annotations.read_text(encoding="utf-8"))
    if doc.get("schema_version") != 1:
        raise ValueError("Unsupported annotation schema")
    pre = SpatialPreprocessor.from_repo_root(
        repo_root=REPO_ROOT, config=make_baseline_config().with_camera_roles("front")
    )
    camera = pre.calibration_bundle.cameras["front"]
    T = camera.T_base_color
    k0 = np.array([
        camera.color_intrinsics.fx, camera.color_intrinsics.fy,
        camera.color_intrinsics.cx, camera.color_intrinsics.cy,
    ], dtype=float)
    profile = {"schema_version": 1}
    if args.base_profile is not None:
        profile = json.loads(args.base_profile.read_text(encoding="utf-8"))
    delta = np.asarray(profile.get("front_extrinsic_translation_base_m", [0, 0, 0]), dtype=float)
    k = (
        np.array([profile["front_color_intrinsics"][key] for key in ("fx", "fy", "cx", "cy")], float)
        if "front_color_intrinsics" in profile else k0
    )
    rows = [row for row in doc["annotations"] if row["group"] == args.stage and row["observed_uv"] is not None]
    if len(rows) < 6:
        raise ValueError(f"Need at least six marked {args.stage} observations across diverse frames/landmarks")
    if len({int(row["frame_index"]) for row in rows}) < 3:
        raise ValueError("Use annotations from at least three distinct frames")
    points = np.asarray([row["base_xyz_m"] for row in rows], dtype=float)
    observed = np.asarray([row["observed_uv"] for row in rows], dtype=float)
    if (points.shape != (len(rows), 3) or observed.shape != (len(rows), 2)
            or not np.all(np.isfinite(points)) or not np.all(np.isfinite(observed))):
        raise ValueError("Invalid annotated coordinates")
    initial = _project(points, T, k, delta)
    if args.stage == "base":
        # The base/proximal pass only adjusts a base-frame translation; K stays frozen.
        result = least_squares(
            lambda x: (_project(points, T, k, x) - observed).ravel(),
            delta, bounds=(-0.05, 0.05), loss="soft_l1", f_scale=3.0,
        )
        profile["front_extrinsic_translation_base_m"] = result.x.tolist()
        fitted = _project(points, T, k, result.x)
    else:
        # The distal pass freezes extrinsic and fits color projection intrinsics only.
        pc = (points - T[:3, 3] - delta) @ T[:3, :3]
        if np.any(pc[:, 2] <= 0) or np.linalg.matrix_rank(np.c_[pc[:, 0] / pc[:, 2], np.ones(len(rows))]) < 2 or np.linalg.matrix_rank(np.c_[pc[:, 1] / pc[:, 2], np.ones(len(rows))]) < 2:
            raise ValueError("Tip observations lack depth/view diversity for a four-parameter K fit")
        result = least_squares(
            lambda x: (_project(points, T, x, delta) - observed).ravel(),
            k, bounds=([0.5*k0[0], 0.5*k0[1], k0[2]-100, k0[3]-100], [1.5*k0[0], 1.5*k0[1], k0[2]+100, k0[3]+100]),
            loss="soft_l1", f_scale=3.0,
        )
        profile["front_color_intrinsics"] = dict(zip(("fx", "fy", "cx", "cy"), result.x.tolist(), strict=True))
        fitted = _project(points, T, result.x, delta)
    if not result.success or np.linalg.matrix_rank(result.jac) < len(result.x):
        raise ValueError("Calibration fit failed or is not identifiable; collect more varied landmarks/poses")
    frame_ids = np.asarray([int(row["frame_index"]) for row in rows])
    per_frame_rmse = {
        str(frame): float(np.sqrt(np.mean((fitted[frame_ids == frame] - observed[frame_ids == frame]) ** 2)))
        for frame in sorted(set(frame_ids))
    }
    profile["provenance"] = {
        "annotation_file": str(args.annotations.resolve()), "dataset": doc["dataset"],
        "episode": doc["episode"], "stage": args.stage,
        "initial_rmse_px": float(np.sqrt(np.mean((initial - observed) ** 2))),
        "fit_count": len(rows), "fit_rmse_px": float(np.sqrt(np.mean((fitted - observed) ** 2))),
        "per_frame_rmse_px": per_frame_rmse,
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}; RMSE {profile['provenance']['fit_rmse_px']:.2f} px over {len(rows)} marks")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate_parser = sub.add_parser("annotate")
    generate_parser.add_argument("--dataset", type=Path, required=True)
    generate_parser.add_argument("--episode", type=int, default=0)
    generate_parser.add_argument("--frames", required=True)
    generate_parser.add_argument("--landmarks", type=Path, default=DEFAULT_LANDMARKS)
    generate_parser.add_argument("--output", type=Path, required=True)
    fit_parser = sub.add_parser("fit")
    fit_parser.add_argument("--annotations", type=Path, required=True)
    fit_parser.add_argument("--stage", choices=("base", "tip"), required=True)
    fit_parser.add_argument("--base-profile", type=Path, default=None)
    fit_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "annotate":
        generate(args)
    else:
        fit(args)


if __name__ == "__main__":
    main()
