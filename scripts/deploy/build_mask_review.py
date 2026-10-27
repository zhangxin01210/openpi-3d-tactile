#!/usr/bin/env python3
"""Build a local HTML viewer for causal masks, human seeds, and RGBTrack projections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
KEY_FRAMES = (0, 90, 200, 265, 380, 400, 420, 465)
OBJECTS = ("funnel", "target_flask")
COLORS = {"funnel": (0, 220, 70), "target_flask": (240, 120, 20)}
MESHES = {
    "funnel": "assets/object_geometry/synthware_19_22/funnel_F181975.glb",
    "target_flask": "assets/object_geometry/synthware_19_22/flask_F301950.glb",
}
_MESH_VERTEX_CACHE: dict[str, np.ndarray] = {}


def project_mesh_outline(repo: Path, name: str, pose: np.ndarray, K: np.ndarray, T_color_base: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    import trimesh

    mesh_path = str(repo / MESHES[name])
    if mesh_path not in _MESH_VERTEX_CACHE:
        loaded = trimesh.load(mesh_path, force="scene", process=False)
        mesh = loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded
        _MESH_VERTEX_CACHE[mesh_path] = np.asarray(mesh.vertices, dtype=np.float64)
    vertices = _MESH_VERTEX_CACHE[mesh_path]
    vertices_base = vertices @ pose[:3, :3].T + pose[:3, 3]
    vertices_color = vertices_base @ T_color_base[:3, :3].T + T_color_base[:3, 3]
    visible = np.isfinite(vertices_color).all(axis=1) & (vertices_color[:, 2] > 0.05)
    if visible.sum() < 3:
        return np.zeros((*hw, 4), dtype=np.uint8)
    points = (vertices_color[visible, :2] / vertices_color[visible, 2:3]) @ K[:2, :2].T + K[:2, 2]
    points = np.rint(points[np.isfinite(points).all(axis=1)]).astype(np.int32)
    h, w = hw
    points[:, 0] = np.clip(points[:, 0], 0, w - 1)
    points[:, 1] = np.clip(points[:, 1], 0, h - 1)
    hull = cv2.convexHull(points)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    bgr = COLORS[name]
    cv2.polylines(rgba, [hull], True, (*bgr, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=ROOT)
    p.add_argument("--replay-dir", type=Path, required=True)
    p.add_argument("--online-dir", type=Path, required=True)
    p.add_argument("--manual-seed-dir", type=Path, required=True)
    p.add_argument("--calibration-profile", type=Path, default=Path("configs/ur7e_xhand/front_calibration_candidate.json"))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()

    repo = a.repo_root.expanduser().resolve()
    replay = a.replay_dir.expanduser().resolve()
    online = a.online_dir.expanduser().resolve()
    seed_dir = a.manual_seed_dir.expanduser().resolve()
    html_path = a.output.expanduser().resolve()
    if html_path.exists():
        raise FileExistsError(f"Refusing to overwrite {html_path}")
    assets = html_path.parent / (html_path.stem + "_assets")
    if assets.exists():
        raise FileExistsError(f"Refusing to overwrite {assets}")
    for sub in ("rgb", "projection"):
        (assets / sub).mkdir(parents=True, exist_ok=True)
    seed_areas = {}
    for name in OBJECTS:
        for path in (online / "masks" / name, seed_dir / f"{name}.png"):
            if not path.exists():
                raise FileNotFoundError(path)
    seed_out = assets / "manual_seed"
    seed_out.mkdir()
    for name in OBJECTS:
        image = cv2.imread(str(seed_dir / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not decode seed mask {name}")
        seed_areas[name] = int(np.count_nonzero(image))
        cv2.imwrite(str(seed_out / f"{name}.png"), image)

    import sys
    sys.path.insert(0, str(repo / "scripts/object_geometry/pose_tracking"))
    from rgbtrack_adapter import load_front_calibration

    profile = a.calibration_profile
    if not profile.is_absolute():
        profile = repo / profile
    camera = load_front_calibration(repo, profile)
    T_color_base = np.linalg.inv(camera.T_base_color)
    rows = [json.loads(line) for line in (online / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = []
    anomalies = []
    for row in rows:
        frame = int(row["frame_index"])
        raw_path = replay / f"frame_{frame:06d}.npz"
        if not raw_path.is_file():
            raw_path = replay / f"frame_{frame:05d}.npz"
        if not raw_path.is_file():
            raise FileNotFoundError(f"Missing replay RGB packet for frame {frame}")
        with np.load(raw_path, allow_pickle=False) as raw:
            rgb = np.asarray(raw["front_rgb"], dtype=np.uint8)
        h, w = rgb.shape[:2]
        ok, image_blob = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 86])
        if not ok:
            raise IOError(f"RGB encode failed for frame {frame}")
        cv2.imwrite(str(assets / "rgb" / f"{frame:06d}.jpg"), cv2.imdecode(image_blob, cv2.IMREAD_COLOR))
        objects = row["objects"]
        projections = {}
        for name in OBJECTS:
            pose_row = objects[name]
            if pose_row.get("valid"):
                rgba = project_mesh_outline(repo, name, np.asarray(pose_row["T_base_link_object"], dtype=np.float64), camera.K_color, T_color_base, (h, w))
                cv2.imwrite(str(assets / "projection" / f"{frame:06d}_{name}.png"), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
                projections[name] = f"{assets.name}/projection/{frame:06d}_{name}.png"
            else:
                projections[name] = None
        masks = row["masks"]
        review_reasons = []
        if row.get("mask_overlap_px", 0) > 0:
            review_reasons.append("mask_overlap")
        for name in OBJECTS:
            if not masks[name]["valid"]:
                review_reasons.append(f"{name}_mask_invalid")
            if masks[name]["recovery"]:
                review_reasons.append(f"{name}_mask_recovery")
            ratio = float(masks[name]["area_px"]) / max(seed_areas[name], 1)
            if ratio > 4.0 or ratio < 0.25:
                review_reasons.append(f"{name}_area_vs_seed_{ratio:.1f}x_review")
            if not objects[name]["valid"]:
                review_reasons.append(f"{name}_pose_invalid")
        bad = bool(review_reasons)
        if bad:
            anomalies.append(frame)
        payload.append({
            "frame": frame,
            "rgb": f"{assets.name}/rgb/{frame:06d}.jpg",
            "masks": {name: f"{online.name}/masks/{name}/{frame:06d}.png" for name in OBJECTS},
            "projection": projections,
            "mask": masks,
            "pose": {name: {
                "valid": bool(objects[name]["valid"]),
                "status": str(objects[name]["status"]),
                "confidence": float(objects[name]["confidence"]),
                "mask_valid": bool(masks[name]["valid"]),
                "mask_status": str(masks[name]["status"]),
                "mask_area_px": int(masks[name]["area_px"]),
                "recovery": bool(masks[name]["recovery"]),
            } for name in OBJECTS},
            "overlap_px": int(row.get("mask_overlap_px", 0)),
            "geometry_finite": bool(row.get("geometry_finite", False)),
            "geometry_valid": all(bool(objects[name]["valid"]) for name in OBJECTS) and bool(row.get("geometry_finite", False)),
            "geometry_trust": "NOT_VERIFIED",
            "review_reasons": review_reasons,
            "seed_area_ratio": {name: round(float(masks[name]["area_px"]) / max(seed_areas[name], 1), 3) for name in OBJECTS},
        })

    keys = sorted(set(frame for frame in KEY_FRAMES if frame < len(payload)) | set(anomalies))
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    keys_json = json.dumps(keys)
    page = r'''<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Episode 23 mask review</title>
<style>body{margin:0;background:#111827;color:#e5e7eb;font:14px system-ui,Arial}header{position:sticky;top:0;background:#1f2937;padding:12px;z-index:2;display:flex;gap:8px;align-items:center;flex-wrap:wrap}button{background:#374151;color:white;border:1px solid #6b7280;border-radius:5px;padding:7px 10px;cursor:pointer}input[type=range]{width:min(44vw,700px)}main{display:grid;grid-template-columns:minmax(500px,1fr) 310px;gap:14px;padding:14px}.view{position:relative;width:100%;background:#030712}canvas{width:100%;height:auto;display:block}aside{background:#1f2937;border-radius:8px;padding:14px;line-height:1.6}.keyrow{display:flex;gap:5px;flex-wrap:wrap}.bad{color:#fca5a5}.ok{color:#86efac}.label{font-weight:650}.checks label{display:block;margin:7px 0}small{color:#cbd5e1}@media(max-width:900px){main{grid-template-columns:1fr}}</style></head><body>
<header><strong>Episode 23 · 因果 mask / RGBTrack review</strong><button id="prev">◀</button><input id="slider" type="range" min="0" max="''' + str(len(payload)-1) + r'''" value="0"><button id="next">▶</button><span id="frame"></span></header>
<main><section><div class="view"><canvas id="canvas"></canvas></div><p class="keyrow" id="keys"></p></section><aside><div id="status"></div><hr><div class="checks">
<label><input type="checkbox" id="sam_funnel" checked> SAM2 funnel mask</label><label><input type="checkbox" id="sam_target" checked> SAM2 target flask mask</label>
<label><input type="checkbox" id="seed_funnel"> Human seed funnel mask</label><label><input type="checkbox" id="seed_target"> Human seed target flask mask</label>
<label><input type="checkbox" id="proj_funnel" checked> Funnel projected mesh contour</label><label><input type="checkbox" id="proj_target" checked> Target projected mesh contour</label></div>
<hr><small>彩色区域是当前帧 SAM2 mask；虚线是人工首帧 mask；实线为 RGBTrack 位姿下 metric mesh 顶点的投影凸包近似轮廓。面积相对人工首帧变化超过 4 倍仅标记“需复核”，不自动判失败。算法有效不代表分割或物理位姿准确；可信度在人工复核前显示 NOT VERIFIED。可在 <code>corrections.json</code> 提供逐帧人工 mask correction 后重新运行回放。</small><p id="quality"></p></aside></main>
<script>const FRAMES=''' + data + r''';const KEY=''' + keys_json + r''';const N=FRAMES.length,slider=document.querySelector('#slider'),canvas=document.querySelector('#canvas'),ctx=canvas.getContext('2d');let idx=0,cache=new Map();
function img(url){if(!url)return Promise.resolve(null);if(cache.has(url))return cache.get(url);let p=new Promise((resolve,reject)=>{let im=new Image();im.onload=()=>resolve(im);im.onerror=reject;im.src=url});cache.set(url,p);return p}
async function tint(url,color,alpha){let im=await img(url);if(!im)return;let c=document.createElement('canvas');c.width=im.width;c.height=im.height;let x=c.getContext('2d');x.drawImage(im,0,0);let d=x.getImageData(0,0,c.width,c.height),v=d.data;for(let i=0;i<v.length;i+=4){if(v[i]>0){v[i]=color[0];v[i+1]=color[1];v[i+2]=color[2];v[i+3]=Math.round(alpha*255)}else v[i+3]=0}x.putImageData(d,0,0);ctx.drawImage(c,0,0)}
async function draw(i){idx=Math.max(0,Math.min(N-1,i));slider.value=idx;let f=FRAMES[idx],base=await img(f.rgb);canvas.width=base.width;canvas.height=base.height;ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(base,0,0);
if(document.querySelector('#sam_funnel').checked)await tint(f.masks.funnel,[0,220,70],.34);if(document.querySelector('#sam_target').checked)await tint(f.masks.target_flask,[240,120,20],.34);
if(document.querySelector('#seed_funnel').checked)await tint(''' + json.dumps(f'{assets.name}/manual_seed/funnel.png') + r''',[0,230,255],.55);if(document.querySelector('#seed_target').checked)await tint(''' + json.dumps(f'{assets.name}/manual_seed/target_flask.png') + r''',[255,80,220],.55);
for(let [name,id] of [['funnel','proj_funnel'],['target_flask','proj_target']])if(document.querySelector('#'+id).checked&&f.projection[name]){let im=await img(f.projection[name]);ctx.setLineDash([8,5]);ctx.lineWidth=2;ctx.drawImage(im,0,0);ctx.setLineDash([])}
document.querySelector('#frame').textContent='frame '+f.frame+' / '+N;let lines=[];for(let n of ['funnel','target_flask']){let o=f.pose[n],color=o.valid?'ok':'bad';lines.push(`<span class="${color}"><b>${n}</b> pose=${o.status}, valid=${o.valid}, score=${o.confidence.toFixed(3)}<br>mask=${o.mask_status}, valid=${o.mask_valid}, area=${o.mask_area_px}px, recovery=${o.recovery}</span>`)}document.querySelector('#status').innerHTML=lines.join('<br><br>');document.querySelector('#quality').innerHTML=`geometry gate (tracker output): <b class="${f.geometry_valid?'ok':'bad'}">${f.geometry_valid?'valid':'invalid'}</b><br>geometry trust: <b class="bad">NOT VERIFIED</b><br>mask overlap: ${f.overlap_px} px<br>seed area ratio: funnel ${f.seed_area_ratio.funnel}×; target ${f.seed_area_ratio.target_flask}×<br>review flags: ${f.review_reasons.join(', ')||'none by current heuristics'}<br>geometry finite: ${f.geometry_finite}`}
slider.oninput=()=>draw(+slider.value);document.querySelector('#prev').onclick=()=>draw(idx-1);document.querySelector('#next').onclick=()=>draw(idx+1);for(let id of ['sam_funnel','sam_target','seed_funnel','seed_target','proj_funnel','proj_target'])document.querySelector('#'+id).onchange=()=>draw(idx);
let keys=document.querySelector('#keys');for(let k of KEY){let b=document.createElement('button');b.textContent='frame '+k;b.onclick=()=>draw(k);keys.appendChild(b)}document.addEventListener('keydown',e=>{if(e.key==='ArrowRight')draw(idx+1);if(e.key==='ArrowLeft')draw(idx-1)});draw(0);</script></body></html>'''
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(page, encoding="utf-8")
    summary = {
        "frames": len(payload),
        "key_frames": [f for f in KEY_FRAMES if f < len(payload)],
        "anomaly_frames": anomalies,
        "geometry_gate_valid_frames": sum(x["geometry_valid"] for x in payload),
        "geometry_trust": "NOT_VERIFIED",
        "html": str(html_path),
        "assets": str(assets),
        "projected_contour_method": "convex hull of transformed metric mesh vertices; visual QA approximation",
    }
    (html_path.parent / "mask_review_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
