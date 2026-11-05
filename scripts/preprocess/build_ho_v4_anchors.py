#!/usr/bin/env python3
"""Build immutable HO V4 anchor assets and interactive CPU-only reviews."""
import argparse
import json
from pathlib import Path

import numpy as np

from openpi.object_geometry.ho_v4_anchors import ROLES
from openpi.object_geometry.ho_v4_anchors import load_sources
from openpi.object_geometry.ho_v4_anchors import make_asset
from openpi.object_geometry.ho_v4_anchors import validate_frame_geometry
from openpi.object_geometry_dataset.derived import ObjectGeometryDerivedDataset


def write_review(path, points, indices, title):
    # Small standalone canvas viewer: drag to rotate, wheel to zoom; no CDN.
    data = json.dumps({"points": points.tolist(), "indices": indices, "title": title}, separators=(",", ":"))
    html = """<!doctype html><meta charset="utf-8"><title>HO V4 anchor review</title>
<style>body{background:#111827;color:white;font:15px sans-serif}canvas{background:#09111e;touch-action:none}</style>
<h2 id="title"></h2><p>拖动旋转, 滚轮缩放。红色数字是固定锚点顺序。</p><canvas id="view" width="1000" height="760"></canvas>
<script>const D=DATA;document.getElementById('title').textContent=D.title;
const c=document.getElementById('view'),x=c.getContext('2d');let ax=.4,ay=.5,z=1,drag=false,px=0,py=0;
const p=D.points,ctr=[0,1,2].map(i=>p.reduce((s,a)=>s+a[i],0)/p.length);
const scale=300/Math.max(...p.map(a=>Math.hypot(...a.map((v,i)=>v-ctr[i]))));
function project(a){let X=a[0]-ctr[0],Y=a[1]-ctr[1],Z=a[2]-ctr[2],u=Math.cos(ay)*X+Math.sin(ay)*Z,w=-Math.sin(ay)*X+Math.cos(ay)*Z,v=Math.cos(ax)*Y-Math.sin(ax)*w;return [500+u*scale*z,380-v*scale*z]}
function draw(){x.clearRect(0,0,c.width,c.height);x.fillStyle='#7b8aa0';for(const q of p){let [u,v]=project(q);x.fillRect(u,v,3,3)}D.indices.forEach((id,i)=>{let [u,v]=project(p[id]);x.fillStyle='#ff5252';x.beginPath();x.arc(u,v,6,0,7);x.fill();x.fillStyle='white';x.fillText(String(i),u+7,v-7)})}draw();
c.onpointerdown=e=>{drag=true;px=e.clientX;py=e.clientY;c.setPointerCapture(e.pointerId)};c.onpointerup=()=>drag=false;
c.onpointermove=e=>{if(drag){ay+=(e.clientX-px)*.01;ax+=(e.clientY-py)*.01;px=e.clientX;py=e.clientY;draw()}};
c.onwheel=e=>{e.preventDefault();z*=Math.exp(-e.deltaY*.001);draw()};</script>""".replace("DATA", data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--datasets", nargs="+", default=["1028_press_button", "1030_yellow_socket"])
    args = parser.parse_args()
    repo = args.repo.resolve()
    for dataset in args.datasets:
        sources = load_sources(repo, dataset)
        candidates = {role: make_asset(sources[role]) for role in ROLES}
        fingerprints = {}
        for role in ROLES:
            points = sources[role]["points"]
            indices = np.asarray(candidates[role]["indices"], dtype=np.int32)
            fingerprints[role] = {"indices": indices,
                "surface_distances_m": np.linalg.norm(points[:, None] - points[indices][None, :], axis=-1)}
        reader = ObjectGeometryDerivedDataset(repo / "data" / dataset, version="v1")
        found = set()
        for sample in reader:
            geometry = sample.as_dict()
            validate_frame_geometry(geometry, fingerprints)
            found.update(role for role in ROLES if geometry[f"{role}_mask"].any())
            if found == set(ROLES):
                break
        if found != set(ROLES):
            raise ValueError(f"{dataset}: no valid sidecar geometry found for {sorted(set(ROLES) - found)}")
        for role in ROLES:
            asset = candidates[role]
            path = repo / "assets/ho_v4_anchors" / dataset / f"{role}.json"
            payload = json.dumps(asset, ensure_ascii=False, indent=2) + "\n"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.read_text(encoding="utf-8") != payload:
                raise FileExistsError(f"refusing to replace existing anchor asset: {path}")
            if not path.exists():
                path.write_text(payload, encoding="utf-8")
            review = repo / "output/analysis/ho_v4_anchor_review" / f"{dataset}_{role}.html"
            write_review(review, sources[role]["points"], asset["indices"], f"{dataset}: {role} / {asset['object_name']}")
            print(f"{path}\n{review}")


if __name__ == "__main__":
    main()
