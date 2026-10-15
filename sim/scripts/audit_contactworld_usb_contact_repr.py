#!/usr/bin/env python3
"""Render a human review page for the contact-centred USB representation.

The page deliberately shows both retained and excluded local scene points.  It
is a review tool for the tactile-facing half-space filter, not evidence of a
full robot/object segmentation.
"""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from openpi.spatial_dataset.contactworld_contact_repr import (  # noqa: E402
    BACKSIDE_TOLERANCE_M, RELATION_RADIUS_M, SELF_SURFACE_MARGIN_M, build_frame,
)


def png_base64(figure) -> str:
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(figure)
    return base64.b64encode(buffer.getvalue()).decode()


def frame_from_video(dataset: Path, role: str, episode: int, frame: int) -> np.ndarray:
    path = (dataset / ("videos/chunk-%03d" % (episode // 1000)) /
            f"observation.images.{role}" / f"episode_{episode:06d}.mp4")
    reader = cv2.VideoCapture(str(path))
    reader.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, image = reader.read()
    reader.release()
    if not ok:
        return np.zeros((256, 256, 3), np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def force_canvas(local: np.ndarray, selected: np.ndarray, centers_uv: np.ndarray, anchors: np.ndarray) -> str:
    scale = max(float(np.abs(local).max()), 1e-6)
    figure, axes = plt.subplots(2, 3, figsize=(9, 5.7), constrained_layout=True)
    for finger in range(2):
        for axis in range(3):
            ax = axes[finger, axis]
            ax.imshow(local[finger, :, :, axis], cmap="coolwarm", vmin=-scale, vmax=scale, origin="upper")
            for anchor in range(2):
                if not anchors[finger, anchor]:
                    continue
                picks = selected[finger, anchor]
                # Convert normalized [u, v] back to grid coordinates.
                col = (picks[:, 0] + 1) * 0.5 * 13
                row = (picks[:, 1] + 1) * 0.5 * 9
                ax.scatter(col, row, s=8, facecolors="none", edgecolors="lime", linewidths=.45)
                center = centers_uv[finger, anchor]
                ax.scatter((center[0] + 1) * .5 * 13, (center[1] + 1) * .5 * 9,
                           s=32, marker="x", color="black", linewidths=1.2)
            ax.set_title(f"{'L' if finger == 0 else 'R'} F{'xyz'[axis]}")
            ax.set_xticks([]); ax.set_yticks([])
    figure.suptitle("Raw local TacFF; green=retained taxels, black ×=contact-component centre", fontsize=11)
    return png_base64(figure)


def relation_canvas(cloud: np.ndarray, taxel_xyz: np.ndarray, derived: dict[str, np.ndarray]) -> str:
    pad = taxel_xyz.mean(axis=1)
    query = derived["query_xyz_base"]
    # Display nearby points rejected by the half-space filter as grey context.
    nearby = []
    retained = []
    for finger in range(2):
        distance = np.linalg.norm(cloud - query[finger], axis=-1)
        context = cloud[distance <= RELATION_RADIUS_M]
        nearby.append(context)
        retained.append(derived["candidate_xyz_base"][finger][derived["candidate_mask"][finger]])
    figure = plt.figure(figsize=(8.8, 6.2), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    context = np.concatenate(nearby) if any(len(row) for row in nearby) else np.zeros((0, 3))
    if len(context):
        axis.scatter(context[:, 0], context[:, 1], context[:, 2], s=2, c="#a8afb9", alpha=.20,
                     label="nearby cloud before tactile-facing filter")
    colours = ("#46d3ff", "#ffbf3f")
    for finger in range(2):
        axis.scatter(taxel_xyz[finger, :, 0], taxel_xyz[finger, :, 1], taxel_xyz[finger, :, 2],
                     s=5, c=colours[finger], alpha=.7, label=f"{'left' if finger == 0 else 'right'} pad taxels")
        axis.scatter(query[finger, 0], query[finger, 1], query[finger, 2], s=65, marker="x", c=colours[finger])
        if len(retained[finger]):
            axis.scatter(retained[finger][:, 0], retained[finger][:, 1], retained[finger][:, 2],
                         s=14, c=colours[finger], edgecolors="black", linewidths=.2,
                         label=f"{'left' if finger == 0 else 'right'} retained candidates")
        inward = pad[1 - finger] - pad[finger]
        inward /= max(np.linalg.norm(inward), 1e-8)
        axis.quiver(query[finger, 0], query[finger, 1], query[finger, 2],
                    inward[0], inward[1], inward[2], length=.05, color=colours[finger])
    axis.set_xlabel("base X (m)"); axis.set_ylabel("base Y (m)"); axis.set_zlabel("base Z (m)")
    axis.set_title("Local cloud relation candidates; arrows point inward across the two pads")
    axis.legend(loc="upper left", fontsize=7)
    return png_base64(figure)


def relation_points(cloud: np.ndarray, taxel_xyz: np.ndarray,
                    derived: dict[str, np.ndarray]) -> dict:
    """Prepare compact data for the self-contained interactive 3-D viewer."""
    query = np.asarray(derived["query_xyz_base"], dtype=np.float32)
    retained_xyz = np.asarray(derived["candidate_xyz_base"], dtype=np.float32)
    retained_mask = np.asarray(derived["candidate_mask"], dtype=bool)
    local_parts = []
    for finger in range(2):
        distance = np.linalg.norm(cloud - query[finger], axis=-1)
        part = np.asarray(cloud[distance <= RELATION_RADIUS_M], dtype=np.float32)
        if len(part) > 1800:
            part = part[np.linspace(0, len(part) - 1, 1800, dtype=np.int32)]
        local_parts.append(part)
    local = (np.concatenate(local_parts, axis=0) if any(len(part) for part in local_parts)
             else np.zeros((0, 3), dtype=np.float32))
    return {
        "scene": np.round(np.asarray(cloud, dtype=np.float32), 5).tolist(),
        "local": np.round(local, 5).tolist(),
        "pads": [np.round(np.asarray(taxel_xyz[finger]), 5).tolist() for finger in range(2)],
        "retained": [np.round(retained_xyz[finger][retained_mask[finger]], 5).tolist()
                     for finger in range(2)],
        "retained_shell": [np.asarray(derived["candidate_shell"][finger][retained_mask[finger]],
                                        dtype=np.int8).tolist() for finger in range(2)],
        "queries": np.round(query, 5).tolist(),
    }


def chosen_frames(force: np.ndarray, length: int) -> list[int]:
    magnitude = np.linalg.norm(force, axis=-1).sum(axis=(1, 2))
    values = {0, length // 2, length - 1, int(np.argmax(magnitude)), int(np.argmin(magnitude))}
    return sorted(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "data/contactworld_usb_v3_bilateral")
    parser.add_argument("--repr-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--max-episodes", type=int, default=5)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    repr_root = (args.repr_root or dataset / "spatial/contact_repr_v2").resolve()
    manifest = json.loads((dataset / "spatial/manifest.json").read_text())
    all_rows = manifest["episodes"]
    if args.episodes:
        wanted = set(args.episodes)
        rows = [row for row in all_rows if int(row["episode_index"]) in wanted]
    else:
        indices = np.rint(np.linspace(0, len(all_rows) - 1, args.max_episodes)).astype(int)
        rows = [all_rows[index] for index in sorted(set(indices))]
    args.output.mkdir(parents=True, exist_ok=True)
    cards = []
    cloud_data = {}
    for row in rows:
        episode = int(row["episode_index"])
        src = dataset / "spatial/episodes" / f"episode_{episode:06d}"
        cloud = np.load(src / "pointcloud_fused_xyz.npy", mmap_mode="r")
        xyz = np.stack((np.load(src / "tactile_xyz_left_base.npy", mmap_mode="r"),
                        np.load(src / "tactile_xyz_right_base.npy", mmap_mode="r")), axis=1)
        force = np.stack((np.load(src / "tactile_force_left_base.npy", mmap_mode="r"),
                          np.load(src / "tactile_force_right_base.npy", mmap_mode="r")), axis=1)
        local = np.stack((np.load(src / "force_grid_left_local.npy", mmap_mode="r"),
                          np.load(src / "force_grid_right_local.npy", mmap_mode="r")), axis=1)
        with np.load(repr_root / "episodes" / f"episode_{episode:06d}.npz") as saved:
            derived = {key: saved[key] for key in saved.files}
        for frame in chosen_frames(force, len(force)):
            # Recompute once and compare enough fields to catch a stale sidecar.
            recomputed = build_frame(cloud[frame], xyz[frame], force[frame])
            if not np.array_equal(recomputed["candidate_mask"], derived["candidate_mask"][frame]):
                raise ValueError("Contact representation sidecar does not match source frame")
            if not np.array_equal(recomputed["candidate_shell"], derived["candidate_shell"][frame]):
                raise ValueError("Contact representation shell labels do not match source frame")
            front = frame_from_video(dataset, "front", episode, frame)
            wrist = frame_from_video(dataset, "wrist", episode, frame)
            image = np.concatenate((front, wrist), axis=1)
            ok, packed = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                                      [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                raise RuntimeError("JPEG encoding failed")
            selected = derived["patch_xyz_local"][frame] + derived["anchor_uv"][frame, :, :, None, :]
            local_force = local[frame]
            force_plot = force_canvas(local_force, selected, derived["anchor_uv"][frame],
                                      derived["anchor_mask"][frame])
            cloud_id = f"cloud-{episode}-{frame}"
            cloud_data[cloud_id] = relation_points(
                cloud[frame], xyz[frame],
                {key: derived[key][frame] for key in (
                    "query_xyz_base", "candidate_xyz_base", "candidate_mask", "candidate_shell")})
            counts = derived["candidate_mask"][frame].sum(axis=-1)
            shell_counts = [[int(((derived["candidate_shell"][frame, finger] == shell)
                                  & derived["candidate_mask"][frame, finger]).sum())
                             for shell in range(3)] for finger in range(2)]
            distance_ranges = []
            for finger in range(2):
                values = derived["candidate_distance"][frame, finger][derived["candidate_mask"][frame, finger]]
                distance_ranges.append([round(float(values.min()) * 100, 1),
                                        round(float(values.max()) * 100, 1)])
            valid = derived["finger_valid"][frame].tolist()
            cards.append(f'''<details class="case"><summary>episode {episode} · frame {frame} · candidates L/R {counts.tolist()} · shells L/R {shell_counts} · distance cm L/R {distance_ranges} · contact valid {valid}</summary>
<div class="grid"><section><h3>Front / wrist RGB</h3><img src="data:image/jpeg;base64,{base64.b64encode(packed.tobytes()).decode()}"></section>
<section><h3>Contact-centred sparse tactile</h3><img src="data:image/png;base64,{force_plot}"></section>
<section class="wide"><h3>Interactive tactile-to-scene point cloud</h3>
<div class="cloud-toolbar"><button type="button" data-cloud-action="local" data-cloud="{cloud_id}">局部关系点</button><button type="button" data-cloud-action="scene" data-cloud="{cloud_id}">全部场景点</button><button type="button" data-cloud-action="shell0" data-cloud="{cloud_id}">只看近层</button><button type="button" data-cloud-action="shell1" data-cloud="{cloud_id}">只看中层</button><button type="button" data-cloud-action="shell2" data-cloud="{cloud_id}">只看远层</button><button type="button" data-cloud-action="reset" data-cloud="{cloud_id}">重置视角</button><span>拖动旋转 · 滚轮缩放</span></div>
<canvas class="cloud-viewer" id="{cloud_id}" width="1100" height="560" aria-label="可旋转的机械臂 base 坐标系点云"></canvas>
<div class="legend"><span><i class="swatch context"></i>场景点</span><span><i class="swatch left"></i>左指 taxel</span><span><i class="swatch right"></i>右指 taxel</span><span><i class="swatch left-rel"></i>近层 0–8 cm</span><span><i class="swatch right-rel"></i>近层（右指）</span><span><i class="swatch mid-rel"></i>中层 8–18 cm</span><span><i class="swatch far-rel"></i>远层 18–36 cm</span><span><i class="swatch query"></i>力中心</span></div></section></div>
<p>默认显示查询点周围 {float(RELATION_RADIUS_M):.3f} m 的局部点；“全部场景点”显示这一帧的 4096 点云。灰色点是场景点，青色/黄色是左右指 taxel。候选按近层 0–8 cm、中层 8–18 cm、远层 18–36 cm 分配名额，颜色由深到浅表示距离层，页面会同时保留远层点，避免近处 U 盘占满候选。候选仍不是物体分割，也不等于 USB–插座真实碰撞点。</p></details>''')
    data_json = json.dumps(cloud_data, separators=(",", ":"))
    page = f'''<!doctype html><meta charset="utf-8"><title>ContactWorld USB contact representation audit</title>
<style>body{{margin:0 auto;max-width:1500px;padding:24px;background:#101722;color:#e8effa;font:16px system-ui;line-height:1.5}}h1{{margin-top:0}}details{{border:1px solid #40536d;border-radius:8px;margin:16px 0;background:#172334}}summary{{padding:12px;cursor:pointer;font-weight:650}}.case>div{{padding:0 12px 12px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}.wide{{grid-column:1/-1}}img{{max-width:100%;display:block;background:#081018}}p{{color:#bccce1}}.notice{{border-left:4px solid #67b7ff;background:#182a40;padding:12px}}.cloud-toolbar{{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:8px 0}}button{{border:1px solid #5c7fa8;background:#23466f;color:#e8effa;border-radius:5px;padding:7px 11px;cursor:pointer}}button:hover{{background:#2f5c8e}}.cloud-toolbar span{{color:#9fb3cd;font-size:13px}}.cloud-viewer{{width:100%;height:auto;display:block;background:#0a111b;border:1px solid #40536d;touch-action:none;cursor:grab}}.cloud-viewer.dragging{{cursor:grabbing}}.legend{{display:flex;flex-wrap:wrap;gap:12px 18px;color:#c8d5e5;font-size:13px;margin-top:8px}}.legend span{{display:inline-flex;align-items:center;gap:5px}}.swatch{{width:10px;height:10px;border-radius:50%;display:inline-block}}.context{{background:#8d9bab}}.left{{background:#29d6ff}}.right{{background:#ffbd45}}.left-rel{{background:#5af29c}}.right-rel{{background:#f25ac5}}.mid-rel{{background:#22b573}}.far-rel{{background:#9ef2c4}}.query{{background:#ffffff}}</style>
<h1>USB V3: contact-centred tactile and local tactile–scene relation input audit</h1>
<p class="notice">This page must be reviewed before training 16–19. It uses the same deterministic sidecar that training and online policy evaluation read. The contact centre is a finger-pad force centre, not a USB–socket collision point. The relation filter is tactile-facing local geometry, not simulator object identity or a full robot segmentation.</p>
<p>Each finger has up to two force-connected patches × 16 retained taxels; the model receives the two patch tokens plus one full-pad summary token. Relation models receive three learned relation tokens per finger over the coloured candidates.</p>
{''.join(cards)}
<script>
const CLOUD_DATA = {data_json};
(function() {{
  const colours = {{context: '#8d9bab', left: '#29d6ff', right: '#ffbd45'}};
  const shellColours = [
    ['#5af29c', '#f25ac5'],
    ['#22b573', '#d03cab'],
    ['#9ef2c4', '#ff9ae0'],
    ['#d7ffe8', '#ffd2f2'],
  ];
  const viewers = new Map();
  function init(canvas, data) {{
    const ctx = canvas.getContext('2d');
    const state = {{mode: 'local', shell: 'all', yaw: 0.72, pitch: 0.42, zoom: 1, dragging: false, lastX: 0, lastY: 0}};
    function reset() {{ state.yaw = 0.72; state.pitch = 0.42; state.zoom = 1; draw(); }}
    function draw() {{
      const rect = canvas.getBoundingClientRect();
      const ratio = Math.max(1, window.devicePixelRatio || 1);
      const width = Math.max(320, rect.width), height = Math.max(240, rect.width * 0.48);
      canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.fillStyle = '#0a111b'; ctx.fillRect(0, 0, width, height);
      const cloud = state.mode === 'scene' ? data.scene : data.local;
      const all = cloud.concat(data.pads[0], data.pads[1], data.retained[0], data.retained[1], data.queries);
      if (!all.length) return;
      const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
      all.forEach(p => {{ for (let i=0;i<3;i++) {{ min[i]=Math.min(min[i],p[i]); max[i]=Math.max(max[i],p[i]); }} }});
      const center = min.map((v,i)=>(v+max[i])*0.5), span = Math.max(...max.map((v,i)=>max[i]-min[i]), 0.02);
      const scale = Math.min(width,height)*0.78*state.zoom/span;
      function project(p) {{
        let x=p[0]-center[0], y=p[1]-center[1], z=p[2]-center[2];
        const cy=Math.cos(state.yaw), sy=Math.sin(state.yaw), x1=cy*x-sy*y, y1=sy*x+cy*y;
        const cp=Math.cos(state.pitch), sp=Math.sin(state.pitch), y2=cp*y1-sp*z, z2=sp*y1+cp*z;
        return [width/2+x1*scale, height/2-y2*scale, z2];
      }}
      const drawSet = (arr, color, radius, alpha) => arr.map(p => {{ const q=project(p); return [q[2],q[0],q[1]]; }}).sort((a,b)=>a[0]-b[0]).forEach(q => {{ ctx.globalAlpha=alpha; ctx.fillStyle=color; ctx.beginPath(); ctx.arc(q[1],q[2],radius,0,Math.PI*2); ctx.fill(); }});
      drawSet(cloud, colours.context, state.mode==='scene'?1.3:2.1, state.mode==='scene'?.35:.58);
      drawSet(data.pads[0], colours.left, 3.2, .95); drawSet(data.pads[1], colours.right, 3.2, .95);
      for (let finger=0; finger<2; finger++) {{
        for (let shell=0; shell<4; shell++) {{
          const selected = data.retained[finger].filter((_, index) =>
            (state.shell === 'all' || Number(state.shell) === shell) && data.retained_shell[finger][index] === shell);
          drawSet(selected, shellColours[shell][finger], 4.4, 1);
        }}
      }}
      data.queries.forEach((p,i) => {{ const q=project(p); ctx.strokeStyle=i===0?colours.left:colours.right; ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo(q[0]-7,q[1]-7); ctx.lineTo(q[0]+7,q[1]+7); ctx.moveTo(q[0]+7,q[1]-7); ctx.lineTo(q[0]-7,q[1]+7); ctx.stroke(); }});
      ctx.globalAlpha=1; ctx.fillStyle='#dbe8f8'; ctx.font='13px system-ui'; ctx.fillText('base X →', width-78, height-15); ctx.fillText('Y ↑', 13, 25); ctx.fillText('Z depth', 13, height-15);
      ctx.fillStyle='#9fb3cd'; ctx.fillText(state.mode==='scene'?`全部场景点：${{data.scene.length}}`:`局部点：${{data.local.length}}`, 14, height-15);
    }}
    canvas.addEventListener('pointerdown', e => {{ state.dragging=true; state.lastX=e.clientX; state.lastY=e.clientY; canvas.classList.add('dragging'); canvas.setPointerCapture(e.pointerId); }});
    canvas.addEventListener('pointermove', e => {{ if(!state.dragging)return; state.yaw+=(e.clientX-state.lastX)*.008; state.pitch=Math.max(-1.45,Math.min(1.45,state.pitch+(e.clientY-state.lastY)*.008)); state.lastX=e.clientX; state.lastY=e.clientY; draw(); }});
    canvas.addEventListener('pointerup', e => {{ state.dragging=false; canvas.classList.remove('dragging'); canvas.releasePointerCapture(e.pointerId); }});
    canvas.addEventListener('pointercancel', () => {{ state.dragging=false; canvas.classList.remove('dragging'); }});
    canvas.addEventListener('wheel', e => {{ e.preventDefault(); state.zoom=Math.max(.25,Math.min(8,state.zoom*Math.exp(-e.deltaY*.001))); draw(); }}, {{passive:false}});
    window.addEventListener('resize', draw); viewers.set(canvas.id, {{reset, setMode: mode => {{ state.mode=mode; draw(); }}, setShell: shell => {{ state.shell=shell; draw(); }}}}); draw();
  }}
  document.querySelectorAll('.cloud-viewer').forEach(canvas => init(canvas, CLOUD_DATA[canvas.id]));
  document.querySelectorAll('[data-cloud-action]').forEach(button => button.addEventListener('click', () => {{ const viewer=viewers.get(button.dataset.cloud); if(!viewer)return; const action=button.dataset.cloudAction; if(action==='reset')viewer.reset(); else if(action.startsWith('shell'))viewer.setShell(action.slice(5)===''?'all':action.slice(5)); else viewer.setMode(action); }}));
}})();
</script></html>'''
    (args.output / "index.html").write_text(page)
    (args.output / "audit.json").write_text(json.dumps({"episodes": [int(r["episode_index"]) for r in rows],
        "contact_repr_root": str(repr_root), "status": "passed"}, indent=2) + "\n")
    print(str(args.output / "index.html"))


if __name__ == "__main__":
    main()
