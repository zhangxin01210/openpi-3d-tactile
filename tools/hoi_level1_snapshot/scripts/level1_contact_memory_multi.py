#!/usr/bin/env python3
"""Level 1 A/B/C: object-mask depth observations, wrist-motion-supported geometry,
and tactile contact hypotheses. Diagnostic only; never asserts reconstructed surfaces.

Uses previous level1_sequence.py build products in cloud_frames/*.npz, plus
original states for strict robot FK and existing SAM2 funnel masks. DOES NOT
rewrite source datasets, masks, checkpoints or training code.

A = current candidate depth under object masks.
B = A + strictly causal, cross-motion-supported local history for *grasped*
    funnel, and multi-frame-stable baseline for *stationary* target flask.
C = B + recently activated tactile taxels, tentatively associated with funnel
    via proximity to its RGB mask; all active taxels appear separately in red.

Object membership is a HYPOTHESIS, not ground truth. A near-mask taxel may be
occluded or touch something else. Motion support can include robot surfaces.
No surface completeness or contact-ground-truth claims are made.

Run (in openpi root):
  PYTHONPATH=src .venv/bin/python /workspace/mnt/sqzhang26/hoi/scripts/level1_contact_memory.py
  # Optional pilot-phase override if auto grasp detection fails:
  # ... --grasp-start 175 --grasp-end 455 --run-name pilot_window
"""
from __future__ import annotations
import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

HOI = Path('/workspace/mnt/sqzhang26/hoi')
REPO = Path('/workspace/mnt/sqzhang26/openpi-3d-tactile')
ROOT = HOI / 'outputs/level1_sequence_e0'


def as_points(x: Any) -> np.ndarray:
    a = np.asarray(x, np.float32)
    if a.size == 0:
        return np.empty((0, 3), np.float32)
    if a.ndim != 2 or a.shape[1] != 3 or not np.all(np.isfinite(a)):
        raise ValueError(f'Bad point array: {a.shape}')
    return a


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    points = as_points(points)
    return (points @ np.asarray(T)[:3, :3].T + np.asarray(T)[:3, 3]).astype(np.float32)


def to_local(points: np.ndarray, T_base_link: np.ndarray) -> np.ndarray:
    # p_link = R^T (p_base - t), row-major equivalent
    points = as_points(points)
    T = np.asarray(T_base_link)
    return ((points - T[:3, 3]) @ T[:3, :3]).astype(np.float32)


def motion_informative(T1: np.ndarray, T2: np.ndarray, trans_m: float, angle_deg: float) -> bool:
    t = np.linalg.norm(T1[:3, 3] - T2[:3, 3])
    r = T1[:3, :3].T @ T2[:3, :3]
    a = math.degrees(math.acos(float(np.clip((np.trace(r) - 1) / 2, -1, 1))))
    return bool(t >= trans_m or a >= angle_deg)


def sparse_nearest(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if not len(query) or not len(reference):
        return np.full((len(query),), np.inf)
    from scipy.spatial import cKDTree
    return cKDTree(reference).query(query, k=1, workers=1)[0]


def grid_key(p: np.ndarray, voxel_m: float) -> tuple[int, int, int]:
    return tuple(np.floor(p / voxel_m).astype(np.int32).tolist())


def grid_history(snapshots: list['Snapshot'], *, property_name: str, t: int,
                 window_frames: int, voxel_m: float, min_motion_m: float,
                 min_motion_deg: float, min_observations: int,
                 max_points: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Causal voxel support: ≥ min distinct frames and ≥1 informative-motion pair.

    All historical points must already be in their proper shared local frame.
    For the flask this is base_link; for the grasped funnel it is hand palm.
    """
    eligible = [s for s in snapshots if t - window_frames <= s.frame <= t]
    cells = defaultdict(list)
    for s in eligible:
        points = getattr(s, property_name)
        # One point per frame per cell for frame-support counting; keep all for robust median
        per_frame = defaultdict(list)
        for p in points:
            per_frame[grid_key(p, voxel_m)].append(p)
        for k, vals in per_frame.items():
            cells[k].append((s.frame, np.median(vals, axis=0).astype(np.float32), s.T_palm))
    accepted = []
    evidence = []
    count_supported = 0
    for key, observations in cells.items():
        if len(observations) < min_observations:
            continue
        # Distinct observed frames are guaranteed by per-frame dedup above.
        informative = any(
            motion_informative(a[2], b[2], min_motion_m, min_motion_deg)
            for ia, a in enumerate(observations)
            for b in observations[ia + 1:]
        ) if property_name == 'funnel_local' else True
        # The target flask is stationary in base; repeated matching across frames
        # gives visibility support, not proof that points belong to the glass.
        if not informative:
            continue
        count_supported += 1
        accepted.append(np.median([x[1] for x in observations], axis=0))
        evidence.append(len(observations))
    if accepted:
        xyz = np.asarray(accepted, np.float32).reshape(-1, 3)
        supp = np.asarray(evidence, np.int16)
        if len(xyz) > max_points:
            order = np.lexsort((np.arange(len(xyz)), -supp))[:max_points]
            xyz, supp = xyz[order], supp[order]
    else:
        xyz = np.empty((0, 3), np.float32)
        supp = np.empty((0,), np.int16)
    return xyz, supp, dict(eligible_snapshots=len(eligible), voxel_cells=len(cells),
                           supported_cells=count_supported)


def force_stats(force: np.ndarray, *, threshold: float) -> dict:
    f = np.asarray(force, dtype=np.float64).reshape(-1)
    if f.size != 600 or not np.all(np.isfinite(f)) or np.any(f < 0):
        raise ValueError(f'Expected 600 finite nonnegative force norms: {f.shape}')
    nonzero = f[f > 0]
    active = f[f > threshold]
    per_finger = f.reshape(5, 120)
    top = np.sort(f)[-min(10, len(f)):]
    return {'max': float(f.max()), 'p95_all': float(np.percentile(f, 95)),
            'p95_nonzero': float(np.percentile(nonzero, 95)) if len(nonzero) else 0.,
            'active_count': int(len(active)), 'nonzero_count': int(len(nonzero)),
            'sum': float(f.sum()), 'top10_mean': float(np.mean(top)),
            'per_finger_sum': per_finger.sum(axis=1).astype(float).tolist(),
            'per_finger_active': (per_finger > threshold).sum(axis=1).astype(int).tolist()}


def mask_proximity_for_taxels(taxel_base: np.ndarray, mask: np.ndarray, camera,
                               max_dist_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Predict color pixel for each known FK taxel, then query its 2D mask distance.
    No physical contact or object identity claim follows from proximity alone.
    """
    import cv2
    from openpi.spatial.geometry import transform_points as to_frame, project_pinhole
    camera_xyz = to_frame(taxel_base, np.linalg.inv(camera.T_base_color))
    uv = project_pinhole(camera_xyz, camera.color_intrinsics)
    h, w = mask.shape
    valid = np.isfinite(uv).all(axis=1) & (camera_xyz[:, 2] > 0)
    pixels = np.rint(np.nan_to_num(uv, nan=-1e8)).astype(np.int64)
    valid &= ((pixels[:, 0] >= 0) & (pixels[:, 0] < w) &
              (pixels[:, 1] >= 0) & (pixels[:, 1] < h))
    distances = np.full((len(taxel_base),), np.inf, np.float32)
    if np.any(mask):
        # CV2 distanceTransform distances to nearest mask pixel, 0 in mask.
        distance_map = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        distances[valid] = distance_map[pixels[valid, 1], pixels[valid, 0]]
    return distances <= max_dist_px, distances


@dataclass
class Snapshot:
    frame: int
    T_palm: np.ndarray
    funnel_local: np.ndarray
    flask_base: np.ndarray


def read_states(repo: Path, dataset: Path, frames: list[int], episode: int) -> dict[int, np.ndarray]:
    from openpi.spatial_dataset.source import RawSpatialDataset
    ds = RawSpatialDataset(dataset, episode=episode)
    out = {}
    for row in ds.iter_rows(selected=frames, depth_roles=()):
        k = int(row['frame_index'])
        out[k] = np.asarray(row['observation.state'], np.float64)
    if set(out) != set(frames):
        raise ValueError(f'Missing state rows: {sorted(set(frames) - set(out))}')
    return out


def load_clouds(input_dir: Path) -> dict[int, dict]:
    result = {}
    for path in sorted(input_dir.glob('*.npz')):
        with np.load(path, allow_pickle=False) as f:
            frame = int(np.asarray(f['frame_index']).item())
            if frame in result:
                raise ValueError(f'duplicate frame {frame}')
            result[frame] = {
                'funnel': as_points(f['funnel_xyz_base_m']),
                'flask': as_points(f['target_flask_xyz_base_m']),
                'taxels': as_points(f['tactile_xyz_base_m']),
                'forces': np.asarray(f['tactile_force_norm'], np.float32),
                'scene': as_points(f['other_depth_xyz_base_m']),
            }
    if not result:
        raise FileNotFoundError(f'No point cloud NPZ found at {input_dir}')
    return result


def sequence_report_for_clouds(cloud_dir: Path) -> Path:
    """Map the coexistence-safe cloud directory name to its build report."""
    if cloud_dir.name == 'cloud_frames':
        return cloud_dir.parent / 'sequence_report.json'
    prefix = 'cloud_frames_'
    if cloud_dir.name.startswith(prefix):
        return cloud_dir.parent / f'sequence_report_{cloud_dir.name[len(prefix):]}.json'
    raise ValueError(f'Cannot infer sequence report from cloud directory: {cloud_dir}')


def compact(points: np.ndarray, limit: int = 1000) -> list:
    p = as_points(points)
    if len(p) > limit:
        p = p[np.linspace(0, len(p) - 1, limit, dtype=int)]
    return [[round(float(x), 4) for x in p[:, i]] for i in range(3)]


def load_previous_overlays(path: Path) -> dict[int, str]:
    """Reuse original sync RGB+mask overlays embedded in the prior HTML.

    The prior HTML is wholly local: do not download anything or recompute RGB-D.
    Missing file is permitted; missing overlays become a visible empty panel.
    """
    if not path.is_file():
        print(f'[NOTE] Previous 3D HTML not found, RGB+Mask side panel omitted: {path}')
        return {}
    s = path.read_text(encoding='utf8')
    marker = 'const frames='
    i = s.find(marker)
    if i < 0:
        print('[NOTE] Previous HTML has no embedded frame array; RGB panel omitted')
        return {}
    try:
        items, _ = json.JSONDecoder().raw_decode(s[i + len(marker):])
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f'Cannot parse source HTML {path}: {e}') from e
    return {int(f['frame']): f['overlay'] for f in items if f.get('overlay')}


def overlay_for_frame(root: Path, frame: int) -> str | None:
    """Use lightweight RGB+Mask images saved by original build if available.
    Otherwise no overlay is fine; previous sequence_3d.html remains separate.
    """
    for path in (root / 'overlays' / f'{frame:05d}.jpg', root / 'overlays' / f'{frame:05d}.png',
                 root / 'tracking_overlay' / f'{frame:05d}.jpg'):
        if path.is_file():
            import base64
            kind = 'image/png' if path.suffix.lower() == '.png' else 'image/jpeg'
            return 'data:' + kind + ';base64,' + base64.b64encode(path.read_bytes()).decode('ascii')
    return None


def build_html(frames: list[dict], js_lib: str, out: Path, *, title: str):
    data = json.dumps(frames, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    html = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>__TITLE__</title>
<style>body{font:14px system-ui,Arial;background:#f7f8fb;color:#192232;margin:12px}
header{display:flex;gap:12px;align-items:center;flex-wrap:wrap}h2{margin:2px}
.twocol{display:grid;grid-template-columns:minmax(450px,1.5fr) minmax(280px,1fr);gap:10px}.rgbpanel{background:white;padding:8px;border:1px solid #dee3e9;align-self:stretch}.rgbpanel img{width:100%;max-height:68vh;object-fit:contain}@media(max-width:920px){.twocol{display:block}.rgbpanel{margin:10px 0}}
.row{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:10px 0}
button,select{padding:7px;border:1px solid #aaa;border-radius:5px;background:white}
#seek{width:min(67vw,840px)}#plot{height:68vh;min-height:460px;background:white;border:1px solid #dee3e9}
.chart{height:235px;background:white;margin:10px 0;border:1px solid #dee3e9}
.note{line-height:1.65;color:#663c20;background:#fff8ec;padding:10px}
.info{font:13px ui-monospace,Consolas,monospace;white-space:pre-wrap;background:white;padding:12px}
label{margin-right:8px;white-space:nowrap}</style>
<script>__PLOTLY__</script></head><body>
<header><h2>Level 1 / A-B-C 接触几何实验</h2><span id="num"></span></header>
<p class="note">A 当前 Mask+RGB-D；B 额外显示多帧一致性筛选的历史候选点；C 再显示近 Mask 的触觉候选锚点。
红点是真实的触觉激活位置，粉红点只是可能与漏斗关联的历史接触锚点；两者均不等于精确物体表面真值。
任何颜色点都不能自动用于真接触标注。训练前应另外检查对象身份、遮挡、置信度。</p>
<div class="row"><button id="prev">◀ 前帧</button><button id="play">▶ 播放</button><button id="next">后帧 ▶</button>
<input id="seek" type="range" min="0" value="0"><label>显示分级 <select id="mode"><option value="A">A 当前点</option><option value="B">B 当前+历史</option><option value="C" selected>C 当前+历史+触觉锚点</option></select></label></div>
<div class="row"><label><input type="checkbox" id="scene">灰色背景</label><label><input type="checkbox" id="hand" checked>绿色未激活触觉点</label>
<label><input type="checkbox" id="active" checked>红色激活点</label><label><input type="checkbox" id="flask" checked>目标烧瓶</label>
<label><input type="checkbox" id="funnel" checked>漏斗</label><label><input type="checkbox" id="history" checked>历史点</label>
<button id="focus">聚焦手物局部</button><button id="all">显示全场景</button></div>
<div class="twocol"><div id="plot"></div><aside class="rgbpanel"><strong>同步 RGB + 物体 Mask（上一轮已有）</strong><br><img id="overlay" alt="当前帧RGB叠加图"><div id="imageNote"></div></aside></div><div id="metrics" class="chart"></div><div id="finger" class="chart"></div>
<div id="info" class="info"></div>
<script>
const frames=__DATA__;const seek=document.getElementById('seek'),get=id=>document.getElementById(id);
seek.max=frames.length-1;let idx=0,timer=null,focus=true;let camera=null;
const palette={funnel:'#ee8424',flask:'#985ce3',fmem:'#1ebcce',smem:'#7c49c0',anchor:'#e83f9b',hand:'#28aa54',touch:'#ee3136',scene:'#a8b4bb'};
function trace(points,name,color,size,opacity){let p=points||[[],[],[]];return {type:'scatter3d',mode:'markers',name,x:p[0],y:p[1],z:p[2],marker:{size,color,opacity},hovertemplate:'x:%{x:.3f} y:%{y:.3f} z:%{z:.3f}<extra>'+name+'</extra>'};}
function draw(){const f=frames[idx],mode=get('mode').value;
let tracks=[];
if(get('scene').checked)tracks.push(trace(f.scene,'other RGB-D (subsample)','#a4adb4',1,0.17));
if(get('flask').checked){tracks.push(trace(f.flask,'A flask mask-depth',palette.flask,2.7,.75));if(mode!=='A'&&get('history').checked)tracks.push(trace(f.static_flask,'B flask static history',palette.smem,3.8,.6));}
if(get('funnel').checked){tracks.push(trace(f.funnel,'A funnel mask-depth',palette.funnel,3.1,.8));if(mode!=='A'&&get('history').checked)tracks.push(trace(f.hand_memory,'B funnel motion-supported',palette.fmem,4.0,.8));
if(mode==='C'&&get('history').checked)tracks.push(trace(f.anchors,'C candidate contact anchors',palette.anchor,6,1));}
if(get('hand').checked)tracks.push(trace(f.inactive,'inactive taxel FK',palette.hand,2.8,.7));
if(get('active').checked){let t=trace(f.active,'active taxels (measured)',palette.touch,5,1);t.marker.size=f.active_sizes;tracks.push(t);}
let center=f.center||[.3,.3,.3],axis=focus?.22:.68;
let layout={margin:{t:10,r:10,b:10,l:10},showlegend:true,legend:{orientation:'h',y:1.06,font:{size:11}},uirevision:'keepcamera',scene:{aspectmode:'cube',
xaxis:{title:'base X (m)',range:[center[0]-axis,center[0]+axis]},
yaxis:{title:'base Y (m)',range:[center[1]-axis,center[1]+axis]},
zaxis:{title:'base Z (m)',range:[Math.max(-.12,center[2]-axis),center[2]+axis]},
camera:camera||{eye:{x:1.5,y:1.3,z:.85}}}};
if(tracks.length===0)tracks.push(trace([[],[],[]],'empty','#999',1,1));
Plotly.react('plot',tracks,layout,{displaylogo:false,responsive:true,scrollZoom:true}).then(()=>{if(!get('plot').__cameraHook){get('plot').on('plotly_relayout',e=>{if(e['scene.camera'])camera=e['scene.camera'];});get('plot').__cameraHook=true;}});
get('num').textContent=`Episode frame ${f.frame}  (${idx+1}/${frames.length})`;
seek.value=idx;
if(f.overlay){get('overlay').src=f.overlay;get('imageNote').textContent='Episode frame '+f.frame;}else{get('overlay').removeAttribute('src');get('imageNote').textContent='这帧没有可用的RGB叠加图，可对照上一轮回放';}
let s=f.stats,m=f.meta;
get('info').textContent=`Frame ${f.frame} | Mode ${mode} | grip hypothesis: ${m.grasp?'ON':'OFF'} | Mask proximity associated contact taxels: ${m.associated}\n`+
`A funnel ${m.funnel} / flask ${m.flask} | B funnel memory ${m.fmem} / fixed flask ${m.smem} | C candidate contact anchors ${m.anchor}\n`+
`Touch active ${s.active_count} / 600 | max ${s.max.toFixed(2)} | top10 mean ${s.top10_mean.toFixed(2)} | sum ${s.sum.toFixed(2)} | raw P95 ${s.p95_all.toFixed(2)} (often zero)\n`+
`B observed-now: funnel near depth ${m.fmem_current_depth}/${m.fmem}, near mask ${m.fmem_current_mask}/${m.fmem}; flask near depth ${m.smem_current_depth}/${m.smem}, near mask ${m.smem_current_mask}/${m.smem}.\n`+
`These are separate, imperfect checks; through-glass background may satisfy both. Mask occlusion may reject true surface.`;
function cursor(plotId){const p=get(plotId);if(p.data)Plotly.relayout(plotId,{'shapes[0].x0':f.frame,'shapes[0].x1':f.frame});}cursor('metrics');cursor('finger');}
function init(){let xs=frames.map(f=>f.frame);let sh=[{type:'line',x0:xs[0],x1:xs[0],yref:'paper',y0:0,y1:1,line:{width:2,dash:'dot',color:'#777'}}];
Plotly.newPlot('metrics',[
{x:xs,y:frames.map(f=>f.stats.max),mode:'lines',name:'maximum taxel norm',line:{color:'#df4848'}},
{x:xs,y:frames.map(f=>f.stats.top10_mean),mode:'lines',name:'top10 mean',line:{color:'#e49035'}},
{x:xs,y:frames.map(f=>f.stats.active_count),mode:'lines',name:'active taxel count',yaxis:'y2',line:{color:'#17a05e'}}],
{height:235,margin:{l:65,r:65,t:15,b:32},legend:{orientation:'h',y:1.25},xaxis:{title:'episode frame'},
yaxis:{title:'sensor magnitude (not N)'},yaxis2:{title:'active count',overlaying:'y',side:'right'},shapes:sh},{responsive:true,displaylogo:false});
let fingers=['thumb','index','middle','ring','pinky'];let colors=['#bc4b50','#3082bb','#44a16c','#8b61b1','#d89a30'];
Plotly.newPlot('finger',fingers.map((nm,i)=>({x:xs,y:frames.map(f=>f.stats.per_finger_sum[i]),mode:'lines',name:nm,line:{color:colors[i]}})),
{height:235,margin:{l:65,r:35,t:20,b:32},legend:{orientation:'h',y:1.25},xaxis:{title:'episode frame'},yaxis:{title:'force norm sum by finger (raw) '},shapes:sh},{responsive:true,displaylogo:false});
get('metrics').on('plotly_click',e=>seekNearest(e.points[0].x));get('finger').on('plotly_click',e=>seekNearest(e.points[0].x));
set(0);}
function seekNearest(frame){let best=0,err=Infinity;frames.forEach((f,i)=>{let d=Math.abs(f.frame-frame);if(d<err){best=i;err=d}});set(best);}
function set(i){idx=Math.max(0,Math.min(frames.length-1,i));draw()}
function stop(){if(timer!==null){clearInterval(timer);timer=null};get('play').textContent='▶ 播放'}
get('play').onclick=()=>{if(timer!==null){stop();return};get('play').textContent='⏸ 暂停';timer=setInterval(()=>{if(idx>=frames.length-1){stop();return};set(idx+1)},260)};
get('prev').onclick=()=>set(idx-1);get('next').onclick=()=>set(idx+1);seek.oninput=()=>set(+seek.value);
for(let id of ['mode','scene','hand','active','flask','funnel','history'])get(id).onchange=draw;
get('focus').onclick=()=>{focus=true;draw()};get('all').onclick=()=>{focus=false;draw()};init();
</script></body></html>'''
    html = html.replace('__PLOTLY__', js_lib).replace('__DATA__', data).replace('__TITLE__', title)
    out.write_text(html, encoding='utf8')


def run(args):
    import cv2
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / 'src'))
    # The formal path is versioned in this checkout; do not silently pick an
    # older helper copied into the external HOI workspace.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(HOI / 'scripts'))
    from openpi.object_geometry.calibration import calibration_provenance_error
    from openpi.object_geometry.calibration import front_calibration_provenance
    from openpi.object_geometry.calibration import resolve_front_calibration_profile
    from openpi.spatial.preprocess import SpatialPreprocessor
    from openpi.spatial.kinematics import joint_positions_from_state
    import level1_object_cloud_probe as cloud_common
    from plotly.offline import get_plotlyjs

    cloud_dir = Path(args.input_dir)
    profile = (
        resolve_front_calibration_profile(repo, args.front_calibration_profile)
        if args.front_calibration_profile is not None else None
    )
    if profile is not None and args.run_name == 'abc_v1':
        raise ValueError(
            'Formal profiled memory requires a new --run-name (for example '
            'abc_v1_front_candidate_stride1); refusing to reuse abc_v1.'
        )
    source_report_path = (
        Path(args.source_sequence_report)
        if args.source_sequence_report is not None else sequence_report_for_clouds(cloud_dir)
    )
    source_report = None
    if source_report_path.is_file():
        source_report = json.loads(source_report_path.read_text(encoding='utf8'))
    if profile is not None:
        if source_report is None:
            raise FileNotFoundError(
                f'Formal memory requires the stride=1 cloud report: {source_report_path}'
            )
        cfg_for_check, _ = cloud_common._make_calibration(repo, profile)
        expected_calibration = front_calibration_provenance(repo, profile, config=cfg_for_check)
        mismatch = calibration_provenance_error(source_report.get('calibration'), expected_calibration)
        if mismatch:
            raise ValueError(f'Source cloud report is incompatible: {mismatch}')
        if int(source_report.get('stride', -1)) != 1:
            raise ValueError('Formal memory requires stride=1 source clouds')
    elif source_report is not None and source_report.get('calibration', {}).get('calibration_mode') == 'profile':
        raise ValueError(
            'Source clouds were built with a front profile; pass the same '
            '--front-calibration-profile for mask association.'
        )
    clouds = load_clouds(cloud_dir)
    frame_ids = sorted(clouds)
    output = Path(args.output_root) / args.run_name
    if output.exists():
        raise FileExistsError(f'Will not overwrite previous output {output}. Use --run-name another_name')
    masks_dir = Path(args.output_root) / 'masks/funnel'
    if not masks_dir.is_dir():
        raise FileNotFoundError(f'Funnel SAM2 mask dir missing: {masks_dir}')
    print(f'[LOAD] {len(frame_ids)} frame clouds from {cloud_dir}', flush=True)
    states = read_states(repo, args.dataset, frame_ids, args.episode)
    overlays = load_previous_overlays(Path(args.source_html)) if not args.no_rgb else {}
    print(f'[RGB] Reused {len(overlays)} previously exported overlay frames', flush=True)
    cfg, cam = cloud_common._make_calibration(repo, profile)
    calibration = front_calibration_provenance(repo, profile, config=cfg)
    prep = SpatialPreprocessor.from_repo_root(repo_root=repo, config=cfg)
    if args.palm_link not in prep.kinematics.links:
        raise ValueError(f'Invalid --palm-link {args.palm_link}, absent in URDF')

    # Process FK and tactile first. Test existing exported samples against recomputed tactile.
    features = {}
    for fid in frame_ids:
        data = clouds[fid]
        state = states[fid]
        q = joint_positions_from_state(state, mapping=prep.state_mapping, kinematics=prep.kinematics)
        transforms = prep.kinematics.compute(q, root='base_link')
        T = np.asarray(transforms[args.palm_link], np.float64)
        xyz = data['taxels']
        force = data['forces']
        if xyz.shape != (600,3) or force.shape != (600,):
            raise ValueError(f'Expected tactile XYZ [600,3] and forces [600], frame {fid}: {xyz.shape}/{force.shape}')
        maskfile = masks_dir / f'{fid:05d}.png'
        if not maskfile.is_file():
            print(f'[MASK] Missing funnel frame {fid}, contact association unavailable', flush=True)
            associated = np.zeros(600, bool)
            proximity = np.full(600, np.inf, np.float32)
        else:
            mask = cv2.imread(str(maskfile), cv2.IMREAD_GRAYSCALE)
            if mask is None: raise RuntimeError(f'Cannot load {maskfile}')
            associated, proximity = mask_proximity_for_taxels(xyz, mask > 127, cam, args.mask_radius_px)
        active = force > args.force_threshold
        n_assoc = int(np.sum(active & associated))
        st = force_stats(force, threshold=args.force_threshold)
        features[fid] = dict(T=T, associated=associated, proximity=proximity,
                             active=active, stats=st, n_assoc=n_assoc)

    # Optional phase windows are *experiment labels*, never hidden ground-truth contact.
    forced = args.grasp_start is not None
    if forced and args.grasp_end is None:
        raise ValueError('--grasp-start requires --grasp-end')
    if args.grasp_end is not None and not forced:
        raise ValueError('--grasp-end requires --grasp-start')
    in_grasp = False
    last_contact = -10**9
    last_associated = -10**9
    grasp_by_frame = {}
    for fid in frame_ids:
        f = features[fid]
        possible = f['stats']['active_count'] >= args.min_active_taxels
        visual_assoc = f['n_assoc'] >= args.min_associated_taxels
        if forced:
            in_grasp = bool(args.grasp_start <= fid <= args.grasp_end)
        else:
            if possible and visual_assoc:
                last_associated = fid
                in_grasp = True
            if possible:
                last_contact = fid
            if in_grasp and (fid - last_contact > args.release_gap_frames or
                             fid - last_associated > args.max_unconfirmed_frames):
                in_grasp = False
        grasp_by_frame[fid] = in_grasp
    if not any(grasp_by_frame.values()):
        print('[WARN] No automatic hand–funnel association. C still shows real activated taxels, '
              'but historical contact anchors/B hand memory will remain EMPTY. '
              'You may run explicitly marked pilot-phase bounds.', flush=True)
    else:
        i = [f for f in frame_ids if grasp_by_frame[f]]
        print(f'[GRASP-HYPOTHESIS] frames {min(i)}..{max(i)}, {len(i)} snapshots; '
              f'forced bounds: {forced}', flush=True)

    output.mkdir(parents=True, exist_ok=False)
    (output / 'frames').mkdir()
    recent_funnel: deque[Snapshot] = deque()
    early_flask: list[Snapshot] = []
    prior_anchors: deque[tuple[int, np.ndarray, float]] = deque()
    view_data, report_rows = [], []
    prev_grasp = False
    static_cache = None
    for seq, fid in enumerate(frame_ids):
        data = clouds[fid]
        extra = features[fid]
        T = extra['T']
        A_funnel, A_flask = data['funnel'], data['flask']
        current_grasp = grasp_by_frame[fid]
        stats = extra['stats']
        mask_assoc_active = extra['associated'] & extra['active']
        # A recording of active finger surface positions in current base_link.
        active_xyz = data['taxels'][extra['active']]
        inactive_xyz = data['taxels'][~extra['active']]
        # Temporal memory must never bridge a release/regrasp event.
        if prev_grasp and not current_grasp:
            recent_funnel.clear()
            prior_anchors.clear()
        if current_grasp and not prev_grasp:
            recent_funnel.clear()
            prior_anchors.clear()
        prev_grasp = current_grasp

        # The target flask is physically fixed in this episode (task-level assumption).
        # Reuse only early candidates, before robot insertion. Stably-repeating
        # background THROUGH glass can survive, so label as measured candidates.
        if fid <= args.flask_reference_end and len(A_flask):
            early_flask.append(Snapshot(fid, T, np.empty((0,3),np.float32), A_flask))
        if fid <= args.flask_reference_end or static_cache is None:
            static_cache = grid_history(
                early_flask, property_name='flask_base', t=fid, window_frames=10**8,
                voxel_m=args.flask_voxel_mm/1000., min_motion_m=0, min_motion_deg=0,
                min_observations=args.min_static_observations, max_points=args.max_flask_points)
        static_flask, static_support, static_meta = static_cache

        B_local = np.empty((0,3),np.float32)
        B_support = np.empty((0,),np.int16)
        B_meta = {}
        if current_grasp:
            near_palm = np.linalg.norm(A_funnel - T[:3, 3],axis=1) < args.hand_roi_radius_m
            # Optionally flag points near tactile taxels as robot/true-contact ambiguous;
            # don't silently call them glass. The B memory uses only other points.
            away_taxel = sparse_nearest(A_funnel, data['taxels']) > args.exclude_taxel_mm/1000.
            funnel_local = to_local(A_funnel[near_palm & away_taxel], T)
            recent_funnel.append(Snapshot(fid, T, funnel_local, np.empty((0,3),np.float32)))
            while recent_funnel and fid - recent_funnel[0].frame > args.history_frames:
                recent_funnel.popleft()
            B_local, B_support, B_meta = grid_history(
                list(recent_funnel), property_name='funnel_local', t=fid,
                window_frames=args.history_frames, voxel_m=args.history_voxel_mm/1000.,
                min_motion_m=args.min_motion_mm/1000., min_motion_deg=args.min_motion_deg,
                min_observations=args.min_history_observations, max_points=args.max_memory_points)
        B_funnel = transform_points(B_local, T)

        # Observational corroboration, not surface verification:
        # (1) near currently *observed* candidate depth in 3D;
        # (2) projected inside/near the CURRENT RGB mask. The second check can
        # be false under hand occlusion, even for physically correct geometry.
        observed_radius = args.observed_mm / 1000.
        B_funnel_near_current_depth = sparse_nearest(B_funnel, A_funnel) <= observed_radius
        B_flask_near_current_depth = sparse_nearest(static_flask, A_flask) <= observed_radius
        def projected_mask_agreement(points, object_name):
            mask_file = Path(args.output_root)/'masks'/object_name/f'{fid:05d}.png'
            if not mask_file.is_file() or not len(points):
                return np.zeros((len(points),), dtype=bool)
            binary = cv2.imread(str(mask_file),cv2.IMREAD_GRAYSCALE)
            if binary is None:
                raise RuntimeError(f'Could not read {mask_file}')
            flags, _ = mask_proximity_for_taxels(points, binary>127, cam, args.mask_radius_px)
            return flags
        B_funnel_in_mask = projected_mask_agreement(B_funnel, 'funnel')
        B_flask_in_mask = projected_mask_agreement(static_flask, 'target_flask')

        # C: recent contact candidates are remembered in palm local coordinates;
        # they are associated only weakly by proximity to the funnel RGB mask.
        # Contact may be at the hand–funnel interface; mask is often occluded.
        if current_grasp:
            for p, f_mag in zip(data['taxels'][mask_assoc_active],data['forces'][mask_assoc_active]):
                p_local = to_local(np.asarray(p,np.float32)[None],T)[0]
                prior_anchors.append((fid,p_local,float(f_mag)))
            while prior_anchors and fid - prior_anchors[0][0] > args.anchor_ttl_frames:
                prior_anchors.popleft()
        else:
            prior_anchors.clear()
        contacts = defaultdict(list)
        for afid,p_local,fmag in prior_anchors:
            contacts[grid_key(p_local,args.anchor_voxel_mm/1000.)].append((afid,p_local,fmag))
        anchor_local=[]; anchor_strength=[]; anchor_last_seen=[]
        for vals in contacts.values():
            anchor_local.append(np.median([q[1] for q in vals],axis=0))
            anchor_strength.append(max(q[2] for q in vals))
            anchor_last_seen.append(max(q[0] for q in vals))
        if anchor_local:
            a_local=np.asarray(anchor_local,np.float32)
            astrength=np.asarray(anchor_strength,np.float32)
            arecent=np.asarray(anchor_last_seen,np.int32)
            if len(a_local)>args.max_anchor_points:
                keep=np.argsort(-arecent)[:args.max_anchor_points]
                a_local,astrength,arecent = a_local[keep],astrength[keep],arecent[keep]
            anchors=transform_points(a_local,T)
        else:
            anchors=np.empty((0,3),np.float32)
            astrength=np.empty((0,),np.float32)
            arecent=np.empty((0,),np.int32)

        out_file = output/'frames'/f'{fid:05d}.npz'
        np.savez_compressed(out_file,
            frame_index=np.int32(fid),
            A_funnel_candidate_xyz_base_m=A_funnel,
            A_flask_candidate_xyz_base_m=A_flask,
            B_funnel_supported_history_xyz_base_m=B_funnel,
            B_funnel_support_count=B_support,
            B_funnel_near_current_depth=B_funnel_near_current_depth,
            B_funnel_near_current_mask=B_funnel_in_mask,
            B_flask_static_history_xyz_base_m=static_flask,
            B_flask_support_count=static_support,
            B_flask_near_current_depth=B_flask_near_current_depth,
            B_flask_near_current_mask=B_flask_in_mask,
            C_candidate_funnel_contact_xyz_base_m=anchors,
            C_contact_activation_norm=astrength,
            C_anchor_last_seen_frame=arecent,
            XHand_all_taxel_xyz_base_m=data['taxels'],
            XHand_all_taxel_force_norm=data['forces'],
            XHand_active_taxel_mask=extra['active'],
            XHand_associated_taxel_mask=extra['associated'],
            T_base_palm=T,
            grasp_hypothesis=np.bool_(current_grasp),
        )
        local_pts = np.concatenate([A_funnel,active_xyz,B_funnel],axis=0)
        center = np.median(local_pts, axis=0).tolist() if len(local_pts) else T[:3,3].tolist()
        mn = dict(grasp=bool(current_grasp),associated=extra['n_assoc'],
                  funnel=len(A_funnel),flask=len(A_flask),fmem=len(B_funnel),
                  smem=len(static_flask),anchor=len(anchors),
                  fmem_current_depth=int(B_funnel_near_current_depth.sum()),
                  fmem_current_mask=int(B_funnel_in_mask.sum()),
                  smem_current_depth=int(B_flask_near_current_depth.sum()),
                  smem_current_mask=int(B_flask_in_mask.sum()))
        # Avoid massive HTML by subsampling to visualization limits. Full outputs remain in npz.
        size = np.clip(4 + 3*np.log1p(data['forces'][extra['active']]),5,13).round(2).tolist()
        view_data.append(dict(frame=fid, funnel=compact(A_funnel,args.display_points),
                              flask=compact(A_flask,args.display_points),
                              hand_memory=compact(B_funnel,args.display_points),
                              static_flask=compact(static_flask,args.display_points),
                              anchors=compact(anchors,args.display_points),
                              inactive=compact(inactive_xyz,600),
                              active=compact(active_xyz,600),active_sizes=size,
                              scene=compact(data['scene'],args.scene_points),
                              stats=stats,meta=mn,center=[float(x) for x in center],
                              overlay=overlays.get(fid)))
        report_rows.append(dict(frame=fid,**mn,**{f'force_{k}':v for k,v in stats.items()},
                                B_support_details=B_meta, flask_support_details=static_meta,
                                contact_association='2D mask proximity only; not ground truth'))
        if seq%20==0 or seq==len(frame_ids)-1:
            print(f'[FRAME] {seq+1}/{len(frame_ids)} f{fid}: active={stats["active_count"]} '
                  f'asso={extra["n_assoc"]} B={len(B_funnel)} C={len(anchors)}', flush=True)

    # Save report/HTML last, with explicit metadata/assumptions for downstream code.
    manifest = dict(schema_version=1,source_cloud_frames=str(cloud_dir),
        source_cloud_run=cloud_dir.name,
        source_sequence_report=str(source_report_path) if source_report is not None else None,
        source_calibration=source_report.get('calibration') if source_report is not None else None,
        calibration=calibration,episode=args.episode,
        output_frames=str(output/'frames'),frame_count=len(frame_ids),
        params={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()},
        semantic_notes=[
            'A: depths projected into RGB masks, not necessarily object surface.',
            'B funnel: past A points supported across meaningful palm motion, not verified object identity.',
            'B flask: multiframe stationary points, background seen through glass may persist.',
            'C: inferred mask-near activated taxel samples; no direct taxel-to-object assignment ground truth.',
            'C contact anchors are cached in palm frame only during estimated/marked stable grasp.',
            'All contact events and tactile force magnitudes are observational proxies, not physical Newtons.',
            'All processing causal: use only current/past frames. No future for historical memory.',
        ],frames=report_rows)
    (output/'report.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    html_file = output/'contact_memory_abc.html'
    build_html(view_data,get_plotlyjs(),html_file,title='Level 1 — A/B/C 接触约束与几何记忆')
    print('[DONE]',html_file, f'({html_file.stat().st_size/1e6:.1f} MB)')
    print('[DATA]',output/'frames')
    print('[REPORT]',output/'report.json')


def main():
    a=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    a.add_argument('--episode',type=int,default=0)
    a.add_argument('--repo',type=Path,default=REPO)
    a.add_argument('--dataset',type=Path,default=REPO/'data/1018_peg_all')
    a.add_argument('--output-root',type=Path,default=None)
    a.add_argument('--input-dir',type=Path,default=None)
    a.add_argument('--source-html',type=Path,default=None)
    a.add_argument('--source-sequence-report',type=Path,default=None,
                   help='Build report for --input-dir; inferred from cloud_frames_<run-name> when omitted')
    a.add_argument('--front-calibration-profile',type=Path,default=None,
                   help='Required when consuming formal profiled stride=1 clouds')
    a.add_argument('--no-rgb',action='store_true',help='Skip extracting the prior RGB+mask overlays')
    a.add_argument('--run-name',default='abc_v1')
    a.add_argument('--palm-link',default='right_hand_link')
    a.add_argument('--force-threshold',type=float,default=0.5)
    a.add_argument('--min-active-taxels',type=int,default=3)
    a.add_argument('--min-associated-taxels',type=int,default=1)
    a.add_argument('--mask-radius-px',type=float,default=24.)
    a.add_argument('--grasp-start',type=int,default=None,help='OPTIONAL pilot annotation; paired with --grasp-end')
    a.add_argument('--grasp-end',type=int,default=None)
    a.add_argument('--release-gap-frames',type=int,default=18)
    a.add_argument('--max-unconfirmed-frames',type=int,default=75)
    a.add_argument('--history-frames',type=int,default=75)
    a.add_argument('--hand-roi-radius-m',type=float,default=0.18)
    a.add_argument('--history-voxel-mm',type=float,default=12.)
    a.add_argument('--min-history-observations',type=int,default=2)
    a.add_argument('--min-motion-mm',type=float,default=25.)
    a.add_argument('--min-motion-deg',type=float,default=6.)
    a.add_argument('--exclude-taxel-mm',type=float,default=7.)
    a.add_argument('--max-memory-points',type=int,default=800)
    a.add_argument('--flask-reference-end',type=int,default=120)
    a.add_argument('--observed-mm',type=float,default=15.,help='Display only: agreement with current candidate depth')
    a.add_argument('--flask-voxel-mm',type=float,default=10.)
    a.add_argument('--min-static-observations',type=int,default=3)
    a.add_argument('--max-flask-points',type=int,default=800)
    a.add_argument('--anchor-ttl-frames',type=int,default=45)
    a.add_argument('--anchor-voxel-mm',type=float,default=7.)
    a.add_argument('--max-anchor-points',type=int,default=80)
    a.add_argument('--display-points',type=int,default=900)
    a.add_argument('--scene-points',type=int,default=1500)
    args=a.parse_args()
    if args.output_root is None: args.output_root = HOI/f'outputs/level1_sequence_e{args.episode}'
    if args.input_dir is None: args.input_dir = args.output_root/'cloud_frames'
    if args.source_html is None: args.source_html = args.output_root/'sequence_3d.html'
    if not args.run_name.replace('_','').replace('-','').isalnum():a.error('bad run-name')
    if args.history_voxel_mm<=0 or args.flask_voxel_mm<=0 or args.anchor_voxel_mm<=0 or args.force_threshold<0:
        a.error('invalid threshold')
    run(args)

if __name__=='__main__':main()
