#!/usr/bin/env python3
"""Photo-to-metric-mesh, task-local human-gated workflow.

prepare/approve-masks/inspect/calibrate/approve/status: OpenPI Python env
segment: HOI SAM2 Python env, cwd hoi/repos/sam2
reconstruct: HOI SAM3D Python env, cwd hoi/repos/sam-3d-objects

The task output is output/preprocess/<dataset>/geometry; no LeRobot data edited.
"""
from __future__ import annotations

import argparse
import io
import zipfile
import base64
import hashlib
import html
import inspect as pyinspect
import json
import math
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA = 'task_photo_geometry.v1'
OBJECT_RE = re.compile(r'^[A-Za-z][A-Za-z0-9_\-]*$')
IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png')


def die(msg: str):
    raise ValueError(msg)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding='utf-8'))


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def manifest(root: Path):
    m = read_json(root / 'manifest.json')
    if m.get('schema') != SCHEMA or len(m.get('objects', [])) != 2:
        die(f'Unexpected geometry manifest: {root}')
    if root.resolve() != Path(m['output_root']).resolve():
        die('Task geometry directory moved; update/reprepare explicitly rather than silently breaking provenance')
    return m


def photo_path(root: Path, name: str):
    matches = [root / 'photos' / (name + ext) for ext in IMAGE_SUFFIXES]
    actual = [p for p in matches if p.is_file()]
    if len(actual) != 1:
        die(f'Expected one photo for {name} at photos/{name}.jpg/.jpeg/.png; found {actual}')
    return actual[0]


def verify_images(root: Path, m: dict):
    for name in m['objects']:
        item = m['photos'][name]
        if digest(Path(item['source'])) != item['source_sha256']:
            die(f'{name} source photo changed since prepare. Use a new output version to replace it.')
        if digest(root / 'reference' / name / 'image.jpg') != item['reference_sha256']:
            die(f'{name} prepared reference image changed')


def write_html_photo(image: Path, name: str, width: int, height: int, dest: Path):
    uri = 'data:image/jpeg;base64,' + base64.b64encode(image.read_bytes()).decode('ascii')
    # Whitelist object identifiers and escape variable interpolation as JSON.
    name_js = json.dumps(name)
    source = '''<!DOCTYPE html><html lang="zh"><meta charset="utf-8"><title>Photo Mask • __NAME__</title>
<style>body{font:15px sans-serif;margin:18px;background:#f6f7fb}main{display:flex;gap:20px;flex-wrap:wrap}
canvas{cursor:crosshair;max-width:90vw;height:auto;border:1px solid #999}button{padding:9px;margin:4px}
pre{max-width:320px;max-height:220px;overflow:auto}</style>
<h3>__NAME__: 点击物体内部（绿色正点）；误分割时可以加红色负点</h3>
<main><canvas id="cv" width="__W__" height="__H__"></canvas><aside>
<p><button onclick="positive=true;draw()">正点 +</button><button onclick="positive=false;draw()">负点 −</button></p>
<p><button onclick="undo()">撤销</button><button onclick="clearPoints()">清空</button></p>
<p><button onclick="download()">下载 prompts.json</button></p>
<p>上传到 geometry/reference/__NAME__/prompts.json（注意每个物体各有一个 JSON）。</p>
<pre id="list"></pre></aside></main>
<script>
const name=__NAMEJS__, W=__W__, H=__H__, cv=document.getElementById('cv'), ctx=cv.getContext('2d');
const points={positive:[],negative:[]}, pic=new Image();let positive=true;
pic.src='__IMAGE__';
function draw(){ctx.clearRect(0,0,W,H);ctx.drawImage(pic,0,0,W,H);
for(const key of ['positive','negative']) for(const [x,y] of points[key]){
ctx.beginPath();ctx.arc(x,y,7,0,7);ctx.fillStyle=key==='positive'?'#00ff44':'#ff3333';ctx.fill();ctx.lineWidth=2;ctx.strokeStyle='white';ctx.stroke();}
document.getElementById('list').textContent=JSON.stringify({selected:positive?'positive':'negative',...points},null,2);}
pic.onload=draw;
cv.onclick=e=>{const r=cv.getBoundingClientRect();
const x=Math.max(0,Math.min(W-1,Math.round((e.clientX-r.left)*W/r.width)));
const y=Math.max(0,Math.min(H-1,Math.round((e.clientY-r.top)*H/r.height)));
points[positive?'positive':'negative'].push([x,y]);draw();};
function undo(){const q=points[positive?'positive':'negative'];q.pop();draw()}
function clearPoints(){points.positive=[];points.negative=[];draw()}
function download(){if(!points.positive.length){alert('至少点击一个正点');return;}
const obj={schema:'__SCHEMA__',object:name,image_width:W,image_height:H,points};
const u=URL.createObjectURL(new Blob([JSON.stringify(obj,null,2)],{type:'application/json'}));
const a=document.createElement('a');a.href=u;a.download='prompts.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);}
</script></html>'''
    for key, val in {'__NAME__':html.escape(name),'__NAMEJS__':name_js,'__W__':str(width),'__H__':str(height),'__IMAGE__':uri,'__SCHEMA__':SCHEMA}.items():
        source = source.replace(key, val)
    dest.write_text(source, encoding='utf-8')


def prepare(a):
    from PIL import Image, ImageOps
    root = a.out.resolve()
    names = [x.strip() for x in a.objects.split(',')]
    if len(names) != 2 or len(set(names)) != 2 or not all(OBJECT_RE.fullmatch(n) for n in names):
        die('Exactly two distinct simple names required: --objects button,box')
    if not (512 <= a.max_side <= 4096):
        die('--max-side must be between 512 and 4096')
    if (root / 'manifest.json').exists() or (root / 'reference').exists():
        raise FileExistsError(f'Already prepared: {root}; do not silently overwrite photos/reviews')
    photos = {name: photo_path(root, name) for name in names}
    for name in names:
        if not photos[name].stat().st_size:
            die(f'Empty photo for {name}')
    records = {}
    for name in names:
        refdir = root / 'reference' / name
        refdir.mkdir(parents=True, exist_ok=False)
        with Image.open(photos[name]) as orig:
            frame = ImageOps.exif_transpose(orig).convert('RGB')
        if max(frame.size) > a.max_side:
            frame.thumbnail((a.max_side, a.max_side), Image.Resampling.LANCZOS)
        ref = refdir / 'image.jpg'
        frame.save(ref, quality=95, subsampling=0)
        write_html_photo(ref, name, frame.width, frame.height, refdir / 'click.html')
        records[name] = {'source':str(photos[name].resolve()), 'source_sha256':digest(photos[name]),
                         'reference_sha256':digest(ref), 'size_px':[frame.width, frame.height]}
        print(f'[PHOTO] {name}: {frame.width}x{frame.height}; label {refdir / "click.html"}')
    m = {'schema':SCHEMA, 'dataset_id':root.parent.name, 'output_root':str(root),
         'objects':names, 'photos':records, 'prepared_utc':timestamp(),
         'note':'one photo per object; scale/pose of generative mesh are NOT metric'}
    save_json(root / 'manifest.json', m)
    print('[NEXT] Upload two prompts.json files into reference/<object>/; run segment in HOI SAM2 env')


def get_prompts(root: Path, m: dict, name: str):
    rec = read_json(root / 'reference' / name / 'prompts.json')
    if rec.get('schema') != SCHEMA or rec.get('object') != name:
        die(f'{name} prompts schema/object mismatch')
    if [rec.get('image_width'), rec.get('image_height')] != m['photos'][name]['size_px']:
        die(f'{name} prompts dimensions mismatch')
    width, height = m['photos'][name]['size_px']
    pos = np.asarray(rec.get('points', {}).get('positive', []), dtype=np.float32).reshape(-1, 2)
    neg = np.asarray(rec.get('points', {}).get('negative', []), dtype=np.float32).reshape(-1, 2)
    if not len(pos):
        die(f'{name} needs positive click(s)')
    for pts in (pos, neg):
        if not np.isfinite(pts).all() or np.any(pts < 0) or np.any(pts[:, 0] >= width) or np.any(pts[:, 1] >= height):
            die(f'{name}: invalid coordinates')
    return pos, neg


def mask_path(root: Path, name: str):
    return root / 'reference' / name / 'mask.png'


def verify_masks(root: Path, m: dict, *, approved: bool):
    import cv2
    verify_images(root, m)
    for name in m['objects']:
        refdir = root / 'reference' / name
        record = read_json(refdir / 'mask_report.json')
        if record.get('image_sha256') != m['photos'][name]['reference_sha256']:
            die(f'{name}: mask generated from wrong photo')
        prompts = refdir / 'prompts.json'
        get_prompts(root, m, name)
        if record.get('prompts_sha256') != digest(prompts) or record.get('mask_sha256') != digest(mask_path(root, name)):
            die(f'{name}: prompts/mask changed since segmentation')
        seg = cv2.imread(str(mask_path(root, name)), cv2.IMREAD_GRAYSCALE)
        if seg is None or seg.shape[::-1] != tuple(m['photos'][name]['size_px']) or not np.any(seg > 127):
            die(f'{name}: empty or wrong-sized mask')
        if approved:
            rev = read_json(refdir / 'mask_review.json')
            if rev.get('status') != 'human_accepted' or rev.get('mask_report_sha256') != digest(refdir / 'mask_report.json'):
                die(f'{name}: mask approval absent/stale')


def segment(a):
    import cv2
    # Reuse the SAM2 image interface that already ran on this A800.
    from sam2_task import image_predictor, sam_predict
    root = a.out.resolve(); m = manifest(root); verify_images(root, m)
    if any((root / 'reference' / n / 'mask_review.json').exists() for n in m['objects']):
        raise FileExistsError('A reference mask is already approved; do not overwrite it')
    predictor = image_predictor(a.checkpoint.expanduser().resolve())
    for name in m['objects']:
        pos, neg = get_prompts(root, m, name)
        refdir = root / 'reference' / name
        frame = cv2.imread(str(refdir / 'image.jpg'), cv2.IMREAD_COLOR)
        seg, score = sam_predict(predictor, frame, pos, neg)
        if not cv2.imwrite(str(mask_path(root, name)), np.uint8(seg) * 255):
            raise IOError(mask_path(root, name))
        overlay = frame.copy(); overlay[seg.astype(bool)] = (0.6 * overlay[seg.astype(bool)] + 0.4 * np.array([0, 220, 25])).astype(np.uint8)
        cv2.imwrite(str(refdir / 'preview.jpg'), overlay)
        save_json(refdir / 'mask_report.json', {'schema':SCHEMA, 'object':name, 'sam2_score':float(score),
                  'mask_pixels':int(seg.sum()), 'image_sha256':m['photos'][name]['reference_sha256'],
                  'prompts_sha256':digest(refdir / 'prompts.json'),'mask_sha256':digest(mask_path(root, name))})
        print(f'[MASK CANDIDATE] {name}: {int(seg.sum())} px, score={score:.3f}; view {refdir / "preview.jpg"}')
    print('[NEXT] Check both photo masks, then approve-masks')


def approve_masks(a):
    root=a.out.resolve(); m=manifest(root)
    verify_masks(root, m, approved=False)
    for name in m['objects']:
        refdir=root/'reference'/name
        if (refdir/'mask_review.json').exists():
            raise FileExistsError(f'{name} mask already approved')
    for name in m['objects']:
        refdir=root/'reference'/name
        save_json(refdir/'mask_review.json',{'schema':SCHEMA,'object':name,'status':'human_accepted',
                                           'mask_report_sha256':digest(refdir/'mask_report.json'),
                                           'reviewer':a.reviewer,'utc':timestamp()})
    print('[APPROVED] both photo masks; ready to reconstruct')


def _sam3d_local_dino():
    """Reuse DINOv2 cache workaround from the previously working e23 runner."""
    import torch
    repo = Path(os.environ.get('SAM3D_DINO_CACHE', '/root/.cache/torch/hub/facebookresearch_dinov2_main'))
    weights = Path(os.environ.get('SAM3D_DINO_WEIGHTS', '/root/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth'))
    if not (repo/'hubconf.py').is_file() or not weights.is_file():
        print('[DINO] No complete local cache; using installed TorchHub behavior')
        return
    original = torch.hub.load
    def cached(repo_or_dir, model, *args, **kwargs):
        if repo_or_dir == 'facebookresearch/dinov2':
            repo_or_dir = str(repo); kwargs['source'] = 'local'
            print('[DINO] loading cached local DINOv2 source and weights')
        return original(repo_or_dir, model, *args, **kwargs)
    torch.hub.load = cached


def _configure_sam3d_runtime(pipeline_config: Path) -> Path:
    """Make the existing HOI SAM 3D checkout importable before Hydra instantiation.

    The model lives in a source checkout, not necessarily site-packages. Its
    package __init__ needs LIDRA_SKIP_INIT on this cloud installation. These
    defaults reproduce the previously validated A800/e23 environment without
    requiring a newcomer to remember extra shell exports.
    """
    # The standard config is <sam3d_source>/checkpoints/hf/pipeline.yaml.
    # Custom checkouts can override this explicitly.
    configured = os.environ.get('SAM3D_SOURCE_ROOT')
    source = (Path(configured).expanduser().resolve() if configured
              else pipeline_config.parent.parent.parent.resolve())
    if not (source / 'sam3d_objects' / '__init__.py').is_file():
        raise FileNotFoundError(
            f'SAM 3D source package not found at {source / "sam3d_objects"}; '
            'set SAM3D_SOURCE_ROOT to your sam-3d-objects checkout'
        )

    os.environ.setdefault('LIDRA_SKIP_INIT', 'true')
    os.environ.setdefault('ATTN_BACKEND', 'sdpa')
    os.environ.setdefault('SPARSE_ATTN_BACKEND', 'sdpa')

    # sys.path takes effect in THIS process, including Hydra's importlib calls.
    # PYTHONPATH additionally covers child processes spawned by dependencies.
    source_string = str(source)
    if source_string not in sys.path:
        sys.path.insert(0, source_string)
    old_pythonpath = os.environ.get('PYTHONPATH', '')
    parts = [p for p in old_pythonpath.split(os.pathsep) if p]
    if source_string not in parts:
        os.environ['PYTHONPATH'] = os.pathsep.join([source_string, *parts])

    print(f'[SAM3D ENV] source={source_string}, '
          f'LIDRA_SKIP_INIT={os.environ["LIDRA_SKIP_INIT"]}, '
          f'ATTN_BACKEND={os.environ["ATTN_BACKEND"]}, '
          f'SPARSE_ATTN_BACKEND={os.environ["SPARSE_ATTN_BACKEND"]}', flush=True)
    return source


def reconstruct(a):
    root=a.out.resolve(); m=manifest(root); verify_masks(root, m, approved=True)
    selected = m['objects'] if a.object is None else [a.object]
    if any(n not in m['objects'] for n in selected):
        die('--object must be one of the task objects')
    pending=[]
    for name in selected:
        dest=root/'reconstruction'/name
        raw=dest/'raw.glb'; report=dest/'report.json'
        if raw.is_file() and report.is_file():
            rec=read_json(report)
            if rec.get('raw_sha256')==digest(raw) and rec.get('mask_sha256')==digest(mask_path(root,name)):
                print(f'[SKIP] verified existing raw mesh for {name}')
                continue
            die(f'{name}: existing reconstruction modified; refuse overwrite')
        if dest.exists() and list(dest.iterdir()):
            raise FileExistsError(f'{name}: partial reconstruction at {dest}; inspect before retry')
        pending.append(name)
    if not pending:
        return
    config_path=a.pipeline.expanduser().resolve(); moge=a.moge.expanduser().resolve()
    if not config_path.is_file() or not moge.is_file():
        raise FileNotFoundError(f'SAM3D/MoGe missing: {config_path} / {moge}')
    _configure_sam3d_runtime(config_path)
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    import torch
    from PIL import Image
    if not torch.cuda.is_available():
        die('SAM 3D reconstruction needs CUDA; run in HOI sam3d env on A800')
    _sam3d_local_dino()
    config=OmegaConf.load(config_path)
    config.workspace_dir=str(config_path.parent)
    config.rendering_engine='pytorch3d'
    config.compile_model=False
    config.depth_model.model.pretrained_model_name_or_path=str(moge)
    print('[SAM3D] loading existing SAM 3D Objects + local MoGe',flush=True)
    pipeline=instantiate(config)
    for name in pending:
        ref=root/'reference'/name
        rgb=np.asarray(Image.open(ref/'image.jpg').convert('RGB')).copy()
        mask=np.asarray(Image.open(ref/'mask.png').convert('L')).copy()
        mask=np.uint8(mask>127)*255
        if rgb.shape[:2]!=mask.shape or not np.any(mask):
            die(f'{name}: invalid photo/mask dimensions')
        out=root/'reconstruction'/name
        out.mkdir(parents=True,exist_ok=True)
        print(f'[SAM3D] {name}, image={rgb.shape}, mask_pixels={int(np.count_nonzero(mask))}',flush=True)
        # Only pass kwargs supported by the pinned installed pipeline version.
        options={'seed':int(a.seed),'stage1_only':False,'with_mesh_postprocess':False,
                 'with_texture_baking':False,'with_layout_postprocess':False,'use_vertex_color':True}
        signature=pyinspect.signature(pipeline.run)
        kwargs={k:v for k,v in options.items() if k in signature.parameters}
        try:
            result=pipeline.run(rgb,mask=mask,**kwargs)
            glb=result.get('glb')
            if glb is None:
                die(f'{name}: SAM3D did not return GLB; keys={list(result.keys())}')
            tmp=out/'raw.partial.glb'
            glb.export(str(tmp))
            if not tmp.is_file() or tmp.stat().st_size<100:
                die(f'{name}: empty GLB exported')
            tmp.replace(out/'raw.glb')
            save_json(out/'report.json',{'schema':SCHEMA,'object':name,'status':'raw_unreviewed',
                      'photo_sha256':m['photos'][name]['reference_sha256'],
                      'mask_sha256':digest(mask_path(root,name)),
                      'mask_review_sha256':digest(ref/'mask_review.json'),
                      'raw_sha256':digest(out/'raw.glb'),'pipeline_config':str(config_path),
                      'pipeline_config_sha256':digest(config_path),'moge_path':str(moge),
                      'sam3d_options':kwargs,'warning':'RAW MESH IS NOT METRIC; pose/axes arbitrary','utc':timestamp()})
            print(f'[RAW MESH] {name}: {out / "raw.glb"}',flush=True)
        except Exception:
            print(f'[FAILED] {name}: inspect partial directory {out}; not overwritten automatically',flush=True)
            raise
    print('[NEXT] OpenPI env: inspect --out <geometry_root>')


def scene_load(path: Path):
    import trimesh
    scene=trimesh.load(str(path), force='scene', process=False)
    if not isinstance(scene,trimesh.Scene) or not scene.geometry:
        die(f'GLB contains no geometry: {path}')
    bounds=np.asarray(scene.bounds,dtype=float)
    if bounds.shape!=(2,3) or not np.isfinite(bounds).all() or np.any(bounds[1]-bounds[0] <= 1e-10):
        die(f'Invalid mesh bounds: {path}: {bounds}')
    return scene


def rotation_xyz(angles):
    from scipy.spatial.transform import Rotation
    vals=np.asarray(angles,dtype=float)
    if vals.shape!=(3,) or not np.isfinite(vals).all() or np.max(np.abs(vals))>3600:
        die('rotation_xyz_deg must be three finite values in degrees')
    mat=np.eye(4)
    mat[:3,:3]=Rotation.from_euler('xyz', vals, degrees=True).as_matrix()
    return mat


def viewer(scene, path: Path, *, title: str):
    """Self-contained geometry-only HTML, with axes and size. No network dependency."""
    import plotly.graph_objects as go
    import plotly.offline as po
    import trimesh
    meshes=[g for g in scene.dump() if isinstance(g,trimesh.Trimesh) and len(g.faces)]
    if not meshes:
        die('Scene has no triangular meshes for preview')
    bounds=np.asarray(scene.bounds,dtype=float)
    extent=bounds[1]-bounds[0]; axis_length=float(np.max(extent))*0.65
    traces=[]
    budget=100000
    for mesh in meshes:
        if len(mesh.faces)>budget:
            # Preview only: sample faces deterministically; approved source mesh unchanged.
            take=np.linspace(0,len(mesh.faces)-1,budget,dtype=int)
            f=mesh.faces[take]
        else:
            f=mesh.faces
        v=np.asarray(mesh.vertices)
        traces.append(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],
                         color='#79a9d0',opacity=1,flatshading=False,lighting=dict(ambient=.55,diffuse=.65),name='mesh',showlegend=False))
    for idx,(axis,color) in enumerate([('X','#df4242'),('Y','#329a40'),('Z','#396ce6')]):
        coords=np.zeros((2,3));coords[1,idx]=axis_length
        traces.append(go.Scatter3d(x=coords[:,0],y=coords[:,1],z=coords[:,2],
                      mode='lines+text',text=['',axis],line=dict(color=color,width=7),name=axis))
    fig=go.Figure(traces)
    fig.update_layout(title=f'{title}  |  bbox = {extent[0]:.4g} × {extent[1]:.4g} × {extent[2]:.4g}',
               scene=dict(aspectmode='data',xaxis_title='X',yaxis_title='Y',zaxis_title='Z'),
               margin=dict(l=0,r=0,t=55,b=0))
    path.parent.mkdir(parents=True, exist_ok=True)
    po.plot(fig,filename=str(path),auto_open=False,include_plotlyjs=True)



def write_metric_editor(root: Path, m: dict) -> Path:
    """Offline all-objects GLB calibration UI: rotate, measure, preview, ZIP export.

    The HTML embeds the raw GLBs; it needs no server or network connection.
    The server still independently applies calibration and writes approved GLBs.
    """
    import plotly.graph_objects as go
    import plotly.offline as po
    import trimesh

    traces = []
    indices = {}
    encoded_glb = {}
    raw_hashes = {}
    vertex_buffers = {}
    max_extent = 0.0
    for object_index, name in enumerate(m['objects']):
        raw_path = root / 'reconstruction' / name / 'raw.glb'
        verify_raw(root, m, name)
        scene = scene_load(raw_path)
        max_extent = max(max_extent, float(np.max(np.asarray(scene.extents))))
        encoded_glb[name] = base64.b64encode(raw_path.read_bytes()).decode('ascii')
        raw_hashes[name] = digest(raw_path)
        indices[name] = []
        vertex_buffers[name] = []
        for mesh in scene.dump():
            if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
                continue
            vertices = np.asarray(mesh.vertices)
            vertex_buffers[name].append(base64.b64encode(np.asarray(vertices,dtype="<f4").tobytes()).decode("ascii"))
            faces = np.asarray(mesh.faces)
            if len(faces) > 100000:
                faces = faces[np.linspace(0, len(faces)-1, 100000, dtype=int)]
            indices[name].append(len(traces))
            traces.append(go.Mesh3d(x=vertices[:,0],y=vertices[:,1],z=vertices[:,2],
                                    i=faces[:,0],j=faces[:,1],k=faces[:,2],
                                    color='#78a9d0',opacity=1,flatshading=False,
                                    lighting=dict(ambient=.55,diffuse=.65),
                                    visible=(object_index==0),name=name,showlegend=False))
        if not indices[name]:
            die(f'No triangles in {name} for calibration editor')
    axis_indices = []
    for j, (label, color) in enumerate((('X','#d64545'),('Y','#3eaa51'),('Z','#396ce6'))):
        points=np.zeros((2,3));points[1,j]=max_extent*0.6
        axis_indices.append(len(traces))
        traces.append(go.Scatter3d(x=points[:,0],y=points[:,1],z=points[:,2],
                       mode='lines+text',text=['',label],line=dict(color=color,width=7),
                       name=label,showlegend=False))
    fig=go.Figure(traces)
    fig.update_layout(title='Photo geometry · interactive metric calibration',
                      scene=dict(aspectmode='data',xaxis_title='X',yaxis_title='Y',zaxis_title='Z'),
                      margin=dict(l=0,r=0,t=65,b=0),height=720,
                      paper_bgcolor='#f6f8fc')
    graph_div=po.plot(fig,output_type='div',auto_open=False,include_plotlyjs=True)
    # Only safe identifiers and generated SHA/base64 strings appear in JSON.
    data={'schema':SCHEMA,'objects':m['objects'],'indices':indices,
          'axis_indices':axis_indices,'raw':encoded_glb,'hashes':raw_hashes,'vertices':vertex_buffers}
    encoded=json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('<','\u003c')
    controls="""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8" />
<title>3D Mesh 尺度与方向校准</title><style>
*{box-sizing:border-box}body{margin:0;background:#f6f8fc;color:#20293b;font:14px system-ui,-apple-system,Arial,sans-serif}
header{padding:20px 30px;background:#172b4d;color:white}header h2{margin:0 0 6px;font-size:21px}header small{color:#bed3f2}
.layout{display:grid;grid-template-columns:minmax(350px,430px) minmax(500px,1fr);gap:16px;padding:18px}
.box{background:#fff;border-radius:13px;box-shadow:0 2px 12px #22334418;padding:20px}
label{font-weight:600;display:block;margin:13px 0 5px}input[type=number],select{width:100%;padding:9px;border:1px solid #ccd4e0;border-radius:7px;background:white}
.axisrow{display:grid;grid-template-columns:23px 85px minmax(80px,1fr) 38px 38px;align-items:center;gap:7px}
.axisrow input[type=range]{width:100%}.axisrow button{padding:7px 3px}
.dimrow{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.dimrow label{margin:5px 0}
button{cursor:pointer;border:1px solid #c4d1e0;background:#eff4fa;border-radius:7px;padding:8px}
button.primary{background:#2367d1;color:white;padding:14px;border:0;font-weight:700;width:100%;font-size:15px;margin-top:18px}
button:disabled{opacity:.5;cursor:wait}.hint{color:#576479;font-size:12px;line-height:1.6}.indicator{background:#edf5ff;border-left:3px solid #2878de;padding:12px;white-space:pre-line;margin-top:16px;border-radius:6px}
#export-status{color:#235a3f;font-weight:600;margin-top:10px;line-height:1.5}.chart{min-height:720px}
@media(max-width:900px){.layout{grid-template-columns:1fr}.chart{min-height:550px}} </style></head><body>
<header><h2>3D 模型校准</h2><small>旋转坐标轴 · 输入实际长度 · 实时预览 · 一键导出米制 GLB</small></header>
<div class="layout"><aside class="box" id="geometry-editor-controls">
<label>当前物体</label><select id="object-select"></select><p class="hint">每个物体独立保存设置；页面支持离线打开。</p>
<h3 id="active-label"></h3>
<label>旋转模型（度，右侧图可拖动视角）</label>
<div class="axisrow"><b style="color:#d64545">X</b><input id="rx" type="number" step="5" value="0"><input id="rxslide" type="range" min="-180" max="180" step="5" value="0"><button id="minusx">−90°</button><button id="plusx">+90°</button></div>
<div class="axisrow"><b style="color:#3eaa51">Y</b><input id="ry" type="number" step="5" value="0"><input id="ryslide" type="range" min="-180" max="180" step="5" value="0"><button id="minusy">−90°</button><button id="plusy">+90°</button></div>
<div class="axisrow"><b style="color:#396ce6">Z</b><input id="rz" type="number" step="5" value="0"><input id="rzslide" type="range" min="-180" max="180" step="5" value="0"><button id="minusz">−90°</button><button id="plusz">+90°</button></div>
<p class="hint">红 X、绿 Y、蓝 Z 是固定坐标轴。调好物体方向，约定 Z 向上，必要时用 ±90° 快捷按钮。</p>
<label>真实尺寸（厘米，至少填写一个方向）</label>
<div class="dimrow"><div><label>X 长度</label><input id="lenx" type="number" step="any" min="0" placeholder="可留空"></div>
<div><label>Y 长度</label><input id="leny" type="number" step="any" min="0" placeholder="可留空"></div>
<div><label>Z 长度</label><input id="lenz" type="number" step="any" min="0" placeholder="可留空"></div></div>
<label>以哪个实测长度确定整体尺度？</label><select id="anchor"><option value="x">X</option><option value="y">Y</option><option value="z">Z</option></select>
<label>坐标原点</label><select id="origin"><option value="bottom_center">底面中心（默认）</option><option value="bbox_center">包围盒中心</option><option value="mesh_origin">保留原始原点</option></select>
<div id="dimensions" class="indicator"></div>
<button class="primary" id="export">确认两个物体并导出模型 ZIP</button>
<div id="export-status"></div><p class="hint">ZIP 内含两个米制 GLB 和 calibration.json。下载后上传到云端 geometry/ 目录，运行一条 finalize 命令即可自动验证、写入 approved/。不需手改 JSON。</p>
</aside><main class="box chart">__GRAPH__</main></div>
<script type="application/json" id="geometry-editor-data">__DATA__</script>
<script>__UI_JS__</script></body></html>"""
    js_path=Path(__file__).with_name('geometry_editor.js')
    if not js_path.is_file():
        raise FileNotFoundError(f'Missing companion offline UI source: {js_path}')
    output=root/'preview'/'metric_editor.html'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(controls.replace('__GRAPH__',graph_div).replace('__DATA__',encoded)
                      .replace('__UI_JS__',js_path.read_text(encoding='utf8')),encoding='utf8')
    print('[INTERACTIVE EDITOR]',output)
    return output


def finalize_ui(a):
    """Import one HTML-exported ZIP, reproduce metric meshes on cloud and approve.

    The browser GLBs are independently checked but never blindly installed:
    approved/ is always produced from the verified SAM3D raw meshes on cloud.
    """
    import trimesh
    root=a.out.resolve();m=manifest(root);verify_masks(root,m,approved=True)
    path=(a.bundle or root/'geometry_metric_export.zip').expanduser().resolve()
    if any((root/'approved'/f'{n}.glb').exists() for n in m['objects']):
        raise FileExistsError('Already approved; refusing to replace existing metric mesh')
    if not path.is_file(): raise FileNotFoundError(f'Upload the browser export first: {path}')
    if path.stat().st_size > 1024*1024*800: die('Unreasonably large calibration bundle')
    with zipfile.ZipFile(path) as z:
        expected={'calibration.json',*(f'meshes/{n}.glb' for n in m['objects'])}
        if set(z.namelist())!=expected:die(f'Unexpected ZIP entries, expected exactly {sorted(expected)}')
        if any(z.getinfo(file).file_size > 1024*1024*400 for file in expected):
            die('Unreasonably large calibration bundle entry')
        calibration=json.loads(z.read('calibration.json'))
        provenance=calibration.get('editor_provenance',{})
        if provenance.get('ui_version')!=1 or provenance.get('review')!='calibrated_in_3d_editor':
            die('Missing browser approval/provenance')
        if calibration.get('schema')!=SCHEMA or calibration.get('unit')!='meter' or set(calibration.get('objects',{}))!=set(m['objects']):
            die('Wrong calibration schema/objects/units')
        for name in m['objects']:
            verify_raw(root,m,name)
            if provenance.get('raw_mesh_sha256',{}).get(name)!=digest(root/'reconstruction'/name/'raw.glb'):
                die(f'{name}: cloud RAW mesh differs from the HTML editor source')
        # Keep all bytes available while we deterministically regenerate the models.
        client_meshes={name:z.read(f'meshes/{name}.glb') for name in m['objects']}
    save_json(root/'calibration.json',calibration)
    calibrate(argparse.Namespace(out=root,max_shape_error=a.max_shape_error,allow_shape_mismatch=False))
    for name in m['objects']:
        client=trimesh.load(io.BytesIO(client_meshes[name]),file_type='glb',force='scene',process=False)
        expected_mesh=scene_load(root/'calibrated'/f'{name}.glb')
        if client.is_empty or not np.allclose(np.asarray(client.bounds),np.asarray(expected_mesh.bounds),rtol=2e-3,atol=1e-5):
            die(f'{name}: exported mesh world-space bounds do not match server calibration; NOT approved')
    approve(argparse.Namespace(out=root,reviewer=a.reviewer))
    print('[FINALIZED FROM UI]',root/'approved'/'manifest.json')

def verify_raw(root:Path,m:dict,name:str):
    rec=read_json(root/'reconstruction'/name/'report.json')
    if rec.get('schema')!=SCHEMA or rec.get('object')!=name or rec.get('raw_sha256')!=digest(root/'reconstruction'/name/'raw.glb'):
        die(f'{name}: raw reconstruction file/report mismatch')
    if rec.get('photo_sha256')!=m['photos'][name]['reference_sha256'] or rec.get('mask_sha256')!=digest(mask_path(root,name)):
        die(f'{name}: raw reconstruction no longer matches its mask/photo')
    if rec.get('mask_review_sha256')!=digest(root/'reference'/name/'mask_review.json'):
        die(f'{name}: mask review changed')
    return rec


def inspect_meshes(a):
    root=a.out.resolve();m=manifest(root);verify_masks(root,m,approved=True)
    for name in m['objects']:
        verify_raw(root,m,name)
        source=root/'reconstruction'/name/'raw.glb'
        scene=scene_load(source);bounds=np.asarray(scene.bounds)
        viewer(scene,root/'preview'/f'{name}_raw.html',title=f'{name}: UNCALIBRATED (arbitrary units)')
        print(f'[RAW] {name}: extents={np.round(bounds[1]-bounds[0],5).tolist()}, view {root / "preview" / (name+"_raw.html")}')
    config=root/'calibration.json'
    if not config.exists():
        spec={'schema':SCHEMA,'unit':'meter','objects':{name:{'rotation_xyz_deg':[0,0,0],
                   'scale_anchor_axis':'x','measurements_m':{'x':None,'y':None,'z':None},
                   'origin':'bottom_center','notes':'Fill measured x/y/z AFTER rotation; at least anchor axis. Z should point up.'}
                    for name in m['objects']}}
        save_json(config,spec)
        print('[CALIBRATION TEMPLATE]',config)
    else:
        print('[KEEP] existing calibration.json not overwritten')
    write_metric_editor(root,m)
    print('[NEXT] Open preview/metric_editor.html: rotate XYZ, enter measured lengths (cm), export ZIP; then finalize')


def calibrate(a):
    root=a.out.resolve();m=manifest(root);verify_masks(root,m,approved=True)
    spec_path=root/'calibration.json';spec=read_json(spec_path)
    if spec.get('schema')!=SCHEMA or spec.get('unit')!='meter' or set(spec.get('objects',{}))!=set(m['objects']):
        die('Invalid calibration.json schema/unit/object names')
    from trimesh.transformations import translation_matrix
    for name in m['objects']:
        if (root/'approved'/f'{name}.glb').exists():
            raise FileExistsError(f'{name} already approved; do not overwrite')
        verify_raw(root,m,name)
        cfg=spec['objects'][name]
        axis=cfg.get('scale_anchor_axis')
        if axis not in ('x','y','z'):
            die(f'{name}: invalid scale_anchor_axis')
        measurements=cfg.get('measurements_m',{})
        measured={}
        for k,v in measurements.items():
            if k not in ('x','y','z'):
                die(f'{name}: unknown measured dimension {k}')
            if v is not None:
                if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0:
                    die(f'{name}: {k} measurement must be positive meters')
                measured[k]=float(v)
        if axis not in measured:
            die(f'{name}: fill measurements_m.{axis} in METERS (not cm)')
        origin=cfg.get('origin','bottom_center')
        if origin not in ('bottom_center','bbox_center','mesh_origin'):
            die(f'{name}: invalid origin')
        raw=root/'reconstruction'/name/'raw.glb'
        scene=scene_load(raw)
        rot=rotation_xyz(cfg.get('rotation_xyz_deg',[0,0,0]))
        scene.apply_transform(rot)
        b=np.asarray(scene.bounds,dtype=float)
        raw_size=b[1]-b[0]
        scale=measured[axis]/raw_size['xyz'.index(axis)]
        if not 1e-12<scale<1e12:
            die(f'{name}: unreasonable scale {scale}')
        sm=np.eye(4);sm[:3,:3]*=scale
        scene.apply_transform(sm)
        bbox=np.asarray(scene.bounds,dtype=float)
        if origin=='bottom_center':
            offset=-np.array([(bbox[0,0]+bbox[1,0])/2,(bbox[0,1]+bbox[1,1])/2,bbox[0,2]])
        elif origin=='bbox_center':
            offset=-bbox.mean(axis=0)
        else:
            offset=np.zeros(3)
        scene.apply_transform(translation_matrix(offset))
        dims=np.asarray(scene.bounds)[1]-np.asarray(scene.bounds)[0]
        check={k:{'measured_m':v,'mesh_m':float(dims['xyz'.index(k)]),
                  'relative_error':float(abs(dims['xyz'.index(k)]-v)/v)} for k,v in measured.items()}
        max_error=max([r['relative_error'] for r in check.values()],default=0)
        if max_error>a.max_shape_error and not a.allow_shape_mismatch:
            die(f'{name}: shape inconsistent with measured dimensions (max error {max_error:.1%}); inspect/reconstruct, or use --allow-shape-mismatch explicitly')
        candidate=root/'calibrated'/f'{name}.glb'
        candidate.parent.mkdir(parents=True, exist_ok=True)
        tmp=candidate.with_name(name+'.partial.glb')
        scene.export(str(tmp),file_type='glb')
        tmp.replace(candidate)
        out=root/'calibrated'/f'{name}_report.json'
        save_json(out,{'schema':SCHEMA,'object':name,'status':'calibrated_unreviewed','source_raw_sha256':digest(raw),
                     'calibration_json_sha256':digest(spec_path),'calibrated_sha256':digest(candidate),
                     'scale_m_per_raw_unit':float(scale),'rotation_xyz_deg':list(map(float,cfg.get('rotation_xyz_deg',[0,0,0]))),
                     'origin':origin,'units':'meter','bbox_extents_m':dims.tolist(),'measured_vs_mesh':check,
                     'shape_mismatch_override':bool(max_error>a.max_shape_error and a.allow_shape_mismatch),
                     'warning':'Single-view hidden geometry may be wrong; visual inspection required'})
        viewer(scene,root/'preview'/f'{name}_metric.html',title=f'{name}: METRIC CANDIDATE (m)')
        print(f'[METRIC CANDIDATE] {name}: dims_m={np.round(dims,5).tolist()}, scale={scale:.6g}, error_max={max_error:.1%}')
    print('[NEXT] Review both *_metric.html against real objects; then approve')


def approve(a):
    root=a.out.resolve();m=manifest(root);verify_masks(root,m,approved=True)
    spec=root/'calibration.json'
    for name in m['objects']:
        verify_raw(root,m,name)
        report=read_json(root/'calibrated'/f'{name}_report.json')
        source=root/'calibrated'/f'{name}.glb'
        if report.get('schema')!=SCHEMA or report.get('status')!='calibrated_unreviewed':
            die(f'{name}: missing calibration report')
        if report.get('source_raw_sha256')!=digest(root/'reconstruction'/name/'raw.glb') or report.get('calibration_json_sha256')!=digest(spec) or report.get('calibrated_sha256')!=digest(source):
            die(f'{name}: source or calibration changed after candidate creation')
        scene_load(source)
        if (root/'approved'/f'{name}.glb').exists():
            raise FileExistsError(f'{name}: approved mesh exists; do not overwrite')
    approved=root/'approved';approved.mkdir(parents=True,exist_ok=True)
    assets={}
    for i,name in enumerate(m['objects']):
        source=root/'calibrated'/f'{name}.glb';dest=approved/f'{name}.glb'
        shutil.copyfile(source,dest)
        report=read_json(root/'calibrated'/f'{name}_report.json')
        assets[name]={'mesh':str(dest.resolve()),'version':f'{m["dataset_id"]}_photo_geometry_v1',
                      'scale_m_per_unit':1.0,'sha256':digest(dest),'sampling_seed':i+1,
                      'bbox_extents_m':report['bbox_extents_m']}
    final={'schema':SCHEMA,'dataset_id':m['dataset_id'],'status':'human_accepted',
           'objects':list(m['objects']),'assets':assets,
           'reconstruction_report_hashes':{n:digest(root/'reconstruction'/n/'report.json') for n in m['objects']},
           'calibration_report_hashes':{n:digest(root/'calibrated'/f'{n}_report.json') for n in m['objects']},
           'calibration_sha256':digest(spec),'reviewer':a.reviewer,'utc':timestamp(),
           'important':'Mesh frame is object-local with Z-up by explicit human calibration; NOT robot/camera pose.'}
    save_json(approved/'manifest.json',final)
    save_json(approved/'assets_config_snippet.json',{'assets':{n:{k:v for k,v in a.items() if k not in ('bbox_extents_m',)} for n,a in assets.items()}})
    print('[APPROVED] two metric meshes:',*(str(approved/f'{n}.glb') for n in m['objects']))
    print('[HANDOFF] ',approved/'assets_config_snippet.json')


def status(a):
    root=a.out.resolve();m=manifest(root)
    print('[TASK]',m['dataset_id'],m['objects'])
    for name in m['objects']:
        def exists(rel):return (root/rel).is_file()
        print(f'[{name}] photo=YES mask={"APPROVED" if exists(f"reference/{name}/mask_review.json") else "PENDING"} '
              f'raw={"YES" if exists(f"reconstruction/{name}/raw.glb") else "NO"} '
              f'metric={"YES" if exists(f"calibrated/{name}.glb") else "NO"} '
              f'approved={"YES" if exists(f"approved/{name}.glb") else "NO"}')
    if (root/'approved'/'manifest.json').is_file():
        approval=read_json(root/'approved'/'manifest.json')
        intact=all(digest(Path(approval['assets'][n]['mesh']))==approval['assets'][n]['sha256'] for n in m['objects'])
        print('[FINAL]',approval['status'],'hashes_ok=',intact)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    for cmd in ('prepare','segment','approve-masks','reconstruct','inspect','calibrate','approve','finalize','status'):
        s=sub.add_parser(cmd)
        s.add_argument('--out',type=Path,required=True,help='output/preprocess/<dataset>/geometry')
        if cmd=='prepare':
            s.add_argument('--objects',required=True,help='two names, e.g. button,box')
            s.add_argument('--max-side',type=int,default=1280)
        if cmd=='segment':
            s.add_argument('--checkpoint',type=Path,default=Path('/workspace/mnt/sqzhang26/hoi/weights/sam2/sam2.1_hiera_small.pt'))
        if cmd=='reconstruct':
            s.add_argument('--object',default=None,help='optional single object')
            s.add_argument('--pipeline',type=Path,default=Path('/workspace/mnt/sqzhang26/hoi/repos/sam-3d-objects/checkpoints/hf/pipeline.yaml'))
            s.add_argument('--moge',type=Path,default=Path('/workspace/mnt/sqzhang26/hoi/weights/moge-vitl/model.pt'))
            s.add_argument('--seed',type=int,default=42)
        if cmd=='finalize':
            s.add_argument('--bundle',type=Path,default=None,help='browser-exported ZIP (default: <out>/geometry_metric_export.zip)')
            s.add_argument('--max-shape-error',type=float,default=.25)
            s.add_argument('--reviewer',default=os.environ.get('USER','unknown'))
        if cmd=='calibrate':
            s.add_argument('--max-shape-error',type=float,default=.25)
            s.add_argument('--allow-shape-mismatch',action='store_true')
        if cmd in ('approve-masks','approve'):
            s.add_argument('--reviewer',default=os.environ.get('USER','unknown'))
    a=p.parse_args()
    {'prepare':prepare,'segment':segment,'approve-masks':approve_masks,'reconstruct':reconstruct,
     'inspect':inspect_meshes,'calibrate':calibrate,'approve':approve,'finalize':finalize_ui,'status':status}[a.command](a)


if __name__=='__main__':
    main()
