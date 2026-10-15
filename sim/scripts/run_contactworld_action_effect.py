#!/usr/bin/env python3
"""Audited ContactWorld action effects from matched bilateral tactile states.

Stages: prepare (including the old 520-rollout symptom audit), branch (Isaac
Gym), analyze, expand, probe. The physical intervention is short and is never
presented as a complete insertion success test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import run_contactworld_tactile_action_branches_v3 as replay
from run_contactworld_k4_down_probe import frozen_model, k4_label, parent_model

REPO = Path(__file__).resolve().parents[2]
OLD = Path('/home/sai/zx/openpi-sim-runtime/runs/contactworld_k4_down_probe_20261009')
EVAL = Path('/home/sai/zx/openpi-sim-runtime/runs/contactworld_pi0_v3_13models_40trials_20261003')
OUTPUT = Path('/home/sai/zx/openpi-sim-runtime/runs/contactworld_action_effect_20261009')
VERSION = 'cw-action-effects-v1'
ACTIVE_TILT_DEG = 2.0
BRANCHES = (
    'hold', 'down', 'retreat_down', 'x_plus_down', 'x_minus_down',
    'y_plus_down', 'y_minus_down', 'pitch_plus_down', 'pitch_minus_down',
)
QUOTA = {3: 15, 0: 5, 1: 5, 2: 5}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    replay.write_json(Path(path), obj)


def failure_audit(args):
    """Describe observed rollout symptoms; do not assign physical causes."""
    path = args.output / 'failure_audit.json'
    if path.exists():
        return read(path)
    entries = sorted(args.eval_root.glob('pi0_cw3_usb_*/summary.json'))
    if args.smoke:
        entries = entries[:1]
    if len(entries) != (1 if args.smoke else 13):
        raise ValueError(f'Expected 13 completed model summaries, got {len(entries)}')
    rows = []
    parent_centers, scale, _ = parent_model()
    for i, summary_file in enumerate(entries, 1):
        model = summary_file.parent.name
        summary = read(summary_file)
        for outcome in summary['results'][:1 if args.smoke else None]:
            seed = int(outcome['seed'])
            folder = summary_file.parent / f'seed_{seed}'
            trajectory = read(folder / 'trajectory.json')['rows']
            force = np.asarray([r['plug_socket_contact_force_xyz'] for r in trajectory])
            hits = np.flatnonzero(np.linalg.norm(force, axis=1) > .1)
            first = int(hits[0]) if len(hits) else None
            ever = bool(outcome['source_ever'])
            final = bool(outcome['source_final'])
            event = ('final_success' if final else 'transient_success' if ever else
                     'contact_no_success' if first is not None else 'no_detected_contact')
            best_start = first if first is not None else 0
            best = min(range(best_start, len(trajectory)),
                       key=lambda j: trajectory[j]['keypoint_error_mm'])
            at = trajectory[best]
            if first is not None:
                a, b = trajectory[first], trajectory[-1]
                inv = Rotation.from_quat(a['socket_quat']).inv()
                ee_down = -float(inv.apply(np.asarray(b['ee_pos']) - a['ee_pos'])[2]) * 1000
                plug_down = -float(inv.apply(np.asarray(b['plug_pos']) - a['plug_pos'])[2]) * 1000
            else:
                ee_down = plug_down = None
            # A query at simulator step t is pre-action. Physical contact at
            # trajectory row t is post-action, so only query steps > first+1
            # count as post-contact observations here.
            with np.load(folder / 'inputs.npz') as obs:
                steps = obs['steps']
                maps = obs['force_base']
                labels = [k4_label(f, parent_centers, scale)[0] for f in maps]
            after = ([labels[j] for j, t in enumerate(steps) if t > first + 1]
                     if first is not None else [])
            rows.append(dict(model=model, seed=seed, event=event, ever=ever, final=final,
                first_contact_step=first, first_success_step=outcome['first_success_step'],
                best_post_contact_step=best,
                best_keypoint_error_mm=float(at['keypoint_error_mm']),
                best_xy_error_mm=float(at['plug_socket_xy_error_mm']),
                best_socket_z_mm=float(at['plug_socket_z_mm']),
                best_axis_error_deg=float(at['unsigned_axis_error_deg']),
                final_xy_error_mm=float(trajectory[-1]['plug_socket_xy_error_mm']),
                final_socket_z_mm=float(trajectory[-1]['plug_socket_z_mm']),
                ee_down_after_contact_mm=ee_down, plug_down_after_contact_mm=plug_down,
                observed_stall_proxy=bool(first is not None and ee_down > 2 and plug_down < .5),
                post_contact_k14_579_fraction=(sum(x in (5, 7, 9) for x in after) / len(after)
                                                 if after else None),
                post_contact_queries=len(after),
                video=str(folder / 'front_wrist.mp4')))
        print(f'failure audit {i}/{len(entries)} {model}', flush=True)
    if not args.smoke and len(rows) != 520:
        raise ValueError(f'Expected 520 rollout records, got {len(rows)}')
    counts = dict(Counter(x['event'] for x in rows))
    by_model = {model: dict(Counter(x['event'] for x in rows if x['model'] == model))
                for model in sorted({x['model'] for x in rows})}
    report = dict(version=VERSION, unit='model/seed rollout', rows=rows,
                  counts=counts, by_model=by_model,
                  caveat='Event and stall fields describe observed trajectories. '
                         'K14 membership is pre-action and cannot establish collision location or cause.')
    write(path, report)
    with (args.output / 'failure_audit.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    body = []
    for r in rows:
        link = Path(r['video'])
        body.append('<tr><td>%s</td><td>%d</td><td>%s</td><td>%s</td>'
                    '<td>%s</td><td>%.1f</td><td>%.1f</td><td>%.1f</td>'
                    '<td><a href="%s">video</a></td></tr>' %
                    (html.escape(r['model']), r['seed'], r['event'],
                     r['first_contact_step'] if r['first_contact_step'] is not None else '—',
                     r['observed_stall_proxy'], r['best_xy_error_mm'],
                     r['best_socket_z_mm'], r['best_axis_error_deg'], link))
    summary = ''.join('<tr><td>%s</td><td>%s</td></tr>' %
                      (html.escape(k),html.escape(str(v))) for k,v in by_model.items())
    (args.output / 'failure_audit.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>520 rollout symptom audit</title>'
        '<style>body{font:15px system-ui;background:#142032;color:#eaf2fc;max-width:1500px;margin:auto}'
        'a{color:#9ad5ff}td,th{border-bottom:1px solid #466079;padding:5px;text-align:left}</style>'
        '<h1>原 13 模型 × 40 初态：失败现象</h1><p>来源：原始 trajectory 和动作前力图；'
        '接触力是 USB—插座仿真记录。末态、曾成功与接触后未成功分开计数。'
        '“夹爪下移而 USB 少动”仅是症状。下表每行是一次模型评测，不是独立环境初态。</p>'
        '<p>类别计数：%s</p><table><tr><th>模型</th><th>轨迹计数</th></tr>%s</table>'
        '<h2>全部轨迹</h2><table><tr><th>模型</th><th>seed</th><th>事件</th>'
        '<th>首次接触步</th><th>停滞症状</th><th>最佳 XY mm</th><th>最佳 Z mm</th>'
        '<th>最佳轴误差 °</th><th>视频</th></tr>%s</table>'
        % (html.escape(str(counts)), summary, ''.join(body)))
    return report


def choose_states(args):
    old_plan = read(args.old_root / 'plan.json')
    old_report = read(args.old_root / 'analysis.json')
    if old_plan['study_version'] != 'cw-k4-paired-down-v1':
        raise ValueError('Expected frozen prior K4 down experiment')
    available = old_plan['all_selected']
    accepted = {(s['source'],s['frame']) for s in old_report['states'] if s['paired']}
    used, pilot = set(), []
    slots = [(3,'mouth',2),(3,'inside',2),(3,'bottom',1),
             (0,'mouth',1),(1,'mouth',1),(2,'mouth',1)]
    for k, band, count in slots:
        pool = sorted((x for x in available if x['k4']==k and x['depth_band']==band
                       and x['source'] not in used),
                      key=lambda x: ((x['source'],x['frame']) not in accepted,
                                     x['recorded_replay_error_mm'],x['source']))
        if len(pool) < count:
            raise ValueError(f'Not enough K4 {k} {band} pilot states')
        for x in pool[:count]:
            pilot.append(x);used.add(x['source'])
    if args.smoke:
        return pilot[:1], pilot[:1]
    full = list(pilot)
    for k, quota in QUOTA.items():
        options = sorted((x for x in available if x['k4']==k and x['source'] not in used),
                         key=lambda x: (x['depth_band']=='bottom',
                                        x['recorded_replay_error_mm'],x['source']))
        for x in options[:quota-sum(y['k4']==k for y in full)]:
            full.append(x);used.add(x['source'])
    if len(full)!=30 or len({x['source'] for x in full})!=30:
        raise ValueError(f'Expected 30 distinct-source states; got {len(full)}')
    return pilot, full


def prepare(args):
    args.output.mkdir(parents=True,exist_ok=True)
    failure_audit(args)
    pilot, full = choose_states(args)
    old_plan = read(args.old_root / 'plan.json')
    task = args.source / 'thirdparty/manifeel-isaacgymenvs/isaacgymenvs/tasks/tacsl/tacsl_task_usb.py'
    atlas = Path(old_plan['atlas']) / 'alternative_k4_model.npz'
    parent = parent_model()[2]
    manifest = args.dataset / 'spatial/manifest.json'
    plan = dict(version=replay.VERSION, study_version=VERSION, dataset=str(args.dataset),
        capture_root=old_plan['capture_root'], selected=pilot, all_selected=full,
        branches=list(BRANCHES), repeats=args.repeats, max_attempts=args.max_attempts,
        shift_mm=args.shift_mm, down_mm=args.down_mm, tilt_deg=args.tilt_deg,
        max_replay_error_mm=args.max_replay_error_mm, tactile_mae_ratio=args.tactile_mae_ratio,
        experiment_script_sha256=digest(replay.__file__),
        study_script_sha256=digest(__file__), simulator_task_sha256=digest(task),
        dataset_manifest_sha256=digest(manifest), frozen_k4_sha256=digest(atlas),
        frozen_parent_sha256=digest(parent), old_plan_sha256=digest(args.old_root/'plan.json'),
        old_analysis_sha256=digest(args.old_root/'analysis.json'),
        selection_note='Pilot chooses formerly replayable states by reconstruction quality only. '
                       'Full candidate pool was selected before those branch outcomes. '
                       'Performance inference is conditional on state reconstruction.',
        intervention='Hold / 3-step wait then 8-step axial down / retreat or socket XY or '
                     'socket X-axis tilt then same 8-step axial down; 11 control steps total.',
        smoke=bool(args.smoke))
    plan = json.loads(json.dumps(plan))
    path=args.output/'plan.json'
    if path.exists():
        previous=read(path)
        if previous!=plan and not (previous.get('selected')==full and
                {k:v for k,v in previous.items() if k!='selected'}==
                {k:v for k,v in plan.items() if k!='selected'}):
            raise ValueError('Existing plan differs; choose another output directory')
        print('resume',len(previous['selected']),'states',flush=True)
    else:
        write(path,plan)
    print('prepared',len(pilot),'pilot and',len(full),'total states',flush=True)


def action_sequence(kind, socket_quat, ps, shift_mm, down_mm, rs):
    if rs is None:
        raise ValueError('Rotation action scale is required for tilt branches')
    rot=Rotation.from_quat(socket_quat)
    trans=np.zeros((11,3),np.float32)
    rotation=np.zeros((11,3),np.float32)
    if kind!='hold':
        trans[3:]=rot.apply([0,0,-down_mm/8/1000])
    if kind=='retreat_down':
        trans[:2]=rot.apply([0,0,shift_mm/2/1000])
    elif kind.startswith(('x_','y_')):
        axis=0 if kind[0]=='x' else 1
        sign=1 if '_plus_' in kind else -1
        v=np.zeros(3);v[axis]=sign*shift_mm/2/1000
        trans[:2]=rot.apply(v)
    elif kind.startswith('pitch_'):
        # Socket-local X rotation tilts the plug toward +/- socket Y.
        sign=1 if 'plus' in kind else -1
        rotation[:2]=rot.apply([sign*np.deg2rad(ACTIVE_TILT_DEG)/2,0,0])
    elif kind not in ('down','hold'):
        raise ValueError(kind)
    return [np.r_[t/ps,r/rs].astype(np.float32) for t,r in zip(trans,rotation)]


def branch(args):
    plan=read(args.output/'plan.json')
    if plan['study_version']!=VERSION or plan['study_script_sha256']!=digest(__file__):
        raise ValueError('Action-effect script changed after prepare')
    if plan['experiment_script_sha256']!=digest(replay.__file__):
        raise ValueError('Replay script changed after prepare')
    if plan['dataset_manifest_sha256']!=digest(args.dataset/'spatial/manifest.json'):
        raise ValueError('Dataset manifest changed')
    global ACTIVE_TILT_DEG
    ACTIVE_TILT_DEG=float(plan['tilt_deg'])
    replay.BRANCHES=BRANCHES
    replay.branch_actions=action_sequence
    args.cluster_holds=True
    args.skip_first_replay=True
    replay.branch(args)


def expand(args):
    path=args.output/'plan.json';plan=read(path)
    if plan.get('smoke'):
        return
    report=read(args.output/'analysis.json')
    if report['stage']!='pilot':
        raise ValueError('Expected pilot analysis before expansion')
    usable=[x for x in report['states'] if x['usable']]
    mouth=sum(x['selected']['depth_band']=='mouth' for x in usable)
    inside=sum(x['selected']['depth_band']=='inside' for x in usable)
    passed=len(usable)>=3 and mouth>=1 and inside>=1
    gate=dict(pass_gate=passed,usable=len(usable),mouth=mouth,inside=inside,
              selected=8,required='>=3 usable, including >=1 mouth and >=1 inside')
    write(args.output/'pilot_gate.json',gate)
    if not passed:
        print('PILOT_GATE_FAILED',gate,flush=True)
        return False
    if plan['selected']!=plan['all_selected']:
        plan['selected']=plan['all_selected'];write(path,plan)
    print('PILOT_GATE_PASSED expanded to',len(plan['selected']),flush=True)
    return True


def branch_observation(path, centers, scale, parent_centers):
    with np.load(path) as z:
        force=z['force_base'][0]
        k4,_=k4_label(force,centers,scale)
        parent,_=k4_label(force,parent_centers,scale)
        rel=replay.socket_relative(np.c_[z['plug_pos'],z['plug_quat']],
                                   np.c_[z['socket_pos'],z['socket_quat']])
        progress=float(rel[0,2]-rel[-1,2])
        xy=float(np.linalg.norm(rel[-1,:2]))
        grasp=Rotation.from_quat(z['ee_quat']).inv().apply(z['plug_pos']-z['ee_pos'])
        slip=float(np.linalg.norm(grasp-grasp[0],axis=1).max()*1000)
        q1=Rotation.from_quat(z['plug_quat'][-1]);q0=Rotation.from_quat(z['plug_quat'][0])
        qs1=Rotation.from_quat(z['socket_quat'][-1]);qs0=Rotation.from_quat(z['socket_quat'][0])
        angular_delta=float(np.degrees(abs((qs1.inv()*q1).magnitude() -
                                           (qs0.inv()*q0).magnitude())))
        success=bool(z['success'].any())
        curve=(rel[0,2]-rel[:,2]).tolist()
        contact=np.linalg.norm(z['contact_force'],axis=-1).tolist()
    return dict(k4=k4,parent=parent,progress_mm=progress,xy_final_mm=xy,
                slip_mm=slip,angular_change_deg=angular_delta,
                success_ever=success,progress_curve_mm=curve,
                pair_contact_curve=contact)


def analyze(args):
    plan=read(args.output/'plan.json')
    progress=read(args.output/'branch_progress.json')
    # The frozen path is in the prior plan, avoiding assumptions about layout.
    old=read(args.old_root/'plan.json')
    centers,scale=frozen_model(Path(old['atlas']))
    parents,parent_scale,_=parent_model()
    if not np.allclose(scale,parent_scale):
        raise ValueError('K4/K14 tactile scales differ')
    indexed=defaultdict(list)
    for row in progress:
        indexed[(row['source'],row['frame'])].append(row)
    states=[]
    for item in plan['selected']:
        key=(item['source'],item['frame'])
        group=indexed[key]
        accepted=defaultdict(list)
        for r in group:
            if not r.get('accepted'):continue
            obs=branch_observation(r['file'],centers,scale,parents)
            if obs['k4']==item['k4'] and obs['parent'] in (5,7,9):
                accepted[r['kind']].append({**r,**obs})
        counts={kind:len(accepted[kind]) for kind in BRANCHES}
        holds=accepted['hold']
        hold_mean=float(np.mean([x['progress_mm'] for x in holds])) if holds else None
        result={}
        for kind in BRANCHES:
            arr=accepted[kind]
            if len(arr)<min(2,args.repeats):
                continue
            values=np.array([x['progress_mm'] for x in arr])
            if np.ptp(values)>.6:
                continue
            def mean(k):return float(np.mean([x[k] for x in arr]))
            extra=mean('progress_mm')-hold_mean if hold_mean is not None else None
            slip=max(x['slip_mm'] for x in arr)
            xy=max(x['xy_final_mm'] for x in arr)
            angle=max(x['angular_change_deg'] for x in arr)
            safe=slip<.75 and xy<3 and angle<3
            effective=bool(kind!='hold' and extra is not None and extra>=.5 and
                           mean('progress_mm')>=.5 and safe)
            utility=(extra - 2*max(0,slip-.75) - max(0,xy-3)
                     - .5*max(0,angle-3)) if extra is not None else None
            result[kind]=dict(n=len(arr),progress_mm=mean('progress_mm'),
                extra_mm=extra,spread_mm=float(np.ptp(values)),slip_mm=slip,
                xy_final_mm=xy,angle_change_deg=angle,effective=effective,
                utility_mm=utility,success_ever=any(x['success_ever'] for x in arr),
                video=arr[0].get('video'),file=arr[0]['file'],
                curve_mm=np.mean([x['progress_curve_mm'] for x in arr],axis=0).tolist())
        alternatives=[k for k in BRANCHES[2:] if k in result]
        usable=('hold' in result and 'down' in result and len(alternatives)>=2)
        state=dict(source=item['source'],frame=item['frame'],selected=item,
                   counts=counts,accepted_branches=result,usable=bool(usable),
                   alternatives=alternatives,rejected_attempts=sum(not x.get('accepted') for x in group),
                   class_mismatch_attempts=sum(x.get('accepted',False) for x in group)-
                                           sum(len(v) for v in accepted.values()))
        states.append(state)
    stage=('pilot' if len(plan['selected'])<len(plan['all_selected']) or plan.get('smoke')
           else 'full')
    report=dict(version=VERSION,stage=stage,states=states,selected=len(states),
                usable=sum(x['usable'] for x in states),
                branch_names=list(BRANCHES),
                accepted_attempts=sum(x.get('accepted',False) for x in progress),
                rejected_attempts=sum(not x.get('accepted',False) for x in progress),
                selection_note=plan['selection_note'],
                effect_definition='Action final axial progress minus same-state hold mean. '
                                  'Short effect is not a full insertion outcome.',
                safety_definition='Effective requires >=0.5 mm extra and absolute axial '
                                  'progress, <0.75 mm gripper slip, <3 mm final socket XY, '
                                  'and <3 degree axis change.')
    write(args.output/'analysis.json',report)
    render(args,report,scale)
    print('ANALYSIS',stage,'usable',report['usable'],'/',len(states),flush=True)
    return report


def render(args,report,scale):
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from run_contactworld_tactile_cluster_atlas import six_force_image
    asset=args.output/'assets';asset.mkdir(exist_ok=True)
    rows=[]
    for s in report['states']:
        counts=' '.join(f'{k}:{s["counts"][k]}' for k in BRANCHES)
        rows.append(f'<tr><td><a href="#s{s["source"]:03d}f{s["frame"]:03d}">'
                    f'{s["source"]}/{s["frame"]}</a></td><td>{s["selected"]["k4"]}</td>'
                    f'<td>{s["selected"]["depth_band"]}</td><td>{int(s["usable"])}</td>'
                    f'<td>{html.escape(counts)}</td></tr>')
    pieces=['<!doctype html><meta charset="utf-8"><title>ContactWorld action effects</title>',
      '<style>body{font:15px/1.55 system-ui;background:#132034;color:#e7f1fc;'
      'max-width:1600px;margin:auto;padding:20px}a{color:#9bd8ff}'
      'section{background:#20324a;padding:15px;margin:20px 0;border-radius:8px}'
      'table{border-collapse:collapse}td,th{padding:6px;border-bottom:1px solid #536d85}'
      'img,video{max-width:100%}.clips{display:grid;grid-template-columns:repeat(3,1fr);gap:9px}'
      'video{width:100%}@media(max-width:900px){.clips{grid-template-columns:1fr}}</style>',
      '<h1>同初态动作效果：接触后的短动作</h1>',
      f'<p>{report["stage"]}：{report["usable"]}/{report["selected"]} 个状态具备可比较的保持、下插及至少两种调整。'
      f'动作从同一示范前缀恢复；每个分支 11 步。六张力图按统一标尺显示。'
      f'旧模型失败分层见 <a href="failure_audit.html">520 条轨迹审计</a>。</p>',
      '<p>候选：保持、下插、回撤再下插、插座 ±X/±Y 横移再下插、绕插座 X 轴 ±2° 倾斜再下插。'
      '下插部分统一在第 4～11 步。位置变化使用插座坐标系。</p>',
      '<p>可信状态需重建起点、触觉类别、重复响应与参考示范一致。若失败，保留初始图片和门槛记录。'
      '轴向进展与夹持滑移是连续量；“有效”仅表示此短窗口满足预设几何和夹持条件。</p>',
      '<table><tr><th>source/frame</th><th>K4</th><th>深度阶段</th><th>可比较</th><th>各动作接受次数</th></tr>',
      ''.join(rows),'</table>']
    for s in report['states']:
        item=s['selected'];sid=f's{s["source"]:03d}f{s["frame"]:03d}'
        entries=s['accepted_branches']
        if entries:
            first=entries.get('hold',next(iter(entries.values())))['file']
            with np.load(first) as z:
                rgb=z['rgb'][0].copy();force=z['force_base'][0].copy()
                cloud=z['cloud_fused_xyz'].copy()
            cv2.imwrite(str(asset/f'{sid}_rgb.jpg'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(asset/f'{sid}_force.jpg'),six_force_image(force,scale))
        else:
            folder=Path(read(args.old_root/'plan.json')['capture_root'])/f'insertion_usb_episode_{s["source"]:03d}'
            cap=replay.load_capture(folder);_,bf=replay.load_forces(folder);t=s['frame']
            rgb=np.concatenate([cap['front'][t],cap['wrist'][t]],axis=1)
            cloud=cap['cloud_fused_xyz'][t]
            cv2.imwrite(str(asset/f'{sid}_rgb.jpg'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(asset/f'{sid}_force.jpg'),six_force_image(bf[t],scale))
        entries_html=[]
        for kind in BRANCHES:
            if kind not in entries:continue
            e=entries[kind]
            video=Path(e['video']).relative_to(args.output) if e.get('video') else None
            entries_html.append(f'<div><b>{kind}</b><p>额外轴向 {e["extra_mm"]:+.2f} mm；'
                f'滑移 {e["slip_mm"]:.2f} mm；结束 XY {e["xy_final_mm"]:.2f} mm；'
                f'有效 {int(e["effective"])}</p>'+
                (f'<video controls preload="metadata" src="{html.escape(str(video))}"></video>'
                 if video else '')+'</div>')
        cloud=np.asarray(cloud).reshape(-1,3)
        cloud=cloud[np.isfinite(cloud).all(1)][::5]
        fig_cloud,ax_cloud=plt.subplots(1,2,figsize=(8,3.4),layout='constrained')
        if len(cloud):
            for axis,ii,jj in ((ax_cloud[0],0,1),(ax_cloud[1],0,2)):
                axis.scatter(cloud[:,ii],cloud[:,jj],c=cloud[:,2],cmap='viridis',s=2)
                axis.set_xlabel('base '+('XYZ'[ii])+' m')
                axis.set_ylabel('base '+('XYZ'[jj])+' m')
                axis.set_aspect('equal',adjustable='box')
        fig_cloud.savefig(asset/f'{sid}_cloud.png',dpi=100);plt.close(fig_cloud)
        if entries:
            fig,ax=plt.subplots(figsize=(9,3.5),layout='constrained')
            for kind,e in entries.items():ax.plot(e['curve_mm'],label=kind)
            ax.set(xlabel='control step',ylabel='socket axial progress (mm)')
            ax.legend(ncol=3,fontsize=7)
            fig.savefig(asset/f'{sid}_curves.png',dpi=120);plt.close(fig)
        pieces.append(f'<section id="{sid}"><h2>source {s["source"]} frame {s["frame"]} · '
                      f'K4 {item["k4"]} · {item["depth_band"]}</h2>'
                      f'<p>原示范相对位姿 {html.escape(str(item["relative_mm"]))} mm；'
                      f'可信比较 {s["usable"]}；重建被拒 {s["rejected_attempts"]} 次，'
                      f'触觉类别不匹配 {s["class_mismatch_attempts"]} 次。</p>'
                      f'<img src="assets/{sid}_rgb.jpg"><img src="assets/{sid}_force.jpg">'
                      f'<img src="assets/{sid}_cloud.png">')
        if entries:
            pieces.append(f'<img src="assets/{sid}_curves.png">'
                          '<button onclick="this.parentNode.querySelectorAll('
                          "'video').forEach(v=>{v.currentTime=0;v.play()})"
                          '">从同一步播放</button>'
                          f'<div class="clips">{"".join(entries_html)}</div>')
        pieces.append('</section>')
    pieces.append('<p><a href="analysis.json">完整数据与拒绝原因</a> · '
                  '<a href="pilot_gate.json">试点验收</a>（通过试点后产生）</p>')
    (args.output/'index.html').write_text('\n'.join(pieces))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('prepare','branch','analyze','expand','probe'),required=True)
    p.add_argument('--dataset',type=Path,default=replay.DATA)
    p.add_argument('--source',type=Path,default=replay.SOURCE)
    p.add_argument('--eval-root',type=Path,default=EVAL)
    p.add_argument('--old-root',type=Path,default=OLD)
    p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--max-attempts',type=int,default=12)
    p.add_argument('--shift-mm',type=float,default=1.5)
    p.add_argument('--down-mm',type=float,default=5.)
    p.add_argument('--tilt-deg',type=float,default=2.)
    p.add_argument('--max-replay-error-mm',type=float,default=1.)
    p.add_argument('--tactile-mae-ratio',type=float,default=1.)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.repeats<1 or a.max_attempts<a.repeats or a.shift_mm<=0 or a.down_mm<=0:
        p.error('Invalid action/repeat budget')
    if a.smoke:
        a.repeats=1;a.max_attempts=max(2,a.max_attempts)
    if a.stage=='prepare':prepare(a)
    elif a.stage=='branch':branch(a)
    elif a.stage=='analyze':analyze(a)
    elif a.stage=='expand':expand(a)
    else:
        from probe_contactworld_action_effect import run
        run(a)


if __name__=='__main__':
    main()
