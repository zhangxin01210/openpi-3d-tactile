#!/usr/bin/env python3
"""Resumable offline/online slip study; ground truth is never a probe input."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

from contactworld_slip_core import VERSION, KINDS, read, write, save, sha, grasp, targets, causal_baseline, branch_effect

REPO = Path(__file__).resolve().parents[2]
RUNTIME = Path('/home/sai/zx/openpi-sim-runtime')
EVAL = RUNTIME/'runs/contactworld_pi0_v3_13models_40trials_20261003'
OLD = RUNTIME/'runs/contactworld_action_effect_20261009'
SOURCE = RUNTIME/'third_party/ContactWorld-bilateral-probe'


def status(args, stage, state, **kw):
    write(args.output/'stages'/f'{stage}.json', dict(stage=stage, state=state, **kw))
    print(stage, state, kw, flush=True)


def cache_path(args, item):
    return args.output/'cache'/f'{item["model"]}_seed_{item["seed"]}.npz'


def prepare(args):
    paths = sorted(args.eval_root.glob('pi0_cw3_usb_*/seed_*/trajectory.json'))
    if args.smoke:
        seeds = sorted({int(p.parent.name.split('_')[-1]) for p in paths})[:6]
        models = sorted({p.parent.parent.name for p in paths})[:2]
        paths = [p for p in paths if int(p.parent.name.split('_')[-1]) in seeds and p.parent.parent.name in models]
    if not paths or (not args.smoke and len(paths) != 520):
        raise ValueError(f'Expected 520 trajectories, got {len(paths)}')
    manifest = []
    for i, path in enumerate(paths):
        item = dict(model=path.parent.parent.name, seed=int(path.parent.name.split('_')[-1]), folder=str(path.parent))
        dest = cache_path(args, item)
        if dest.exists() and dest.with_suffix('.json').exists():
            manifest.append(read(dest.with_suffix('.json')))
            continue
        data = read(path); rows = data['rows']; n = len(rows)
        initial = data['result']['initial_physical_state']
        states = [initial] + rows
        physical = {k: np.asarray([r[k] for r in states]) for k in
                    ('ee_pos', 'ee_quat', 'plug_pos', 'plug_quat', 'socket_pos_gt', 'socket_quat', 'dof_pos', 'dof_vel')}
        p, q = grasp(physical['ee_pos'], physical['ee_quat'], physical['plug_pos'], physical['plug_quat'])
        action = np.asarray([r['action_sim_clipped'] for r in rows], np.float32)
        proprio = np.concatenate([physical['dof_pos'], physical['dof_vel']], axis=1).astype(np.float32)
        with np.load(path.parent/'inputs.npz') as obs:
            steps = obs['steps'].copy()
            force = np.concatenate([obs['force_local'][0:1], obs['force_local_post_step']], axis=0).reshape(n+1, -1)
            if not np.allclose(force[steps], obs['force_local'].reshape(len(steps), -1), atol=1e-7):
                raise ValueError(f'Pre/post tactile time mismatch: {path}')
            planned = np.zeros((n+1, 4, 6), np.float32)
            available = np.zeros(n+1, np.int32)
            chunks = obs['action_chunks']
            for j, t in enumerate(steps):
                end = min(t+4, n)
                raw = np.asarray([r['action_raw'] for r in rows[t:end]])
                if not np.allclose(raw, chunks[j, :end-t], atol=1e-6):
                    raise ValueError('Saved executed plan does not match the known action chunk')
                for k in range(t, end):
                    planned[k, :end-k] = action[k:end]
                    available[k] = end-k
        dt = float(rows[0]['time_s'])
        if len(force) != n+1 or not np.isfinite(force).all():
            raise ValueError(f'Invalid force: {path}')
        baseline, baseline_valid = causal_baseline(force, proprio, action)
        lab, ys, masks = targets(p, q, args.slip_mm, args.slip_deg)
        contact = np.array([np.linalg.norm(data['result']['initial']['plug_socket_contact_force_xyz'])] +
                           [np.linalg.norm(r['plug_socket_contact_force_xyz']) for r in rows])
        rel = Rotation.from_quat(physical['socket_quat']).inv().apply(physical['plug_pos']-physical['socket_pos_gt'])*1000
        sensitivity = {str(factor): int(targets(p, q, args.slip_mm*factor, args.slip_deg*factor)[0]['onset'].sum())
                       for factor in (.5, 1., 2.)}
        arrays = dict(force=force.astype(np.float32), proprio=proprio, action=action, planned=planned,
                      available=available, steps=steps, baseline=baseline, baseline_valid=baseline_valid,
                      relative_position=p, relative_quaternion=q, socket_relative=rel, contact=contact,
                      success=np.asarray([bool(data['result']['initial'].get('source_success',False))]+[bool(r['source_success']) for r in rows]),
                      active=lab['active'], onset=lab['onset'], event_start=lab['event_start'],movement=lab['movement'], dp=lab['dp'], dr=lab['dr'],
                      **physical, **{f'y_{k}': v for k, v in ys.items()}, **{f'valid_{k}': v for k, v in masks.items()})
        save(dest, **arrays)
        item.update(dt=dt, n=n, query_steps=steps.tolist(), onsets=np.flatnonzero(lab['onset']).tolist(),
                    event_starts=np.flatnonzero(lab['event_start']).tolist(),
                    ever=bool(data['result']['source_ever']), final=bool(data['result']['source_final']),
                    baseline_fraction=float(baseline_valid.mean()), sensitivity_onsets=sensitivity,
                    max_relative_translation_mm=float(np.linalg.norm(p-p[0], axis=1).max()),
                    force_timing_verified=True, trajectory_sha256=sha(path))
        write(dest.with_suffix('.json'), item); manifest.append(item)
        if i % 20 == 0:
            print('prepare', i+1, '/', len(paths), flush=True)
    write(args.output/'manifest.json', manifest)
    # Prior branch data are rescored in a moving gripper frame including rotation.
    old = OLD/'analysis.json'
    previous = []
    if old.exists():
        for state in read(old).get('states', []):
            for kind, entry in state.get('accepted_branches', {}).items():
                if entry.get('file') and Path(entry['file']).exists():
                    with np.load(entry['file']) as z:
                        previous.append(dict(source=state['source'], kind=kind, **branch_effect(z)))
    write(args.output/'prior_branch_audit.json', dict(rows=previous,
          note='Historical selected demonstration states; separate from random-reset policy rollouts.'))
    status(args, 'prepare', 'complete', trajectories=len(manifest), seeds=len({r['seed'] for r in manifest}),
           events=sum(len(r['onsets']) for r in manifest))


def views(args):
    from contactworld_neighbor_features import Features
    encoder = Features(device='cpu', batch=32)
    import torch
    torch.set_num_threads(args.threads)
    items = read(args.output/'manifest.json')
    for i, item in enumerate(items):
        dest = cache_path(args, item).with_suffix('.views.npz')
        if dest.exists():
            continue
        with np.load(Path(item['folder'])/'inputs.npz') as z:
            # Fixed image-center crop, never cropped with simulator object positions.
            front = z['front']; wrist = z['wrist']
            visual = np.concatenate([encoder.rgb(front), encoder.rgb(wrist),
                                     encoder.rgb(front[:, 64:224, 48:208]),
                                     encoder.rgb(wrist[:, 48:208, 48:208])], axis=1)
            cloud = encoder.cloud(z['cloud_xyz'])
        save(dest, visual=visual, cloud=cloud)
        print('views', i+1, '/', len(items), flush=True)
    status(args, 'views', 'complete', trajectories=len(items))


def select(args):
    dest = args.output/'branch_plan.json'
    if dest.exists():
        return
    items = read(args.output/'manifest.json')
    pools = dict(pre=[], ongoing=[], stable=[])
    for item in items:
        with np.load(cache_path(args, item)) as z:
            for t in z['steps']:
                t = int(t)
                if t < 8 or t+16 >= item['n'] or abs(z['socket_relative'][t, 2]) > 40 or z['success'][t]:
                    continue
                if z['active'][t]:
                    kind = 'ongoing'
                elif not z['movement'][t] and z['event_start'][t+1:t+5].any():
                    kind = 'pre'
                elif not z['active'][t-4:t+9].any():
                    kind = 'stable'
                else:
                    continue
                pools[kind].append(dict(model=item['model'], seed=item['seed'], step=t, group=kind,
                                        folder=item['folder'], dt=item['dt']))
    rng = np.random.default_rng(10010)
    for pool in pools.values():
        rng.shuffle(pool)
    # Independent seeds, selection uses original observed events, never branch outcomes.
    selected = []; used = set()
    slots = ['pre', 'ongoing', 'stable', 'pre', 'ongoing', 'stable', 'pre', 'ongoing']
    slots += ['pre', 'ongoing', 'stable'] * 10
    for kind in slots:
        choice = next((r for r in pools[kind] if r['seed'] not in used), None)
        if choice:
            used.add(choice['seed']); selected.append(choice)
        if len(selected) >= args.branch_states:
            break
    for i, r in enumerate(selected):
        r['id'] = f'state_{i:02d}'
        r['query_index'] = r['step']//4
    write(dest, dict(selected=selected, kinds=KINDS, pilot=min(8, len(selected)),
                     repeats=args.repeats, horizon=16, max_attempts=args.repeats+2,
                     selection='Targeted event sample, one state per seed, currently successful states excluded; not population prevalence.',
                     command='Repeat four already-known actions for 16 steps; resume is open-loop, not pi0 feedback.'))
    status(args, 'select', 'complete', states=len(selected), pools={k:len(v) for k,v in pools.items()})


def branches(args):
    plan = read(args.output/'branch_plan.json')
    if args.offline_only:
        status(args, 'branches', 'pending', reason='offline-only requested; rerun without this flag to continue')
        return
    check = subprocess.run(['nvidia-smi'], capture_output=True, text=True) if shutil.which('nvidia-smi') else None
    if check is None or check.returncode:
        status(args, 'branches', 'pending_gpu', reason=(check.stderr+check.stdout if check else 'nvidia-smi missing')[:2000])
        return
    accepted = []
    started=time.monotonic()
    for i, item in enumerate(plan['selected']):
        if i == plan['pilot']:
            accepted_groups={r['group'] for r in plan['selected'][:i] if r['id'] in accepted}
            gate = len(accepted) >= max(3, int(np.ceil(plan['pilot']*.6))) and {'pre','stable'}<=accepted_groups
            write(args.output/'pilot_gate.json', dict(passed=gate, usable=len(accepted),
                  selected=plan['pilot'], groups=sorted(accepted_groups),note='Start matching, repeatable continue/hold, and both pre-event/stable controls required.'))
            if not gate:
                status(args, 'branches', 'gate_failed', usable=len(accepted), attempted=i)
                return
        done = args.output/'branches'/item['id']/'result.json'
        if not done.exists():
            remaining=args.sim_budget_hours*3600-(time.monotonic()-started)
            if remaining<60:
                status(args,'branches','budget_checkpoint',usable=len(accepted),next_state=item['id'])
                return
            log = args.output/'logs'/f'{item["id"]}.log'; log.parent.mkdir(parents=True, exist_ok=True)
            command = ['bash', str(REPO/'sim/scripts/run_contactworld_sm120.sh'),
                       str(REPO/'sim/scripts/branch_contactworld_slip.py'), '--output', str(args.output),
                       '--source', str(args.source), '--state-id', item['id']]
            with log.open('a') as stream:
                try:
                    worker = subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    code=worker.wait(timeout=min(args.state_timeout,remaining))
                    if code:
                        status(args, 'branches', 'worker_failed', state_id=item['id'], log=str(log))
                        return
                except subprocess.TimeoutExpired:
                    os.killpg(worker.pid,signal.SIGTERM)
                    try:worker.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(worker.pid,signal.SIGKILL);worker.wait()
                    status(args, 'branches', 'worker_timeout', state_id=item['id'], log=str(log))
                    return
        result = read(done)
        if result['usable']:
            accepted.append(item['id'])
        print('branch state', i+1, '/', len(plan['selected']), 'usable', result['usable'], flush=True)
    status(args, 'branches', 'complete', states=len(plan['selected']), usable=len(accepted))


def render_clip(args, item, center, target):
    import cv2
    path = cache_path(args, item)
    with np.load(path) as z:
        force = z['force'].reshape(-1, 2, 10, 14, 3)
        dp, dr, active = z['dp'], z['dr'], z['active']
    start, end = max(0, center-8), min(len(force), center+17)
    limits = np.asarray(read(RUNTIME/'runs/contactworld_pi0_v3_13models_40trials_20261003/all_sensor_comparison/manifest.json')['scales']['local'])
    cap = cv2.VideoCapture(str(Path(item['folder'])/'front_wrist.mp4'))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.stem+'.partial.mp4')
    proc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                             '-s', '1024x380', '-r', str(1/item['dt']), '-i', '-', '-an', '-c:v', 'libx264',
                             '-threads', '1', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary)], stdin=subprocess.PIPE)
    try:
        for t in range(start, end):
            ok, rgb = cap.read()
            if not ok:
                raise ValueError('Video shorter than aligned data')
            canvas = np.full((380, 1024, 3), 25, np.uint8)
            canvas[45:301, :512] = rgb
            for hand in range(2):
                for axis in range(3):
                    v = np.clip(force[t, hand, :, :, axis]/limits[axis], -1, 1)
                    c = np.stack([np.where(v<=0,245,245-190*v),245-160*np.abs(v),np.where(v>=0,245,245+190*v)],-1).astype(np.uint8)
                    x,y = 515+axis*168,45+hand*145
                    canvas[y:y+110, x:x+160] = cv2.resize(c,(160,110), interpolation=cv2.INTER_NEAREST)
                    cv2.putText(canvas, ('L' if hand==0 else 'R')+' F'+'xyz'[axis], (x,y-8),0,.48,(230,230,230),1)
            lines = [f'{item["model"]} | seed {item["seed"]} | frame {t} | marked {center}',
                     f'2-step relative shift {dp[t]:.3f} mm | rotation {dr[t]:.3f} deg | confirmed {int(active[t])}',
                     'Labels describe gripper-relative motion; tactile is sensor-local. Blue - / Red +.']
            for j,line in enumerate(lines):
                cv2.putText(canvas,line,(7,20 if j==0 else 321+(j-1)*25),0,.43,(235,235,235),1)
            proc.stdin.write(canvas.tobytes())
    finally:
        cap.release(); proc.stdin.close(); code=proc.wait()
    if code:
        raise RuntimeError('ffmpeg failed')
    temporary.replace(target)
    return start, end


def report(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest_path = args.output/'manifest.json'
    items = read(manifest_path) if manifest_path.exists() else []
    stages = [read(p) for p in sorted((args.output/'stages').glob('*.json'))]
    body = ['<!doctype html><meta charset="utf-8"><title>ContactWorld 滑移研究</title>',
            '<style>body{font:16px/1.6 system-ui;max-width:1500px;margin:auto;padding:16px;background:#142032;color:#eaf2fc}'
            'a{color:#9ad5ff}td,th{padding:6px;border:1px solid #4d6780}table{border-collapse:collapse}'
            'video{max-width:100%;width:1024px}img{max-width:100%}section{border-top:1px solid #526b86;padding:20px 0}</style>',
            '<h1>夹爪—USB：相对运动检测、预警与动作干预</h1>',
            '<p>初次输出为自动标签与探索性分析，需先核对下方连续案例。标签依据夹爪坐标中的 USB 平移与转动，'
            '并非对胶垫接触点摩擦滑动的直接测量。任何检测能力不自动等于提前预警能力。</p>',
            f'<p>阈值：两个控制步内相对平移 ≥ {args.slip_mm} mm 或转动 ≥ {args.slip_deg}°，连续两帧确认，连续三帧低于门槛退出。'
            '同时保留半倍／双倍阈值覆盖；没有依据待评估触觉定义标签。</p>',
            '<h2>执行状态</h2><pre>'+html.escape(json.dumps(stages,ensure_ascii=False,indent=2))+'</pre>']
    if items:
        body.append(f'<p>{len(items)} 条轨迹；{len({r["seed"] for r in items})} 个独立初态；'
                    f'{sum(len(r["onsets"]) for r in items)} 次操作定义下的事件。统计拆分以 seed 为单位。</p>')
        body.append('<table><tr><th>阈值倍数</th><th>事件数</th><th>有事件轨迹</th></tr>')
        for factor in (.5,1.,2.):
            values=[r['sensitivity_onsets'][str(factor)] for r in items]
            body.append(f'<tr><td>{factor}</td><td>{sum(values)}</td><td>{sum(v>0 for v in values)}/{len(values)}</td></tr>')
        body.append('</table>')
    for filename, title in [('probe_report.html','检测与预警：分组留出与模态对照'),
                            ('branch_report.html','同状态动作效果与防滑选择')]:
        if (args.output/filename).exists():
            body.append(f'<h2><a href="{filename}">{title}</a></h2>')
    body.append('<h2>先核对事件时间线与反例</h2><p>每段包含双视图及左右 Fx/Fy/Fz，统一局部通道色标。'
                'marked 是选择片段的参考帧；每帧 confirmed 标签随时间变化。</p>')
    candidates = []
    for item in items:
        if item['onsets']:
            candidates.append((item,item['onsets'][0],'首次确认相对运动'))
        else:
            candidates.append((item,min(80,item['n']-1),'未检出持续相对运动对照'))
    prediction_cases = args.output/'probe_cases.json'
    risk_lookup={}
    prediction_file=args.output/'probes/forecast4_TH/predictions.npz'
    if prediction_file.exists() and (args.output/'probe_results.json').exists():
        prediction_rows=read(args.output/'probe_results.json')['rows']
        with np.load(prediction_file) as pred:
            for j,r in enumerate(prediction_rows):
                if np.isfinite(pred['score'][j]):
                    risk_lookup.setdefault((r['model'],r['seed']),[]).append((r['step'],float(pred['score'][j]),float(pred['threshold'][j])))
    if prediction_cases.exists():
        lookup={(r['model'],r['seed']):r for r in items}
        candidates = [(lookup[(r['model'],r['seed'])],r['step'],r['title']) for r in read(prediction_cases)] + candidates
    rng=np.random.default_rng(91)
    tail=candidates[12:];rng.shuffle(tail);candidates=candidates[:12]+tail
    used=set();shown=0
    for item,t,title in candidates:
        key=(item['model'],item['seed'])
        if key in used:continue
        used.add(key)
        if shown>=args.clips:break
        shown+=1
        name=f'{item["model"]}_{item["seed"]}_{t}'
        target=args.output/'clips'/f'{name}.mp4'
        start,end=max(0,t-8),min(item['n']+1,t+17)
        if not target.exists():render_clip(args,item,t,target)
        with np.load(cache_path(args,item)) as z:
            fig,axs=plt.subplots(3,1,figsize=(10,6),sharex=True)
            xs=np.arange(start,end)
            axs[0].plot(xs,z['dp'][start:end],label='2-step relative translation (mm)')
            axs[0].axhline(args.slip_mm,color='r',ls='--');axs[0].legend()
            axs[1].plot(xs,z['dr'][start:end],label='2-step relative rotation (deg)')
            axs[1].axhline(args.slip_deg,color='r',ls='--');axs[1].legend()
            values=[r for r in risk_lookup.get(key,[]) if start<=r[0]<end]
            if values:
                values=np.asarray(values);axs[2].plot(values[:,0],values[:,1],'.-',label='TH forecast 4: held-out score')
                axs[2].plot(values[:,0],values[:,2],'--',label='validation-only threshold');axs[2].legend()
            else:axs[2].text(.1,.5,'No valid held-out forecast at these frames',transform=axs[2].transAxes)
            axs[2].set_ylim(0,1);axs[2].set_xlabel('observation frame (score is not calibrated probability)')
            for ax in axs:ax.axvline(t,color='gray',ls=':')
            fig.tight_layout();fig.savefig(target.with_suffix('.svg'));plt.close(fig)
        link=os.path.relpath(target,args.output)
        body.append(f'<section><h3>{html.escape(title)} · {html.escape(item["model"])} · seed {item["seed"]}</h3>'
                    f'<p>参考帧 {t}；末态成功 {item["final"]}；最大夹持相对位移 {item["max_relative_translation_mm"]:.2f} mm。</p>'
                    f'<video controls preload="metadata" src="{link}"></video>'
                    f'<button onclick="this.previousElementSibling.pause();this.previousElementSibling.currentTime={(t-start)*item["dt"]:.6f}">停在参考帧</button>'
                    f'<p><img src="{link.replace(".mp4",".svg")}"></p></section>')
    (args.output/'index.html').write_text('\n'.join(body))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=RUNTIME/'runs/contactworld_slip_study_20261010')
    p.add_argument('--eval-root',type=Path,default=EVAL)
    p.add_argument('--source',type=Path,default=SOURCE)
    p.add_argument('--stage',choices=['all','prepare','views','probes','select','branches','branch-probes','report'],default='all')
    p.add_argument('--smoke',action='store_true');p.add_argument('--offline-only',action='store_true')
    p.add_argument('--slip-mm',type=float,default=.5);p.add_argument('--slip-deg',type=float,default=1.)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--epochs',type=int,default=35)
    p.add_argument('--branch-states',type=int,default=32);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--clips',type=int,default=30);p.add_argument('--state-timeout',type=int,default=1800)
    p.add_argument('--sim-budget-hours',type=float,default=6.)
    args=p.parse_args();args.output=args.output.resolve()
    if args.slip_mm<=0 or args.slip_deg<=0 or args.repeats<2 or args.branch_states<3:
        p.error('Positive thresholds, at least two repeats and three branch states required')
    args.output.mkdir(parents=True,exist_ok=True)
    if args.smoke:args.epochs=2;args.clips=2;args.branch_states=3
    frozen={k:v for k,v in vars(args).items() if k not in ('stage','offline_only','threads','clips','state_timeout','sim_budget_hours')}
    frozen={k:str(v) if isinstance(v,Path) else v for k,v in frozen.items()}
    frozen.update(version=VERSION,code={f:sha(Path(__file__).with_name(f)) for f in
                  ('contactworld_slip_core.py','run_contactworld_slip_study.py','probe_contactworld_slip.py','branch_contactworld_slip.py')})
    task=args.source/'thirdparty/manifeel-isaacgymenvs/isaacgymenvs/tasks/tacsl/tacsl_task_usb.py'
    frozen['simulator_task_sha256']=sha(task)
    frozen['data_files']={str(p.relative_to(args.eval_root)):{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
                          for pattern in ('pi0_cw3_usb_*/seed_*/trajectory.json','pi0_cw3_usb_*/seed_*/inputs.npz')
                          for p in sorted(args.eval_root.glob(pattern))}
    plan=args.output/'run_contract.json'
    if plan.exists() and read(plan)!=frozen:raise ValueError('Run inputs/code changed; choose a new output folder')
    write(plan,frozen)
    from probe_contactworld_slip import probes, branch_probes
    actions={'prepare':prepare,'views':views,'probes':probes,'select':select,'branches':branches,
             'branch-probes':branch_probes,'report':report}
    requested=list(actions) if args.stage=='all' else [args.stage]
    try:
        for stage in requested:
            marker=args.output/'stages'/f'{stage}.json'
            if stage not in ('report','branch-probes') and marker.exists() and read(marker)['state']=='complete':continue
            status(args,stage,'running')
            stage_start=time.monotonic()
            actions[stage](args)
            if marker.exists():
                detail=read(marker);detail['elapsed_s']=round(time.monotonic()-stage_start,2);write(marker,detail)
            if stage=='report':
                status(args,stage,'complete');report(args)
            if stage=='probes':report(args)
        stages=[read(f) for f in (args.output/'stages').glob('*.json')]
        write(args.output/'run_status.json',dict(state='review_ready',stages=stages,
              note='Offline results may be ready while simulation is pending; inspect stage status.'))
        print('REPORT',args.output/'index.html',flush=True)
    except Exception:
        status(args,stage,'failed',error=traceback.format_exc())
        try:report(args)
        except Exception:pass
        raise


if __name__=='__main__':main()
