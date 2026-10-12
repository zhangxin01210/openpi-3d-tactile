#!/usr/bin/env python3
"""Bilateral tactile baseline audit, matched action branches, and grouped probes.

Stages: prepare (offline) -> branch (Isaac Gym) -> analyze (offline). All
physical branch interventions begin at a replayed demonstration prefix.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / 'data/contactworld_usb_v3_bilateral'
SOURCE = Path('/home/sai/zx/openpi-sim-runtime/third_party/ContactWorld-bilateral-probe')
OUTPUT = Path('/home/sai/zx/openpi-sim-runtime/runs/contactworld_tactile_action_branches_v3_20261008')
BRANCHES = ('hold', 'down', 'retreat_down', 'x_plus_down', 'x_minus_down', 'y_plus_down', 'y_minus_down')
VERSION = 'cw-tactile-branch-v3'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def load_capture(folder):
    with np.load(folder / 'v2_capture.npz') as z:
        return {k: z[k].copy() for k in ('front', 'wrist', 'cloud_fused_xyz', 'state', 'plug_pose', 'socket_pose', 'ee_pose', 'action_executed')}


def load_forces(folder):
    with np.load(folder / 'bilateral_tactile.npz') as z:
        local = np.stack([z[f'force_grid_{s}'] for s in ('left', 'right')], 1)
        base = np.stack([z[f'force_base_{s}'] for s in ('left', 'right')], 1)
    return local.astype(np.float32), base.astype(np.float32)


def socket_relative(plug, socket):
    r = Rotation.from_quat(socket[..., 3:])
    return r.inv().apply(plug[..., :3] - socket[..., :3]) * 1000


def source_contact(folder, n):
    rows = json.loads((folder / 'replay.json').read_text())['rows']
    if len(rows) != n:
        raise ValueError(f'Replay length mismatch: {folder}')
    return np.r_[False, [np.linalg.norm(x['plug_socket_force_post']) > .1 for x in rows[:-1]]]


def tactile_features(force):
    # force shape [T, finger, taxel, xyz] in robot base. Preserve sign and
    # spatial organization; summaries alone can cancel opposing taxels.
    f = np.asarray(force, np.float32).reshape(-1, 2, 10, 14, 3)
    parts = [f.mean((2, 3)).reshape(len(f), -1), np.abs(f).mean((2, 3)).reshape(len(f), -1)]
    for side in (0, 1):
        for row in (slice(0, 5), slice(5, 10)):
            for col in (slice(0, 7), slice(7, 14)):
                parts.append(f[:, side, row, col].mean((1, 2)))
    return np.concatenate(parts, 1)


def stable_baseline(local, contact, action):
    # A stable grasp is sought before first socket contact. No forced frame-0
    # baseline. Compare local sensor channels to avoid rotation artifacts.
    first = int(np.flatnonzero(contact)[0]) if contact.any() else len(local)
    end = min(first, len(local) - 4, 30)
    if end < 8:
        return None
    magnitude = np.abs(local).mean((1, 2, 3, 4))
    delta = np.r_[0, np.abs(np.diff(local, axis=0)).mean((1, 2, 3, 4))]
    candidates = []
    for t in range(3, end - 3):
        window = slice(t, t + 4)
        # Motion is allowed, but gripping force must be stable and nontrivial.
        stability = float(np.median(delta[window]) / max(np.median(magnitude[window]), 1e-7))
        if magnitude[window].mean() > 1e-6 and stability < .25:
            candidates.append((stability, t))
    if not candidates:
        return None
    t = min(candidates)[1]
    return dict(start=t, stop=t + 4, local=np.median(local[t:t+4], axis=0))


def prepare(args):
    manifest = json.loads((args.dataset / 'spatial/manifest.json').read_text())
    entries = manifest['episodes']
    root = Path(manifest['capture_root'])
    if args.smoke:
        entries = entries[:8]
    baseline_rows, candidates = [], []
    baselines = {}
    for i, entry in enumerate(entries, 1):
        source = int(entry['source_episode']); folder = root / f'insertion_usb_episode_{source:03d}'
        capture = load_capture(folder); local, base = load_forces(folder)
        n = len(local)
        if len(capture['plug_pose']) != n:
            raise ValueError(f'Capture length mismatch: {source}')
        contact = source_contact(folder, n)
        baseline = stable_baseline(local, contact, capture['action_executed'])
        rel = socket_relative(capture['plug_pose'], capture['socket_pose'])
        if baseline is None:
            baseline_rows.append(dict(source=source, split=entry['split'], found=False, first_contact=int(np.argmax(contact)) if contact.any() else None))
        else:
            bg = baseline['local']; residual = local - bg[None]
            raw_change = np.abs(np.diff(local, axis=0)).mean((1, 2, 3, 4))
            baselines[source] = bg
            baseline_rows.append(dict(source=source, split=entry['split'], found=True,
                                      window=[baseline['start'], baseline['stop']],
                                      first_contact=int(np.argmax(contact)) if contact.any() else None,
                                      baseline_abs=float(np.abs(bg).mean()),
                                      post_contact_residual_abs=float(np.abs(residual[contact]).mean()) if contact.any() else None,
                                      force_temporal_change=float(raw_change.mean())))
        # Select by physical event category, not by tactile magnitude/cluster.
        speed = np.r_[np.zeros((1, 3)), np.diff(rel, axis=0)]
        actions = capture['action_executed']
        for t in range(8, n - 8):
            xy = float(np.linalg.norm(rel[t, :2]))
            z = float(rel[t, 2]); prior_down = bool(actions[t-1, 2] < -.2)
            cat = None
            if not contact[t] and z > 12 and t < n - 20: cat = 'free'
            elif not contact[t] and xy < 3 and 5 < z < 25: cat = 'near_aligned'
            elif contact[t] and xy > 5 and z > 4: cat = 'contact_offset'
            elif contact[t] and prior_down and speed[t, 2] > -.25 and z > 3: cat = 'blocked'
            elif contact[t] and prior_down and speed[t, 2] < -.4: cat = 'progressing'
            elif contact[t] and np.linalg.norm(actions[t, :2]) > .3: cat = 'adjusting'
            if cat is None:
                continue
            replay_row = json.loads((folder / 'replay.json').read_text())['rows'][t]
            if replay_row['plug_pre_error_mm'] > args.max_replay_error_mm:
                continue
            grasp_mm = Rotation.from_quat(capture['ee_pose'][t, 3:]).inv().apply(
                capture['plug_pose'][t, :3] - capture['ee_pose'][t, :3]) * 1000
            candidates.append(dict(category=cat, source=source, episode=int(entry['episode_index']),
                                   split=entry['split'], frame=t, relative_mm=rel[t].round(3).tolist(),
                                   contact=bool(contact[t]), grasp_mm=grasp_mm.round(3).tolist(),
                                   recorded_replay_error_mm=float(replay_row['plug_pre_error_mm']),
                                   baseline_found=baseline is not None and baseline['stop'] <= t,
                                   force_abs=float(np.abs(base[t]).mean())))
        print(f'prepare {i}/{len(entries)} source={source} baseline={baseline is not None}', flush=True)
    # Deterministic, one source per category where possible. Favor existing
    # stable baseline and low replay error. Coverage shortfall is explicit.
    categories = ('free', 'near_aligned', 'contact_offset', 'blocked', 'progressing', 'adjusting')
    selected, used = [], set()
    per_category = 1 if args.smoke else args.per_category
    for category in categories:
        # One candidate frame per source, then select different grasps and
        # initial geometry within the same physical event category.
        by_source = {}
        ordered = sorted((c for c in candidates if c['category'] == category),
                         key=lambda c: (not c['baseline_found'], c['recorded_replay_error_mm'], abs(c['frame'] - 30)))
        for c in ordered:
            by_source.setdefault(c['source'], c)
        pool = [c for c in by_source.values() if c['source'] not in used and c['baseline_found']]
        if not pool:
            pool = [c for c in by_source.values() if c['source'] not in used]
        for j in range(per_category):
            if not pool: break
            if j == 0:
                picked = min(pool, key=lambda c: (not c['baseline_found'], c['recorded_replay_error_mm']))
            else:
                prior = [x for x in selected if x['category'] == category]
                def diversity(c):
                    return min(np.linalg.norm(np.asarray(c['grasp_mm']) - np.asarray(x['grasp_mm']))
                               + .2 * np.linalg.norm(np.asarray(c['relative_mm']) - np.asarray(x['relative_mm']))
                               for x in prior)
                picked = max(pool, key=lambda c: (diversity(c), -c['recorded_replay_error_mm']))
            selected.append(picked); used.add(picked['source'])
            pool = [c for c in pool if c['source'] != picked['source']]
    if args.smoke: selected = selected[:1]
    if not selected:
        raise ValueError('No replay state passed selection; inspect source capture')
    task_file = args.source / 'thirdparty/manifeel-isaacgymenvs/isaacgymenvs/tasks/tacsl/tacsl_task_usb.py'
    def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    plan = dict(version=VERSION, dataset=str(args.dataset), capture_root=str(root),
                dataset_manifest_sha256=digest(args.dataset/'spatial/manifest.json'),
                experiment_script_sha256=digest(__file__), simulator_task_sha256=digest(task_file),
                categories_requested=list(categories), per_category=per_category, selected=selected,
                missing_categories=sorted(set(categories) - {x['category'] for x in selected}),
                repeats=args.repeats, max_attempts=args.max_attempts, shift_mm=args.shift_mm, down_mm=args.down_mm,
                max_replay_error_mm=args.max_replay_error_mm)
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    plan_path = out / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Existing output has a different experiment plan; choose another --output')
    write_json(plan_path, plan)
    write_json(out / 'baseline_audit.json', dict(rows=baseline_rows,
               with_baseline=sum(x['found'] for x in baseline_rows), total=len(baseline_rows)))
    np.savez_compressed(out / 'baselines_local.npz', **{f'source_{k:03d}': v for k, v in baselines.items()})
    print('selected', len(selected), 'missing', plan['missing_categories'], flush=True)


def snapshot(env):
    return {k: v[0].detach().cpu().numpy().copy() for k, v in env.obs_dict.items()
            if k in ('ee_pos', 'ee_quat', 'plug_pos', 'plug_quat', 'socket_pos_gt', 'socket_quat',
                     'dof_pos', 'dof_vel', 'front', 'wrist',
                     'tactile_force_field_left', 'tactile_force_field_right')}


def pose_feedback_action(demo, t, obs, ps, rs):
    target_pos = demo['ee_pos'][t] + demo['action'][t, :3] * ps
    target_quat = Rotation.from_rotvec(demo['action'][t, 3:6] * rs) * Rotation.from_quat(demo['ee_quat'][t])
    result = np.empty(6, np.float32)
    result[:3] = (target_pos - obs['ee_pos']) / ps
    result[3:] = (target_quat * Rotation.from_quat(obs['ee_quat']).inv()).as_rotvec() / rs
    return result


def branch_actions(kind, socket_quat, ps, shift_mm, down_mm, rs=None):
    rot = Rotation.from_quat(socket_quat)
    if kind == 'hold':
        vecs = [np.zeros(3)] * 11
    elif kind == 'down':
        vecs = [np.zeros(3)] * 3 + [rot.apply([0, 0, -down_mm / 8 / 1000])] * 8
    elif kind == 'retreat_down':
        vecs = [rot.apply([0, 0, shift_mm / 2 / 1000])] * 2 + [np.zeros(3)] + [rot.apply([0, 0, -(down_mm + shift_mm) / 8 / 1000])] * 8
    else:
        axis = np.array([1., 0., 0.]) if kind.startswith('x') else np.array([0., 1., 0.])
        sign = 1 if '_plus_' in kind else -1
        vecs = [rot.apply(axis * sign * shift_mm / 2 / 1000)] * 2 + [np.zeros(3)] + [rot.apply([0, 0, -down_mm / 8 / 1000])] * 8
    return [np.r_[v / ps, np.zeros(3)].astype(np.float32) for v in vecs]


def branch(args):
    # Isaac Gym must load before torch.
    from isaacgym import gymapi  # noqa: F401
    import torch
    from eval_contactworld_proprio_baseline import prepare_environment
    from contactworld_tactile_base import tactile_field_to_base
    plan = json.loads((args.output / 'plan.json').read_text())
    task_file = args.source / 'thirdparty/manifeel-isaacgymenvs/isaacgymenvs/tasks/tacsl/tacsl_task_usb.py'
    checks = {'experiment_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'simulator_task_sha256': hashlib.sha256(task_file.read_bytes()).hexdigest()}
    if plan.get('version') != VERSION or any(plan.get(k) != v for k,v in checks.items()):
        raise ValueError('Experiment code or simulator source changed since prepare; choose a fresh output')
    torch.set_num_threads(4)
    sim_args = argparse.Namespace(source=args.source, output=args.output / 'sim', task='usb',
                                  seed_base=20261008, dual_tactile=True)
    env = prepare_environment(sim_args, torch, np)
    env.reset_idx(torch.arange(1, device=env.device, dtype=torch.long))
    ps = np.asarray(env.cfg_task.rl.pos_action_scale, np.float32)
    rs = np.asarray(env.cfg_task.rl.rot_action_scale, np.float32)
    records = []
    try:
        for item in plan['selected']:
            source, t = item['source'], item['frame']
            folder = Path(plan['capture_root']) / f'insertion_usb_episode_{source:03d}'
            demo_file = Path(json.loads((folder / 'replay.json').read_text())['demo'])
            with np.load(demo_file) as z:
                demo = {k: z[k].copy() for k in ('dof_pos', 'dof_vel', 'plug_pos', 'plug_quat',
                       'socket_quat', 'socket_pos_gt', 'ee_pos', 'ee_quat', 'action')}
            capture = load_capture(folder)
            original_local, _ = load_forces(folder)
            cases = args.output / 'branches' / f'source_{source:03d}_frame_{t:03d}'
            cases.mkdir(parents=True, exist_ok=True)
            repeated_start = []
            repeated_force = []
            required_holds = min(2, plan['repeats'])
            for kind in BRANCHES:
                if kind != 'hold' and len(repeated_start) < required_holds:
                    print('skip_state', source, t, 'fewer than two stable holds', flush=True)
                    break
                for repeat in range(plan['max_attempts']):
                    already = sum(r.get('source') == source and r.get('frame') == t
                                  and r.get('kind') == kind and r.get('accepted') for r in records)
                    if already >= plan['repeats']:
                        break
                    target = cases / f'{kind}_repeat_{repeat}.npz'
                    meta = cases / f'{kind}_repeat_{repeat}.json'
                    if target.exists() and meta.exists():
                        old = json.loads(meta.read_text())
                        if kind == 'hold' and old.get('accepted'):
                            with np.load(target) as saved:
                                repeated_start.append(np.r_[saved['plug_pos'][0], saved['plug_quat'][0],
                                    saved['ee_pos'][0],
                                    saved['ee_quat'][0] if 'ee_quat' in saved.files else capture['ee_pose'][t,3:],
                                    saved['socket_pos'][0]])
                                repeated_force.append(saved['force_local'][0].copy())
                        records.append(old); continue
                    np.random.seed(20261008); torch.manual_seed(20261008)
                    init = {k: demo[k][0:1] for k in ('dof_pos', 'dof_vel', 'plug_pos', 'plug_quat', 'socket_quat')}
                    init['socket_pos'] = demo['socket_pos_gt'][0:1].copy()
                    env.ctrl_target_dof_pos.copy_(torch.as_tensor(init['dof_pos'], device=env.device))
                    env._reset_franka_actuation(env.ctrl_target_dof_pos)
                    env.reset_to_dataset_state(**init)
                    env._reset_franka_actuation(env.ctrl_target_dof_pos)
                    env.reset_buf.zero_()
                    env.progress_buf.zero_()
                    force_history = []
                    action_history = []
                    for j in range(t):
                        pre = snapshot(env)
                        force_history.append(np.stack([pre[f'tactile_force_field_{s}'] for s in ('left', 'right')]))
                        act = pose_feedback_action(demo, j, pre, ps, rs)
                        action_history.append(act.copy())
                        env.step(torch.tensor(act[None], device=env.device))
                    pre = snapshot(env)
                    force_history.append(np.stack([pre[f'tactile_force_field_{s}'] for s in ('left', 'right')]))
                    plug_error = float(np.linalg.norm(pre['plug_pos'] - capture['plug_pose'][t, :3]) * 1000)
                    ee_error = float(np.linalg.norm(pre['ee_pos'] - capture['ee_pose'][t, :3]) * 1000)
                    start = np.r_[pre['plug_pos'], pre['plug_quat'], pre['ee_pos'],
                                   pre['ee_quat'], pre['socket_pos_gt']]
                    live_force = np.stack([pre[f'tactile_force_field_{s}'] for s in ('left','right')])
                    force_mae = float(np.abs(live_force - original_local[t]).mean())
                    force_scale = max(float(np.abs(original_local[t]).mean()), 1e-6)
                    # The first replay after switching source episodes can settle to
                    # a different contact manifold than later deterministic repeats.
                    # Optional warmup discards that first prefix before choosing a
                    # matched hold reference; historical callers keep their default.
                    warmup = bool(getattr(args, 'skip_first_replay', False) and
                                  kind == 'hold' and repeat == 0)
                    # Accepted hold starts form the reference cluster below.
                    position_spread = rotation_spread = force_spread = None
                    hold_progress_range = hold_slip_range = None
                    # Check against the audited capture, not just repeatability.
                    accepted = (not warmup and plug_error <= plan['max_replay_error_mm'] and ee_error <= 1.5
                                and force_mae <= max(1e-4, getattr(args, 'tactile_mae_ratio', .5) * force_scale))
                    if kind == 'hold' and accepted and getattr(args, 'cluster_holds', False) and repeated_start:
                        reference = repeated_start[0]
                        position_spread = max(float(np.linalg.norm(start[i:j] - reference[i:j]))
                                              for i, j in ((0, 3), (7, 10), (14, 17))) * 1000
                        rotation_spread = max(float(np.degrees(2 * np.arccos(np.clip(
                            abs(np.dot(start[i:j], reference[i:j])), 0, 1))))
                            for i, j in ((3, 7), (10, 14)))
                        force_spread = float(np.abs(live_force - repeated_force[0]).mean())
                        accepted = (position_spread <= .5 and rotation_spread <= 1.0 and
                                    force_spread <= max(7e-5, .3 * force_scale))
                    if kind == 'hold' and accepted:
                        repeated_start.append(start)
                        repeated_force.append(live_force)
                    if kind != 'hold':
                        accepted = accepted and len(repeated_start) >= required_holds
                    if kind != 'hold' and len(repeated_start) >= required_holds:
                        refs = repeated_start + [start]
                        position_spread = max(max(float(np.linalg.norm(a[i:j]-repeated_start[0][i:j]))
                                                  for i,j in ((0,3),(7,10),(14,17))) for a in refs) * 1000
                        rotation_spread = max(max(float(np.degrees(2*np.arccos(np.clip(
                                              abs(np.dot(a[i:j],repeated_start[0][i:j])),0,1))))
                                              for i,j in ((3,7),(10,14))) for a in refs)
                        force_spread = max(float(np.abs(f - repeated_force[0]).mean())
                                           for f in repeated_force + [live_force])
                        accepted = (accepted and position_spread <= .5 and rotation_spread <= 1.0
                                    and force_spread <= max(7e-5, .3 * force_scale))
                        holds = [r for r in records if r.get('source') == source and r.get('frame') == t
                                 and r.get('kind') == 'hold' and r.get('accepted')]
                        accepted = accepted and len(holds) >= required_holds
                        if len(holds) >= required_holds:
                            hold_progress_range = np.ptp([r['progress_mm'] for r in holds])
                            hold_slip_range = np.ptp([r['slip_mm'] for r in holds])
                            accepted = accepted and hold_progress_range <= .5 and hold_slip_range <= .5
                    if not accepted:
                        record = dict(source=source, frame=t, category=item['category'], kind=kind,
                                      repeat=repeat, accepted=False, plug_error_mm=plug_error,
                                      ee_error_mm=ee_error, tactile_mae=force_mae,
                                      tactile_limit=max(1e-4, getattr(args, 'tactile_mae_ratio', .5) * force_scale),
                                      position_spread_mm=position_spread, rotation_spread_deg=rotation_spread,
                                      force_spread=force_spread, force_spread_limit=max(7e-5, .3 * force_scale),
                                      hold_progress_range_mm=hold_progress_range, hold_slip_range_mm=hold_slip_range,
                                      reason='warmup_run' if warmup else 'prefix mismatch or repeatability gate')
                        write_json(meta, record); records.append(record)
                        print('skip', source, t, kind, repeat, record['reason'], flush=True)
                        continue
                    actions = branch_actions(kind, pre['socket_quat'], ps, plan['shift_mm'], plan['down_mm'], rs)
                    series = {k: [] for k in ('plug_pos', 'ee_pos', 'ee_quat', 'socket_pos', 'plug_quat', 'socket_quat',
                                              'force_local', 'force_base', 'contact_force', 'success', 'rgb')}
                    def append(obs):
                        local = np.stack([obs[f'tactile_force_field_{s}'] for s in ('left', 'right')])
                        force_base = []
                        bp = env.franka_base_pos[0].detach().cpu().numpy().copy()
                        bq = env.franka_base_quat[0].detach().cpu().numpy().copy()
                        for s, grid in zip(('left', 'right'), local):
                            key = f'tactile_force_field_{s}'
                            pos = env.tactile_pos_world_dict[key][0].detach().cpu().numpy().copy()
                            quat = env.tactile_quat_world_dict[key][0].detach().cpu().numpy().copy()
                            _, f = tactile_field_to_base(grid, pos, quat, bp, bq)
                            force_base.append(f)
                        series['plug_pos'].append(obs['plug_pos']); series['ee_pos'].append(obs['ee_pos'])
                        series['ee_quat'].append(obs['ee_quat'])
                        series['socket_pos'].append(obs['socket_pos_gt']); series['plug_quat'].append(obs['plug_quat'])
                        series['socket_quat'].append(obs['socket_quat'])
                        series['force_local'].append(local); series['force_base'].append(np.stack(force_base))
                        series['contact_force'].append(env.contact_force_pairwise[0, env.plug_body_id_env, env.socket_body_id_env].detach().cpu().numpy().copy())
                        series['success'].append(bool(env._check_success()[0].item()))
                        series['rgb'].append(np.uint8(np.clip(np.concatenate((obs['front'], obs['wrist']), 1) * 255, 0, 255)))
                    append(pre)
                    for a in actions:
                        env.step(torch.tensor(a[None], device=env.device))
                        append(snapshot(env))
                    arrays = {k: np.asarray(v) for k, v in series.items()}
                    arrays['action'] = np.stack(actions)
                    arrays['force_history_local'] = np.asarray(force_history[-4:])
                    arrays['action_history'] = np.asarray(action_history[-3:])
                    # Saved capture cloud corresponds to the same pre-action frame;
                    # the prefix gate above limits pose disagreement.
                    arrays['cloud_fused_xyz'] = capture['cloud_fused_xyz'][t]
                    arrays['state'] = np.r_[pre['dof_pos'], pre['dof_vel']].astype(np.float32)
                    np.savez_compressed(target, **arrays)
                    video_path = cases / f'{kind}.mp4'
                    if not video_path.exists():
                        rgb_frames = arrays['rgb']
                        base_maps = arrays['force_base'].reshape(len(rgb_frames), 2, 10, 14, 3)
                        color_scale = max(float(np.quantile(np.abs(base_maps), .99)), 1e-6)
                        frames = np.full((len(rgb_frames), 456, 512, 3), 255, np.uint8)
                        frames[:, :256] = rgb_frames
                        # Rows: left/right finger. Columns: base Fx/Fy/Fz.
                        # Fixed color scale within each clip preserves timing.
                        for frame_i in range(len(rgb_frames)):
                            for hand in range(2):
                                for axis in range(3):
                                    value = np.clip(base_maps[frame_i, hand, :, :, axis] / color_scale, -1, 1)
                                    red = np.where(value >= 0, 255, 255 * (1 + value))
                                    blue = np.where(value <= 0, 255, 255 * (1 - value))
                                    green = 255 * (1 - np.abs(value))
                                    colors = np.stack((red, green, blue), -1).astype(np.uint8)
                                    patch = np.repeat(np.repeat(colors, 10, axis=0), 12, axis=1)
                                    frames[frame_i, 256+hand*100:356+hand*100, 4+axis*168:172+axis*168] = patch
                        proc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
                            '-pix_fmt', 'rgb24', '-s', '512x456', '-r', '15',
                            '-i', '-', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                            '-movflags', '+faststart', str(video_path)], stdin=subprocess.PIPE)
                        proc.communicate(frames.tobytes())
                        if proc.returncode:
                            raise RuntimeError(f'ffmpeg failed: {video_path}')
                    rel = socket_relative(np.c_[arrays['plug_pos'], arrays['plug_quat']],
                                          np.c_[arrays['socket_pos'], arrays['socket_quat']])
                    initial_z, min_z = float(rel[0, 2]), float(rel[:, 2].min())
                    progress = initial_z - min_z
                    grasp = arrays['plug_pos'] - arrays['ee_pos']
                    slip = float(np.max(np.linalg.norm(grasp - grasp[0], axis=1)) * 1000)
                    record = dict(source=source, frame=t, category=item['category'], kind=kind,
                                  repeat=repeat, accepted=True, plug_error_mm=plug_error,
                                  ee_error_mm=ee_error, tactile_mae=force_mae, progress_mm=progress,
                                  xy_initial_mm=float(np.linalg.norm(rel[0, :2])),
                                  xy_final_mm=float(np.linalg.norm(rel[-1, :2])),
                                  slip_mm=slip, success_ever=bool(arrays['success'].any()),
                                  max_contact_force=float(np.linalg.norm(arrays['contact_force'], axis=-1).max()),
                                  file=str(target), video=str(video_path) if video_path.exists() else None)
                    write_json(meta, record); records.append(record)
                    print('branch', source, t, kind, repeat, 'progress', round(progress, 2), 'slip', round(slip, 2), flush=True)
            write_json(args.output / 'branch_progress.json', records)
    finally:
        env.gym.destroy_sim(env.sim)
    write_json(args.output / 'branch_progress.json', records)


def probe_features(case, baseline):
    f = case['force_base'][0:1]
    f0 = tactile_features(f)[0]
    local = case['force_local'][0]
    current = local.reshape(1, 2, 140, 3)
    delta = (local - baseline).reshape(1, 2, 140, 3) if baseline is not None else np.zeros_like(current)
    hist = case['force_history_local']
    change = (hist[-1] - hist[0]).reshape(1, 2, 140, 3)
    # Tactile comparisons use same representation; no true poses/contacts.
    tac = np.r_[f0, tactile_features(delta)[0], tactile_features(change)[0]]
    rgb = case['rgb'][0]
    # Small grid from both views preserves object location; no GT crop.
    import cv2
    vision = cv2.resize(rgb, (32, 16), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1) / 255
    cloud = case['cloud_fused_xyz']
    finite = np.isfinite(cloud).all(1)
    cloud = cloud[finite]
    if len(cloud):
        # Geometry bins in robot base, with absolute coordinates retained.
        lo, hi = np.quantile(cloud, [.05, .95], axis=0)
        voxel = np.histogramdd(cloud, bins=(6, 6, 4),
                     range=((-0.2, 0.7), (-0.35, 0.35), (0.0, 0.3)))[0]
        geo = np.r_[cloud.mean(0), cloud.std(0), lo, hi, (voxel / len(cloud)).reshape(-1)]
    else: geo = np.zeros(156)
    state = case['state']
    ah = case['action_history'].reshape(-1)
    return np.r_[state, ah, geo, vision], tac


def ridge_predict(train_x, train_y, test_x, penalty=20.):
    mu = train_x.mean(0); sd = train_x.std(0)
    sd = np.where(sd < 1e-5, 1, sd)
    x = np.clip((train_x - mu) / sd, -5, 5)
    q = np.clip((test_x - mu) / sd, -5, 5)
    ym = train_y.mean(0)
    # Dual ridge works for width >> sample count.
    alpha = np.linalg.solve(x @ x.T + penalty * np.eye(len(x)), train_y - ym)
    return ym + q @ x.T @ alpha


def baseline_diagnostics(args):
    """Episode-held-out diagnostic; no physics truth enters the feature vector."""
    manifest = json.loads((args.dataset / 'spatial/manifest.json').read_text())
    root = Path(manifest['capture_root'])
    all_data = {s: [] for s in ('train', 'val', 'test')}
    for entry in manifest['episodes']:
        folder = root / f"insertion_usb_episode_{entry['source_episode']:03d}"
        capture = load_capture(folder); local, _ = load_forces(folder)
        contact = source_contact(folder, len(local))
        baseline = stable_baseline(local, contact, capture['action_executed'])
        if baseline is None: continue
        start = max(baseline['stop'], 3)
        force = local.reshape(len(local), 2, 140, 3)
        bg = baseline['local'].reshape(1, 2, 140, 3)
        residual = force - bg
        change = force - np.roll(force, 3, axis=0)
        rel = socket_relative(capture['plug_pose'], capture['socket_pose'])
        speed = np.r_[np.zeros((1, 3)), np.diff(rel, axis=0)]
        previous_down = np.r_[False, capture['action_executed'][:-1, 2] < -.2]
        eligible = contact & previous_down & (rel[:, 2] > 3)
        blocked = eligible & (speed[:, 2] > -.25)
        progressed = eligible & (speed[:, 2] < -.4)
        keep = np.arange(start, len(local) - 1)
        blocks = dict(raw=tactile_features(force[keep]),
                      residual=tactile_features(residual[keep]),
                      residual_baseline=np.c_[tactile_features(residual[keep]),
                                              tactile_features(np.broadcast_to(bg, force[keep].shape))],
                      residual_history=np.c_[tactile_features(residual[keep]), tactile_features(change[keep])])
        all_data[entry['split']].append(dict(source=int(entry['source_episode']), blocks=blocks,
                     contact=contact[keep].astype(np.float32),
                     blocked=blocked[keep].astype(np.float32),
                     blocked_valid=(blocked | progressed)[keep]))
    def joined(split, mode, target):
        items=all_data[split]
        x=np.concatenate([i['blocks'][mode] for i in items])
        y=np.concatenate([i[target] for i in items])
        valid=np.concatenate([i['blocked_valid'] if target=='blocked' else np.ones(len(i[target]),bool) for i in items])
        return x[valid],y[valid]
    def fit(x,y,query,penalty=50.):
        mu=x.mean(0);sd=x.std(0);sd=np.where(sd<1e-6,1,sd)
        xx=np.clip((x-mu)/sd,-5,5);qq=np.clip((query-mu)/sd,-5,5)
        ym=y.mean();a=np.linalg.solve(xx.T@xx+penalty*np.eye(xx.shape[1]),xx.T@(y-ym))
        return ym+qq@a
    def balanced(y,p,threshold):
        guess=p>=threshold;positive=y>.5
        if positive.sum()==0 or (~positive).sum()==0:return float('nan')
        return float(.5*(guess[positive].mean()+(~guess[~positive]).mean()))
    rows=[]
    for target in ('contact','blocked'):
        for mode in ('raw','residual','residual_baseline','residual_history'):
            train_x,train_y=joined('train',mode,target)
            val_x,val_y=joined('val',mode,target)
            test_x,test_y=joined('test',mode,target)
            if len(np.unique(train_y))<2 or len(val_x)==0 or len(test_x)==0:
                rows.append(dict(target=target,mode=mode,status='insufficient labels'));continue
            pv=fit(train_x,train_y,val_x);pt=fit(train_x,train_y,test_x)
            thresholds=np.quantile(pv,np.linspace(.05,.95,19))
            threshold=max(thresholds,key=lambda x:balanced(val_y,pv,x))
            rows.append(dict(target=target,mode=mode,train_frames=len(train_y),test_frames=len(test_y),
                             test_positive=int(test_y.sum()),validation_balanced_accuracy=balanced(val_y,pv,threshold),
                             test_balanced_accuracy=balanced(test_y,pt,threshold),
                             threshold=float(threshold),test_episodes=len(all_data['test'])))
    return dict(rows=rows,baselines_by_split={s:len(v) for s,v in all_data.items()},
                note='All tests hold out complete source demonstrations. Contact/blocked labels use simulator physics and executed motion, never sensor predictions. This is a light linear probe, not a proof of information absence.')

def analyze(args):
    plan = json.loads((args.output / 'plan.json').read_text())
    progress = json.loads((args.output / 'branch_progress.json').read_text())
    accepted = [r for r in progress if r['accepted']]
    baselines = np.load(args.output / 'baselines_local.npz')
    all_groups = sorted({(r['source'], r['frame']) for r in accepted})
    complete_groups = []
    for group in all_groups:
        local_rows = [r for r in accepted if (r['source'], r['frame']) == group]
        if all(sum(r['kind'] == kind for r in local_rows) >= 2 for kind in BRANCHES):
            complete_groups.append(group)
    # Ground truth outcome is physically observed after the candidate action;
    # this does not certify untried actions or a globally optimal correction.
    samples = []
    for r in accepted:
        with np.load(r['file']) as z:
            case = {k: z[k].copy() for k in z.files}
        baseline = baselines[f"source_{r['source']:03d}"] if f"source_{r['source']:03d}" in baselines else None
        core, tactile = probe_features(case, baseline)
        action = np.zeros(len(BRANCHES)); action[BRANCHES.index(r['kind'])] = 1
        samples.append(dict(group=(r['source'], r['frame']), label=r['category'], kind=r['kind'],
                            core=np.r_[core, action], tactile=tactile,
                            y=np.array([r['progress_mm'], r['slip_mm']], np.float32), row=r))
    scores = []
    samples = [s for s in samples if s['group'] in complete_groups]
    if len(complete_groups) >= 6:
        for mode in ('core', 'core_tactile', 'core_tactile_history'):
            true, pred = [], []
            for group in complete_groups:
                train = [s for s in samples if s['group'] != group]
                test = [s for s in samples if s['group'] == group]
                if not train or not test: continue
                def vec(s):
                    if mode == 'core': return s['core']
                    if mode == 'core_tactile': return np.r_[s['core'], s['tactile'][:36]]
                    return np.r_[s['core'], s['tactile']]
                x = np.stack([vec(s) for s in train]); q = np.stack([vec(s) for s in test])
                y = np.stack([s['y'] for s in train])
                yp = ridge_predict(x, y, q)
                true.extend([s['y'] for s in test]); pred.extend(yp)
            true = np.asarray(true); pred = np.asarray(pred)
            scores.append(dict(mode=mode, groups=len(complete_groups), samples=len(true),
                               progress_mae_mm=float(np.mean(np.abs(true[:, 0] - pred[:, 0]))),
                               slip_mae_mm=float(np.mean(np.abs(true[:, 1] - pred[:, 1])))))
    by_group = []
    for source, frame in all_groups:
        rows = [r for r in accepted if (r['source'], r['frame']) == (source, frame)]
        by_kind = []
        for kind in BRANCHES:
            rr = [r for r in rows if r['kind'] == kind]
            if rr:
                by_kind.append(dict(kind=kind, repeats=len(rr),
                                    progress_mean_mm=float(np.mean([r['progress_mm'] for r in rr])),
                                    progress_std_mm=float(np.std([r['progress_mm'] for r in rr])),
                                    slip_mean_mm=float(np.mean([r['slip_mm'] for r in rr])),
                                    example=rr[0]['file'], video=rr[0].get('video')))
        by_group.append(dict(source=source, frame=frame, category=rows[0]['category'], branches=by_kind))
    baseline_diag = baseline_diagnostics(args)
    report = dict(version=VERSION, plan=plan, baseline_diagnostic=baseline_diag, accepted=len(accepted), rejected=len(progress)-len(accepted),
                  groups=len(all_groups), complete_groups=len(complete_groups),
                  complete_group_ids=[list(x) for x in complete_groups],
                  grouped_prediction=scores, branch_groups=by_group,
                  interpretation='Grouped modality scores require at least six complete paired state groups; even then they are exploratory, not policy success or causal modality ablation.')
    write_json(args.output / 'analysis.json', report)
    render(args, report)


def render(args, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    assets = args.output / 'assets'; assets.mkdir(exist_ok=True)
    pieces = ['<html><head><meta charset="utf-8"><style>body{font:16px sans-serif;max-width:1400px;margin:auto;background:#111827;color:#e5e7eb;padding:25px}table{border-collapse:collapse}td,th{border:1px solid #64748b;padding:7px}img{max-width:100%}section{border:1px solid #475569;padding:16px;margin:20px 0}a{color:#93c5fd}</style></head><body>',
              '<h1>ContactWorld 双指触觉与同状态动作分支</h1>',
              '<p>每个分支先重放同一示范前缀。源视频、点云和触觉图均为分支前观察；进展与滑移由实际短动作试验测量。结论先看每个状态的重复一致性，再看跨状态预测。</p>',
              f'<p>有效分支 {report["accepted"]}，被门槛排除 {report["rejected"]}，至少一个有效动作的初态组 {report["groups"]}，七动作均至少两次的完整初态组 {report["complete_groups"]}；缺少类别：{html.escape(str(report["plan"]["missing_categories"]))}</p>',
              '<h2>留出整个初态的轻量预测</h2><table><tr><th>输入</th><th>进展 MAE mm</th><th>滑移 MAE mm</th></tr>']
    if not report['grouped_prediction']:
        pieces.append('<tr><td colspan="3">完整初态不足 6 组：本轮不输出跨状态模态分数。</td></tr>')
    for x in report['grouped_prediction']:
        pieces.append(f'<tr><td>{x["mode"]}</td><td>{x["progress_mae_mm"]:.3f}</td><td>{x["slip_mae_mm"]:.3f}</td></tr>')
    pieces.append('</table><p>core 含双视图降采样、机器人状态、机械臂 base 下固定体素占据与几何统计、已执行动作历史和候选动作；触觉分别加入当前力、夹持基线差与短时变化。只有约六个初态，分数仅用于寻找进一步实验方向。</p>')
    pieces.append('<h2>夹持基线与触觉变化：完整示范留出</h2><table><tr><th>目标</th><th>表示</th><th>验证平衡准确率</th><th>测试平衡准确率</th><th>测试正例/帧</th></tr>')
    for row in report['baseline_diagnostic']['rows']:
        if 'status' in row: continue
        pieces.append(f'<tr><td>{row["target"]}</td><td>{row["mode"]}</td><td>{row["validation_balanced_accuracy"]:.3f}</td><td>{row["test_balanced_accuracy"]:.3f}</td><td>{row["test_positive"]}/{row["test_frames"]}</td></tr>')
    pieces.append('</table><p>raw=双指原始局部三通道；residual=减去同一指的稳定夹持基线；residual_baseline=两者同时输入；residual_history=再加入最近三步变化。仅用已接触、上一动作下推且实际运动明确的帧区分受阻和推进。指标按轨迹划分，相关帧不能当作独立样本。</p>')
    pieces.append('<h2>分支点目录</h2><ul>')
    for group in report['branch_groups']:
        pieces.append(f'<li><a href="#s{group["source"]:03d}f{group["frame"]:03d}">{html.escape(group["category"])} · source {group["source"]} frame {group["frame"]}</a></li>')
    pieces.append('</ul>')
    for group in report['branch_groups']:
        source, frame = group['source'], group['frame']
        first = group['branches'][0]['example']
        with np.load(first) as z:
            rgb=z['rgb'][0]; force=z['force_base'][0]; cloud=z['cloud_fused_xyz'].copy()
        from PIL import Image
        Image.fromarray(rgb).save(assets/f's{source:03d}_f{frame:03d}_rgb.png')
        finite=np.isfinite(cloud).all(1); cloud=cloud[finite][::3]
        fig_cloud,ax_cloud=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for a,ii,jj,title in [(ax_cloud[0],0,1,'base XY'),(ax_cloud[1],0,2,'base XZ')]:
            a.scatter(cloud[:,ii],cloud[:,jj],c=cloud[:,2],s=2,alpha=.6,cmap='viridis')
            a.set_xlabel(['X','Y','Z'][ii]+' m');a.set_ylabel(['X','Y','Z'][jj]+' m');a.set_title(title);a.set_aspect('equal',adjustable='box')
        fig_cloud.savefig(assets/f's{source:03d}_f{frame:03d}_cloud.png',dpi=125);plt.close(fig_cloud)
        fig, ax = plt.subplots(2, 3, figsize=(10, 5), layout='constrained')
        lim = max(float(np.quantile(np.abs(force), .99)), 1e-7)
        for hand in range(2):
            for axis in range(3):
                ax[hand,axis].imshow(force[hand,:,axis].reshape(10,14),cmap='coolwarm',vmin=-lim,vmax=lim)
                ax[hand,axis].set_title(f'{["left","right"][hand]} F{["x","y","z"][axis]}')
        fig.savefig(assets/f's{source:03d}_f{frame:03d}_force.png',dpi=125);plt.close(fig)
        key=f'source_{source:03d}'
        with np.load(args.output/'baselines_local.npz') as baseline_file:
            baseline=baseline_file[key] if key in baseline_file else None
        if baseline is not None:
            with np.load(first) as z: current_local=z['force_local'][0]; hist=z['force_history_local']
            fig2,ax2=plt.subplots(2,3,figsize=(10,5),layout='constrained')
            delta=current_local-baseline
            clim=max(float(np.quantile(np.abs(delta),.99)),1e-7)
            for hand in range(2):
                for axis in range(3):
                    ax2[hand,axis].imshow(delta[hand,:,:,axis],cmap='coolwarm',vmin=-clim,vmax=clim)
                    ax2[hand,axis].set_title(f'{["left","right"][hand]} Δ local {axis}')
            fig2.savefig(assets/f's{source:03d}_f{frame:03d}_delta.png',dpi=125);plt.close(fig2)
        pieces.append(f'<section id="s{source:03d}f{frame:03d}"><h2>{html.escape(group["category"])} · source {source} frame {frame}</h2>')
        pieces.append(f'<img src="assets/s{source:03d}_f{frame:03d}_rgb.png"><img src="assets/s{source:03d}_f{frame:03d}_cloud.png"><img src="assets/s{source:03d}_f{frame:03d}_force.png">')
        if baseline is not None: pieces.append(f'<p>相对稳定夹持基线的局部力变化：</p><img src="assets/s{source:03d}_f{frame:03d}_delta.png">')
        pieces.append('<table><tr><th>动作分支</th><th>进展均值±波动 mm</th><th>最大滑移均值 mm</th><th>重复</th></tr>')
        for branch in group['branches']:
            pieces.append(f'<tr><td>{branch["kind"]}</td><td>{branch["progress_mean_mm"]:.2f} ± {branch["progress_std_mm"]:.2f}</td><td>{branch["slip_mean_mm"]:.2f}</td><td>{branch["repeats"]}</td></tr>')
        fig_time,ax_time=plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
        for branch in group['branches']:
            with np.load(branch['example']) as zz:
                pp=zz['plug_pos'].copy();ss=zz['socket_pos'].copy();qq=zz['socket_quat'].copy()
                rel=socket_relative(np.c_[pp,zz['plug_quat']],np.c_[ss,qq])
                force_curve=np.abs(zz['force_base']).mean((2,3))
                contact_curve=np.linalg.norm(zz['contact_force'],axis=-1)
            ax_time[0].plot(np.arange(len(rel)),rel[0,2]-rel[:,2],label=branch['kind'])
            ax_time[1].plot(np.arange(len(force_curve)),force_curve[:,0],label=branch['kind'])
            ax_time[2].plot(np.arange(len(contact_curve)),contact_curve,label=branch['kind'])
        for a,title,ylabel in [(ax_time[0],'USB axial progress','mm'),(ax_time[1],'Left pad mean |force|','sim units'),(ax_time[2],'USB-socket contact','sim units')]:
            a.set_title(title);a.set_ylabel(ylabel);a.set_xlabel('control step')
        ax_time[0].legend(fontsize=7,ncol=2)
        fig_time.savefig(assets/f's{source:03d}_f{frame:03d}_timeline.png',dpi=125);plt.close(fig_time)
        pieces.append(f'</table><img src="assets/s{source:03d}_f{frame:03d}_timeline.png"><div style="display:grid;grid-template-columns:repeat(2,1fr);gap:10px">')
        for branch in group['branches']:
            if branch.get('video'):
                relpath = Path(branch['video']).relative_to(args.output)
                pieces.append(f'<div><b>{html.escape(branch["kind"])}</b><video controls preload="metadata" width="100%" src="{html.escape(str(relpath))}"></video></div>')
        pieces.append('</div></section>')
    pieces.append('<p>图中双指力为机械臂 base 坐标。分支动作沿插座局部 ±X/±Y/Z 执行；触觉可能主要反映夹持及先前动作，不能仅凭相关性断言它决定纠偏方向。<a href="analysis.json">完整机器可读结果</a>。</p></body></html>')
    (args.output / 'index.html').write_text('\n'.join(pieces))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('prepare','branch','analyze'),required=True)
    p.add_argument('--dataset',type=Path,default=DATA)
    p.add_argument('--source',type=Path,default=SOURCE)
    p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--max-attempts',type=int,default=5,help='Repeat attempts per branch to recover transient replay mismatches')
    p.add_argument('--per-category',type=int,default=4,help='Candidate states per category; default 24 total')
    p.add_argument('--shift-mm',type=float,default=2.)
    p.add_argument('--down-mm',type=float,default=6.)
    p.add_argument('--max-replay-error-mm',type=float,default=1.0)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if args.repeats < 1 or args.max_attempts < args.repeats or args.per_category < 1 or args.shift_mm <= 0 or args.down_mm <= 0:
        p.error('repeats/per-category/shift-mm/down-mm must be positive')
    if args.stage=='prepare':prepare(args)
    elif args.stage=='branch':branch(args)
    else:analyze(args)


if __name__=='__main__':main()
