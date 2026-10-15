#!/usr/bin/env python3
"""Paired hold/down interventions from frozen ContactWorld K=4 tactile states.

Prepare selects states from the reviewed K=4 assignments without consulting future
outcomes. Branch reuses the existing replay and sensor gates. Analyze compares
paired trajectories and preserves individual videos, maps, and failure reasons.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

import run_contactworld_tactile_action_branches_v3 as replay

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ATLAS = Path('/home/sai/zx/openpi-sim-runtime/runs/contactworld_contact_subclusters_20261009')
DEFAULT_OUTPUT = Path('/home/sai/zx/openpi-sim-runtime/runs/contactworld_k4_down_probe_20261009')
QUOTAS = {0: 8, 1: 8, 2: 8, 3: 16}
PILOT = {0: 2, 1: 2, 2: 2, 3: 6}
VERSION = 'cw-k4-paired-down-v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def frozen_model(atlas):
    with np.load(atlas / 'alternative_k4_model.npz') as z:
        centers, scale = z['centers'].copy(), z['scale'].copy()
    if centers.shape != (4, 840) or scale.shape != (2, 3):
        raise ValueError('Unexpected K4 center/scale shapes')
    return centers, scale


def parent_model():
    path = Path('/home/sai/zx/openpi-sim-runtime/runs/contactworld_tactile_clusters_k14_20261008/cluster_model.npz')
    with np.load(path) as z:
        return z['centers'].copy(), z['scale'].copy(), path


def k4_label(force, centers, scale):
    from run_contactworld_tactile_cluster_atlas import assignment
    normalized = np.clip(force.reshape(1, 2, 140, 3) / scale[None, :, None, :], -3, 3)
    labels, distances = assignment(normalized.reshape(1, 840).astype(np.float32), centers)
    return int(labels[0]), float(distances[0])


def source_candidates(args):
    manifest = json.loads((args.dataset / 'spatial/manifest.json').read_text())
    lookup = {(e['split'], int(e['source_episode'])): e for e in manifest['episodes']}
    rows = json.loads((args.atlas / 'alternative_assignments.json').read_text())
    rows = [x for x in rows if x['k'] == 4]
    centers, scale = frozen_model(args.atlas)
    parent_centers, parent_scale, parent_path = parent_model()
    if parent_centers.shape != (14, 840) or not np.allclose(parent_scale, scale):
        raise ValueError('Original K14 and K4 scales differ')
    by_source = defaultdict(list)
    for row in rows:
        by_source[(row['split'], int(row['source']))].append(row)
    candidates, counts = [], defaultdict(int)
    for (split, source), group in sorted(by_source.items()):
        entry = lookup[(split, source)]
        folder = Path(manifest['capture_root']) / f'insertion_usb_episode_{source:03d}'
        capture = replay.load_capture(folder)
        local, base = replay.load_forces(folder)
        replay_rows = json.loads((folder / 'replay.json').read_text())['rows']
        contact = replay.source_contact(folder, len(local))
        baseline = replay.stable_baseline(local, contact, capture['action_executed'])
        relative = replay.socket_relative(capture['plug_pose'], capture['socket_pose'])
        motion = np.r_[np.zeros((1, 3)), np.diff(relative, axis=0)]
        prior_success = np.r_[False, np.maximum.accumulate([bool(r['source_success']) for r in replay_rows[:-1]])]
        if len(prior_success) != len(local):
            raise ValueError(f'Success length mismatch: {source}')
        for row in group:
            t = int(row['frame']); cat = int(row['subcluster']); counts[cat] += 1
            if not (8 <= t < len(local)) or (cat != 3 and prior_success[t]):
                continue
            pre_error = float(replay_rows[t]['plug_pre_error_mm'])
            if pre_error > args.max_replay_error_mm:
                continue
            assigned, distance = k4_label(base[t], centers, scale)
            if assigned != cat:
                raise ValueError(f'Frozen K4 mismatch source={source} frame={t}: {assigned} != {cat}')
            grasp = Rotation.from_quat(capture['ee_pose'][t, 3:]).inv().apply(
                capture['plug_pose'][t, :3] - capture['ee_pose'][t, :3]) * 1000
            candidates.append(dict(category=f'k4_{cat}', k4=cat, source=source,
                 episode=int(entry['episode_index']), split=split, frame=t,
                 relative_mm=relative[t].round(3).tolist(),
                 prior_motion_mm=motion[t].round(3).tolist(),
                 grasp_mm=grasp.round(3).tolist(), contact=bool(contact[t]),
                 baseline_found=bool(baseline and baseline['stop'] <= t),
                 recorded_replay_error_mm=pre_error, distance=distance,
                 prior_success=bool(prior_success[t]),
                 depth_band=('bottom' if relative[t, 2] <= 3 else
                             'inside' if relative[t, 2] <= 8 else 'mouth'),
                 near_bottom=bool(relative[t, 2] <= 3.0)))
    return candidates, dict(counts), manifest


def pick_states(candidates, quotas, smoke=False):
    # Pick one frame per source, using only pre-action geometry, grasp and
    # tactile distance. Outcomes and future demonstration actions are excluded.
    chosen, used = [], set()
    for cat in (3, 0, 1, 2):
        options = [c for c in candidates if c['k4'] == cat and c['source'] not in used]
        per_source = {}
        for c in options:
            key = (c['source'], c['depth_band']) if cat == 3 else c['source']
            rank = (not c['baseline_found'], c['recorded_replay_error_mm'],
                    c['near_bottom'], c['distance'])
            if key not in per_source or rank < per_source[key][0]:
                per_source[key] = (rank, c)
        pool = [x[1] for x in per_source.values()]
        class3_bands = ['mouth','mouth','inside','mouth','inside','mouth',
                        'inside','mouth','inside','mouth','inside','mouth',
                        'inside','mouth','bottom','bottom']
        for slot in range(quotas[cat]):
            if not pool:
                break
            band = (class3_bands[slot] if cat == 3 and slot < len(class3_bands)
                    else 'mouth' if cat != 3 and slot < PILOT[cat] else None)
            choices = [c for c in pool if c['depth_band'] == band] if band else pool
            if not choices:
                choices = pool
            if slot == 0:
                item = min(choices, key=lambda c: (c['near_bottom'], c['recorded_replay_error_mm'],
                                                    c['distance']))
            else:
                selected = [c for c in chosen if c['k4'] == cat]
                def feature(c):
                    r = np.asarray(c['relative_mm']); g = np.asarray(c['grasp_mm']); v = np.asarray(c['prior_motion_mm'])
                    return np.r_[r / [5, 5, 20], g / [20, 20, 20],
                                 v / [2, 2, 2], c['distance'], float(c['contact']),
                                 float(c['near_bottom'])]
                item = max(choices, key=lambda c: (min(np.linalg.norm(feature(c) - feature(p)) for p in selected),
                                                -c['recorded_replay_error_mm']))
            chosen.append(item); used.add(item['source'])
            pool = [c for c in pool if c['source'] != item['source']]
    if smoke:
        chosen = chosen[:1]
    return chosen


def prepare(args):
    candidates, original_counts, manifest = source_candidates(args)
    _, _, parent_path = parent_model()
    full = pick_states(candidates, QUOTAS, args.smoke)
    pilot = [c for cat in (3, 0, 1, 2) for c in [x for x in full if x['k4'] == cat][:PILOT[cat]]]
    if args.smoke:
        pilot = full
    availability = {str(k): sum(c['k4'] == k for c in full) for k in range(4)}
    if len(full) < (1 if args.smoke else 30) or (not args.smoke and any(availability[str(k)] < PILOT[k] for k in range(4))):
        raise ValueError(f'Insufficient distinct-source K4 states: {availability}')
    task_file = args.source / 'thirdparty/manifeel-isaacgymenvs/isaacgymenvs/tasks/tacsl/tacsl_task_usb.py'
    plan = dict(version=replay.VERSION, study_version=VERSION, smoke=bool(args.smoke), dataset=str(args.dataset),
                atlas=str(args.atlas), capture_root=manifest['capture_root'],
                dataset_manifest_sha256=sha(args.dataset / 'spatial/manifest.json'),
                frozen_k4_sha256=sha(args.atlas / 'alternative_k4_model.npz'),
                frozen_parent_sha256=sha(parent_path),
                assignments_sha256=sha(args.atlas / 'alternative_assignments.json'),
                experiment_script_sha256=sha(Path(replay.__file__)),
                study_script_sha256=sha(__file__), simulator_task_sha256=sha(task_file),
                selected=pilot, all_selected=full, original_class_counts=original_counts,
                eligible_counts={str(k): sum(c['k4'] == k for c in candidates) for k in range(4)},
                quotas=QUOTAS, pilot_quotas=PILOT, missing_categories=[],
                repeats=args.repeats, max_attempts=args.max_attempts, tactile_mae_ratio=args.tactile_mae_ratio, shift_mm=2.0,
                down_mm=args.down_mm, max_replay_error_mm=args.max_replay_error_mm,
                intervention='hold zero 11 steps; down socket -Z for 11 steps, total down_mm')
    # Normalize integer JSON keys before comparing a persisted plan on resume.
    plan = json.loads(json.dumps(plan, ensure_ascii=False))
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    path = out / 'plan.json'
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != plan:
            resumed_full = (existing.get('selected') == plan['all_selected'] and
                            {k: v for k, v in existing.items() if k != 'selected'} ==
                            {k: v for k, v in plan.items() if k != 'selected'})
            if not resumed_full:
                raise ValueError('Existing plan differs; use another --output directory')
            print('Resuming expanded full plan', flush=True)
        else:
            print('Resuming pilot plan', flush=True)
    else:
        replay.write_json(path, plan)
    print('K4 candidates', original_counts, 'eligible', plan['eligible_counts'],
          'selected', availability, 'pilot', len(pilot), flush=True)


def paired_actions(kind, socket_quat, pos_scale, shift_mm, down_mm):
    if kind == 'hold':
        return [np.zeros(6, np.float32) for _ in range(11)]
    if kind != 'down':
        raise ValueError(f'Unexpected action branch: {kind}')
    xyz = Rotation.from_quat(socket_quat).apply([0, 0, -down_mm / 11 / 1000])
    return [np.r_[xyz / pos_scale, np.zeros(3)].astype(np.float32) for _ in range(11)]


def branch(args):
    plan = json.loads((args.output / 'plan.json').read_text())
    if plan['study_version'] != VERSION or plan['study_script_sha256'] != sha(__file__):
        raise ValueError('Study script changed since prepare')
    if sha(args.atlas / 'alternative_k4_model.npz') != plan['frozen_k4_sha256']:
        raise ValueError('Frozen K4 model changed')
    if sha(args.atlas / 'alternative_assignments.json') != plan['assignments_sha256']:
        raise ValueError('Frozen K4 assignments changed')
    if sha(parent_model()[2]) != plan['frozen_parent_sha256']:
        raise ValueError('Frozen K14 parent model changed')
    replay.BRANCHES = ('hold', 'down')
    replay.branch_actions = paired_actions
    replay.branch(args)


def expand(args):
    path = args.output / 'plan.json'; plan = json.loads(path.read_text())
    if plan['study_version'] != VERSION or plan['study_script_sha256'] != sha(__file__):
        raise ValueError('Study script changed since prepare')
    if plan.get('smoke', False):
        print('smoke has no expansion', flush=True); return
    report = json.loads((args.output / 'analysis.json').read_text())
    if report.get('stage') != 'pilot':
        raise ValueError('Pilot analysis missing')
    complete = [r for r in report['states'] if r['paired']]
    per_class = {k: sum(r['k4'] == k for r in complete) for k in range(4)}
    if len(complete) < 4 or per_class[3] < 1:
        raise ValueError(f'Pilot replay gate failed: paired={len(complete)} class3={per_class[3]}; inspect pilot report')
    plan['selected'] = plan['all_selected']
    replay.write_json(path, plan)
    print('Pilot gate passed. Expanded to', len(plan['selected']), 'states', flush=True)


def pose_rotation_deg(plug_q, socket_q):
    relative = Rotation.from_quat(socket_q).inv() * Rotation.from_quat(plug_q)
    return np.degrees(relative.magnitude())


def score_state(rows, centers, scale, parent_centers, args):
    by_kind = {kind: sorted((r for r in rows if r['accepted'] and r['kind'] == kind),
                            key=lambda r: r['repeat']) for kind in ('hold', 'down')}
    kinds = {}
    for kind, accepted in by_kind.items():
        curves = []
        for row in accepted:
            with np.load(row['file']) as z:
                force0 = z['force_base'][0].copy()
                live_class, live_distance = k4_label(force0, centers, scale)
                live_parent, _ = k4_label(force0, parent_centers, scale)
                rel = replay.socket_relative(np.c_[z['plug_pos'], z['plug_quat']],
                                             np.c_[z['socket_pos'], z['socket_quat']])
                disp = rel[0, 2] - rel[:, 2]
                xy = np.linalg.norm(rel[:, :2], axis=1)
                plug_ee_local = Rotation.from_quat(z['ee_quat']).inv().apply(
                    z['plug_pos'] - z['ee_pos'])
                slip = np.linalg.norm(plug_ee_local - plug_ee_local[0], axis=1) * 1000
                angle = pose_rotation_deg(z['plug_quat'], z['socket_quat'])
                valid_axis = (xy[0] < 3 and xy[-1] < 3 and
                              abs(angle[-1] - angle[0]) < 3)
                curves.append(dict(repeat=int(row['repeat']), progress_mm=disp.tolist(),
                    xy_mm=xy.tolist(), rotation_deg=angle.tolist(), slip_mm=slip.tolist(),
                    contact=np.linalg.norm(z['contact_force'], axis=-1).tolist(),
                    force_abs=np.abs(z['force_base']).mean((1, 2, 3)).tolist(),
                    success=bool(z['success'].any()), live_k4=live_class,
                    live_parent_k14=live_parent,
                    live_distance=live_distance, near_axis=bool(valid_axis),
                    replay_error_mm=float(row['plug_error_mm']),
                    tactile_mae=float(row['tactile_mae']),
                    file=str(row['file']), video=row['video']))
        kinds[kind] = curves
    selected = rows[0]['selected']
    live_match = all(x['live_k4'] == selected['k4'] and x['live_parent_k14'] in (5, 7, 9)
                     for x in kinds['hold'] + kinds['down'])
    paired = len(kinds['hold']) >= args.repeats and len(kinds['down']) >= args.repeats and live_match
    result = dict(source=selected['source'], frame=selected['frame'], split=selected['split'],
                  k4=selected['k4'], selected=selected, accepted_counts={k: len(v) for k, v in kinds.items()},
                  rejected=[r for r in rows if not r['accepted']],
                  live_match=live_match, paired=paired, branches=kinds)
    if not paired:
        result['outcome'] = 'incomplete_or_reconstructed_class_mismatch'
        return result
    # Compare fixed action horizon from the same reconstructed state. Existing
    # nonzero velocity contributes to both branches and is removed by pairing.
    h = np.stack([c['progress_mm'] for c in kinds['hold']]).mean(0)
    d = np.stack([c['progress_mm'] for c in kinds['down']]).mean(0)
    delta = d - h
    spreads = [np.ptp([c['progress_mm'][-1] for c in kinds[k]]) for k in ('hold', 'down')]
    down = kinds['down']; headroom = float(selected['relative_mm'][2])
    slip = max(c['slip_mm'][-1] for c in down)
    near_axis = all(c['near_axis'] for c in down)
    reliable = max(spreads) <= .6
    # Thresholds describe a short mechanical response, not complete insertion.
    if headroom <= 3:
        outcome = 'near_bottom_requires_review'
    elif not reliable:
        outcome = 'variable_repeats'
    elif slip > .75:
        outcome = 'grasp_slip'
    elif not near_axis:
        outcome = 'off_axis_or_rotating'
    elif delta[-1] >= .5 and d[-1] >= .5:
        outcome = 'short_axial_advance'
    elif delta[-1] <= .2 and d[-1] <= .3:
        outcome = 'no_additional_advance'
    else:
        outcome = 'small_or_ambiguous_response'
    result.update(outcome=outcome, headroom_mm=headroom, hold_progress_mm=h.tolist(),
                  down_progress_mm=d.tolist(), extra_progress_mm=delta.tolist(),
                  down_slip_max_mm=float(slip), repeat_spread_mm=float(max(spreads)),
                  down_near_axis=bool(near_axis))
    return result


def analyze(args):
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    centers, scale = frozen_model(args.atlas)
    parent_centers, parent_scale, _ = parent_model()
    if not np.allclose(parent_scale, scale):
        raise ValueError('Original K14 and K4 scales differ')
    plan = json.loads((args.output / 'plan.json').read_text())
    progress = json.loads((args.output / 'branch_progress.json').read_text())
    by_key = defaultdict(list)
    chosen = {(x['source'], x['frame']): x for x in plan['selected']}
    for row in progress:
        key = (row['source'], row['frame'])
        if key in chosen:
            row['selected'] = chosen[key]
            by_key[key].append(row)
    states = [score_state(by_key[(x['source'], x['frame'])], centers, scale, parent_centers, args)
              if by_key[(x['source'], x['frame'])] else
              dict(source=x['source'], frame=x['frame'], split=x['split'], k4=x['k4'],
                   selected=x, paired=False, outcome='no_branch_records', branches={}, accepted_counts={})
              for x in plan['selected']]
    stage = 'pilot' if len(plan['selected']) < len(plan['all_selected']) or plan.get('smoke', False) else 'full'
    by_class = {}
    for k in range(4):
        group = [s for s in states if s['k4'] == k]
        valid = [s for s in group if s['paired']]
        counts = {label: sum(s['outcome'] == label for s in valid)
                  for label in sorted({s['outcome'] for s in valid})}
        by_class[str(k)] = dict(selected=len(group), paired=len(valid), outcomes=counts,
            source_ids=[s['source'] for s in valid],
            median_extra_mm=float(np.median([s['extra_progress_mm'][-1] for s in valid])) if valid else None)
    by_class_depth = {}
    for k in range(4):
        for band in ('mouth', 'inside', 'bottom'):
            group = [s for s in states if s['k4'] == k and s['selected']['depth_band'] == band]
            valid = [s for s in group if s['paired']]
            by_class_depth[f'{k}:{band}'] = dict(selected=len(group), paired=len(valid),
                outcomes={label: sum(s['outcome'] == label for s in valid)
                          for label in sorted({s['outcome'] for s in valid})},
                median_extra_mm=float(np.median([s['extra_progress_mm'][-1] for s in valid])) if valid else None)
    report = dict(version=VERSION, stage=stage, intervention=plan['intervention'],
                  sample_unit='source demonstration/state', repeat_unit='replay diagnostic',
                  manifest_sha256=plan['dataset_manifest_sha256'], frozen_k4_sha256=plan['frozen_k4_sha256'],
                  selected=len(states), paired=sum(s['paired'] for s in states),
                  repeats=args.repeats, down_mm=args.down_mm,
                  by_class=by_class, by_class_depth=by_class_depth, states=states,
                  limitations=['K4 groups were selected from original raw K14 classes 5/7/9 only',
                               'Short axial response is not a full insertion success rate',
                               'Class comparison is observational; depth/grasp/velocity may differ',
                               'No assigned class is assumed to be a valid correction direction'])
    replay.write_json(args.output / 'analysis.json', report)
    render(args, report, scale)
    print('ANALYSIS', stage, 'paired', report['paired'], '/', len(states),
          'by_class', {k: v['outcomes'] for k, v in by_class.items()}, flush=True)


def render(args, report, scale):
    from run_contactworld_tactile_cluster_atlas import six_force_image
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    assets = args.output / 'assets'; assets.mkdir(exist_ok=True)
    style = '''<style>body{font:16px/1.5 system-ui;background:#121a28;color:#edf3fa;max-width:1500px;margin:auto;padding:18px}
    section{background:#1e2b40;padding:16px;border-radius:8px;margin:18px 0}.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
    img,video{max-width:100%}video{width:100%}button{background:#276da7;color:white;border:0;padding:8px;border-radius:5px}
    a{color:#9bd4ff}table{border-collapse:collapse}td,th{border-bottom:1px solid #526780;padding:7px;text-align:left}
    .clip{background:#293950;padding:10px}@media(max-width:900px){.cards{grid-template-columns:1fr}}</style>'''
    summary = ''.join(f'<tr><td>K4 {k}</td><td>{v["selected"]}</td><td>{v["paired"]}</td><td>{html.escape(str(v["outcomes"]))}</td><td>{v["median_extra_mm"] if v["median_extra_mm"] is not None else "—"}</td></tr>'
                      for k,v in report['by_class'].items())
    phase_rows = ''.join(f'<tr><td>K4 {key.replace(":", " · ")}</td><td>{value["selected"]}</td><td>{value["paired"]}</td><td>{html.escape(str(value["outcomes"]))}</td><td>{value["median_extra_mm"] if value["median_extra_mm"] is not None else "—"}</td></tr>' for key,value in report['by_class_depth'].items() if value['selected'])
    blocks=[]
    for k in (3,0,1,2):
        group=[x for x in report['states'] if x['k4']==k]
        links=''.join(f'<a href="#s{x["source"]:03d}f{x["frame"]:03d}">source {x["source"]} frame {x["frame"]} · {x["outcome"]}</a><br>' for x in group)
        blocks.append(f'<section><h2>K4 {k}</h2>{links}</section>')
        for state in group:
            sid=f's{state["source"]:03d}_f{state["frame"]:03d}'
            header=f'<h3>source {state["source"]} frame {state["frame"]} · {html.escape(state["outcome"])}</h3>'
            header+=f'<p>初始阶段 {state["selected"]["depth_band"]}；此前达到成功标志 {int(state["selected"]["prior_success"])}；原始接触 {int(state["selected"]["contact"])}；初始插座轴向 z {state["selected"]["relative_mm"][2]:.2f} mm；相对位姿 {html.escape(str(state["selected"]["relative_mm"]))} mm；重建仍属原始 5/7/9 且 K4 类别一致 {state.get("live_match",False)}；成对重复 {state["paired"]}</p>'
            if not state['paired']:
                manifest = json.loads((args.dataset / 'spatial/manifest.json').read_text())
                folder = Path(manifest['capture_root']) / f'insertion_usb_episode_{state["source"]:03d}'
                capture = replay.load_capture(folder)
                _, original_base = replay.load_forces(folder)
                t = state['frame']
                source_rgb = np.concatenate((capture['front'][t], capture['wrist'][t]), axis=1)
                cv2.imwrite(str(assets/f'{sid}_source_rgb.jpg'),
                            cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(assets/f'{sid}_source_force.jpg'),
                            six_force_image(original_base[t], scale))
                blocks.append(f'<section id="{sid}">{header}<p>无可信成对结果；接受次数 {html.escape(str(state["accepted_counts"]))}。以下是原始示范当前帧，未用于给出分支结论；拒绝原因见 JSON。</p><img src="assets/{sid}_source_rgb.jpg" alt="原始双视图"><img src="assets/{sid}_source_force.jpg" alt="原始双指六力图"></section>')
                continue
            hold=state['branches']['hold'][0];down=state['branches']['down'][0]
            with np.load(down['file']) as z:
                rgb=z['rgb'][0].copy(); force=z['force_base'][0].copy()
            cv2.imwrite(str(assets/f'{sid}_rgb.jpg'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(assets/f'{sid}_force.jpg'), six_force_image(force,scale))
            fig, ax=plt.subplots(1,3,figsize=(12,3.3),layout='constrained')
            for kind,color in [('hold','#59b9e8'),('down','#f5ad58')]:
                curves=state['branches'][kind]
                for c in curves:
                    ax[0].plot(c['progress_mm'],color=color,alpha=.25)
                    ax[1].plot(c['slip_mm'],color=color,alpha=.25)
                    ax[2].plot(c['force_abs'],color=color,alpha=.25)
                mean=np.mean([c['progress_mm'] for c in curves],axis=0)
                ax[0].plot(mean,color=color,label=kind,linewidth=2)
            ax[0].set_title('USB socket-Z progress');ax[0].set_ylabel('mm');ax[0].legend()
            ax[1].set_title('Plug-gripper displacement');ax[1].set_ylabel('mm')
            ax[2].set_title('mean |tactile|');ax[2].set_ylabel('sim units')
            for axis in ax:axis.set_xlabel('control step')
            fig.savefig(assets/f'{sid}_curves.png',dpi=120);plt.close(fig)
            videos=[]
            for name,c in [('保持',hold),('直接下插',down)]:
                url=Path(c['video']).relative_to(args.output)
                videos.append(f'<div class="clip"><h4>{name}</h4><video controls preload="metadata" src="{html.escape(str(url))}"></video></div>')
            extra=state['extra_progress_mm'];prog=state['down_progress_mm']
            paragraph=f'<p>下插相对保持的额外推进：第 1 步 {extra[1]:+.2f} mm；第 11 步 {extra[-1]:+.2f} mm。下插总推进 {prog[-1]:+.2f} mm；最大滑移 {state["down_slip_max_mm"]:.2f} mm；重复终点差 {state["repeat_spread_mm"]:.2f} mm。</p>'
            blocks.append(f'<section id="{sid}">{header}{paragraph}<img src="assets/{sid}_rgb.jpg" alt="真实初始双视图"><img src="assets/{sid}_force.jpg" alt="双指六张真实初始力图"><img src="assets/{sid}_curves.png" alt="重复试验曲线"><button onclick="const v=this.parentNode.querySelectorAll(\'video\');v.forEach(x=>{{x.currentTime=0;x.play()}})">从同一步播放</button><div class="cards">{"".join(videos)}</div></section>')
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>K4 下插动作分支</title>{style}
    <h1>K4 触觉类别：当前是否可以直接下插</h1>
    <p>冻结原始 K14 的 5/7/9 内部 K4 标签。每个初态重放相同示范前缀，比较保持和沿插座轴向下插 11 步；每种重复 {report['repeats']} 次，总下插指令 {report['down_mm']:.1f} mm。只按开始时的 K4 类别分组，之后类别变化不会使试验停止。类 3 同时覆盖孔口、已插入且有剩余空间、接近底部三个阶段；结果按阶段标记。</p>
    <p>阶段 {report['stage']}；选中 {report['selected']} 状态，重建并成对通过 {report['paired']} 状态。类别没有视觉或动作标签参与拟合。</p>
    <section><h2>状态概览</h2><table><tr><th>初始类别</th><th>选中</th><th>可信成对</th><th>短响应类别与数量</th><th>额外推进中位数 mm</th></tr>{summary}</table>
    <h3>按初始插入阶段展开</h3><table><tr><th>类别与阶段</th><th>选中</th><th>可信成对</th><th>短响应类别与数量</th><th>额外推进中位数 mm</th></tr>{phase_rows}</table>
    <p>短轴向推进：11 步下插比保持多推进至少 0.5 mm，USB 仍靠近轴线且夹持相对移动小。该标签不等于整段成功插入；初始 z≤3 mm 的近底部状态单列人工核对，不凭该阈值断定已经物理触底；重建不一致和重复不稳定的状态也分开报告。连续位移和视频比单个阈值更重要。</p></section>
    <section><h2>阅读提示</h2><p>力图是本次仿真重建在分支开始时的真实左右指 base Fx/Fy/Fz，所有状态使用同一 K4 色标。视频显示双视图和逐帧力图；每段视频内部的力图色标可能随峰值变化，跨状态请比较上面的静态图。下插动作从第一个控制步开始，十一小步总计同一轴向位移。是否仍属 K4 类 3 不参与终点判定。</p></section>
    {''.join(blocks)}<section><h2>数据与限制</h2><p>先按完整来源示范计数。仅在相近插入深度、姿态和运动条件下比较 K4 类别；观察到类别关联不能证明触觉本身决定结果。原始数据和重建拒绝原因在 <a href="analysis.json">analysis.json</a>，选样规则在 <a href="plan.json">plan.json</a>。</p></section></html>'''
    (args.output/'index.html').write_text(page)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('prepare','branch','analyze','expand'),required=True)
    p.add_argument('--dataset',type=Path,default=replay.DATA)
    p.add_argument('--source',type=Path,default=replay.SOURCE)
    p.add_argument('--atlas',type=Path,default=DEFAULT_ATLAS)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--max-attempts',type=int,default=5)
    p.add_argument('--tactile-mae-ratio',type=float,default=1.0)
    p.add_argument('--down-mm',type=float,default=6)
    p.add_argument('--max-replay-error-mm',type=float,default=1.0)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if args.repeats<1 or args.max_attempts<args.repeats or args.down_mm<=0 or args.tactile_mae_ratio<=0:
        p.error('Invalid repeat/attempt/down parameters')
    if args.stage=='prepare':prepare(args)
    elif args.stage=='branch':branch(args)
    elif args.stage=='analyze':analyze(args)
    else:expand(args)

if __name__=='__main__':main()
