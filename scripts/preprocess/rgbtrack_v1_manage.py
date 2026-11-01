#!/usr/bin/env python3
"""Read-only QC, canonical publication, and opt-in cleanup for RGBTrack v1.

Unlike tracking itself, this command never invokes GPU inference. Source-of-truth
remains in rgbtrack/runs/<tag>/episodes; rgbtrack/episodes contains RELATIVE
symlinks to the approved version, avoiding duplicated frames. No data is deleted
unless `cleanup --apply` is explicitly requested after successful publication.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import sys

DEFAULT_REPO = Path('/workspace/mnt/sqzhang26/openpi-3d-tactile')
DEFAULT_TASK = '1028_press_button'
DEFAULT_TAG = 'press_button_v1'
EXPECTED = {
    'static_object': 'box',
    'dynamic_object': 'button',
    'dynamic_init_up_axis': '+z',
    'mode': 'depth',
}


def _load_task(repo: Path):
    path = repo / 'scripts/preprocess/rgbtrack_task.py'
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location('_rgbtrack_task_release_v1', path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _is_release_policy(policy: dict) -> bool:
    if any(policy.get(k) != v for k, v in EXPECTED.items()):
        return False
    checks = {
        'dynamic_init_max_tilt_deg': 25.,
        'guard_max_step_m': .02,
        'guard_max_turn_deg': 25.,
        'guard_recovery_max_turn_deg': 60.,
    }
    for key, target in checks.items():
        try:
            if abs(float(policy[key]) - target) > 1e-7:
                return False
        except (KeyError, TypeError, ValueError):
            return False
    return True


def _report(repo: Path, task_name: str, tag: str):
    task = _load_task(repo)
    info = task.task_inputs(repo, task_name)
    track_root = info['base'] / 'rgbtrack'
    run_root = track_root / 'runs' / tag / 'episodes'
    calibration = repo / 'configs/ur7e_xhand/front_calibration_candidate.json'
    completed, issues, totals = [], [], Counter()
    policies = []
    for ep in info['episode_ids']:
        name = f'e{ep:05d}'
        folder = run_root / name
        if not folder.is_dir():
            issues.append((ep, 'missing episode output'))
            continue
        try:
            mask, frames = task.verify_episode(info, ep)
            task.completed(info, ep, folder, mask, frames, calibration)
            provenance = task.read_json(folder / 'task_input_provenance.json')
            policy = provenance.get('policy', {})
            if not _is_release_policy(policy):
                raise ValueError('policy differs from the accepted press_button v1')
            policies.append(policy)
            quality_path = folder / 'quality_summary.json'
            if not quality_path.is_file():
                raise ValueError('quality_summary.json missing')
            quality = task.read_json(quality_path)
            for obj in info['objects']:
                q = quality.get('objects', {}).get(obj, {})
                if int(q.get('frames', -1)) != frames:
                    raise ValueError(f'{obj}: quality frame count mismatch')
                valid = int(q.get('valid_frames', -1))
                if not (0 <= valid <= frames):
                    raise ValueError(f'{obj}: invalid QC count')
                totals[f'{obj}_frames'] += frames
                totals[f'{obj}_valid'] += valid
                for status, count in q.get('statuses', {}).items():
                    totals[f'{obj}:status:{status}'] += int(count)
            completed.append(ep)
        except Exception as exc:
            issues.append((ep, str(exc)))
    if policies and any(p != policies[0] for p in policies):
        issues.append((-1, 'episode policies are inconsistent'))
    return info, track_root, run_root, completed, issues, totals, policies[0] if policies else {}


def _brief(completed, issues, totals, total):
    print(f'[RGBTRACK v1] completed={len(completed)}/{total}, issues={len(issues)}')
    for obj in ('button', 'box'):
        frames = totals[f'{obj}_frames']
        valid = totals[f'{obj}_valid']
        if frames:
            print(f'  {obj}: valid={valid}/{frames} ({valid/frames:.1%}); '
                  f'invalid={frames-valid}')
        statuses = {k.split(':', 2)[2]: v for k, v in totals.items()
                    if k.startswith(f'{obj}:status:')}
        print(f'  {obj} statuses: {dict(sorted(statuses.items()))}')
    for ep, desc in issues[:15]:
        print(f'  [ISSUE] e{ep:05d}: {desc}' if ep >= 0 else f'  [ISSUE] {desc}')
    if len(issues) > 15:
        print(f'  ... {len(issues)-15} additional issues')


def _canonical_target(tag: str, episode: int) -> str:
    return f'../runs/{tag}/episodes/e{episode:05d}'


def _verify_published(episodes_root: Path, tag: str, ids: list[int]):
    for ep in ids:
        dst = episodes_root / f'e{ep:05d}'
        expected = _canonical_target(tag, ep)
        if not dst.is_symlink() or dst.readlink().as_posix() != expected:
            raise RuntimeError(f'Canonical episode missing/mismatched: {dst}; expected symlink -> {expected}')
        if not dst.resolve().is_dir():
            raise RuntimeError(f'Canonical symlink is broken: {dst}')


def _promote(track_root: Path, tag: str, ids: list[int], totals, policy: dict, *, apply: bool):
    episodes_root = track_root / 'episodes'
    release = track_root / 'release_v1.json'
    allowed_names = {f'e{ep:05d}' for ep in ids}
    if episodes_root.exists():
        unexpected = [p for p in episodes_root.iterdir() if p.name not in allowed_names]
        if unexpected:
            raise RuntimeError(f'Unexpected files in canonical directory (not touching): {unexpected[:5]}')
        for ep in ids:
            p = episodes_root / f'e{ep:05d}'
            if p.exists() or p.is_symlink():
                expected = _canonical_target(tag, ep)
                if not p.is_symlink() or p.readlink().as_posix() != expected:
                    raise RuntimeError(f'Existing canonical path differs; do not overwrite: {p}')
    if release.exists():
        prev = json.loads(release.read_text(encoding='utf-8'))
        if prev.get('dataset') != DEFAULT_TASK or prev.get('run_tag') != tag:
            raise RuntimeError(f'Release already points to another run: {release}')
    print(f'[PROMOTE PLAN] {len(ids)} relative links at {episodes_root} -> runs/{tag}/episodes/')
    print('  release manifest:', release)
    if not apply:
        print('  [DRY RUN] Add --apply only after the complete episode audit and representative visual QA.')
        return
    episodes_root.mkdir(parents=True, exist_ok=True)
    for ep in ids:
        dst = episodes_root / f'e{ep:05d}'
        if not dst.is_symlink():
            dst.symlink_to(_canonical_target(tag, ep), target_is_directory=True)
    _verify_published(episodes_root, tag, ids)
    payload = {
        'schema_version': 1,
        'dataset': DEFAULT_TASK,
        'run_tag': tag,
        'published_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'episodes': ids,
        'objects': ['button', 'box'],
        'policy': policy,
        'note': ('Canonical episodes are relative symlinks into an immutable run. '
                 'valid=false (including occluded_causal_prediction) is NOT ground-truth 6D pose.'),
        'totals': dict(totals),
    }
    release.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'[PUBLISHED] {len(ids)} episodes -> {episodes_root}')


def _cleanup(track_root: Path, tag: str, ids: list[int], *, apply: bool):
    release = track_root / 'release_v1.json'
    if not release.is_file():
        raise RuntimeError('No release_v1.json. Promote and verify first; cleanup blocked.')
    payload = json.loads(release.read_text(encoding='utf-8'))
    if payload.get('run_tag') != tag or payload.get('episodes') != ids:
        raise RuntimeError('Published release differs from selected run or episode set')
    _verify_published(track_root / 'episodes', tag, ids)
    candidates = []
    for name in ('experiments', 'reviews'):
        path = track_root / name
        if path.exists() or path.is_symlink():
            candidates.append(path)
    runs = track_root / 'runs'
    for item in sorted(runs.iterdir()) if runs.is_dir() else []:
        if item.name != tag:
            candidates.append(item)
    print('[CLEANUP PLAN] Only press_button RGBTrack test artifacts (never SAM2/geometry/raw data):')
    for p in candidates:
        print('  ', p)
    if not candidates:
        print('[NOTHING TO DELETE]')
        return
    if not apply:
        print('[DRY RUN] Use cleanup --apply to actually remove the exact listed paths.')
        return
    for p in candidates:
        if p.is_symlink() or p.is_file():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)
    print(f'[CLEANED] {len(candidates)} test paths deleted; canonical v1 retained')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=DEFAULT_REPO)
    p.add_argument('--dataset', default=DEFAULT_TASK)
    p.add_argument('--run-tag', default=DEFAULT_TAG)
    p.add_argument('action', choices=['audit', 'promote', 'cleanup'])
    p.add_argument('--apply', action='store_true', help='Explicitly publish symlinks or delete test paths')
    args = p.parse_args(argv)
    if args.dataset != DEFAULT_TASK:
        raise SystemExit('This release manager is intentionally restricted to 1028_press_button')
    if args.run_tag != DEFAULT_TAG:
        raise SystemExit('This v1 release manager requires the canonical tag press_button_v1')
    repo = args.repo.expanduser().resolve()
    info, track_root, _, completed, issues, totals, policy = _report(repo, args.dataset, args.run_tag)
    _brief(completed, issues, totals, len(info['episode_ids']))
    if args.action == 'audit':
        if issues:
            raise SystemExit('[NOT READY] Fix listed episodes before publishing')
        print('[AUDIT OK] Every episode has complete pose streams, QC, and matching versioned inputs')
        return
    if issues or len(completed) != len(info['episode_ids']):
        raise SystemExit('[BLOCKED] Batch is not complete; promotion/cleanup prohibited')
    if args.action == 'promote':
        _promote(track_root, args.run_tag, info['episode_ids'], totals, policy, apply=args.apply)
    else:
        _cleanup(track_root, args.run_tag, info['episode_ids'], apply=args.apply)


if __name__ == '__main__':
    main()
