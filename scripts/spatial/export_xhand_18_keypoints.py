#!/usr/bin/env python3
"""Export XHand's verified 18-point FK skeleton as a *new* sidecar.

No changes to spatial/v1, object_geometry/v1 or training code.
Put next to audit_xhand_keypoints.py under scripts/spatial/.
Run from repo root:
 PYTHONPATH=src .venv/bin/python scripts/spatial/export_xhand_18_keypoints.py \
   --dataset data/1028_press_button

Outputs hand_keypoints/v1/episodes/eXXXXX.npz and manifest.json.
Uses identical anchor ordering to the approved audit viewer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

from audit_xhand_keypoints import select_keypoints
from openpi.spatial.kinematics import RobotKinematics, joint_positions_from_state, load_state_mapping_csv
from openpi.spatial_dataset.source import RawSpatialDataset
from openpi.spatial_dataset.derived import SpatialDerivedDataset
from openpi.object_geometry_dataset.derived import ObjectGeometryDerivedDataset


def digest(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, value: object):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, path)


def checked_anchors(kin):
    # The QA script is the single source of anchor ordering.
    # FK at all-zero q is used only to read names / topology, not to test poses.
    q = {j: 0.0 for j in kin.required_measured_joints(root='base_link')}
    tf = kin.compute(q, root='base_link')
    names, groups, kinds, xyz, edges = select_keypoints(kin, tf)
    assert xyz.shape == (18, 3)
    return names, groups, kinds, edges


def load_done(path: Path, expected_count: int, expected_names: list[str]) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            frames = z['frame_index']
            points = z['xyz_m']
            names = z['names'].tolist()
            return (points.shape == (expected_count, 18, 3)
                    and frames.shape == (expected_count,)
                    and np.array_equal(frames, np.arange(expected_count))
                    and names == expected_names
                    and np.isfinite(points).all())
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--version', default='v1')
    p.add_argument('--episodes', default='all', help='all or comma-separated episode ids')
    p.add_argument('--overwrite', action='store_true', help='replace an existing episode output')
    a = p.parse_args()
    repo = a.repo.resolve()
    dataset = (a.dataset if a.dataset.is_absolute() else repo / a.dataset).resolve()
    if not dataset.is_dir():
        raise FileNotFoundError(dataset)
    urdf = repo / 'configs/ur7e_xhand/ur7e_xhand_verified.urdf'
    mapping = repo / 'configs/ur7e_xhand/state_mapping.csv'
    kin = RobotKinematics(urdf)
    name_list, group_list, kind_list, edges = checked_anchors(kin)
    mapping_data = load_state_mapping_csv(mapping)
    source_episodes = list(RawSpatialDataset.discover_episode_ids(dataset))
    if a.episodes == 'all':
        selected = source_episodes
    else:
        selected = [int(x.strip()) for x in a.episodes.split(',')]
        if len(selected) != len(set(selected)) or not set(selected).issubset(source_episodes):
            raise ValueError('Invalid or repeated episode ID')
    spatial = SpatialDerivedDataset(dataset, version='v1')
    objects = ObjectGeometryDerivedDataset(dataset, version='v1')
    root = dataset / 'hand_keypoints' / a.version
    eps_root = root / 'episodes'
    eps_root.mkdir(parents=True, exist_ok=True)
    total = 0
    records = []
    for episode in selected:
        raw = RawSpatialDataset(dataset, episode=episode)
        # Only lightweight state columns are loaded, never RGB/depth.
        tab = raw.states()
        frame_ids = np.asarray(tab['frame_index'], dtype=np.int32)
        if not np.array_equal(frame_ids, np.arange(len(frame_ids))):
            raise ValueError(f'e{episode}: noncontiguous or reordered frames')
        count = len(frame_ids)
        if count < 1:
            raise ValueError(f'e{episode}: empty episode')
        # Indexed joins already passed on original datasets; double check boundaries.
        for frame in sorted({0, count - 1}):
            ss = spatial.get(episode_index=episode, frame_index=frame)
            oo = objects.get(episode_index=episode, frame_index=frame)
            if int(ss.frame_index) != frame or int(oo.frame_index) != frame:
                raise AssertionError(f'e{episode} f{frame}: join mismatch')
        out = eps_root / f'e{episode:05d}.npz'
        if out.exists() and not a.overwrite:
            if not load_done(out, count, name_list):
                raise RuntimeError(f'{out}: existing output failed verification; inspect and use --overwrite explicitly')
            print(f'[SKIP VERIFIED] e{episode:05d} {count} frames', flush=True)
        else:
            xyz = np.empty((count, 18, 3), dtype=np.float32)
            states = tab['observation.state']
            for f in range(count):
                q = joint_positions_from_state(np.asarray(states[f], dtype=np.float64),
                                               mapping=mapping_data, kinematics=kin, root='base_link')
                tf = kin.compute(q, root='base_link')
                names, _, _, points, _ = select_keypoints(kin, tf)
                if names != name_list or points.shape != (18, 3):
                    raise AssertionError(f'Unexpected skeleton topology e{episode} f{f}')
                xyz[f] = points
            if not np.isfinite(xyz).all():
                raise ValueError(f'e{episode}: nonfinite xyz')
            tmp = out.with_suffix('.npz.tmp')
            with tmp.open('wb') as handle:
                np.savez_compressed(handle, frame_index=frame_ids,
                                    timestamp_s=np.asarray(tab['timestamp'], dtype=np.float64),
                                    xyz_m=xyz, names=np.asarray(name_list),
                                    groups=np.asarray(group_list), kinds=np.asarray(kind_list),
                                    edges=np.asarray(edges, dtype=np.int32))
            os.replace(tmp, out)
            if not load_done(out, count, name_list):
                raise AssertionError(f'e{episode}: exported file failed verification')
            print(f'[DONE] e{episode:05d} frames={count}', flush=True)
        total += count
        records.append({'episode_index': episode, 'frames': count, 'file': out.relative_to(root).as_posix(),
                        'sha256': digest(out)})
    is_complete = selected == source_episodes
    manifest = {
        'schema': 'xhand_18_fk_anchors_v1',
        'status': 'complete' if is_complete else 'partial',
        'dataset': dataset.name,
        'version': a.version,
        'coordinate_frame': 'base_link', 'unit': 'm',
        'keypoint_count': 18, 'names': name_list, 'groups': group_list,
        'kinds': kind_list, 'edges': edges,
        'urdf_sha256': digest(urdf), 'state_mapping_sha256': digest(mapping),
        'spatial_version': 'v1', 'object_geometry_version': 'v1',
        'episodes': records, 'total_frames': total,
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'caution': 'FK joint axes and tip-link origins; these are not measured skin contact points.',
    }
    atomic_json(root / 'manifest.json', manifest)
    print(f'[FINISHED] {dataset.name} episodes={len(records)} frames={total} status={manifest["status"]}')
    print(f'[OUTPUT] {root}')


if __name__ == '__main__':
    main()
