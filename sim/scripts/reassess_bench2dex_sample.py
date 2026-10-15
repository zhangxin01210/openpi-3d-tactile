#!/usr/bin/env python3
"""Summarize initial variation and tactile availability near assembly targets.

Near-target masks are geometric proxies, not detected object-object contacts or
verified recovery labels. Uses the previously audited task-73 state replays.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np


TARGETS = {2: (-.065, .300), 3: (-.150, .270),
           4: (-.150, .190), 5: (-.065, .215)}


def inspect(path):
    with h5py.File(path) as f:
        before = f['time/sim_step'][:] < int(f['meta/homing_start_sim_step'][()])
        n = len(before)
        active = np.zeros(n, bool)
        per_site = {}
        for site, dataset in f['robot/tactile/distance_along_normal_m'].items():
            # 0.1 mm and 10 pixels are explicit noise-screening choices.
            flags = np.zeros(n, bool)
            for start in range(0, n, 32):
                block = dataset[start:start + 32]
                flags[start:start + len(block)] = (block > .0001).sum(axis=(1, 2)) >= 10
            active |= flags
            per_site[site] = int((flags & before).sum())
        result = {'path': str(path), 'before_homing_frames': int(before.sum()),
                  'any_finger_active_frames': int((active & before).sum()),
                  'per_site_active_frames': per_site, 'objects': {}}
        anchor_z = f['objects/obj_326_piece_0/pose_world'][:, 2]
        for piece, target in TARGETS.items():
            pose = f[f'objects/obj_326_piece_{piece}/pose_world'][:]
            # collector/state_reader.py exports pose_world as xyz + xyzw.
            x, y, z, w = pose[0, 3:7]
            yaw = np.degrees(np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z)))
            error = np.linalg.norm(pose[:, :2] - target, axis=1)
            near = (error < .04) & (np.abs(pose[:, 2] - anchor_z) < .04) & before
            delta = np.linalg.norm(np.diff(pose[:, :2], axis=0), axis=1)
            near_motion = near[1:] & near[:-1] & (delta > .002)
            result['objects'][str(piece)] = {
                'initial_xyz_m': pose[0, :3].tolist(), 'initial_yaw_deg': float(yaw),
                'near_target_frames': int(near.sum()),
                'near_target_with_any_finger_active_frames': int((near & active).sum()),
                'near_target_xy_motion_over_2mm_frame_pairs': int(near_motion.sum()),
                'near_target_motion_and_tactile_frame_pairs': int((near_motion & active[1:]).sum()),
            }
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('files', type=Path, nargs='+')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    episodes = [inspect(x) for x in args.files]
    spans = {}
    for piece in TARGETS:
        items = [e['objects'][str(piece)] for e in episodes]
        xyz = np.array([x['initial_xyz_m'] for x in items])
        yaw = np.degrees(np.unwrap(np.radians([x['initial_yaw_deg'] for x in items])))
        spans[str(piece)] = {'xyz_range_m': np.ptp(xyz, axis=0).tolist(),
                             'yaw_range_deg': float(np.ptp(yaw))}
    result = {'episodes': episodes, 'initial_variation': spans,
              'thresholds': {'tactile_distance_m': .0001, 'minimum_pixels': 10,
                             'near_target_xy_m': .04, 'near_target_relative_z_m': .04,
                             'motion_per_frame_m': .002},
              'limits': ['Three selected trajectories only.',
                         'Any finger can contact any surface, not necessarily this puzzle piece.',
                         'Near-target motion is not proof of jamming, recovery, or tactile causality.',
                         'Tactile is reconstructed geometry, not measured force.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(spans))


if __name__ == '__main__':
    main()
