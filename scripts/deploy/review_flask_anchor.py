#!/usr/bin/env python3
"""Prepare/explicitly approve a CURRENT episode's training-upright flask pose.

Requires the source run from rgbtrack_flask_pivot.py (its target_init/ is
the independently constrained frame-0 RGBTrack registration). Does not run
RGBTrack or approve images automatically. Approval requires a specific
reviewed RGB projection PNG and explicit CLI flag AFTER human inspection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--flask-pivot-run',type=Path,required=True)
    p.add_argument('--episode',type=int,required=True)
    p.add_argument('--front-calibration-profile',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--approve-projection',type=Path,
                   help='Human inspected exact target_init/rgb_overlays/f000000.png and accepts current scene pose')
    a=p.parse_args()
    run=a.flask_pivot_run.expanduser().resolve()
    pose_file=run/'target_init/target_flask_poses.jsonl'
    png=run/'target_init/rgb_overlays/f000000.png'
    metadata_file=run/'target_init/manifest.json'
    for q in (pose_file,png,metadata_file):
        if not q.is_file():
            raise FileNotFoundError(q)
    metadata=json.loads(metadata_file.read_text())
    if metadata.get('episode')!=a.episode:
        raise ValueError('Pose episode mismatch')
    row=json.loads(pose_file.read_text().splitlines()[0])
    if int(row['frame_index'])!=0 or row['object_name']!='target_flask' or not row['valid']:
        raise ValueError('Upright candidate must be valid observed target flask at f0')
    if (run/'flask_pivot_report.json').is_file():
        report=json.loads((run/'flask_pivot_report.json').read_text())
        if report.get('init_candidate',{}).get('chosen_rank') is None:
            raise ValueError('Missing training upright chosen-rank proof')
    else:
        raise FileNotFoundError(run/'flask_pivot_report.json')
    T=np.asarray(row['T_base_link_object'],dtype=float)
    if T.shape!=(4,4) or not np.isfinite(T).all():
        raise ValueError('Bad target f0 transform')
    profile=a.front_calibration_profile.expanduser().resolve()
    from openpi.object_geometry.calibration import front_calibration_provenance,calibration_provenance_error
    root=Path(__file__).resolve().parents[2]
    err=calibration_provenance_error(metadata.get('calibration'),front_calibration_provenance(root,profile))
    if err:
        raise ValueError(f'Calibration differs: {err}')
    approved=False
    if a.approve_projection is not None:
        if a.approve_projection.expanduser().resolve()!=png:
            raise ValueError('Approval PNG must equal original projected frame-0 image from this pose run')
        approved=True
    dest=a.output.expanduser().resolve()
    if dest.exists():
        raise FileExistsError('Refusing to overwrite existing reviewed anchor')
    rec={
        'schema_version':1,'reviewed':approved,
        'source':'human_reviewed' if approved else 'pending_review',
        'episode_index':a.episode,
        'T_base_link_object':T.tolist(),
        'front_calibration_sha256':hashlib.sha256(profile.read_bytes()).hexdigest(),
        'visual_review_projection':str(png),
        'visual_review_projection_sha256':hashlib.sha256(png.read_bytes()).hexdigest(),
        'source_pose_run':str(run),
        'source_frame':0,
        'training_upright_candidate_rank':report['init_candidate']['chosen_rank'],
        'warning':'Review candidate orientation and 2D projection before enabling as policy geometry prior.',
    }
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(rec,indent=2)+'\n',encoding='utf-8')
    print('[ANCHOR]',dest,'reviewed=',approved)
    print('[PROJECTION]',png)
    if not approved:
        print('[PENDING] Inspect projection, then rerun using a DIFFERENT output path and --approve-projection.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
