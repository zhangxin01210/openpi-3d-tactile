#!/usr/bin/env python3
"""Read-only distribution audit; historical XHand peaks are NOT glass breakage limits.

Reports per-finger raw taxel-vector maxima and quantiles. Keep this offline;
use a separate human-reviewed configuration plus measured robot arm wrench
before enabling any action. Cannot infer safe Newtons from uncalibrated taxels.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episodes',type=int,nargs='*',default=None)
    a=p.parse_args()
    from openpi.spatial_dataset.source import RawSpatialDataset
    from deploy.contact_safety import raw_tactile_finger_peaks
    if a.output.exists():
        raise FileExistsError(a.output)
    episodes=a.episodes
    if episodes is None:
        ds=RawSpatialDataset(a.dataset,episode=None)
        episodes=list(ds.episode_indices)
    rows=[]
    for ep in episodes:
        ds=RawSpatialDataset(a.dataset,episode=ep)
        previous=None
        for row in ds.iter_rows(depth_roles=()):
            pks=raw_tactile_finger_peaks(row['observation.state'])
            rows.append((int(ep),int(row['frame_index']),pks,
                         np.abs(pks-previous) if previous is not None else None))
            previous=pks
    if not rows:
        raise ValueError('No dataset samples')
    peaks=np.stack([r[2] for r in rows])
    steps=np.stack([r[3] for r in rows if r[3] is not None])
    def summary(x):
        return {k:np.percentile(x,q,axis=0).round(6).tolist() for k,q in
                [('p95',95),('p99',99),('p99_9',99.9),('max',100)]}
    report={
      'status':'DISTRIBUTION_ONLY_NOT_SAFE_LIMITS','episodes':episodes,
      'frames':len(rows),'units':'raw XHand sensor vector units; NOT Newtons',
      'finger_peak_per_frame':summary(peaks),
      'finger_peak_step_absolute_change':summary(steps) if len(steps) else None,
      'peak_frames':[
          {'finger':i,'episode':rows[int(np.argmax(peaks[:,i]))][0],
           'frame':rows[int(np.argmax(peaks[:,i]))][1],
           'peak':float(np.max(peaks[:,i]))}
          for i in range(5)
      ],
      'warning':'Do not set breakage thresholds directly to a percentage of historical maxima. '
                'Data may contain crushed-glass events or sensor saturation. '
                'Independently calibrate with non-fragile surrogates and review arm/TCP wrench limits.',
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'frames':len(rows),'peak_max_raw':report['finger_peak_per_frame']['max'],
                      'output':str(a.output)},ensure_ascii=False))
if __name__=='__main__':main()
