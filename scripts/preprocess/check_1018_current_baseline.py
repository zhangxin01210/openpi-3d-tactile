#!/usr/bin/env python3
"""Read-only preflight to ensure the *current* Spatial exporter reproduces 1028's frozen v1 config/assets.
Prints 1018/1028 provenance and, if possible, reconstructs the old camera config from git history.
Run from openpi-3d-tactile root: PYTHONPATH=src .venv/bin/python scripts/preprocess/check_1018_current_baseline.py
No files are changed. Exit nonzero on current baseline mismatch.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import subprocess


def load(p: Path):
    return json.loads(p.read_text(encoding='utf-8'))


def sha256(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest()


def diff_obj(a,b,prefix='',output=None):
    if output is None:output=[]
    if isinstance(a,dict) and isinstance(b,dict):
        for k in sorted(set(a)|set(b)):
            key=f'{prefix}.{k}' if prefix else str(k)
            if k not in a:output.append((key,'<absent>',b[k]))
            elif k not in b:output.append((key,a[k],'<absent>'))
            else:diff_obj(a[k],b[k],key,output)
    elif a!=b:output.append((prefix,a,b))
    return output


def main():
    repo=Path.cwd().resolve()
    old=load(repo/'data/1018_peg_all/spatial/v1_front_candidate/manifest.json')
    new=load(repo/'data/1028_press_button/spatial/v1/manifest.json')
    from openpi.spatial.config import make_baseline_config
    current_cfg=make_baseline_config().with_camera_roles('front').with_visual_sampling(num_points=4096)
    current=json.loads(json.dumps(asdict(current_cfg), default=str))
    diffs=diff_obj(current,new.get('preprocess_config',{}))
    print('[CONFIG] current exporter defaults vs 1028 v1:', 'MATCH' if not diffs else 'DIFFERENT')
    for item in diffs[:15]:print(' ',item)
    for name,m in [('1018 OLD',old),('1028 NEW',new)]:
        d=m.get('preprocess_config',{}).get('diagnostics',{})
        print(f'[{name}] diagnostic flags: front_extrinsic={d.get("enable_front_extrinsic_correction")} '
              f'front_K1={d.get("enable_front_intrinsic_k1")} correction={d.get("front_extrinsic_translation_base_m")}')
        c=m.get('asset_provenance',{}).get('camera_internal_config',{})
        print(' camera_internal_config:',c.get('path'), c.get('sha256'))
        print(' generated commit:',m.get('generator',{}).get('openpi_repo_git',{}).get('commit'))
    bad=[]
    for key,item in new.get('asset_provenance',{}).items():
        if not isinstance(item,dict) or not item.get('path') or not item.get('sha256'):
            bad.append(f'{key}: no reliable path/hash');continue
        p=(repo/item['path']).resolve()
        actual=sha256(p) if p.is_file() else None
        if actual!=item['sha256']:bad.append(f'{key}: expected={item["sha256"]}, current={actual}')
    print('[ASSETS] current files vs 1028 v1:', 'MATCH' if not bad else 'DIFFERENT')
    for item in bad:print(' ',item)
    path=old.get('asset_provenance',{}).get('camera_internal_config',{}).get('path')
    expected=old.get('asset_provenance',{}).get('camera_internal_config',{}).get('sha256')
    commit=old.get('generator',{}).get('openpi_repo_git',{}).get('commit')
    if isinstance(commit,str) and commit and isinstance(path,str) and path:
        p=subprocess.run(['git','show',f'{commit}:{path}'],cwd=repo,capture_output=True)
        if p.returncode==0 and hashlib.sha256(p.stdout).hexdigest()==expected:
            try:
                past=json.loads(p.stdout)
                now=load(repo/path)
                change=diff_obj(past,now)
                print('[CAMERA FILE] old snapshot recovered with VERIFIED old hash; changed fields:',len(change))
                for x in change[:40]:print(' ',x)
            except (UnicodeDecodeError,json.JSONDecodeError) as exc:
                print('[CAMERA FILE] old snapshot available but JSON parse failed:',exc)
        else:print('[CAMERA FILE] old snapshot NOT recovered/hash mismatch; do not infer changed numeric fields from hash alone')
    else:print('[CAMERA FILE] old git revision or path unavailable; cannot recover historical asset values')
    if diffs or bad:
        raise SystemExit('[STOP] Current source files do not reproduce the new frozen v1 configuration; do not run probe until reconciled')
    print('[PREFLIGHT OK] Current exporter matches new 1028 v1 configuration and asset hashes. Geometric pose validation still required.')

if __name__=='__main__':main()
