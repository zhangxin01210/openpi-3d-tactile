#!/usr/bin/env python3
"""Multi-episode diagnostics for object-mask candidate depth and A/B/C memory.
Reports observations, NOT object-surface truth or contact-event ground truth.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np

HOI=Path('/workspace/mnt/sqzhang26/hoi')
def summary_one(root,episode,run_name):
    e=root/f'level1_sequence_e{episode}';result={'episode':episode,'root':str(e),'status':'missing'}
    mf=e/'masks/masks_report.json';sf=e/'sequence_report.json';bf=e/run_name/'report.json'
    if not mf.exists():return result
    masks=json.loads(mf.read_text())
    if masks.get('status')!='complete':result['status']='incomplete_masks';return result
    r=masks.get('mask_pixels',{});length=int(masks.get('num_frames',0));result['n_frames']=length
    stats={};warnings=[]
    for name in ('funnel','target_flask'):
        values=np.array([float(r.get(name,{}).get(str(i),0)) for i in range(length)])
        if not len(values):warnings.append('no_mask_rows');continue
        stats[name+'_mask_median_px']=int(np.median(values))
        stats[name+'_mask_min_px']=int(np.min(values))
        nearzero=int(np.sum(values<25));stats[name+'_mask_small_frames']=nearzero
        if nearzero>warnfrac(length,.03):warnings.append(f'{name}: {nearzero} frames <25 mask px')
        v=np.maximum(values,1);ratio=np.maximum(v[1:]/v[:-1],v[:-1]/v[1:])
        large=np.where((ratio>4)&(np.maximum(values[1:],values[:-1])>100))[0]
        stats[name+'_mask_4x_jumps']=int(len(large))
        if len(large):warnings.append(f'{name}: abrupt mask jumps near frames {large[:6].tolist()}')
    if sf.exists():
        a=json.loads(sf.read_text());data=a.get('frames',[])
        stats['cloud_sampled_frames']=len(data)
        if data:
            f=np.array([x.get('funnel_candidates',0) for x in data]);g=np.array([x.get('target_flask_candidates',0) for x in data]);
            stats['A_funnel_candidates_median']=float(np.median(f));stats['A_flask_candidates_median']=float(np.median(g))
            stats['A_funnel_low_fraction']=float(np.mean(f<250));stats['A_flask_low_fraction']=float(np.mean(g<250))
            for name in ('funnel','target_flask'):
                total=sum(x.get('mask_depth_projected',{}).get(name,0) for x in data)
                missing=sum(x.get('mask_depth_outside_roi',{}).get(name,0) for x in data)
                if total:
                    rate=float(missing/total)
                    stats[name+'_roi_clip_fraction']=round(rate,4)
                    if rate>.25:warnings.append(f'{name}: {rate:.0%} mask-projected valid depths excluded by fixed ROI; compare across target positions before training')
    else:warnings.append('missing cloud sequence_report.json')
    if bf.exists():
        b=json.loads(bf.read_text());data=b.get('frames',[])
        stats['abc_sampled_frames']=len(data)
        if data:
            bo=np.array([x.get('fmem',0) for x in data]);active=np.array([x.get('force_active_count',0) for x in data]);grasp=np.array([x.get('grasp',False) for x in data]);
            stats['B_funnel_mean_points']=float(bo.mean());stats['B_funnel_nonempty_fraction']=float(np.mean(bo>0))
            suspicious=int(np.sum(grasp & (active==0)));stats['grasp_hypothesis_no_active_taxels_frames']=suspicious
            if suspicious>warnfrac(len(data),.07):warnings.append(f'grasp hypothesis stayed on for {suspicious} zero-force sampled frames')
    else:warnings.append(f'missing {run_name}/report.json')
    result.update(stats);result['warnings']=warnings
    result['status']='needs_review' if warnings else 'diagnostics_ready'
    return result

def warnfrac(n,frac):return max(2,round(n*frac))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes',required=True,help='comma-separated IDs, e.g. 1,2,3,4,5,6');p.add_argument('--outputs',type=Path,default=HOI/'outputs')
    p.add_argument('--run-name',default='abc_v1');p.add_argument('--output',type=Path,default=None)
    a=p.parse_args();ids=[int(x.strip()) for x in a.episodes.split(',')]
    reports=[summary_one(a.outputs,i,a.run_name) for i in ids]
    out=a.output or a.outputs/'level1_multi_report'
    out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2))
    cols=sorted({key for r in reports for key in r})
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for r in reports:
            x=r.copy();x['warnings']='; '.join(x.get('warnings',[]));w.writerow(x)
    print('[REPORT]',out/'summary.csv')
    for r in reports:print(f'e{r["episode"]}: {r["status"]};', '; '.join(r.get('warnings',[])))
if __name__=='__main__':main()
