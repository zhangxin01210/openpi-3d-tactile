#!/usr/bin/env python3
"""Audit paired target shifts, same policy noise, action response and task outcomes."""
import argparse
import csv
import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(folder):
    s=json.loads((folder/'summary.json').read_text())
    t=json.loads((folder/'trajectory.json').read_text())
    with np.load(folder/'inputs.npz') as z:
        fields={key:z[key][0].copy() for key in ['front','wrist','state','cloud_xyz','force_local']}
        steps=z['steps'].copy();chunks=z['action_chunks'].copy()
    return s,t['rows'],fields,steps,chunks


def first_contact(rows):
    c=np.linalg.norm([r['plug_socket_contact_force_xyz'] for r in rows],axis=1)
    ix=np.flatnonzero(c>.1)
    return int(ix[0]) if len(ix) else None


def analyze(folder,plan):
    output=folder/'analysis';output.mkdir(exist_ok=True)
    expected={(m,s,c) for m in plan['configs'] for s in plan['seeds'] for c in plan['conditions']}
    found={(m,s,c) for m in plan['configs'] for s in plan['seeds'] for c in plan['conditions']
           if (folder/m/f'seed_{s}'/c/'summary.json').is_file()}
    if found!=expected:raise RuntimeError(f'Missing {len(expected-found)} conditions: {sorted(expected-found)}')
    lines=[];checks=[]
    for m in plan['configs']:
      for s in plan['seeds']:
        base=folder/m/f'seed_{s}';bs,br,bf,bsteps,bchunks=read(base/'zero')
        for c in plan['conditions']:
            trial=base/c;ss,rr,fields,steps,chunks=read(trial)
            if not np.array_equal(steps,bsteps):raise ValueError('Replan schedule changed')
            if not np.array_equal([r['noise_seed'] for r in rr],[r['noise_seed'] for r in br]):raise ValueError('Noise schedule changed')
            init_diff={key:float(np.max(np.abs(np.asarray(ss['initial_unperturbed_state'][key])-bs['initial_unperturbed_state'][key])))
                       for key in bs['initial_unperturbed_state']}
            if max(init_diff.values())>1e-6:raise ValueError(f'Reset mismatch {m} {s} {c}: {init_diff}')
            pre_diff={key:float(np.max(np.abs(np.asarray(ss['initial_perturbed_state'][key])-bs['initial_perturbed_state'][key])))
                      for key in ['plug_pos','dof_pos','ee_pos']}
            if max(pre_diff.values())>2e-4:raise ValueError(f'Non-target perturbation {m} {s} {c}: {pre_diff}')
            if not np.allclose(np.array(ss['offset_actual_xyz_mm'])[:2],ss['offset_requested_xy_mm'],atol=.001):raise ValueError('Target shift mismatch')
            a=np.array([r['action_sim_clipped'] for r in rr]);b=np.array([r['action_sim_clipped'] for r in br]);diff=a-b
            # Translation commands are scaled by 0.01 m per unit; 1 normalized unit = 10 mm.
            rms20=float(np.sqrt(np.mean(diff[:20,:3]**2))*10)
            first4=float(np.sqrt(np.mean(diff[:4,:3]**2))*10)
            bychunk=np.array([np.sqrt(np.mean(diff[k:k+4,:3]**2))*10 for k in range(0,len(diff),4)])
            responsive=np.flatnonzero(bychunk>=.5)
            onset=int(responsive[0]*4) if len(responsive) else None
            ca=first_contact(rr);cb=first_contact(br)
            sourcep=base/c/'front_wrist.mp4'
            item={'model':m,'seed':s,'condition':c,'dx_mm':ss['offset_requested_xy_mm'][0],'dy_mm':ss['offset_requested_xy_mm'][1],
                  'ever':int(ss['source_ever']),'final':int(ss['source_final']),
                  'baseline_ever':int(bs['source_ever']),'baseline_final':int(bs['source_final']),
                  'first_success_step':ss['first_success_step'],'first_contact_step':ca,'baseline_first_contact_step':cb,
                  'first4_translation_command_rms_diff_mm':first4,'first20_translation_command_rms_diff_mm':rms20,
                  'first_replan_with_0p5mm_command_rms_difference':onset,
                  'front_rgb_initial_mae':float(np.mean(np.abs(fields['front'].astype(np.int16)-bf['front'].astype(np.int16)))),
                  'wrist_rgb_initial_mae':float(np.mean(np.abs(fields['wrist'].astype(np.int16)-bf['wrist'].astype(np.int16)))),
                  'initial_cloud_mean_point_displacement_mm':float(np.mean(np.linalg.norm(fields['cloud_xyz']-bf['cloud_xyz'],axis=-1))*1000),
                  'initial_force_local_max_difference':float(np.max(np.abs(fields['force_local']-bf['force_local']))),
                  'min_keypoint_error_mm':ss['min_keypoint_error_mm'],'final_keypoint_error_mm':ss['final_keypoint_error_mm'],
                  'final_plug_socket_xy_mm':ss['final_plug_socket_xy_mm'],'final_plug_socket_z_mm':ss['final_plug_socket_z_mm'],
                  'video':str(sourcep)}
            lines.append(item)
            checks.append({'model':m,'seed':s,'condition':c,'reset_max_abs_differences':init_diff,'non_target_max_abs_differences':pre_diff})
    with (output/'paired_trials.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(lines[0]));w.writeheader();w.writerows(lines)
    summary={'planned':len(expected),'completed':len(found),'checks':checks,'trials':lines,
             'model_noise':'Fixed Gaussian diffusion noise with seed base + replan index, shape 16×32; same seed for paired conditions.',
             'difference_threshold':'First 4-action replan with per-component RMS difference in translated command >=0.5 mm.'}
    (output/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
    for m in plan['configs']:
      fig,axes=plt.subplots(len(plan['seeds']),2,figsize=(11,4*len(plan['seeds'])),layout='constrained')
      axes=np.atleast_2d(axes)
      for i,s in enumerate(plan['seeds']):
        for c in plan['conditions']:
            item=next(x for x in lines if x['model']==m and x['seed']==s and x['condition']==c)
            rr=json.loads((folder/m/f'seed_{s}'/c/'trajectory.json').read_text())['rows']
            x=np.arange(len(rr))+1
            axes[i,0].plot(x,[r['keypoint_error_mm'] for r in rr],label=c)
            axes[i,1].plot(x,[r['plug_socket_xy_error_mm'] for r in rr],label=c)
        axes[i,0].set(title=f'{m} seed {s}',xlabel='Step',ylabel='Keypoint error (mm)')
        axes[i,1].set(xlabel='Step',ylabel='USB lateral offset (mm)')
        for ax in axes[i]:ax.grid(alpha=.2);ax.legend(fontsize=7)
      fig.savefig(output/f'{m}.png',dpi=150);plt.close(fig)
    print(json.dumps({'planned':len(expected),'completed':len(found),
                      'outcomes':[{k:x[k] for k in ['model','seed','condition','ever','first20_translation_command_rms_diff_mm','first_replan_with_0p5mm_command_rms_difference']} for x in lines]},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    a=p.parse_args();plan=json.loads((a.root/'plan.json').read_text());analyze(a.root,plan)

if __name__=='__main__':main()
