#!/usr/bin/env python3
"""One random-reset seed's counterfactual slip branches in private Isaac Gym."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np

from contactworld_slip_core import (read, write, save, sha, grasp, rotation_distance,
                                   action_schedule, branch_effect, causal_baseline)


def match(actual, expected, local, reference_force):
    pos = max(float(np.linalg.norm(actual[k]-expected[k])*1000) for k in ('plug_pos','ee_pos','socket_pos_gt'))
    angle = max(float(rotation_distance(actual[k],expected[k])) for k in ('plug_quat','ee_quat','socket_quat'))
    joint = float(np.max(np.abs(actual['dof_pos']-expected['dof_pos'])))
    velocity = float(np.max(np.abs(actual['dof_vel']-expected['dof_vel'])))
    force_error = float(np.abs(local-reference_force).mean())
    force_limit = max(2e-5,.25*float(np.abs(reference_force).mean()))
    return dict(pass_gate=bool(pos<=.25 and angle<=.5 and joint<=.005 and velocity<=.08 and force_error<=force_limit),
                position_mm=pos,angle_deg=angle,joint_rad=joint,joint_velocity=velocity,
                force_mae=force_error,force_limit=force_limit)


def encode(path, rgb, force, dt, limits):
    import cv2
    target=path.with_name(path.stem+'.partial.mp4')
    proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
                           '-s','1024x350','-r',str(1/dt),'-i','-','-an','-c:v','libx264','-threads','1',
                           '-pix_fmt','yuv420p','-movflags','+faststart',str(target)],stdin=subprocess.PIPE)
    try:
        for t,(image,field) in enumerate(zip(rgb,force)):
            frame=np.full((350,1024,3),25,np.uint8);frame[40:296,:512]=image
            for hand in range(2):
                for axis in range(3):
                    v=np.clip(field[hand,:,:,axis]/limits[axis],-1,1)
                    color=np.stack([np.where(v>=0,245,245+190*v),245-160*np.abs(v),np.where(v<=0,245,245-190*v)],-1).astype(np.uint8)
                    x,y=515+axis*168,40+hand*145
                    frame[y:y+110,x:x+160]=cv2.resize(color,(160,110),interpolation=cv2.INTER_NEAREST)
                    cv2.putText(frame,('L' if hand==0 else 'R')+' F'+'xyz'[axis],(x,y-8),0,.5,(235,235,235),1)
            cv2.putText(frame,f'{path.stem} | frame {t} | fixed LOCAL channel scale',(8,20),0,.5,(235,235,235),1)
            cv2.putText(frame,'Resume = fixed known commands; this is NOT closed-loop pi0 recovery.',(8,335),0,.5,(235,235,235),1)
            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close();code=proc.wait()
    if code:raise RuntimeError('Video encoding failed')
    target.replace(path)


def finalize(out,item,source,plan,records,reference_force,t):
    accepted={kind:[r for r in records if r['kind']==kind and r['accepted']] for kind in plan['kinds']}
    all_repeated=all(len(v)>=plan['repeats'] for v in accepted.values())
    effects={}
    for kind,arr in accepted.items():
        if arr:
            effects[kind]={k:float(np.mean([r['effect'][k] for r in arr])) for k in
                           ('slip_mm','rotation_deg','progress_mm','xy_final_mm','late_slip_mm')}
            effects[kind]['repeat_slip_range_mm']=float(np.ptp([r['effect']['slip_mm'] for r in arr]))
            effects[kind]['repeat_progress_range_mm']=float(np.ptp([r['effect']['progress_mm'] for r in arr]))
            effects[kind]['all_repeats']=[r['effect'] for r in arr]
            effects[kind]['n']=len(arr)
    reproducible=all(kind in effects and effects[kind]['repeat_slip_range_mm']<=.5 and
                     effects[kind]['repeat_progress_range_mm']<=.5 for kind in ('continue','hold'))
    original_response=bool(accepted['continue']) and all(r.get('continue_first4_match',{}).get('passed',False)
                                                           for r in accepted['continue'])
    for e in effects.values():
        e['extra_progress_mm']=e['progress_mm']-effects['hold']['progress_mm'] if 'hold' in effects else None
    # Compare start states across branches, independent of their outcomes.
    starts=[]
    for arr in accepted.values():
        for r in arr:
            with np.load(r['file']) as z:starts.append({k:z[k][0] for k in ('plug_pos','ee_pos','plug_quat','ee_quat','force_local')})
    spread=angle=force_spread=0.
    if starts:
        for a in starts:
            spread=max(spread,*[float(np.linalg.norm(a[k]-starts[0][k])*1000) for k in ('plug_pos','ee_pos')])
            angle=max(angle,*[float(rotation_distance(a[k],starts[0][k])) for k in ('plug_quat','ee_quat')])
            force_spread=max(force_spread,float(np.abs(a['force_local']-starts[0]['force_local']).mean()))
    same_start=spread<=.25 and angle<=.5 and force_spread<=max(2e-5,.25*float(np.abs(reference_force[t]).mean()))
    write(out/'result.json',dict(item,folder=str(out),original_folder=str(source),kinds=plan['kinds'],effects=effects,
                                usable=bool(all_repeated and reproducible and same_start and original_response),
                                gate=dict(all_repeated=all_repeated,reproducible=reproducible,same_start=same_start,
                                          original_first4_response_matches=original_response,
                                          start_spread_mm=spread,start_spread_deg=angle,start_force_mae=force_spread),records=records))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--state-id',required=True)
    p.add_argument('--one-branch');p.add_argument('--attempt',type=int,default=0);args=p.parse_args()
    contract=read(args.output/'run_contract.json')
    if contract['code']['branch_contactworld_slip.py']!=sha(__file__):raise ValueError('Branch code changed')
    task=args.source/'thirdparty/manifeel-isaacgymenvs/isaacgymenvs/tasks/tacsl/tacsl_task_usb.py'
    if contract['simulator_task_sha256']!=sha(task):raise ValueError('Simulator source changed')
    plan=read(args.output/'branch_plan.json');item=next(r for r in plan['selected'] if r['id']==args.state_id)
    out=args.output/'branches'/item['id'];out.mkdir(parents=True,exist_ok=True)
    source=Path(item['folder']);data=read(source/'trajectory.json')
    original=[data['result']['initial_physical_state']]+data['rows'];t=item['step']
    expected={k:np.asarray(original[t][k]) for k in ('ee_pos','ee_quat','plug_pos','plug_quat','socket_pos_gt','socket_quat','dof_pos','dof_vel')}
    with np.load(source/'inputs.npz') as z:
        reference_force=np.concatenate([z['force_local'][:1],z['force_local_post_step']],0)
        planned=z['action_chunks'][item['query_index'],:4].copy()
    if args.one_branch is None:
        records=[]
        for kind in plan['kinds']:
            accepted=[]
            for attempt in range(plan['max_attempts']):
                meta=out/f'{kind}_{attempt}.json'
                if not meta.exists():
                    subprocess.run([sys.executable,str(Path(__file__).resolve()),'--output',str(args.output),
                        '--source',str(args.source),'--state-id',args.state_id,'--one-branch',kind,
                        '--attempt',str(attempt)],check=True)
                record=read(meta);records.append(record)
                if record['accepted']:
                    if not Path(record['file']).exists():raise FileNotFoundError(record['file'])
                    accepted.append(record)
                if len(accepted)>=plan['repeats']:break
            if kind in ('continue','hold') and len(accepted)<plan['repeats']:break
        finalize(out,item,source,plan,records,reference_force,t)
        return
    from isaacgym import gymapi  # MUST precede torch
    import torch
    from eval_contactworld_proprio_baseline import prepare_environment, video_frame
    from contactworld_v3_eval_inputs import obs_array, snapshot
    torch.set_num_threads(4)
    sim_args=argparse.Namespace(source=args.source,output=out/'sim',seed_base=item['seed'],task='usb',dual_tactile=True)
    env=None;records=[]
    try:
        env=prepare_environment(sim_args,torch,np)
        dt=float(env.cfg_base.sim.dt)*int(env.cfg_task.env.controlFrequencyInv)
        if abs(dt-item['dt'])>1e-6:raise ValueError('Control period changed')
        ps=np.asarray(env.cfg_task.rl.pos_action_scale);rs=np.asarray(env.cfg_task.rl.rot_action_scale)
        planned=np.clip(planned,-float(env.clip_actions),float(env.clip_actions))
        limits=np.asarray(read(Path('/home/sai/zx/openpi-sim-runtime/runs/contactworld_pi0_v3_13models_40trials_20261003/all_sensor_comparison/manifest.json'))['scales']['local'])
        def physical():return {k:obs_array(env,k) for k in expected}
        def field():return np.stack([obs_array(env,'tactile_force_field_'+side) for side in ('left','right')])
        def reset_prefix():
            np.random.seed(item['seed']);torch.manual_seed(item['seed'])
            env.reset_idx(torch.arange(1,device=env.device,dtype=torch.long))
            env.refresh_all_tensors();env.compute_observations()
            initial={k:np.asarray(original[0][k]) for k in expected}
            first=physical();init_gate=match(first,initial,field(),reference_force[0])
            forces=[];states=[];phys=[]
            for j in range(t):
                if j%4==0:snapshot(env)  # preserve the original observation call sequence
                forces.append(field());a=physical();phys.append(a);states.append(np.r_[a['dof_pos'],a['dof_vel']])
                env.step(torch.as_tensor(np.asarray(data['rows'][j]['action_raw'])[None],device=env.device,dtype=torch.float32))
            pre=snapshot(env);a=physical();forces.append(field());phys.append(a);states.append(np.r_[a['dof_pos'],a['dof_vel']])
            gate=match(a,expected,forces[-1],reference_force[t]);gate['initial']=init_gate
            motion_error=max(float(np.linalg.norm((phys[-1][k]-phys[-2][k])-(np.asarray(original[t][k])-original[t-1][k]))*1000)
                             for k in ('ee_pos','plug_pos'))
            gate['last_step_motion_error_mm']=motion_error
            gate['pass_gate']=bool(gate['pass_gate'] and init_gate['pass_gate'] and motion_error<=.2)
            return pre,np.asarray(forces),np.asarray(states),gate
        kind=args.one_branch;attempt=args.attempt
        if kind not in plan['kinds']:raise ValueError('Unknown branch')
        meta=out/f'{kind}_{attempt}.json';npz=out/f'{kind}_{attempt}.npz'
        pre,fh,sh,gate=reset_prefix()
        record=dict(kind=kind,attempt=attempt,accepted=gate['pass_gate'],gate=gate)
        if gate['pass_gate']:
            act=action_schedule(kind,planned,expected['socket_quat'],ps,rs,plan['horizon'])
            # Clipping could silently change the physical intervention; reject instead.
            if np.abs(act).max()>float(env.clip_actions)+1e-6:
                record.update(accepted=False,reason='action_clip_would_change_schedule')
            else:
                values={k:[] for k in ('ee_pos','ee_quat','plug_pos','plug_quat','socket_pos','socket_quat','success','force_local','rgb','contact_force')}
                def append():
                    for k in ('ee_pos','ee_quat','plug_pos','plug_quat','socket_quat'):values[k].append(obs_array(env,k))
                    values['socket_pos'].append(obs_array(env,'socket_pos_gt'));values['success'].append(bool(env._check_success()[0].item()))
                    values['force_local'].append(field());values['rgb'].append(video_frame(env))
                    values['contact_force'].append(env.contact_force_pairwise[0,env.plug_body_id_env,env.socket_body_id_env].detach().cpu().numpy().copy())
                append()
                for a in act:
                    env.step(torch.as_tensor(a[None],device=env.device));append()
                values={k:np.asarray(v) for k,v in values.items()};values['action']=act
                save(npz,**values);effect=branch_effect(values)
                record['effect']=effect;record['file']=str(npz)
                if kind=='continue':
                    error=max(float(np.linalg.norm(values[k][4]-np.asarray(original[t+4][k]))*1000)
                              for k in ('plug_pos','ee_pos'))
                    angle_error=max(float(rotation_distance(values[k][4],np.asarray(original[t+4][k])))
                                    for k in ('plug_quat','ee_quat'))
                    force_error=float(np.abs(values['force_local'][4]-reference_force[t+4]).mean())
                    record['continue_first4_match']=dict(position_mm=error,angle_deg=angle_error,
                        force_mae=force_error,passed=bool(error<=.5 and angle_error<=1. and
                            force_error<=max(3e-5,.5*float(np.abs(reference_force[t+4]).mean()))))
                if kind=='continue' and not (out/'pre.npz').exists():
                    ah=np.asarray([r['action_sim_clipped'] for r in data['rows'][:t]],np.float32)
                    ff=fh.reshape(len(fh),-1);baseline,valid=causal_baseline(ff,sh,ah)
                    save(out/'pre.npz',force_history=ff,proprio_history=sh,action_history=ah,
                         baseline_history=baseline,baseline_valid=valid,front=pre['front'],wrist=pre['wrist'],cloud_xyz=pre['cloud_xyz'])
                video=out/f'{kind}.mp4'
                if not video.exists():encode(video,values['rgb'],values['force_local'],dt,limits)
        write(meta,record)
        print(item['id'],kind,attempt,'accepted',record['accepted'],flush=True)
    finally:
        if env is not None:env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
