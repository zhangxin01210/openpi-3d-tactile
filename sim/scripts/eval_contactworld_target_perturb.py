#!/usr/bin/env python3
"""One ContactWorld USB rollout after a paired millimetre socket shift."""
from __future__ import annotations
import argparse
import json
from multiprocessing.connection import Client
from pathlib import Path
import subprocess
import numpy as np

from contactworld_v3_eval_inputs import policy_input, snapshot, tactile_summary
from eval_contactworld_pi0_v3 import metrics, physical_state, save_inputs
from eval_contactworld_proprio_baseline import prepare_environment, video_frame


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--socket',type=Path,required=True);p.add_argument('--config',required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--delta-x-mm',type=float,default=0)
    p.add_argument('--delta-y-mm',type=float,default=0);p.add_argument('--noise-seed',type=int,default=20261004)
    p.add_argument('--max-steps',type=int,default=180)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    a.task='usb';a.dual_tactile=True;a.seed_base=a.seed
    from isaacgym import gymapi,gymtorch  # noqa: F401
    import torch
    torch.set_num_threads(4)
    env=None;vid=None
    try:
        env=prepare_environment(a,torch,np)
        dt=float(env.cfg_base.sim.dt)*int(env.cfg_task.env.controlFrequencyInv)
        np.random.seed(a.seed);torch.manual_seed(a.seed)
        env.reset_idx(torch.arange(1,device=env.device,dtype=torch.long))
        env.refresh_all_tensors();env.compute_observations()
        unperturbed=physical_state(env)
        offset=np.array([a.delta_x_mm,a.delta_y_mm,0],np.float32)*.001
        env.root_pos[0,env.socket_actor_id_env,:3]+=torch.as_tensor(offset,device=env.device)
        actor_id=env.actor_ids_sim_tensors['socket'].to(dtype=torch.int32)
        env.gym.set_actor_root_state_tensor_indexed(env.sim,gymtorch.unwrap_tensor(env.root_state),
                                                    gymtorch.unwrap_tensor(actor_id),len(actor_id))
        env.gym.simulate(env.sim);env.gym.fetch_results(env.sim,True)
        env.refresh_all_tensors();env.compute_observations()
        perturbed=physical_state(env)
        actual_delta=(np.asarray(perturbed['socket_pos_gt'])-unperturbed['socket_pos_gt'])*1000
        assert np.allclose(actual_delta,[a.delta_x_mm,a.delta_y_mm,0],atol=.0001),actual_delta
        assert np.allclose(perturbed['plug_pos'],unperturbed['plug_pos'],atol=.0001),'Plug moved during setup'
        assert np.allclose(perturbed['dof_pos'],unperturbed['dof_pos'],atol=.0001),'Robot moved during setup'
        before,_=metrics(env);first=snapshot(env)
        vid=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
                              '-s','512x256','-r',f'{1/dt:.6f}','-i','-','-an','-c:v','libx264',
                              '-pix_fmt','yuv420p','-movflags','+faststart',str(a.output/'front_wrist.mp4')],stdin=subprocess.PIPE)
        vid.stdin.write(video_frame(env).tobytes())
        rows=[];snaps=[];chunks=[];forces=[];ever=bool(before['source_success']);first_success=None
        with Client(str(a.socket),family='AF_UNIX',authkey=b'contactworld-rgb') as conn:
            for step in range(a.max_steps):
                if step%4==0:
                    snap=first if step==0 else snapshot(env)
                    request=policy_input(snap,a.config)
                    request['__fixed_noise_seed__']=a.noise_seed+step//4
                    conn.send(request);reply=conn.recv()
                    if 'error' in reply:raise RuntimeError(reply['error'])
                    chunk=np.asarray(reply['actions'],np.float32)
                    if chunk.shape!=(16,6) or not np.isfinite(chunk).all():raise ValueError('Bad policy output')
                    snaps.append((step,snap));chunks.append(chunk.copy());pre_tactile=tactile_summary(snap)
                action=chunk[step%4]
                env.step(torch.as_tensor(action[None],device=env.device))
                current,local=metrics(env);forces.append(local)
                if current['source_success'] and first_success is None:first_success=step
                ever|=bool(current['source_success'])
                rows.append({'step':step,'action_raw':action.tolist(),
                             'action_sim_clipped':np.clip(action,-env.clip_actions,env.clip_actions).tolist(),
                             'noise_seed':a.noise_seed+step//4 if step%4==0 else None,
                             'tactile_pre_action':pre_tactile if step%4==0 else None,
                             **current,**physical_state(env)})
                vid.stdin.write(video_frame(env).tobytes())
        vid.stdin.close()
        if vid.wait():raise RuntimeError('ffmpeg failed')
        vid=None
        save_inputs(a.output/'inputs.npz',snaps,chunks,forces)
        final=rows[-1]
        result={'seed':a.seed,'config':a.config,'max_steps':a.max_steps,
                'offset_requested_xy_mm':[a.delta_x_mm,a.delta_y_mm],
                'offset_actual_xyz_mm':actual_delta.tolist(),
                'fixed_noise_seed_base':a.noise_seed,
                'initial_unperturbed_state':unperturbed,'initial_perturbed_state':perturbed,
                'source_ever':ever,'source_final':bool(final['source_success']),
                'first_success_step':first_success,
                'min_keypoint_error_mm':min([before['keypoint_error_mm']]+[r['keypoint_error_mm'] for r in rows]),
                'final_keypoint_error_mm':final['keypoint_error_mm'],
                'final_plug_socket_xy_mm':final['plug_socket_xy_error_mm'],
                'final_plug_socket_z_mm':final['plug_socket_z_mm']}
        (a.output/'trajectory.json').write_text(json.dumps({'result':result,'rows':rows},indent=2)+'\n')
        (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:result[k] for k in ['seed','config','offset_requested_xy_mm','source_ever','source_final','min_keypoint_error_mm']},indent=2),flush=True)
    finally:
        if vid is not None and vid.poll() is None:vid.stdin.close();vid.wait()
        if env is not None:env.gym.destroy_sim(env.sim)

if __name__=='__main__':main()
