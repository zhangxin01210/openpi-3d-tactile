#!/usr/bin/env python3
"""Replay saved simulator actions, verify state match, then run a policy for extra steps."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from multiprocessing.connection import Client
import subprocess
import sys
import time
import numpy as np

from contactworld_v3_eval_inputs import policy_input, snapshot
from eval_contactworld_proprio_baseline import prepare_environment, video_frame
from eval_contactworld_pi0_v3 import metrics, physical_state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--config',required=True)
    p.add_argument('--trajectory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-steps',type=int,default=300)
    p.add_argument('--gpu',default='0')
    p.add_argument('--policy-python',type=Path,default=Path('.venv/bin/python'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    old=json.loads(a.trajectory.read_text());prefix=old['rows'];seed=old['result']['seed'];length=len(prefix)
    if not length<a.max_steps:raise ValueError('max-steps must exceed prefix length')
    a.task='usb';a.dual_tactile=True;a.seed_base=seed
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
    os.environ['JAX_PLATFORMS']='cuda'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='0.55'
    sock=Path('/tmp')/('cw3-cont-'+hashlib.sha256(str(a.output).encode()).hexdigest()[:14]+'.sock')
    if sock.exists():raise FileExistsError(sock)
    logfile=a.output/'policy.log'
    policy_env=os.environ.copy()
    policy_env.pop('PYTHONPATH',None)
    policy_env.pop('PYTHONNOUSERSITE',None)
    with logfile.open('w') as out:
        # Keep the .venv symlink in argv[0]; resolving it loses virtualenv discovery.
        server=subprocess.Popen([str(a.policy_python.absolute()),'-u','sim/scripts/serve_contactworld_rgb_policy.py',
                                 '--config',a.config,'--checkpoint',str(a.checkpoint),'--socket',str(sock),'--num-steps','10'],stdout=out,stderr=subprocess.STDOUT,env=policy_env)
        env=None;ffmpeg=None
        try:
            for _ in range(240):
                if sock.exists():break
                if server.poll() is not None:raise RuntimeError(f'Policy server failed: {logfile}')
                time.sleep(.5)
            else:raise TimeoutError('Policy socket not ready')
            from isaacgym import gymapi  # noqa: F401
            import torch
            torch.set_num_threads(4)
            env=prepare_environment(a,torch,np)
            dt=float(env.cfg_base.sim.dt)*int(env.cfg_task.env.controlFrequencyInv)
            np.random.seed(seed);torch.manual_seed(seed)
            env.reset_idx(torch.arange(1,device=env.device,dtype=torch.long))
            env.refresh_all_tensors();env.compute_observations()
            before=physical_state(env)
            old_init=old['result']['initial_physical_state']
            initial_differences={k:float(np.linalg.norm(np.asarray(before[k])-old_init[k])) for k in before}
            ffmpeg=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
                                     '-s','512x256','-r',f'{1/dt:.6f}','-i','-','-an','-c:v','libx264',
                                     '-pix_fmt','yuv420p','-movflags','+faststart',str(a.output/'front_wrist.mp4')],stdin=subprocess.PIPE)
            ffmpeg.stdin.write(video_frame(env).tobytes())
            records=[]
            for step in range(length):
                action=np.asarray(prefix[step]['action_sim_clipped'],dtype=np.float32)
                env.step(torch.as_tensor(action[None],device=env.device))
                current,_=metrics(env)
                records.append({'step':step,'phase':'replay','action':action.tolist(),**current,**physical_state(env)})
                ffmpeg.stdin.write(video_frame(env).tobytes())
            state180=physical_state(env);last=prefix[-1]
            key_errors={k:float(np.linalg.norm(np.asarray(state180[k])-last[k])) for k in state180}
            extra=[];ever=any(x['source_success'] for x in records)
            with Client(str(sock),family='AF_UNIX',authkey=b'contactworld-rgb') as conn:
                for step in range(length,a.max_steps):
                    if (step-length)%4==0:
                        conn.send(policy_input(snapshot(env),a.config));res=conn.recv()
                        if 'error' in res:raise RuntimeError(res['error'])
                        actions=np.asarray(res['actions'],np.float32)
                        if actions.shape!=(16,6) or not np.isfinite(actions).all():raise ValueError('Bad policy actions')
                    action=actions[(step-length)%4]
                    env.step(torch.as_tensor(action[None],device=env.device))
                    current,_=metrics(env);ever|=bool(current['source_success'])
                    extra.append({'step':step,'phase':'policy_continuation','action':action.tolist(),**current,**physical_state(env)})
                    ffmpeg.stdin.write(video_frame(env).tobytes())
            ffmpeg.stdin.close()
            if ffmpeg.wait():raise RuntimeError('ffmpeg failed')
            ffmpeg=None
            summary={'seed':seed,'config':a.config,'prefix_steps':length,'steps':a.max_steps,
                     'original_ever':old['result']['source_ever'],'initial_differences':initial_differences,
                     'prefix_end_state_differences':key_errors,
                     'prefix_end_keypoint_mm':records[-1]['keypoint_error_mm'],
                     'original_end_keypoint_mm':last['keypoint_error_mm'],
                     'continuation_ever_success':any(x['source_success'] for x in extra),
                     'continuation_final_success':extra[-1]['source_success'],
                     'continuation_min_keypoint_mm':min(x['keypoint_error_mm'] for x in extra),
                     'continuation_end_keypoint_mm':extra[-1]['keypoint_error_mm'],
                     'continuation_end_plug_socket_xy_mm':extra[-1]['plug_socket_xy_error_mm'],
                     'continuation_end_plug_socket_z_mm':extra[-1]['plug_socket_z_mm'],
                     'continuation_first_success_step':next((x['step'] for x in extra if x['source_success']),None)}
            (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            (a.output/'trajectory.json').write_text(json.dumps({'summary':summary,'rows':records+extra})+'\n')
            print(json.dumps(summary,indent=2))
        finally:
            if ffmpeg and ffmpeg.poll() is None:
                ffmpeg.stdin.close();ffmpeg.wait()
            if env is not None:env.gym.destroy_sim(env.sim)
            server.terminate()
            try:server.wait(timeout=10)
            except subprocess.TimeoutExpired:server.kill();server.wait()
            sock.unlink(missing_ok=True)

if __name__=='__main__':main()
