#!/usr/bin/env python3
"""Run paired target-shift interventions with fixed diffusion noise per query step."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from contactworld_v3_eval_inputs import CONFIGS

REPO=Path(__file__).resolve().parents[2]
SOURCE=Path('/home/sai/zx/openpi-sim-runtime/third_party/ContactWorld-bilateral-probe')
CHECKPOINTS=Path('/home/sai/zsq/FactileLDM/1002_ckpt')
OUTPUT=Path('/home/sai/zx/openpi-sim-runtime/runs/contactworld_target_perturb_20261003')
CONDITIONS=[('zero',0,0),('x_plus3',3,0),('x_minus3',-3,0),('y_plus3',0,3),('y_minus3',0,-3)]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--configs',nargs='+',choices=CONFIGS,default=[CONFIGS[0],CONFIGS[12]])
    p.add_argument('--seeds',nargs='+',type=int,default=[10005,10017])
    p.add_argument('--conditions',nargs='+',choices=[x[0] for x in CONDITIONS],default=[x[0] for x in CONDITIONS])
    p.add_argument('--source',type=Path,default=SOURCE);p.add_argument('--checkpoints',type=Path,default=CHECKPOINTS)
    p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--gpu',default='0')
    p.add_argument('--noise-seed',type=int,default=20261004);p.add_argument('--max-steps',type=int,default=180)
    p.add_argument('--limit-new',type=int);p.add_argument('--check-only',action='store_true')
    a=p.parse_args();conditions={x[0]:x for x in CONDITIONS}
    plan={'configs':a.configs,'seeds':a.seeds,'conditions':a.conditions,'noise_seed':a.noise_seed,
          'max_steps':a.max_steps,'source':str(a.source.resolve()),'checkpoints':str(a.checkpoints.resolve())}
    a.output.mkdir(parents=True,exist_ok=True);planfile=a.output/'plan.json'
    if planfile.exists() and json.loads(planfile.read_text())!=plan:raise ValueError('Existing plan differs')
    planfile.write_text(json.dumps(plan,indent=2)+'\n')
    for model in a.configs:
        ckpt=a.checkpoints/model/'cw3_round1_20k/19999'
        if not (ckpt/'_CHECKPOINT_METADATA').is_file() or not (ckpt/'params/_METADATA').is_file():raise FileNotFoundError(ckpt)
        print('READY',model,ckpt,flush=True)
    if a.check_only:return
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=a.gpu,JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_MEM_FRACTION='0.55')
    new=0
    for model in a.configs:
        pending=[]
        for seed in a.seeds:
            for label in a.conditions:
                item=a.output/model/f'seed_{seed}'/label
                if (item/'summary.json').exists():
                    summary=json.loads((item/'summary.json').read_text())
                    if (summary['seed']!=seed or summary['config']!=model or
                        summary['offset_requested_xy_mm']!=list(conditions[label][1:]) or
                        summary['fixed_noise_seed_base']!=a.noise_seed or summary['max_steps']!=a.max_steps):
                        raise ValueError(f'Existing trial differs: {item}')
                    continue
                pending.append((seed,label,item))
        if not pending:continue
        ckpt=a.checkpoints/model/'cw3_round1_20k/19999'
        sock=Path('/tmp')/('cw3-pert-'+hashlib.sha256((str(a.output)+model).encode()).hexdigest()[:18]+'.sock')
        if sock.exists():raise FileExistsError(sock)
        folder=a.output/model;folder.mkdir(exist_ok=True)
        with (folder/'policy.log').open('a') as out:
            server=subprocess.Popen([sys.executable,'-u','sim/scripts/serve_contactworld_rgb_policy.py',
                                     '--config',model,'--checkpoint',str(ckpt),'--socket',str(sock),'--num-steps','10'],
                                     cwd=REPO,env=env,stdout=out,stderr=subprocess.STDOUT)
            try:
                for _ in range(240):
                    if sock.exists():break
                    if server.poll() is not None:raise RuntimeError(f'Policy server failed: {folder}/policy.log')
                    time.sleep(.5)
                else:raise TimeoutError('Policy server not ready')
                for seed,label,item in pending:
                    if a.limit_new is not None and new>=a.limit_new:break
                    item.mkdir(parents=True,exist_ok=True)
                    _,dx,dy=conditions[label]
                    cmd=['bash','sim/scripts/run_contactworld_sm120.sh','sim/scripts/eval_contactworld_target_perturb.py',
                         '--source',str(a.source),'--output',str(item),'--socket',str(sock),
                         '--config',model,'--seed',str(seed),'--delta-x-mm',str(dx),'--delta-y-mm',str(dy),
                         '--noise-seed',str(a.noise_seed),'--max-steps',str(a.max_steps)]
                    print('START',model,seed,label,flush=True)
                    with (item/'eval.log').open('w') as log:
                        status=subprocess.run(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
                    if status:raise RuntimeError(f'Simulator failed: {item}/eval.log')
                    result=json.loads((item/'summary.json').read_text())
                    print('DONE',model,seed,label,'ever',result['source_ever'],'min_mm',round(result['min_keypoint_error_mm'],2),flush=True)
                    new+=1
            finally:
                server.terminate()
                try:server.wait(timeout=10)
                except subprocess.TimeoutExpired:server.kill();server.wait()
                sock.unlink(missing_ok=True)
        if a.limit_new is not None and new>=a.limit_new:break
    print('NEW_TRIALS',new,flush=True)

if __name__=='__main__':main()
