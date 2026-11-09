#!/usr/bin/env python3
"""Standalone V2 experiment: (RGB, raw unaligned depth, ONE CAD) -> ONE SE(3).

Designed to run at the root of openpi-3d-tactile beside fast_cad_pose_spatial.py.
No existing source files, data or checkpoints are modified.

Commands:
    python cad_pose_rgbd_single_v2.py cache --tasks 1030_yellow_socket --max-episodes 2
    python cad_pose_rgbd_single_v2.py train --allow-partial --epochs 1 --steps-per-epoch 10
    python cad_pose_rgbd_single_v2.py cache
    python cad_pose_rgbd_single_v2.py train --epochs 100 --patience 30
    python cad_pose_rgbd_single_v2.py evaluate --checkpoint PATH_TO_BEST_PT

RGB and depth have separate small CNN encoders because the raw front depth is NOT
registered to the RGB image. Do not interpret same pixel locations as aligned.
All CAD queries share one set of model weights. No object IDs, segmentation,
localization head, extra supervision, or tracking.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

TASK_NAMES = ("1018_peg_all", "1028_press_button", "1030_yellow_socket", "1102_eraser")
ROLE_NAMES = ("manipulated", "target")
SCHEMA = "single_rgbd_cad_pose_v2_v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def jwrite(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default) + "\n")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"JSON unsupported type: {type(o)}")


def episode_frames(task, ep):
    reader = getattr(task, "reader", None) or getattr(task, "geometry_reader", None)
    if reader is None:
        raise RuntimeError(f"{task.dataset}: no audited geometry reader")
    frames = [int(v) for v in reader.frame_ids(int(ep))]
    if frames != list(range(len(frames))):
        raise RuntimeError(f"{task.dataset} e{ep}: geometry frame IDs non-contiguous")
    return frames


def _open_tasks(names, repo, data_root):
    import fast_cad_pose_spatial as fast
    fast.install()
    from openpi.cad_pose.pipeline import open_tasks
    tasks, reports = open_tasks(
        repo, data_root, names, quick=True, peg_legacy_intrinsics=True
    )
    if set(tasks) != set(names):
        raise RuntimeError("Data contracts failed: " + "; ".join(
            f"{n}: {reports.get(n, {}).get('errors')}" for n in names if n not in tasks))
    return tasks, reports


def safe_label_samples(task, ep):
    """Same fixed-order rigid fit as the original legacy adapter, but exclude
    individual partial-mask/outlier roles rather than inventing an SE(3) label.
    Published PoseRecords keep the existing audited adapter verbatim.
    """
    if task.dataset not in ("1018_peg_all", "1028_press_button"):
        yield from task.iter_episode(int(ep))
        return
    from openpi.cad_pose.legacy import fit_corresponding_rigid_pose
    for fid in episode_frames(task, ep):
        geo=task.reader.get(episode_index=int(ep),frame_index=int(fid))
        gt=np.zeros((2,4,4),dtype=np.float32)
        valid=np.zeros((2,),dtype=bool)
        for role_index,role in enumerate(ROLE_NAMES):
            mask=np.asarray(getattr(geo,f"{role}_mask"),dtype=bool)
            if not mask.any():
                continue
            if not mask.all():
                # An incomplete ordered CAD array cannot define an unambiguous fit.
                continue
            try:
                transform,_=fit_corresponding_rigid_pose(
                    task.ordered_cad[role_index],
                    getattr(geo,f"{role}_xyz_m"),
                    getattr(geo,f"{role}_mask"),
                    max_rmse_m=.0001,
                )
            except ValueError:
                continue
            if not np.isfinite(transform).all():
                continue
            gt[role_index]=transform
            valid[role_index]=True
        if valid.any():
            yield {"frame":int(fid),"valid":valid,"target_transform":gt}


def cache_one_episode(task, ep, cache_dir, *, width, height):
    import cv2
    from openpi.spatial_dataset.source import RawSpatialDataset

    final = cache_dir / task.dataset / f"e{ep:05d}"
    if (final / "meta.json").is_file():
        meta = json.loads((final / "meta.json").read_text())
        if (meta.get("schema"), meta.get("width"), meta.get("height")) != (SCHEMA, width, height):
            raise RuntimeError(f"Existing cache shape/schema mismatch: {final}")
        required = ("rgb.npy", "depth.npy", "transform.npy", "valid.npy")
        if not all((final / x).is_file() for x in required):
            raise RuntimeError(f"Incomplete existing cache: {final}")
        return "skip", int(meta["n"]), int(meta["valid_roles"])
    if final.exists():
        raise RuntimeError(f"Incomplete cache directory, inspect manually: {final}")

    frames = episode_frames(task, ep)
    n = len(frames)
    tmp = final.with_name(final.name + f".incomplete_{os.getpid()}")
    if tmp.exists():
        raise RuntimeError(f"Incomplete temporary cache, inspect manually: {tmp}")
    tmp.mkdir(parents=True)
    rgb = np.lib.format.open_memmap(tmp / "rgb.npy", mode="w+", dtype=np.uint8, shape=(n,height,width,3))
    depth = np.lib.format.open_memmap(tmp / "depth.npy", mode="w+", dtype=np.uint16, shape=(n,height,width))
    transforms = np.lib.format.open_memmap(tmp / "transform.npy", mode="w+", dtype=np.float32, shape=(n,2,4,4))
    valid = np.lib.format.open_memmap(tmp / "valid.npy", mode="w+", dtype=np.bool_, shape=(n,2))
    transforms[:] = 0
    valid[:] = False

    t0 = time.perf_counter()
    raw = RawSpatialDataset(task.root, episode=int(ep))
    count = 0
    # This is a ONE-TIME decode, not part of the training loop.
    for row, video in zip(raw.iter_rows(depth_roles=("front",)), raw.iter_video_frames("front"), strict=True):
        fid = int(row["frame_index"])
        if fid != count or int(video.frame_index) != fid or fid >= n:
            raise RuntimeError(f"RGB/depth/geometry frame mismatch: {task.dataset} e{ep} frame {fid}")
        src_rgb = np.asarray(video.rgb)
        src_d = np.asarray(row["observation.depths.cam_front"])
        if src_rgb.ndim != 3 or src_rgb.shape[2] != 3 or src_rgb.dtype != np.uint8:
            raise RuntimeError(f"Unexpected RGB input: {src_rgb.shape} {src_rgb.dtype}")
        if src_d.ndim != 2 or src_d.dtype != np.uint16:
            raise RuntimeError(f"Unexpected raw depth: {src_d.shape} {src_d.dtype}")
        rgb[fid] = cv2.resize(src_rgb, (width,height), interpolation=cv2.INTER_AREA)
        depth[fid] = cv2.resize(src_d, (width,height), interpolation=cv2.INTER_NEAREST)
        count += 1
    if count != n:
        raise RuntimeError(f"RGB-D {count} frames vs Geometry {n}: {task.dataset} e{ep}")

    # Reuse the ACTUAL audited existing GT adapter; it handles Button's cleaned
    # frame index, Peg fixed-point fitting, and Published pose/geometry validation.
    labels = 0
    for sample in safe_label_samples(task, int(ep)):
        fid = int(sample["frame"])
        if fid < 0 or fid >= n:
            raise RuntimeError(f"GT frame out of range: {task.dataset} e{ep} f{fid}")
        label_valid = np.asarray(sample["valid"], dtype=bool)
        if label_valid.shape != (2,) or not label_valid.any():
            raise RuntimeError(f"invalid label: {task.dataset} e{ep} f{fid}")
        if valid[fid].any():
            raise RuntimeError(f"duplicate label: {task.dataset} e{ep} f{fid}")
        pose = np.asarray(sample["target_transform"], dtype=np.float32)
        if pose.shape != (2,4,4) or not np.isfinite(pose).all():
            raise RuntimeError(f"bad GT transform: {task.dataset} e{ep} f{fid}")
        valid[fid] = label_valid
        transforms[fid] = pose
        labels += int(label_valid.sum())
    if labels == 0:
        raise RuntimeError(f"No valid labels in {task.dataset} e{ep}")
    for a in (rgb,depth,transforms,valid):
        a.flush()
    del rgb,depth,transforms,valid
    meta = {
        "schema":SCHEMA,"task":task.dataset,"episode":int(ep),"n":n,
        "valid_roles":labels,"width":width,"height":height,
        "rgb_depth_alignment":"unaligned; separate encoder branches",
        "image_source":"raw front RGB+raw uint16 front Depth, resized once",
        "pose_gt":"published labels unchanged; legacy per-role fixed-point fit, skip invalid/unfittable",
        "invalid_role_slots":int(n*2-labels),
        "seconds":round(time.perf_counter()-t0,3)
    }
    jwrite(tmp / "meta.json", meta)
    final.parent.mkdir(parents=True, exist_ok=True)
    os.rename(tmp, final)  # atomic on same filesystem, no replacement
    return "created", n, labels


def do_cache(args):
    width,height=args.width,args.height
    if min(width,height)<64 or width%16 or height%16:
        raise ValueError("width/height must be >=64 and divisible by 16")
    names = tuple(args.tasks)
    tasks, reports = _open_tasks(names, args.repo, args.data_root)
    cache_dir = args.cache_dir.resolve()
    config_path=cache_dir/"cache_config.json"
    if config_path.exists():
        cfg=json.loads(config_path.read_text())
        if cfg["width"]!=width or cfg["height"]!=height or cfg["schema"]!=SCHEMA:
            raise RuntimeError("Cache config mismatch; use a new --cache-dir")
    else:
        jwrite(config_path, {"schema":SCHEMA,"width":width,"height":height,
                             "depth_dtype":"uint16 raw, scaled inside model",
                             "script_sha256":sha256(Path(__file__))})
    # Capture the same episode split policy as V1; absent cache episodes remain
    # absent rather than leaking across train and test.
    from openpi.cad_pose.pipeline import splits_for
    all_splits=splits_for(tasks,seed=args.seed)
    split_file=cache_dir/"episode_splits.json"
    split_data={k:{s:[int(x) for x in v] for s,v in ss.items()} for k,ss in all_splits.items()}
    if split_file.exists():
        saved=json.loads(split_file.read_text())
        for n in names:
            if n in saved and saved[n]!=split_data[n]:
                raise RuntimeError(f"Episode split mismatch in cache: {n}")
            saved[n]=split_data[n]
        split_data=saved
    jwrite(split_file,split_data)
    for task_name,task in tasks.items():
        cad=np.asarray(task.ordered_cad,dtype=np.float32)
        if cad.shape!=(2,512,3) or not np.isfinite(cad).all():
            raise RuntimeError(f"Bad CAD shape: {task_name}: {cad.shape}")
        cad_path=cache_dir/task_name/"cad.npy"
        cad_path.parent.mkdir(parents=True,exist_ok=True)
        if cad_path.exists():
            old=np.load(cad_path,allow_pickle=False)
            if not np.array_equal(old,cad):
                raise RuntimeError(f"Cached CAD mismatch: {task_name}")
        else:
            np.save(cad_path,cad,allow_pickle=False)
        ep_ids=list(map(int,task.episodes))
        if args.max_episodes:
            ep_ids=ep_ids[:args.max_episodes]
        for j,ep in enumerate(ep_ids,1):
            kind,n,v=cache_one_episode(task,ep,cache_dir,width=width,height=height)
            print(f"[{task_name}] {j}/{len(ep_ids)} e{ep:05d} {kind} {n} frames {v} labels",flush=True)
    print(f"[OK] Cache ready in {cache_dir}",flush=True)


class CachedPoseDataset(Dataset):
    def __init__(self, cache, tasks, split, *, max_per_role=0):
        self.cache=Path(cache)
        splits=json.loads((self.cache/"episode_splits.json").read_text())
        self.entries=[]
        self.groups=defaultdict(list)
        self.cads={}
        for t_idx,task in enumerate(tasks):
            self.cads[task]=np.load(self.cache/task/"cad.npy",allow_pickle=False).astype(np.float32)
            ep_list=splits[task][split]
            for ep in ep_list:
                p=self.cache/task/f"e{ep:05d}"
                if not (p/"meta.json").exists():
                    continue
                valid=np.load(p/"valid.npy",mmap_mode="r",allow_pickle=False)
                for frame in range(len(valid)):
                    for role in range(2):
                        if valid[frame,role]:
                            idx=len(self.entries)
                            self.entries.append((task,t_idx,int(ep),int(frame),role))
                            self.groups[(task,role)].append(idx)
        if max_per_role:
            keep=set()
            for ids in self.groups.values():
                choose=np.linspace(0,len(ids)-1,num=min(max_per_role,len(ids)),dtype=int)
                keep.update(ids[i] for i in choose)
            self.entries=[v for i,v in enumerate(self.entries) if i in keep]
            self.groups=defaultdict(list)
            for i,(task,_,_,_,role) in enumerate(self.entries):
                self.groups[(task,role)].append(i)
        print(f"[INDEX {split}] items={len(self.entries)} groups="+
              str({f"{t}/{ROLE_NAMES[r]}":len(self.groups[(t,r)]) for t in tasks for r in range(2)}),flush=True)

    @lru_cache(maxsize=12)
    def arrays(self,task,ep):
        p=self.cache/task/f"e{ep:05d}"
        return tuple(np.load(p/f"{x}.npy",mmap_mode="r",allow_pickle=False)
                     for x in ("rgb","depth","transform","valid"))

    def __len__(self):return len(self.entries)

    def __getitem__(self,i):
        task,t_idx,ep,f,role=self.entries[i]
        rgb,depth,pose,valid=self.arrays(task,ep)
        if not valid[f,role]:
            raise RuntimeError("Cache validity changed during training")
        # Copy before returning: memory-mapped arrays are read-only.
        return {
            "rgb":np.array(rgb[f],copy=True),
            "depth":np.array(depth[f],copy=True),
            "cad":np.array(self.cads[task][role],copy=True),
            "gt":np.array(pose[f,role],copy=True),
            "other_t":np.array(pose[f,1-role,:3,3],copy=True),
            "other_valid":bool(valid[f,1-role]),
            "task":t_idx,"role":role,"episode":ep,"frame":f,
        }


class BalancedRoleBatchSampler(Sampler):
    """Each batch equally samples 4 tasks x 2 roles with replacement."""
    def __init__(self,ds,tasks,batch,steps,seed=42):
        self.groups=[ds.groups[(t,r)] for t in tasks for r in range(2)]
        if any(len(g)==0 for g in self.groups):
            raise RuntimeError("Some task/role has no training data; cache full task first")
        if batch%len(self.groups):
            raise ValueError(f"batch size {batch} must be divisible by {len(self.groups)}")
        self.batch,self.steps,self.seed=batch,steps,seed
        self.epoch=0
    def __len__(self): return self.steps
    def __iter__(self):
        rng=np.random.default_rng(self.seed+self.epoch)
        for _ in range(self.steps):
            ids=[]
            for g in self.groups:
                ix=rng.integers(0,len(g),size=self.batch//len(self.groups))
                ids.extend(g[int(j)] for j in ix)
            rng.shuffle(ids)
            yield ids


def _g(n,c_in,c_out):
    return nn.Sequential(nn.Conv2d(c_in,c_out,3,stride=2,padding=1),
                         nn.GroupNorm(min(8,c_out),c_out),nn.SiLU())


class SingleCadRGBDPose(nn.Module):
    """Two unregistered image branches; one CAD query; one 6D-pose head."""
    def __init__(self, dim=128, heads=4):
        super().__init__()
        self.rgb_net=nn.Sequential(_g(0,3,24),_g(0,24,48),_g(0,48,96),_g(0,96,128))
        self.depth_net=nn.Sequential(_g(0,1,16),_g(0,16,32),_g(0,32,64),_g(0,64,96))
        # Never concatenate RGB/depth features pixelwise: the raw depth is
        # unregistered to the RGB image. Both instead remain separate tokens.
        self.rgb_proj=nn.Conv2d(128,dim,1)
        self.depth_proj=nn.Conv2d(96,dim,1)
        self.modality=nn.Parameter(torch.zeros(2,dim))
        self.pos=nn.Sequential(nn.Linear(2,dim),nn.SiLU(),nn.Linear(dim,dim))
        self.cad=nn.Sequential(nn.Linear(3,64),nn.SiLU(),nn.Linear(64,dim),nn.SiLU())
        self.cad_fuse=nn.Linear(2*dim,dim)
        self.cross1=nn.MultiheadAttention(dim,heads,batch_first=True)
        self.cross2=nn.MultiheadAttention(dim,heads,batch_first=True)
        self.norm1=nn.LayerNorm(dim)
        self.norm2=nn.LayerNorm(dim)
        self.norm_vis=nn.LayerNorm(dim)
        self.ffn=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,2*dim),nn.SiLU(),nn.Linear(2*dim,dim))
        self.head=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim),nn.SiLU(),nn.Linear(dim,9))
        nn.init.zeros_(self.head[-1].weight)
        with torch.no_grad():
            self.head[-1].bias.copy_(torch.tensor([0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0]))

    def forward(self,rgb,depth,cad):
        # rgb: [B,3,H,W] float [0,1]; depth: raw meters clipped/normalized [0,1].
        ir=self.rgb_proj(self.rgb_net(rgb))
        dr=self.depth_proj(self.depth_net(depth))
        b,d,h,w=ir.shape
        ys=torch.linspace(-1,1,h,device=ir.device,dtype=ir.dtype)
        xs=torch.linspace(-1,1,w,device=ir.device,dtype=ir.dtype)
        grid=torch.stack(torch.meshgrid(ys,xs,indexing="ij"),dim=-1).reshape(-1,2)
        pos=self.pos(grid).unsqueeze(0)
        vis_rgb=ir.flatten(2).transpose(1,2)+pos+self.modality[0]
        vis_dep=dr.flatten(2).transpose(1,2)+pos+self.modality[1]
        vis=self.norm_vis(torch.cat([vis_rgb,vis_dep],dim=1))
        cf=self.cad(cad)
        q=self.cad_fuse(torch.cat([cf.mean(dim=1),cf.amax(dim=1)],dim=-1)).unsqueeze(1)
        q=q+self.cross1(self.norm1(q),vis,vis,need_weights=False)[0]
        q=q+self.ffn(q)
        q=q+self.cross2(self.norm2(q),vis,vis,need_weights=False)[0]
        raw=self.head(q.squeeze(1))
        t=raw[:,:3]
        r6=raw[:,3:]
        b1=F.normalize(r6[:,:3],dim=-1,eps=1e-6)
        a2=r6[:,3:]
        b2=F.normalize(a2-(a2*b1).sum(-1,keepdim=True)*b1,dim=-1,eps=1e-6)
        b3=torch.cross(b1,b2,dim=-1)
        r=torch.stack([b1,b2,b3],dim=-1)
        return t,r


def get_tensors(batch,device,*,depth_scale=0.001,max_depth_m=2.0):
    rgb=batch["rgb"].to(device,non_blocking=True).permute(0,3,1,2).float().div_(255)
    dep=batch["depth"].to(device,non_blocking=True).unsqueeze(1).float()
    dep=(dep*depth_scale).clamp_(0,max_depth_m)/max_depth_m
    cad=batch["cad"].to(device,non_blocking=True).float()
    gt=batch["gt"].to(device,non_blocking=True).float()
    return rgb,dep,cad,gt


def pose_objective(t,r,cad,gt):
    tgt=gt[:,:3,3]
    rot=gt[:,:3,:3]
    pred_xyz=torch.einsum("bij,bnj->bni",r,cad)+t[:,None,:]
    gt_xyz=torch.einsum("bij,bnj->bni",rot,cad)+tgt[:,None,:]
    trans=torch.linalg.vector_norm(t-tgt,dim=-1).mean()
    add=torch.linalg.vector_norm(pred_xyz-gt_xyz,dim=-1).mean()
    rotm=(r-rot).square().mean()
    return 3*trans+add+0.1*rotm,{"trans_m":float(trans.detach()),"add_m":float(add.detach())}


def build_loader(ds,*,batch_size,num_workers,steps=0,seed=42,tasks=TASK_NAMES):
    kwargs=dict(num_workers=num_workers,pin_memory=torch.cuda.is_available())
    if num_workers:
        kwargs.update(persistent_workers=True,prefetch_factor=2)
    if steps:
        sampler=BalancedRoleBatchSampler(ds,tasks,batch_size,steps,seed)
        return DataLoader(ds,batch_sampler=sampler,**kwargs),sampler
    return DataLoader(ds,batch_size=batch_size,shuffle=False,**kwargs),None


def _device(name):
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def _cache_preflight(args,*,allow_partial):
    cfg=json.loads((args.cache_dir/"cache_config.json").read_text())
    if cfg["schema"]!=SCHEMA:
        raise RuntimeError("Unexpected cache schema")
    splits=json.loads((args.cache_dir/"episode_splits.json").read_text())
    for task in TASK_NAMES:
        for sp in ("train","val","test"):
            expected=splits[task][sp]
            have=sum((args.cache_dir/task/f"e{x:05d}"/"meta.json").is_file() for x in expected)
            if not allow_partial and have!=len(expected):
                raise RuntimeError(f"Incomplete cache {task}/{sp}: {have}/{len(expected)}; run cache")
            print(f"[CACHE] {task} {sp} {have}/{len(expected)} episodes",flush=True)
    return cfg


def train(args):
    cfg=_cache_preflight(args,allow_partial=args.allow_partial)
    out=args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory nonempty: {out}")
    out.mkdir(parents=True,exist_ok=True)
    jwrite(out/"config.json",{"schema":SCHEMA,"script_sha256":sha256(Path(__file__)),
        "args":{k:v for k,v in vars(args).items() if k!="fn"},"cache_config":cfg,"depth_scale":args.depth_scale})
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device=_device(args.device)
    tr=CachedPoseDataset(args.cache_dir,TASK_NAMES,"train")
    val=CachedPoseDataset(args.cache_dir,TASK_NAMES,"val",max_per_role=args.val_per_role)
    if len(val)==0:
        raise RuntimeError("Empty validation split; cache more episodes")
    train_loader,batchsampler=build_loader(tr,batch_size=args.batch_size,num_workers=args.workers,
        steps=args.steps_per_epoch,seed=args.seed,tasks=TASK_NAMES)
    val_loader,_=build_loader(val,batch_size=args.batch_size,num_workers=args.workers)
    model=SingleCadRGBDPose().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    scaler=None # bf16 automatic mixed precision does not need GradScaler
    total=sum(p.numel() for p in model.parameters())
    print(f"[MODEL] params={total:,}; {device}; rgb/depth separate raw branches",flush=True)
    best=float("inf")
    bad=0
    for epoch in range(args.epochs):
        t0=time.monotonic()
        model.train()
        batchsampler.epoch=epoch
        losses=[]
        for i,batch in enumerate(train_loader):
            rgb,dep,cad,gt=get_tensors(batch,device,depth_scale=args.depth_scale)
            rgb=(rgb*torch.empty((rgb.shape[0],1,1,1),device=device).uniform_(.9,1.1)).clamp_(0,1)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=="cuda",dtype=torch.bfloat16):
                t,r=model(rgb,dep,cad)
                loss,_=pose_objective(t,r,cad,gt)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss: e{epoch} step{i}")
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            if args.log_steps and (i+1)%args.log_steps==0:
                print(f"[STEP] e{epoch} {i+1}/{len(train_loader)} train_loss={np.mean(losses[-args.log_steps:]):.5f}",flush=True)
        train_s=time.monotonic()-t0
        model.eval()
        val_losses=[]
        with torch.inference_mode():
            for batch in val_loader:
                rgb,dep,cad,gt=get_tensors(batch,device,depth_scale=args.depth_scale)
                with torch.autocast(device_type=device.type,enabled=device.type=="cuda",dtype=torch.bfloat16):
                    t,r=model(rgb,dep,cad)
                    loss,_=pose_objective(t,r,cad,gt)
                val_losses.append(float(loss.detach()))
        vl=float(np.mean(val_losses))
        improved=vl < best-1e-6
        if improved:
            best=vl; bad=0
            ckpt={"schema":SCHEMA,"model_state_dict":model.state_dict(),"epoch":epoch,
                  "optimizer_state_dict":opt.state_dict(),"best_validation_loss":best,
                  "params":total,"depth_scale":args.depth_scale,"width":cfg["width"],"height":cfg["height"],
                  "script_sha256":sha256(Path(__file__)),"tasks":TASK_NAMES}
            torch.save(ckpt,out/"best.pt")
            jwrite(out/"best.json",{"checkpoint":str((out/"best.pt").resolve()),
                                    "epoch":epoch,"validation_loss":best})
        else:
            bad+=1
        log={"epoch":epoch,"train_loss":float(np.mean(losses)),"validation_loss":vl,
             "train_seconds":round(train_s,2),"epoch_seconds":round(time.monotonic()-t0,2),
             "samples_per_second":round(args.steps_per_epoch*args.batch_size/train_s,1),
             "best_validation_loss":best,"best_epoch":json.loads((out/"best.json").read_text())["epoch"],
             "patience_counter":bad,"params":total}
        with (out/"training.jsonl").open("a") as f:
            f.write(json.dumps(log)+"\n")
        print(json.dumps(log),flush=True)
        if bad>=args.patience:
            print(f"[EARLY_STOP] patience={args.patience}, best={best:.6f}",flush=True)
            break
    print(f"[DONE] best: {out/'best.json'}",flush=True)


def summarize(a):
    a=np.asarray(a,dtype=float)
    return {"count":int(len(a)),"mean":round(float(a.mean()),3),
        "median":round(float(np.median(a)),3),"p90":round(float(np.quantile(a,.9)),3)} if len(a) else {"count":0}


def evaluate(args):
    cfg=_cache_preflight(args,allow_partial=args.allow_partial)
    device=_device(args.device)
    ds=CachedPoseDataset(args.cache_dir,TASK_NAMES,args.split,max_per_role=args.max_per_role)
    if not len(ds):raise RuntimeError("No evaluation samples")
    loader,_=build_loader(ds,batch_size=args.batch_size,num_workers=args.workers)
    ckpt=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    if ckpt.get("schema")!=SCHEMA:
        raise RuntimeError("Checkpoint is not the RGB-D single CAD model")
    if (ckpt["width"],ckpt["height"])!=(cfg["width"],cfg["height"]):
        raise RuntimeError("Cache/checkpoint image resolution mismatch")
    model=SingleCadRGBDPose().to(device)
    model.load_state_dict(ckpt["model_state_dict"],strict=True)
    model.eval()
    records=[]
    per=defaultdict(lambda:defaultdict(list))
    t0=time.monotonic()
    with torch.inference_mode():
        for i,batch in enumerate(loader):
            rgb,dep,cad,gt=get_tensors(batch,device,depth_scale=ckpt["depth_scale"])
            with torch.autocast(device_type=device.type,enabled=device.type=="cuda",dtype=torch.bfloat16):
                t,r=model(rgb,dep,cad)
            t=t.float();r=r.float()
            geot=torch.linalg.vector_norm(t-gt[:,:3,3],dim=-1)*1000
            rt=gt[:,:3,:3]
            cos=((r.transpose(1,2)@rt).diagonal(dim1=1,dim2=2).sum(-1)-1)/2
            deg=torch.rad2deg(torch.acos(cos.clamp(-1+1e-6,1-1e-6)))
            pxyz=torch.einsum("bij,bnj->bni",r,cad)+t[:,None,:]
            gxyz=torch.einsum("bij,bnj->bni",rt,cad)+gt[:,:3,3][:,None,:]
            add=torch.linalg.vector_norm(pxyz-gxyz,dim=-1).mean(-1)*1000
            wrong=torch.linalg.vector_norm(t-batch["other_t"].to(device),dim=-1)*1000
            for j in range(len(geot)):
                name=TASK_NAMES[int(batch["task"][j])]
                role=ROLE_NAMES[int(batch["role"][j])]
                key=f"{name}/{role}"
                for key2 in ("overall",name,key):
                    per[key2]["translation_mm"].append(float(geot[j]))
                    per[key2]["rotation_deg"].append(float(deg[j]))
                    per[key2]["add_mm"].append(float(add[j]))
                other_valid=bool(batch["other_valid"][j])
                cross=bool(other_valid and wrong[j]<geot[j])
                for key2 in ("overall",name,key):
                    if other_valid:
                        per[key2]["wrong_object_closer"].append(int(cross))
                if args.save_rows:
                    records.append({"task":name,"role":role,"episode":int(batch["episode"][j]),
                        "frame":int(batch["frame"][j]),"translation_mm":float(geot[j]),
                        "rotation_deg":float(deg[j]),"add_mm":float(add[j]),
                        "other_object_closer":cross,"other_valid":other_valid,
                        "pred_t":t[j].cpu().tolist(),"gt_t":gt[j,:3,3].cpu().tolist()})
            if (i+1)%25==0:
                print(f"[EVAL] batches {i+1}/{len(loader)}",flush=True)
    results={"checkpoint":str(args.checkpoint.resolve()),"split":args.split,
             "model":SCHEMA,"params":sum(p.numel() for p in model.parameters()),
             "evaluation_seconds":round(time.monotonic()-t0,2),"metrics":{}}
    for name,p in per.items():
        results["metrics"][name]={metric:summarize(values) for metric,values in p.items()}
    jwrite(args.output,results)
    if args.save_rows:
        path=args.output.with_suffix(".rows.jsonl")
        with path.open("w") as f:
            for v in records:f.write(json.dumps(v)+"\n")
    for name in ("overall",*TASK_NAMES):
        s=results["metrics"].get(name,{})
        print(f"[{name}] t={s.get('translation_mm',{}).get('mean','?')}mm "+
              f"rot={s.get('rotation_deg',{}).get('mean','?')}deg "+
              f"ADD={s.get('add_mm',{}).get('mean','?')}mm "+
              f"wrong={s.get('wrong_object_closer',{}).get('mean','?')}",flush=True)
    print(f"[DONE] {args.output}",flush=True)


def build_parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=p.add_subparsers(dest="command",required=True)
    c=sub.add_parser("cache",help="One-time RGB and raw depth cache, audited unchanged GT labels")
    c.add_argument("--repo",type=Path,default=ROOT)
    c.add_argument("--data-root",type=Path,default=ROOT/"data")
    c.add_argument("--cache-dir",type=Path,default=ROOT/"output/cad_pose_rgbd_v2/cache")
    c.add_argument("--tasks",nargs="+",choices=TASK_NAMES,default=list(TASK_NAMES))
    c.add_argument("--max-episodes",type=int,default=0,help="0=all; 1-2 for pipeline smoke")
    c.add_argument("--width",type=int,default=256)
    c.add_argument("--height",type=int,default=192)
    c.add_argument("--seed",type=int,default=42)
    c.set_defaults(fn=do_cache)
    tr=sub.add_parser("train",help="Balanced 4-task x 2-object training")
    tr.add_argument("--cache-dir",type=Path,default=ROOT/"output/cad_pose_rgbd_v2/cache")
    tr.add_argument("--output-dir",type=Path,required=True)
    tr.add_argument("--device",default="cuda:0")
    tr.add_argument("--batch-size",type=int,default=32)
    tr.add_argument("--steps-per-epoch",type=int,default=1000)
    tr.add_argument("--epochs",type=int,default=100)
    tr.add_argument("--patience",type=int,default=30)
    tr.add_argument("--lr",type=float,default=0.0003)
    tr.add_argument("--workers",type=int,default=4)
    tr.add_argument("--seed",type=int,default=42)
    tr.add_argument("--val-per-role",type=int,default=128)
    tr.add_argument("--depth-scale",type=float,default=.001)
    tr.add_argument("--log-steps",type=int,default=100)
    tr.add_argument("--allow-partial",action="store_true",help="smoke only; allow incomplete dataset cache")
    tr.set_defaults(fn=train)
    e=sub.add_parser("evaluate",help="Test per task/object + cross-object confusion")
    e.add_argument("--cache-dir",type=Path,default=ROOT/"output/cad_pose_rgbd_v2/cache")
    e.add_argument("--checkpoint",type=Path,required=True)
    e.add_argument("--output",type=Path,required=True)
    e.add_argument("--split",choices=("train","val","test"),default="test")
    e.add_argument("--device",default="cuda:0")
    e.add_argument("--batch-size",type=int,default=32)
    e.add_argument("--workers",type=int,default=4)
    e.add_argument("--max-per-role",type=int,default=0,help="0=all")
    e.add_argument("--save-rows",action="store_true")
    e.add_argument("--allow-partial",action="store_true")
    e.set_defaults(fn=evaluate)
    return p


def main():
    args=build_parser().parse_args()
    # CPU oversubscription was a significant issue in original v1 data processing.
    torch.set_num_threads(min(4,max(1,os.cpu_count() or 1)))
    args.fn(args)

if __name__=="__main__":main()
