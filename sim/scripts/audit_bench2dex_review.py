#!/usr/bin/env python3
"""Audit a small Bench2Dex sample without treating metadata as success truth."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def scalar(value):
    if isinstance(value, bytes):
        return value.decode()
    return value.item() if isinstance(value, np.generic) else value


def inspect(path, success_module=None):
    with h5py.File(path, "r") as f:
        meta = f["meta"]
        fields = ["collection_mode", "demo_eligible", "success", "success_available",
                  "instruction", "robot_key", "frame_count", "fps", "physics_dt", "step_stride"]
        result = {"path": str(path), "metadata": {k:scalar(meta[k][()]) for k in fields if k in meta}}
        steps = f["time/sim_step"][:]
        cutoff = int(meta["homing_start_sim_step"][()])
        before_home = steps < cutoff
        actions = f["action/commanded"][:]
        valid_action = f["action/action_valid"][:].astype(bool)
        finite_action = np.isfinite(actions).all(axis=1)
        source, count = np.unique(f["action/source"][:], return_counts=True)
        qpos = f["robot/qpos"][:]
        result.update(
            before_homing_frames=int(before_home.sum()),
            usable_action_frames_before_home=int((before_home & valid_action & finite_action).sum()),
            nonfinite_action_rows=np.flatnonzero(~finite_action).tolist(),
            nonfinite_action_rows_marked_valid=int((~finite_action & valid_action).sum()),
            action_sources={scalar(k):int(v) for k,v in zip(source,count)},
            qpos_all_finite=bool(np.isfinite(qpos).all()),
            qpos_max_range_rad=float(np.ptp(qpos[before_home], axis=0).max()),
            sim_step_delta_counts={str(int(k)):int(v) for k,v in zip(*np.unique(np.diff(steps),return_counts=True))},
            objects={}, cameras={}, tactile={})
        for key in f["objects"]:
            poses = f[f"objects/{key}/pose_world"][:]
            result["objects"][key] = dict(initial_xyz=poses[0,:3].tolist(),
                final_before_home_xyz=poses[before_home][-1,:3].tolist(),
                max_position_range_before_home_m=float(np.ptp(poses[before_home,:3],axis=0).max()))
        if success_module is not None:
            trajectories={k:{field:f[f"objects/{k}/{field}"][:] for field in
                             ("pose_world","lin_vel_world","ang_vel_world") if field in f["objects"][k]}
                          for k in f["objects"]}
            flags=[]
            for row in np.flatnonzero(before_home):
                states={k:{field:data[row] for field,data in values.items()} for k,values in trajectories.items()}
                flags.append(bool(success_module.check_success(states,{},{})))
            longest=run=0
            for flag in flags:
                run=run+1 if flag else 0
                longest=max(longest,run)
            result["recomputed_current_source_success"]={
                "before_homing_pass_frames":sum(flags),"last_before_homing_pass":flags[-1],
                "longest_consecutive_pass_frames":longest,
                "threshold_xy_m":success_module._PUZZLE_XY_TOL,
                "threshold_z_m":success_module._PUZZLE_Z_TOL,
                "uses_recorded_velocities":True,
                "checks_orientation":False,
                "targets_absolute_world_xy":True}
        for key, g in f.get("cameras", {}).items():
            info = {k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in g.items() if isinstance(v,h5py.Dataset)}
            if "depth" in g:
                samples = g["depth"][sorted(set([0,len(steps)//2,len(steps)-1]))]
                finite = np.isfinite(samples) & (samples > 0)
                info["depth_valid_fraction_sampled"] = float(finite.mean())
                info["depth_range_m_sampled"] = [float(samples[finite].min()),float(samples[finite].max())] if finite.any() else None
            result["cameras"][key] = info
        if "robot/tactile" in f:
            def visit(name, value):
                if isinstance(value,h5py.Dataset):
                    info=dict(shape=list(value.shape),dtype=str(value.dtype))
                    if value.ndim >= 3 and np.issubdtype(value.dtype,np.number):
                        data=value[:]
                        info.update(finite=bool(np.isfinite(data).all()),nonzero_fraction=float((data!=0).mean()),
                                    maximum=float(np.nanmax(data)))
                    result["tactile"][name]=info
            f["robot/tactile"].visititems(visit)
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("files",type=Path,nargs="+")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--source",type=Path,help="Pinned Bench2Dex source for task-73 success recomputation")
    args=p.parse_args()
    success_module=None
    if args.source:
        import importlib.util
        spec=importlib.util.spec_from_file_location("bench_success73",args.source/"success/custom/task_73_jigsaw_puzzle_assembly.py")
        success_module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(success_module)
    report={"episodes":[inspect(x,success_module) for x in args.files],
            "caveat":"Stored success/demo_eligible flags are reported, not independently verified task outcomes."}
    target_names=["obj_326_piece_0","obj_326_piece_1"]
    report["anchor_initial_xyz_range_m"]={k:np.ptp([e["objects"][k]["initial_xyz"] for e in report["episodes"] if k in e["objects"]],axis=0).tolist()
                                         for k in target_names if all(k in e["objects"] for e in report["episodes"])}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"episodes":len(report["episodes"]),"anchor_initial_xyz_range_m":report["anchor_initial_xyz_range_m"]}))


if __name__=="__main__":
    main()
