#!/usr/bin/env python3
"""Robot-free tactile intervention + FK motion-intent analysis for UR7e/XHand.

Modes:
  scan: inspect recorded per-taxel causal force, propose (NOT label) change points.
  run:  run ONE checkpoint on same recorded images/pose/geometry with tactile
        variants (actual / historical donor / zero-force); render Cartesian
        predicted hand-motion intent.  Run each checkpoint in a SEPARATE process.
  compare: plot four model results from those saved CSVs.

No robot SDK, no WebSocket, no actuator commands. These are predicted absolute
joint targets; FK assumes the robot could reach them, not actual executed motion.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import numpy as np

from scripts.deploy.run_policy_dry_run import _geometry_from_npz

TACTILE_HISTORY_FIELDS = (
    "grasp_reference_force_norm",
    "grasp_short_delta_force_norm",
    "grasp_contact_valid",
    "grasp_quality",
)
CAUSAL_FIELDS = (
    "ema_before_force_norm", "ema_after_force_norm",
    "current_minus_ema_force_norm", "current_minus_previous_force_norm",
    "reference_valid", "previous_valid", "sample_valid", "alpha",
)
DEFAULT_TASK = ("Pick up the glass funnel, align its stem with the opening of the small "
                "beaker, and insert it until it is fully seated.")
ARM_NAMES = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
             "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
FINGER_LINK_NAMES = (
    "right_hand_thumb_rota_link2", "right_hand_index_rota_link2",
    "right_hand_mid_link2", "right_hand_ring_link2", "right_hand_pinky_link2",
)


def _get_saved(root: Path, frame: int):
    file = root / f"frame_{frame:06d}.npz"
    if not file.is_file():
        raise FileNotFoundError(file)
    with np.load(file, allow_pickle=False) as npz:
        names = ("front_rgb", "left_rgb", "right_rgb", "front_depth_native",
                 "robot_state", "timestamp_s", "frame_index")
        data = {k: np.asarray(npz[k]).copy() for k in names}
        geometry = _geometry_from_npz(npz)
    if int(data["frame_index"]) != frame:
        raise ValueError(f"Frame mismatch: {file}")
    return data, geometry


def _force_norm_from_geometry(geo):
    causal = geo["hand_scene"].get("causal_tactile")
    if causal is None:
        raise ValueError("Processed frame has no causal_tactile; DTCR counterfactual unsupported")
    before = np.asarray(causal["ema_before_force_norm"], dtype=np.float32)
    delta = np.asarray(causal["current_minus_ema_force_norm"], dtype=np.float32)
    valid = np.asarray(causal["sample_valid"], dtype=bool)
    if before.shape != (600,) or delta.shape != (600,):
        raise ValueError(f"Bad tactile causal shape {before.shape}/{delta.shape}")
    return np.maximum(before + delta, 0) * valid


def scan_frames(root: Path, out: Path):
    files = sorted(root.glob("frame_*.npz"))
    if not files:
        raise FileNotFoundError(root)
    rows = []
    for file in files:
        idx = int(file.stem.split("_")[-1])
        # Scan ONLY the saved causal force arrays; avoid decoding all RGB-D
        # and raw state again from the large compressed perception archive.
        pre = "object_geometry__hand_scene__causal_tactile__"
        with np.load(file, allow_pickle=False) as npz:
            geo = {"hand_scene":{"causal_tactile":{
                k: np.asarray(npz[pre+k])
                for k in ("ema_before_force_norm","current_minus_ema_force_norm",
                          "current_minus_previous_force_norm","sample_valid")
            }}}
        norm = _force_norm_from_geometry(geo)
        changes = np.asarray(
            geo["hand_scene"]["causal_tactile"]["current_minus_previous_force_norm"],
            dtype=np.float64)
        rows.append({
            "frame": idx,
            "force_mean": float(norm.mean()),
            "force_p95": float(np.percentile(norm, 95)),
            "force_peak": float(norm.max()),
            "positive_force_change_sum": float(np.maximum(changes, 0).sum()),
            "absolute_force_change_sum": float(np.abs(changes).sum()),
            **{f"finger_{f}_mean": float(norm.reshape(5,120)[f].mean()) for f in range(5)},
        })
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "tactile_scan.csv", rows)
    candidates = sorted(rows, key=lambda r:r["positive_force_change_sum"], reverse=True)[:12]
    print("Largest measured tactile rises (CANDIDATES ONLY, not verified flask contact):")
    for r in sorted(candidates, key=lambda r:r["frame"]):
        print(f'  frame {r["frame"]:4} rise_sum={r["positive_force_change_sum"]:.3f} '
              f'force_mean={r["force_mean"]:.4f}')
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,6), sharex=True)
        ax1.plot([r["frame"] for r in rows],[r["force_mean"] for r in rows])
        ax1.set_ylabel("mean tactile force norm (raw representation)")
        ax2.plot([r["frame"] for r in rows],[r["positive_force_change_sum"] for r in rows])
        ax2.set_ylabel("positive force change (sum)")
        ax2.set_xlabel("recorded frame")
        fig.suptitle("Tactile changes: do not infer flask-contact times without video")
        fig.tight_layout();fig.savefig(out/"tactile_scan.png", dpi=150);plt.close(fig)
    except ImportError:
        pass
    print("Review RGB overlays near candidate frames to mark FIRST contact with the FLASK.")


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to save: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def _change_raw_taxel_force(state, donor, *, zero=False):
    """Change ONLY tactile raw force vectors; preserve proprio, EE pose, temps."""
    from openpi.spatial.config import make_baseline_config
    config = make_baseline_config().tactile
    base = 52
    if state.shape != (1972,) or donor.shape != (1972,):
        raise ValueError("Only the verified 1972-D UR7e/XHand state is allowed")
    if config.num_fingers != 5 or config.taxels_per_finger != 120 or config.tactile_block_size != 384:
        raise ValueError("Unsupported tactile schema")
    result = state.copy()
    for finger in range(config.num_fingers):
        start = base + finger * config.tactile_block_size + config.raw_force_offset
        end = start + 3*config.taxels_per_finger
        if end > base+(finger+1)*config.tactile_block_size:
            raise ValueError("Tactile force offsets outside block")
        result[start:end] = 0 if zero else donor[start:end]
    np.testing.assert_array_equal(result[:52], state[:52])
    return result


def _vary(raw, geo, donor_raw, donor_geo, *, zero=False):
    new_raw = dict(raw)
    new_raw["robot_state"] = _change_raw_taxel_force(raw["robot_state"], donor_raw["robot_state"], zero=zero)
    new_geo = dict(geo)
    orig = geo["hand_scene"]
    donor = donor_geo["hand_scene"]
    changed = dict(orig)
    for name in TACTILE_HISTORY_FIELDS:
        if name not in donor:
            raise KeyError(f"Donor lacks {name}")
        value = np.asarray(donor[name]).copy()
        if zero:
            value = np.zeros_like(value)
        changed[name] = value
    if "causal_tactile" not in donor:
        raise ValueError("Donor has no causal_tactile")
    causal = {}
    for name in CAUSAL_FIELDS:
        value = np.asarray(donor["causal_tactile"][name]).copy()
        if zero:
            if name in ("sample_valid","reference_valid","previous_valid"):
                value = np.ones_like(value, dtype=bool)  # valid physical zero force
            else:
                value = np.zeros_like(value)
        causal[name] = value
    changed["causal_tactile"] = causal
    new_geo["hand_scene"] = changed
    # Object poses and geometry, hand joint positions, images, camera depth,
    # and base-frame tactile taxel xyz stay the SAME for every condition.
    return new_raw, new_geo


def _observation(raw, geo, prompt=DEFAULT_TASK):
    return {
        "prompt": prompt,
        "observation.images.cam_front": raw["front_rgb"],
        "observation.images.cam_left": raw["left_rgb"],
        "observation.images.cam_right": raw["right_rgb"],
        "observation.depths.cam_front": raw["front_depth_native"],
        "observation.state": np.asarray(raw["robot_state"], dtype=np.float32),
        "object_geometry": geo,
        "frame_index": int(raw["frame_index"]),
        "timestamp_s": float(raw["timestamp_s"]),
    }


def _fk_intent(fk, mapping, robot_state, actions, target_xyz):
    from openpi.spatial.kinematics import joint_positions_from_state
    q0 = joint_positions_from_state(robot_state, mapping=mapping, kinematics=fk)
    missing = [name for name in ARM_NAMES if name not in q0]
    if missing: raise ValueError(f"URDF mapping lacks arm joints {missing}")
    initial = fk.compute(q0, root="base_link")
    T0 = initial["right_hand_link"]
    def fingertip_origins_in_palm(transforms, palm):
        # Finger link2 origin is an approximate KINEMATIC landmark, not a
        # calibrated taxel contact point or physical fingertip center.
        return np.stack([
            palm[:3,:3].T @ (transforms[link][:3,3]-palm[:3,3])
            for link in FINGER_LINK_NAMES
        ])
    finger0 = fingertip_origins_in_palm(initial,T0)
    center = np.asarray(target_xyz, dtype=float).mean(axis=0)
    center_xy = center[:2]
    distances=[]; xs=[]; ys=[]; zs=[]; angles=[]
    finger_motion=[]; thumb_index_change=[];thumb_middle_change=[]
    for action in actions:
        q = q0.copy()
        for i, name in enumerate(ARM_NAMES): q[name] = float(action[i])
        for name, index in mapping.items():
            if 28 <= index < 52 and (index - 28) % 2 == 0:
                q[name] = float(action[6+(index-28)//2])
        # Compute BOTH palm displacement and finger articulation. Palm is a
        # fixed child of UR tool0; finger joints do not translate the palm.
        transforms=fk.compute(q, root="base_link")
        T = transforms["right_hand_link"]
        finger = fingertip_origins_in_palm(transforms,T)
        finger_motion.append(float(np.linalg.norm(finger-finger0,axis=1).mean()*1000))
        thumb_index_change.append(float((np.linalg.norm(finger[0]-finger[1])-np.linalg.norm(finger0[0]-finger0[1]))*1000))
        thumb_middle_change.append(float((np.linalg.norm(finger[0]-finger[2])-np.linalg.norm(finger0[0]-finger0[2]))*1000))
        d = (T[:3,3]-T0[:3,3])*1000
        xs.append(float(d[0]));ys.append(float(d[1]));zs.append(float(d[2]))
        distances.append(float((np.linalg.norm(T[:2,3]-center_xy)-np.linalg.norm(T0[:2,3]-center_xy))*1000))
        cos = np.clip((np.trace(T0[:3,:3].T @ T[:3,:3])-1)/2,-1,1)
        angles.append(float(np.degrees(np.arccos(cos))))
    return dict(x_mm=xs,y_mm=ys,z_mm=zs,xy_mm=np.hypot(xs,ys).tolist(),
                rotation_deg=angles,target_centroid_xy_gap_change_mm=distances,
                finger_link2_mean_motion_in_palm_mm=finger_motion,
                thumb_index_link2_aperture_change_mm=thumb_index_change,
                thumb_middle_link2_aperture_change_mm=thumb_middle_change)


def _classify_z(z):
    if z >= 0.5: return "UP"
    if z <= -0.5: return "DOWN"
    return "LEVEL"


def run_model(args):
    import jax
    from openpi.policies.policy_config import create_trained_policy
    from openpi.spatial.kinematics import RobotKinematics, load_state_mapping_csv
    from openpi.training.config import get_config
    root = args.repo_root.resolve()
    fk = RobotKinematics(root/"configs/ur7e_xhand/ur7e_xhand_verified.urdf")
    mapping = load_state_mapping_csv(root/"configs/ur7e_xhand/state_mapping.csv")
    cfg = get_config(args.config)
    checkpoint=args.checkpoint.resolve()
    if not checkpoint.is_dir(): raise FileNotFoundError(checkpoint)
    policy=create_trained_policy(cfg,checkpoint)
    frames = list(range(args.start_frame,args.end_frame+1,args.stride))
    if args.contact_frame not in frames:
        frames=sorted(set([*frames,args.contact_frame]))
    if not (args.start_frame<=args.contact_frame<=args.end_frame):
        raise ValueError("Contact frame must lie in chosen interval")
    donor_pre,geo_pre=_get_saved(args.frames_dir,args.pre_tactile_frame)
    donor_post,geo_post=_get_saved(args.frames_dir,args.post_tactile_frame)
    if args.pre_tactile_frame>=args.contact_frame or args.post_tactile_frame<=args.contact_frame:
        raise ValueError("Choose pre donor strictly BEFORE contact and post donor strictly AFTER")
    noise=np.random.default_rng(args.noise_seed).normal(
        size=(int(cfg.model.action_horizon),int(cfg.model.action_dim))).astype(np.float32)
    outputs=[];comparisons=[]
    for frame in frames:
        raw,geo=_get_saved(args.frames_dir,frame)
        force_actual=_force_norm_from_geometry(geo)
        readings = {}
        variants = [
            ("actual",raw,geo),
            ("pre_touch",*_vary(raw,geo,donor_pre,geo_pre)),
            ("post_touch",*_vary(raw,geo,donor_post,geo_post)),
            ("zero_force",*_vary(raw,geo,raw,geo,zero=True)),
        ]
        for condition,obs_raw,obs_geo in variants:
            # Independent fixed diffusion noise + reset online tactile EMA.
            # Crucial for paired intervention: prohibit noise/state leakage.
            policy._rng=jax.random.key(args.noise_seed)
            pre=getattr(policy,"_input_transform",None)
            for transform in getattr(pre,"transforms",()):
                reset=getattr(transform,"reset_tactile_ema",None)
                if callable(reset): reset()
            result=policy.infer(_observation(obs_raw,obs_geo,args.prompt),noise=noise)["actions"]
            actions=np.asarray(result,dtype=np.float64)
            if actions.ndim!=2 or actions.shape[1]!=18 or not np.isfinite(actions).all():
                raise ValueError(f"Bad predicted actions {actions.shape}")
            actions=actions[:args.chunk_steps]
            intent=_fk_intent(fk,mapping,raw["robot_state"],actions,geo["target_xyz_m"])
            # Store lightweight vectors and absolute 18D targets; DO NOT
            # copy RGBD or prediction checkpoint bytes into the output.
            readings[condition]=(actions,intent)
            np.savez_compressed(args.output_dir/f"actions_{frame:06d}_{condition}.npz",actions=actions)
            row = {
                "model":args.name,"frame":frame,"phase":("before" if frame<args.contact_frame else ("contact" if frame==args.contact_frame else "after")),
                "condition":condition,"donor_frame":(args.pre_tactile_frame if condition=="pre_touch" else (args.post_tactile_frame if condition=="post_touch" else "")),
                "force_actual_mean":float(force_actual.mean()),
                "force_actual_max":float(force_actual.max()),
                "first_z_mm":intent["z_mm"][0],
                "end_z_mm":intent["z_mm"][-1],
                "first_z_label":_classify_z(intent["z_mm"][0]),
                "end_z_label":_classify_z(intent["z_mm"][-1]),
                "end_x_mm":intent["x_mm"][-1],
                "end_y_mm":intent["y_mm"][-1],
                "end_xy_mm":intent["xy_mm"][-1],
                "end_rotation_deg":intent["rotation_deg"][-1],
                "end_finger_link2_mean_motion_in_palm_mm":intent["finger_link2_mean_motion_in_palm_mm"][-1],
                "end_thumb_index_aperture_change_mm":intent["thumb_index_link2_aperture_change_mm"][-1],
                "end_thumb_middle_aperture_change_mm":intent["thumb_middle_link2_aperture_change_mm"][-1],
                "end_target_centroid_xy_gap_change_mm":intent["target_centroid_xy_gap_change_mm"][-1],
            }
            outputs.append(row)
        actual=readings["actual"][1]
        for name in ("pre_touch","post_touch","zero_force"):
            cf=readings[name][1]
            comparisons.append({
                "model":args.name,"frame":frame,
                "phase":("before" if frame<args.contact_frame else ("contact" if frame==args.contact_frame else "after")),
                "intervention":name,
                "change_end_z_mm":cf["z_mm"][-1]-actual["z_mm"][-1],
                "change_end_x_mm":cf["x_mm"][-1]-actual["x_mm"][-1],
                "change_end_y_mm":cf["y_mm"][-1]-actual["y_mm"][-1],
                "change_end_rotation_deg":cf["rotation_deg"][-1]-actual["rotation_deg"][-1],
                "change_end_thumb_index_aperture_mm":cf["thumb_index_link2_aperture_change_mm"][-1]-actual["thumb_index_link2_aperture_change_mm"][-1],
                "actual_end_z_mm":actual["z_mm"][-1],
                "counterfactual_end_z_mm":cf["z_mm"][-1],
                "flip_up_down":_classify_z(actual["z_mm"][-1]) != _classify_z(cf["z_mm"][-1]),
            })
        print(f"[{args.name}] frame {frame}: actual Δz={actual['z_mm'][-1]:+.2f} mm, "
              f"pre-touch={readings['pre_touch'][1]['z_mm'][-1]:+.2f}, "
              f"post-touch={readings['post_touch'][1]['z_mm'][-1]:+.2f}, "
              f"zero={readings['zero_force'][1]['z_mm'][-1]:+.2f}",flush=True)
    _write_csv(args.output_dir/"intent.csv",outputs)
    _write_csv(args.output_dir/"tactile_effect.csv",comparisons)
    (args.output_dir/"analysis_meta.json").write_text(json.dumps({
        "model":args.name,"checkpoint":str(checkpoint),"config":args.config,
        "source_frames":str(args.frames_dir.resolve()),"contact_frame":args.contact_frame,
        "pre_tactile_frame":args.pre_tactile_frame,"post_tactile_frame":args.post_tactile_frame,
        "noise_seed":args.noise_seed,"chunk_steps":args.chunk_steps,"frames":frames,
        "prompt":args.prompt,
        "motion_frame":"right_hand_link (palm), URDF FK, base_link xyz",
        "z_sign":"positive = +base_link Z, negative = -base_link Z; NOT verified insertion axis",
        "disclaimer":"Open-loop predicted ABSOLUTE joint targets. FK assumes perfect following; no robot dynamics, no execution, no confirmed contact.",
    },indent=2)+"\n",encoding="utf-8")


def compare_runs(args):
    runs={}
    meta=None
    for spec in args.run:
        name,path=spec.split("=",1)
        directory=Path(path).resolve()
        current=json.loads((directory/"analysis_meta.json").read_text())
        if meta is None: meta=current
        for field in ("source_frames","contact_frame","pre_tactile_frame","post_tactile_frame","noise_seed","chunk_steps","frames","prompt"):
            if current[field]!=meta[field]:
                raise ValueError(f"Non-matched {field}: {name}, cannot claim same input")
        with (directory/"intent.csv").open(newline="") as f: rows=list(csv.DictReader(f))
        with (directory/"tactile_effect.csv").open(newline="") as f: diffs=list(csv.DictReader(f))
        runs[name]=(rows,diffs)
    if len(runs)<2: raise ValueError("Need >=2 models")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    all_rows = [dict(r, model=name) for name,(rows,_) in runs.items() for r in rows]
    all_diffs = [dict(r,model=name) for name,(_,diffs) in runs.items() for r in diffs]
    _write_csv(args.output_dir/"all_models_intent.csv",all_rows)
    _write_csv(args.output_dir/"all_models_tactile_effect.csv",all_diffs)
    print(f"Confirmed matched recorded frames and fixed intervention donors; contact_frame={meta['contact_frame']}")
    print("Motion is FK of predicted joint TARGETS, not physically executed commands.")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; read all_models_*.csv");return
    for metric,title,filename in [
        ("end_z_mm","Predicted 16-step palm ΔZ (+base Z is up)","end_z_by_model.png"),
        ("end_rotation_deg","Predicted 16-step palm rotation","end_rotation_by_model.png"),
        ("end_target_centroid_xy_gap_change_mm","Change in palm XY gap to flask centroid (NOT mouth)","end_target_xy_gap_change.png"),
        ("end_thumb_index_aperture_change_mm","Predicted thumb-index link2 distance change (negative = closing)","thumb_index_aperture.png"),
    ]:
        fig,ax=plt.subplots(figsize=(12,5))
        for name,(rows,_) in runs.items():
            actual=sorted((r for r in rows if r["condition"]=="actual"),key=lambda r:int(r["frame"]))
            ax.plot([int(r["frame"]) for r in actual],[float(r[metric]) for r in actual],label=name,marker=".")
        ax.axvline(meta["contact_frame"],linestyle="--",label="video-annotated flask contact")
        ax.axhline(0,linewidth=0.7)
        ax.set(xlabel="recorded frame",ylabel=metric,title=title);ax.legend();fig.tight_layout()
        fig.savefig(args.output_dir/filename,dpi=155);plt.close(fig)
    for condition in ("pre_touch","post_touch","zero_force"):
        fig,ax=plt.subplots(figsize=(12,5))
        for name,(_,diffs) in runs.items():
            rows=sorted((r for r in diffs if r["intervention"]==condition),key=lambda r:int(r["frame"]))
            ax.plot([int(r["frame"]) for r in rows],
                    [float(r["change_end_z_mm"]) for r in rows],label=name,marker=".")
        ax.axvline(meta["contact_frame"],linestyle="--",label="video-annotated flask contact")
        ax.axhline(0,linewidth=0.7)
        ax.set(xlabel="recorded frame",ylabel="ΔZ intervention - actual (mm)",
               title=f"Effect on next 16-step palm-Z intention: {condition}")
        ax.legend();fig.tight_layout()
        fig.savefig(args.output_dir/f"tactile_effect_{condition}.png",dpi=155);plt.close(fig)
    print(f"Saved diagnostic plots + paired intervention table: {args.output_dir}")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("scan","run","compare"),required=True)
    p.add_argument("--repo-root",type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument("--frames-dir",type=Path)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--name",type=str)
    p.add_argument("--config",type=str)
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--contact-frame",type=int)
    p.add_argument("--pre-tactile-frame",type=int)
    p.add_argument("--post-tactile-frame",type=int)
    p.add_argument("--start-frame",type=int)
    p.add_argument("--end-frame",type=int)
    p.add_argument("--stride",type=int,default=8)
    p.add_argument("--chunk-steps",type=int,default=16)
    p.add_argument("--noise-seed",type=int,default=42)
    p.add_argument("--prompt",default=DEFAULT_TASK)
    p.add_argument("--run",action="append",default=[],metavar="NAME=DIR")
    args=p.parse_args()
    if args.mode=="compare":
        compare_runs(args);return
    if args.frames_dir is None or not args.frames_dir.is_dir():
        raise ValueError("--frames-dir must point to saved perception/frames")
    if args.mode=="scan":
        scan_frames(args.frames_dir,args.output_dir);return
    required=("name","config","checkpoint","contact_frame","pre_tactile_frame","post_tactile_frame","start_frame","end_frame")
    for arg in required:
        if getattr(args,arg) is None:
            p.error(f"--{arg.replace('_','-')} is required in run mode")
    if args.stride<=0 or not 1<=args.chunk_steps<=50: p.error("invalid stride/chunk-steps")
    args.output_dir.mkdir(parents=True,exist_ok=False)
    run_model(args)


if __name__=="__main__":
    main()
