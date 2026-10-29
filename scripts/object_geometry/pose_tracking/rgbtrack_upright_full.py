#!/usr/bin/env python3
"""RGBTrack upright first registration + full episode tracking, existing refined candidates.

No modification of repository files or third-party code; wraps the project's
RGBTrackObject runtime only inside this process. The funnel object-local +Z
(bowl-up) axis is constrained ONLY ONCE at first initialization. If RGBTrack
re-registers after occlusion, allow unrestricted estimates; forcing upright then
would be physically incorrect.
Run with the exact same arguments as rgbtrack_pose_eval.py plus:
 --upright-max-tilt-deg 25
 --upright-max-rank 100

Initially run --max-frames 1 to inspect the alternate pose and its 2D reprojection.
Then omit --max-frames for full sequence if the initial pose is acceptable.
Requires the established RGBTrack environment (nvdiffrast, pydeps etc.).
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path
from typing import Any
import numpy as np

# Import beside this file, where the official adapter lives.
try:
    import rgbtrack_adapter as adapter
except ImportError as exc:
    raise SystemExit("Put this file next to rgbtrack_adapter.py inside scripts/object_geometry/pose_tracking") from exc


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _axis_report(candidates: np.ndarray, base_R_cam: np.ndarray, scores: np.ndarray) -> list[dict]:
    out=[]
    for i, pose in enumerate(candidates):
        z_base=base_R_cam @ pose[:3, :3] @ np.array([0.,0.,1.])
        cosine=float(np.clip(z_base[2],-1,1))
        ang=float(np.degrees(np.arccos(cosine)))
        out.append({"rank":i,"signed_bowl_up_tilt_deg":round(ang,4),
                    "bowl_up_vertical_cosine":round(cosine,7),
                    "raw_score":float(scores[i]),
                    "candidate_centered_translation_cam_m":pose[:3,3].tolist()})
    return out


def _project_mesh_canvas(rgb:np.ndarray, vertices:np.ndarray, T_cam_object:np.ndarray, K:np.ndarray,
                         label:str, color:tuple[int,int,int])->np.ndarray:
    import cv2
    canvas=rgb.copy()
    # Fixed deterministic vertex subset; all poses draw the exact same mesh vertices.
    ids=np.linspace(0,len(vertices)-1,min(len(vertices),4000),dtype=np.int64)
    pts_cam=adapter.transform_points(vertices[ids], T_cam_object)
    uv=adapter.project_points(pts_cam,K)
    h,w=canvas.shape[:2]
    good=np.isfinite(uv).all(axis=1)&(pts_cam[:,2]>0.05)
    x=np.rint(np.nan_to_num(uv[:,0],nan=-1)).astype(np.int64)
    y=np.rint(np.nan_to_num(uv[:,1],nan=-1)).astype(np.int64)
    good &= (x>=0)&(y>=0)&(x<w)&(y<h)
    for xx,yy in zip(x[good],y[good]):
        cv2.circle(canvas,(int(xx),int(yy)),1,color,-1)
    cv2.rectangle(canvas,(3,3),(min(w-1,650),32),(15,15,15),-1)
    cv2.putText(canvas,label,(10,23),cv2.FONT_HERSHEY_SIMPLEX,0.53,(255,255,255),1,cv2.LINE_AA)
    return canvas


def main()->int:
    parser=adapter.build_arg_parser()
    parser.description=(parser.description or "")+"\nUpright funnel init: wrap the existing RGBTrack candidate scorer."
    parser.add_argument("--upright-max-tilt-deg",type=float,default=25.,
                        help="Max angle from base_link +Z of funnel CAD +Z at initialization")
    parser.add_argument("--upright-max-rank",type=int,default=150,
                        help="Reject a candidate if ranked worse than this in RGBTrack's original scorer")
    parser.add_argument("--install-runtime-only",action="store_true",
                        help="Install the training pose hooks in-process without running an episode.")
    args=parser.parse_args()
    if not (0<args.upright_max_tilt_deg<90):parser.error("tilt must be (0,90) degrees")
    if args.upright_max_rank<1:parser.error("upright-max-rank must be >=1")
    if "funnel" not in args.objects:parser.error("--objects must include funnel")
    out=Path(args.output).expanduser().resolve()
    if out.exists():
        # Original adapter might make output dirs; do not accidentally overwrite previous runs.
        parser.error(f"Output directory exists; choose a new run name: {out}")
    # Uses exactly the calibrated base<-color transform already used by RGBTrack.
    calib=adapter.load_front_calibration(args.repo_root,args.front_calibration_profile)
    T_base_color=np.asarray(calib.T_base_color,dtype=np.float64)
    base_R_cam=T_base_color[:3,:3]

    ensure_orig=adapter.RGBTrackObject._ensure_estimator
    hooked_ids=set()
    decisions=[]
    selected_once = False
    recovery_register_calls = 0

    def ensure_with_selection(self):
        ensure_orig(self)
        if self.object_name!="funnel" or id(self._estimator) in hooked_ids:
            return
        est=self._estimator
        hooked_ids.add(id(est))
        register_orig=est.register

        def register_with_selection(*reg_args,**reg_kwargs):
            nonlocal selected_once, recovery_register_calls
            default=register_orig(*reg_args,**reg_kwargs)
            if selected_once:
                recovery_register_calls += 1
                print(f"[UPRIGHT INIT] recovery registration #{recovery_register_calls}: "
                      "no upright constraint (object may be tilted after grasp)",flush=True)
                return default
            ranked=_numpy(est.poses)
            scores=_numpy(est.scores).reshape(-1)
            if ranked.ndim!=3 or ranked.shape[1:]!=(4,4) or len(ranked)!=len(scores):
                raise RuntimeError(f"Unexpected RGBTrack candidates/scores: {ranked.shape}/{scores.shape}")
            center_T=_numpy(est.get_tf_to_centered_mesh())
            if not np.allclose(ranked[0]@center_T, np.asarray(default),rtol=1e-3,atol=1e-3):
                raise RuntimeError("RGBTrack centered pose convention differs; refusing to select a candidate")
            ranked_report=_axis_report(ranked,base_R_cam,scores)
            eligible=[r for r in ranked_report[:args.upright_max_rank]
                      if r["signed_bowl_up_tilt_deg"]<=args.upright_max_tilt_deg]
            chosen=eligible[0] if eligible else None
            out.mkdir(parents=True,exist_ok=True)
            report={
              "schema_version":1,"mode":"RGBTrack-scored upright init only",
              "frame_initialization_constraint":"CAD +Z is bowl-up, base_link +Z is gravity-up",
              "constraint_max_tilt_deg":args.upright_max_tilt_deg,
              "max_scored_rank":args.upright_max_rank,
              "num_candidates":len(ranked_report),
              "default_rank0":ranked_report[0],
              "best_allowed_candidate":chosen,
              "eligible_in_top_rank":len(eligible),
              "all_candidates":ranked_report,
              "important_note":"Candidate scores are RGBTrack scorer outputs, not adapter mask-area confidence."
            }
            (out/"upright_init_candidate_audit.json").write_text(
                json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
            vertices=np.asarray(self._mesh.vertices,dtype=np.float64)
            raw_rgb=np.asarray(reg_kwargs.get("rgb",reg_args[1] if len(reg_args)>1 else None))
            K=np.asarray(reg_kwargs.get("K",reg_args[0] if reg_args else None),dtype=np.float64)
            if raw_rgb.ndim==3:
                import cv2
                normal=_project_mesh_canvas(raw_rgb,vertices,np.asarray(default),K,
                        "RGBTrack default | local +Z tilt %.1f deg"%ranked_report[0]["signed_bowl_up_tilt_deg"],(30,220,30))
                cv2.imwrite(str(out/"init_default_mesh_projection.png"),normal[...,::-1])
            if chosen is None:
                near=sorted(ranked_report,key=lambda r:r["signed_bowl_up_tilt_deg"])[:5]
                print("[UPRIGHT INIT] No physically upright candidate in allowed rank range.",flush=True)
                print("[UPRIGHT INIT] Best tilted candidates:",near,flush=True)
                raise RuntimeError("Upright candidate unavailable: see upright_init_candidate_audit.json; no fabricated rotation applied")
            pick=int(chosen["rank"])
            selected_centered=ranked[pick]
            corrected=selected_centered@center_T
            if raw_rgb.ndim==3:
                import cv2
                overlay=_project_mesh_canvas(raw_rgb,vertices,corrected,K,
                     f"RGBTrack upright candidate rank={pick} | tilt={chosen['signed_bowl_up_tilt_deg']:.1f}deg",(30,220,30))
                cv2.imwrite(str(out/"init_upright_mesh_projection.png"),overlay[...,::-1])
            # Upstream register() stores the centered pose and seeds its own PoseTracker
            # with the corresponding translation and Euler angles.  Update BOTH.
            import torch
            from scipy.spatial.transform import Rotation
            est.pose_last=torch.as_tensor(selected_centered,device="cuda",dtype=torch.float32)
            xyz=selected_centered[:3,3].reshape(3,1)
            euler=Rotation.from_matrix(selected_centered[:3,:3]).as_euler("xyz").reshape(3,1)
            est.tracker.initialize(xyz,euler)
            est.xyz=est.pose_last[:3,3]
            est.best_id=pick
            est.track_good=True
            decisions.append(report)
            selected_once = True
            print(f"[UPRIGHT INIT] candidate #{pick} of {len(ranked)}; "
                  f"angle {chosen['signed_bowl_up_tilt_deg']:.1f}deg vs default "
                  f"{ranked_report[0]['signed_bowl_up_tilt_deg']:.1f}deg; "
                  f"scorer={chosen['raw_score']:.5g} vs top={ranked_report[0]['raw_score']:.5g}",flush=True)
            return corrected

        est.register=register_with_selection

    adapter.RGBTrackObject._ensure_estimator=ensure_with_selection
    print("[UPRIGHT INIT] Existing RGBTrack candidate set, scorer and refiner unchanged.",flush=True)
    print("[UPRIGHT INIT] Only funnel's FIRST registration is constrained; later re-registration is unrestricted.",flush=True)
    if args.install_runtime_only:
        print("[UPRIGHT INIT] Hooks installed for causal caller.",flush=True)
        return 0
    rc=adapter.run(args)
    if not decisions:raise RuntimeError("Funnel initial register was never called")
    print("[UPRIGHT INIT] outputs:",out,flush=True)
    print(f"[UPRIGHT INIT] extra recovery registration calls: {recovery_register_calls}",flush=True)
    return int(rc)

if __name__=="__main__":
    raise SystemExit(main())
