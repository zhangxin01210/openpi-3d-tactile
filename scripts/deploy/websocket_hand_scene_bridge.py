"""A thin wrapper over the VERIFIED offline online-perception -> Pi0 path.

The robot sends its ordinary Pi0 observation plus hand_scene_history, which
contains raw frames seen since the last policy query. Here, raw frames are
written to an INTERNAL inference-host spool; the SAME causal
run_online_perception.py executes in the perception environment and generates
object_geometry. No HTTP bridge, no new segmentation/pose algorithm, and NO
robot actions are implemented here.

This is a correctness-first blocking implementation, NOT a 15-Hz-safe servo.
The full causal backlog can take seconds; use --dry-run --no-home first.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

from scripts.deploy.run_policy_dry_run import _geometry_from_npz
from scripts.deploy.position_initialization import SCENE_POSITION_TO_EPISODE

# Operator-facing positions 1..6; scene 1 combines E0 flask/E23 funnel clicks.
SELECTABLE_SCENE_POSITIONS=tuple(SCENE_POSITION_TO_EPISODE)

FRAME_KEYS=("front_rgb","left_rgb","right_rgb","front_depth_native",
            "robot_state","timestamp_s","frame_index")


def validate_history(history, *, next_index:int, previous_timestamp:float) -> list[dict]:
    if not isinstance(history,list) or not 1<=len(history)<=64:
        raise ValueError("hand_scene_history must contain 1..64 consecutive raw observations")
    checked=[]
    last=previous_timestamp
    for ordinal,item in enumerate(history):
        if not isinstance(item,dict):
            raise TypeError("hand_scene_history items must be mappings")
        missing=set(FRAME_KEYS)-set(item)
        if missing:
            raise KeyError(f"Missing hand_scene_history fields {sorted(missing)}")
        idx=np.asarray(item["frame_index"])
        stamp=np.asarray(item["timestamp_s"])
        if idx.shape!=() or stamp.shape!=():
            raise ValueError("frame_index and timestamp_s must be scalars")
        frame=int(idx)
        timestamp=float(stamp)
        if frame!=next_index+ordinal:
            raise ValueError(f"History frame discontinuity: got {frame}, expected {next_index+ordinal}")
        if not np.isfinite(timestamp) or timestamp<=last:
            raise ValueError("Non-increasing history timestamp")
        last=timestamp
        entry={k:np.asarray(item[k]) for k in FRAME_KEYS}
        for k in ("front_rgb","left_rgb","right_rgb"):
            rgb=entry[k]
            if rgb.shape!=(480,640,3) or rgb.dtype!=np.uint8:
                raise ValueError(f"Bad history RGB {k}: {rgb.shape} {rgb.dtype}")
        depth=entry["front_depth_native"]
        if depth.shape!=(480,640) or depth.dtype!=np.uint16:
            raise ValueError(f"Bad original LeRobot front depth: {depth.shape} {depth.dtype}")
        state=entry["robot_state"]
        if state.shape!=(1972,) or not np.isfinite(state).all():
            raise ValueError("Bad original 1972-state")
        entry["robot_state"]=state.astype(np.float32,copy=False)
        entry["frame_index"]=np.asarray(frame,dtype=np.int64)
        entry["timestamp_s"]=np.asarray(timestamp,dtype=np.float64)
        checked.append(entry)
    return checked


class HandScenePerceptionBridge:
    """One guarded session, one original-perception subprocess, one trained Pi0."""

    def __init__(self, policy, *, repo_root:Path, perception_python:Path,
                 workdir:Path, anchor_json:Path|None=None, episode:int=23,
                 position_template_episode:int=23, prompt_file:Path|None=None,
                 timeout_s:float=180., sam2_checkpoint:Path|None=None,
                 rgbtrack_weights:Path|None=None, sam2_source:Path|None=None,
                 scale_manifest:Path|None=None, geometry_config:Path|None=None,
                 calibration_profile:Path|None=None,
                 surface_points_sha256:dict[str,str]|None=None):
        self.policy=policy
        self.repo_root=Path(repo_root).resolve()
        # Keep the venv's *entry-point path*, including its python symlink.
        # pathlib.resolve() follows .venv-perception/bin/python to the shared
        # base interpreter, which skips the virtualenv's site-packages (SAM2's
        # hydra-core etc.) even though direct venv Python imports succeed.
        self.perception_python=Path(os.path.abspath(os.path.expanduser(str(perception_python))))
        self.workdir=Path(workdir).resolve()
        self.anchor_json=Path(anchor_json).resolve() if anchor_json is not None else None
        self.episode=episode
        self.position_template_episode=position_template_episode
        self.scene_position=None
        self.geometry_config=Path(geometry_config).resolve() if geometry_config else None
        self.calibration_profile=Path(calibration_profile).expanduser() if calibration_profile else None
        self.surface_points_sha256=dict(surface_points_sha256) if surface_points_sha256 is not None else None
        self.generic_geometry = (self.geometry_config is not None and
                                 json.loads(self.geometry_config.read_text()).get("status") == "human_accepted")
        self.task = (json.loads(self.geometry_config.read_text())["dataset_id"] if self.generic_geometry
                     else self.policy.metadata.get("dataset", "1018_peg_all"))
        if self.generic_geometry and self.policy.metadata.get("dataset", self.task) != self.task:
            raise ValueError("Policy training task and approved geometry task differ")
        if self.geometry_config is not None and self.calibration_profile is None:
            raise ValueError("Task geometry requires the training front calibration profile")
        self.session_counter=0
        self.session_root=None
        self.prompt_file=prompt_file
        self.timeout_s=float(timeout_s)
        if self.timeout_s<=0:raise ValueError("timeout_s must be positive")
        self.sam2_checkpoint=Path(sam2_checkpoint or self.repo_root/"checkpoints/sam2/sam2.1_hiera_small.pt").expanduser().resolve()
        self.rgbtrack_weights=Path(rgbtrack_weights or self.repo_root/"checkpoints/rgbtrack").expanduser().resolve()
        self.sam2_source=Path(sam2_source or Path.home()/".cache/openpi_deploy/sam2").expanduser().resolve()
        self.scale_manifest=scale_manifest or self.repo_root/"data/pose_eval/hand_scene_metric_mesh_scales.json"
        self.next_index=0
        self.previous_timestamp=float("-inf")
        self.session_id=None
        self.worker=None
        self._started=False
        if self.workdir.exists() and not self.workdir.is_dir():
            raise NotADirectoryError(f"Perception spool is not a directory: {self.workdir}")
        if self.anchor_json is not None and not self.anchor_json.is_file():
            raise FileNotFoundError(f"Selected-position reviewed flask anchor missing: {self.anchor_json}")
        if not self.perception_python.is_file():
            raise FileNotFoundError("Original perception virtualenv Python not found")
        # Previous sessions are read-only archives; never overwrite them.
        self.workdir.mkdir(parents=True, exist_ok=True)

    @property
    def metadata(self):
        policy_metadata = dict(self.policy.metadata)
        policy_camera_roles = list(policy_metadata.get("spatial_camera_roles", ()))
        acquisition_camera_roles = list(dict.fromkeys((*policy_camera_roles, "front")))
        return {
            **policy_metadata,
            # Older robot clients use spatial_camera_roles for RGB-D sensor
            # acquisition even when spatial_use_visual=False. Advertise the
            # bridge's Front requirement there for wire compatibility, while
            # retaining the model's actual token-input roles separately.
            "policy_spatial_camera_roles":policy_camera_roles,
            "spatial_camera_roles":acquisition_camera_roles,
            "hand_scene_geometry_service":True,
            "hand_scene_history_required":True,
            "hand_scene_history_max_frames":64,
            # Sensor acquisition is independent of Pi0 spatial token inputs.
            # ho_v3 has spatial_camera_roles=[] but its perception worker still
            # consumes the original Front RGB-D stream.
            "hand_scene_camera_roles":["front"],
            # Retain 1 only as a wire-compatibility key for older clients.
            "hand_scene_positions":[1] if self.generic_geometry else list(SELECTABLE_SCENE_POSITIONS),
            "hand_scene_position_required":not self.generic_geometry,
            "hand_scene_selection_source":"task_single_reference_e00000" if self.generic_geometry else "manual_peg_six_reference",
            "hand_scene_task":self.task,
            "hand_scene_actuation_allowed":True,
            "hand_scene_server_status":"blocking_worker_starts_on_first_request",
            "hand_scene_realtime_validated":False,
        }

    def _worker_command(self):
        cmd=[
            str(self.perception_python),"-u",
            str(self.repo_root/"scripts/deploy/run_online_perception.py"),
            "--repo-root",str(self.repo_root),
            "--replay-dir",str(self.session_root/"raw"),
            "--output-dir",str(self.session_root/"perception"),
            "--sam2-checkpoint",str(self.sam2_checkpoint),
            "--sam2-source",str(self.sam2_source),
            "--weights-root",str(self.rgbtrack_weights),
            "--scale-manifest",str(self.scale_manifest),
            "--front-calibration-profile",
            str(self.calibration_profile or self.repo_root/"configs/ur7e_xhand/front_calibration_candidate.json"),
            "--episode",str(0 if self.generic_geometry else SCENE_POSITION_TO_EPISODE[self.scene_position]),
            "--register-missing-color-depth",
            "--follow",
        ]
        if self.geometry_config is not None:
            cmd += ["--geometry-config", str(self.geometry_config)]
            if self.generic_geometry:
                return cmd
        if self.anchor_json is not None:
            cmd+=["--static-target-anchor-json",str(self.anchor_json),
                  "--static-target-anchor-source","human_reviewed"]
        else:
            # Selected reviewed 2D SAM2 seed points + current frame-0
            # RGBTrack detection. Never silently reuse Episode23's pose.
            cmd+=["--auto-static-flask-from-frame0"]
        if self.prompt_file is not None:
            cmd+=["--prompts",str(self.prompt_file)]
        else:
            cmd+=["--position-template-episode",str(SCENE_POSITION_TO_EPISODE[self.scene_position]),
                  "--template-reviewed-for-current-scene"]
        return cmd

    def _launch(self):
        # Validate the resources the subprocess will actually open. WebSocket
        # startup alone is not evidence that this lazy worker has run.
        required = [self.sam2_source/"sam2/build_sam.py", self.sam2_checkpoint,
                    self.rgbtrack_weights/"2024-01-11-20-02-45/model_best.pth",
                    self.rgbtrack_weights/"2023-10-28-18-33-37/model_best.pth"]
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(f"Perception worker resource missing: {path}")
        env=os.environ.copy()
        prefix=[
            str(self.perception_python.parent),
            str(self.repo_root/"src"),
            str(self.repo_root),
            str(self.repo_root/"scripts/object_geometry/pose_tracking"),
        ]
        env["PATH"]=os.pathsep.join([str(self.perception_python.parent),env.get("PATH","")])
        env["PYTHONPATH"]=os.pathsep.join(prefix[1:]+[env.get("PYTHONPATH","")])
        # Do not let a JAX model in the server compete with the RGBTrack worker.
        env["JAX_PLATFORMS"]="cpu"
        print(f"[PERCEPTION] task={self.task} python={self.perception_python} "
              f"sam2_source={self.sam2_source} sam2_checkpoint={self.sam2_checkpoint} "
              f"rgbtrack_weights={self.rgbtrack_weights}; starting real worker",flush=True)
        logfile=(self.session_root/"perception.log").open("ab")
        try:
            self.worker=subprocess.Popen(
                self._worker_command(),cwd=str(self.repo_root),env=env,
                stdout=logfile,stderr=subprocess.STDOUT,
            )
        finally:
            logfile.close()
        self._started=True

    def _publish(self,frames):
        for packet in frames:
            idx=int(packet["frame_index"])
            dst=self.session_root/"raw"/f"frame_{idx:06d}.npz"
            temp=dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
            try:
                with temp.open("xb") as f:
                    np.savez_compressed(f,**packet)
                os.replace(temp,dst)
            finally:
                temp.unlink(missing_ok=True)

    def _await_output(self,index):
        output=self.session_root/"perception"/"frames"/f"frame_{index:06d}.npz"
        deadline=time.monotonic()+self.timeout_s
        while time.monotonic()<deadline:
            if output.is_file():
                return output
            if self.worker is not None and self.worker.poll() is not None:
                log=(self.session_root/"perception.log")
                tail=log.read_text(errors="replace")[-5000:] if log.exists() else ""
                raise RuntimeError(f"Online perception exited rc={self.worker.returncode}: {tail}")
            time.sleep(.025)
        raise TimeoutError(f"Online perception timed out waiting for frame {index}; "
                           f"inspect {self.session_root}/perception.log")

    def _reset_causal_policy(self):
        """Reset policy tactile EMA before new session, but do NOT reload Pi0."""
        pipeline=getattr(self.policy,"_input_transform",None)
        if pipeline is None:
            return  # Lightweight test policies have no transform state.
        transforms=getattr(pipeline,"transforms",())
        resets=[x for x in transforms if callable(getattr(x,"reset_tactile_ema",None))]
        if not resets:
            raise RuntimeError("Cannot locate Pi0 spatial/tactile causal reset in loaded policy")
        for item in resets:
            item.reset_tactile_ema()

    def _start_session(self,session:str,position:int):
        allowed = (1,) if self.generic_geometry else SELECTABLE_SCENE_POSITIONS
        if position not in allowed:
            raise ValueError(f"Unknown --scene-position {position}; allowed {allowed}")
        if self.prompt_file is not None:
            raise ValueError("Client --scene-position cannot be combined with fixed server --prompts")
        if self.anchor_json is not None and SCENE_POSITION_TO_EPISODE[position]!=self.position_template_episode:
            raise ValueError("Fixed reviewed flask anchor does not match selected scene position")
        self.close()  # Old SAM2/RGBTrack process, not WebSocket/Pi0 process.
        # A restarted WebSocket process may reuse the same workdir. Allocate
        # a fresh unique session directory without deleting historical data.
        candidate=self.session_counter
        while True:
            candidate+=1
            root=self.workdir/f"scene_{position}_session_{candidate:04d}"
            try:
                root.mkdir(parents=True,exist_ok=False)
            except FileExistsError:
                continue
            self.session_counter=candidate
            self.session_root=root
            break
        (self.session_root/"raw").mkdir()
        (self.session_root/"perception").mkdir()
        self.next_index=0
        self.previous_timestamp=float("-inf")
        self._started=False
        self.worker=None
        self.scene_position=position
        self.session_id=session
        self._reset_causal_policy()
        if self.generic_geometry:
            print(f"[SCENE] task={self.task} reference=e00000 session={self.session_counter}; "
                  "single-reference SAM2 initialization on CURRENT frame0",flush=True)
        else:
            print(f"[SCENE] selected={position} reference_episode={SCENE_POSITION_TO_EPISODE[position]} session={self.session_counter} "
                  "source=manual_reviewed_2d_clicks applied_to_CURRENT_live_frame0 "
                  "flask_pose=RGBTrack_current_frame0_only",flush=True)

    def infer(self,observation):
        obs=dict(observation)
        session=obs.pop("hand_scene_session",None)
        history=obs.pop("hand_scene_history",None)
        requested=obs.pop("hand_scene_position",None)
        if not isinstance(session,str) or not session:
            raise ValueError("hand_scene_session missing; old robot client cannot bypass object geometry")
        allowed = (1,) if self.generic_geometry else SELECTABLE_SCENE_POSITIONS
        if self.generic_geometry and requested is None:
            requested = 1  # Internal legacy session key, never an operator scene choice.
        if not isinstance(requested,(int,np.integer)) or int(requested) not in allowed:
            raise ValueError(f"hand_scene_position must be one of {allowed}, got {requested}")
        position=int(requested)
        if self.session_id != session:
            # Switching at robot-side between different runs starts a new
            # local perception/EMA session; server and model stay loaded.
            self._start_session(session,position)
        elif position!=self.scene_position:
            raise ValueError("Scene selection cannot change midway through an active robot run")
        entries=validate_history(history,next_index=self.next_index,
                                 previous_timestamp=self.previous_timestamp)
        # The raw input used by Pi0 MUST correspond to the LAST frame
        # passed to the causal geometry producer.
        last=entries[-1]
        if not np.array_equal(np.asarray(obs["observation.state"],dtype=np.float32),
                              last["robot_state"]):
            raise ValueError("Policy state and last geometry frame differ")
        if not np.array_equal(np.asarray(obs["observation.depths.cam_front"]),
                              last["front_depth_native"]):
            raise ValueError("Policy depth and geometry input differ")
        for role in ("front","left","right"):
            if not np.array_equal(
                np.asarray(obs[f"observation.images.cam_{role}"]),last[f"{role}_rgb"]
            ):
                raise ValueError(f"Policy {role} RGB and geometry history input differ")
        # Match frame/timestamp used for SAME-frame spatial preprocessing and
        # causal geometry. Old client timestamp was synthetic frame_idx/fps.
        obs["frame_index"]=last["frame_index"]
        obs["timestamp_s"]=last["timestamp_s"]
        self._publish(entries)
        if not self._started:
            self._launch()
        output=self._await_output(int(last["frame_index"]))
        print(f"[PERCEPTION] task={self.task} completed geometry frame={int(last['frame_index'])}",flush=True)
        with np.load(output,allow_pickle=False) as packet:
            obs["object_geometry"]=_geometry_from_npz(packet)
        if self.surface_points_sha256 is not None:
            # The assets were verified against the same approved meshes,
            # release and sidecar before model restoration. The policy's V4
            # transform additionally verifies every frame's ordered geometry.
            obs["object_geometry"]["surface_points_sha256"] = dict(self.surface_points_sha256)
        # Advance only after validated complete geometry and inference.
        result=self.policy.infer(obs)
        self.next_index+=len(entries)
        self.previous_timestamp=float(last["timestamp_s"])
        return result

    def close(self):
        if self.worker is not None and self.worker.poll() is None:
            self.worker.terminate()
            try:self.worker.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.worker.kill()
                self.worker.wait(timeout=3)
        self.worker=None
        self._started=False
