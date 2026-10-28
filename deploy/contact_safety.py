"""Fail-closed application-level contact safety for UR/XHand control.

NOT a substitute for a hardware E-stop, UR protective stop, force limits,
speed/energy reduction, or a validated real-time force controller. Runs before
EVERY policy action, and again after a blocking inference request.

Two tiers: moderate tactile changes advise DISCARD-CHUNK/REPLAN, while hard
absolute force/torque or catastrophic tactile changes latch a STOP. Only the
second tier is a safety trip; the soft trigger is a scheduling intervention.

XHand raw values are vendor sensor units, NOT Newtons. Arm wrench must be
measured/estimated by the arm SDK and explicitly exposed as a current 6D
TCP wrench; the 12 XHand joint torques are NOT arm torques.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Mapping

import numpy as np

TACTILE_START = 52
FINGERS = 5
FINGER_STRIDE = 384
TAXELS = 120
FORCE_OFFSET = 24
EXPECTED_STATE = TACTILE_START + FINGERS * FINGER_STRIDE


class SafetyTrip(RuntimeError):
    """Latched unsafe or unavailable sensor reading; stop the entire run."""


def _positive_scalar(value, name):
    v=float(value)
    if not np.isfinite(v) or v<=0:
        raise ValueError(f"{name} must be a positive finite value")
    return v


def load_safety_limits(path: str | Path) -> dict:
    """Only explicitly reviewed hardware limits can enable robot actuation."""
    path=Path(path).expanduser().resolve()
    cfg=json.loads(path.read_text(encoding="utf-8"))
    required={
        "schema_version","reviewed_for_glass_task","arm_wrench_observation_key",
        "tactile_peak_limits_raw","tactile_emergency_step_change_limits_raw",
        "tactile_replan_step_change_limits_raw", "replan_enabled",
        "replan_min_interval_s", "arm_tcp_force_norm_limit_n",
        "arm_tcp_torque_norm_limit_nm", "max_observation_gap_s"
    }
    if not isinstance(cfg,dict) or set(cfg)!=required or cfg["schema_version"]!=1:
        raise ValueError(f"Safety config must contain exactly {sorted(required)}")
    if cfg["reviewed_for_glass_task"] is not True:
        raise ValueError("Safety configuration is not explicitly reviewed for glass manipulation")
    name=cfg["arm_wrench_observation_key"]
    if not isinstance(name,str) or not name.strip():
        raise ValueError("Arm 6D wrench observation key must be explicitly configured")
    for key in ("tactile_peak_limits_raw","tactile_emergency_step_change_limits_raw",
                "tactile_replan_step_change_limits_raw"):
        v=np.asarray(cfg[key],dtype=float)
        if v.shape!=(FINGERS,) or not np.isfinite(v).all() or np.any(v<=0):
            raise ValueError(f"{key} must specify five positive raw-sensor thresholds")
    emergency=np.asarray(cfg["tactile_emergency_step_change_limits_raw"],dtype=float)
    replan=np.asarray(cfg["tactile_replan_step_change_limits_raw"],dtype=float)
    if np.any(replan>=emergency):
        raise ValueError("All replan thresholds must be strictly below emergency step-change thresholds")
    if type(cfg["replan_enabled"]) is not bool:
        raise ValueError("replan_enabled must be a boolean")
    interval=float(cfg["replan_min_interval_s"])
    if not np.isfinite(interval) or interval<0:
        raise ValueError("replan_min_interval_s must be a finite nonnegative number")
    for key in ("arm_tcp_force_norm_limit_n","arm_tcp_torque_norm_limit_nm","max_observation_gap_s"):
        _positive_scalar(cfg[key],key)
    return cfg


def raw_tactile_finger_peaks(state) -> np.ndarray:
    x=np.asarray(state,dtype=np.float64)
    if x.shape!=(EXPECTED_STATE,) or not np.isfinite(x).all():
        raise SafetyTrip(f"Missing/nonfinite canonical 1972-value XHand state: {x.shape}")
    blocks=x[TACTILE_START:].reshape(FINGERS,FINGER_STRIDE)
    forces=blocks[:,FORCE_OFFSET:].reshape(FINGERS,TAXELS,3)
    return np.linalg.norm(forces,axis=-1).max(axis=1)


class ContactSafetyGuard:
    def __init__(self, config: Mapping, *, clock=time.monotonic):
        self.cfg=dict(config)
        self.clock=clock
        self.last_time=None
        self.last_peaks=None
        self.tripped=None
        self.last_replan_time=None
        self.replan_events=0

    def check(self, *, observation: Mapping, state):
        if self.tripped is not None:
            raise SafetyTrip(f"Safety already latched: {self.tripped}")
        try:
            now=float(self.clock())
            if not np.isfinite(now):
                raise SafetyTrip("Safety monotonic clock invalid")
            if self.last_time is not None and now-self.last_time>float(self.cfg["max_observation_gap_s"]):
                raise SafetyTrip(f"Safety polling gap {now-self.last_time:.3f}s exceeds configured maximum")
            peaks=raw_tactile_finger_peaks(state)
            limits=np.asarray(self.cfg["tactile_peak_limits_raw"],dtype=float)
            overflow=np.flatnonzero(peaks>=limits)
            if len(overflow):
                i=int(overflow[0])
                raise SafetyTrip(f"tactile_peak finger={i} raw={peaks[i]:.5g} >= {limits[i]:.5g}")
            if self.last_peaks is not None:
                step=np.abs(peaks-self.last_peaks)
                emergency=np.asarray(self.cfg["tactile_emergency_step_change_limits_raw"],dtype=float)
                jumps=np.flatnonzero(step>=emergency)
                if len(jumps):
                    i=int(jumps[0])
                    raise SafetyTrip(f"tactile_emergency_step_change finger={i} raw_delta={step[i]:.5g} >= {emergency[i]:.5g}")
            else:
                step=np.zeros(FINGERS,dtype=float)
            key=self.cfg["arm_wrench_observation_key"]
            if key not in observation:
                raise SafetyTrip(f"Arm 6D wrench missing: {key}; XHand joint torque is not arm torque")
            wrench=np.asarray(observation[key],dtype=np.float64)
            if wrench.shape!=(6,) or not np.isfinite(wrench).all():
                raise SafetyTrip(f"Arm wrench {key} missing/nonfinite/invalid: {wrench.shape}")
            force=float(np.linalg.norm(wrench[:3]))
            torque=float(np.linalg.norm(wrench[3:]))
            if force>=float(self.cfg["arm_tcp_force_norm_limit_n"]):
                raise SafetyTrip(f"arm_tcp_force_norm {force:.5g} N exceeds threshold")
            if torque>=float(self.cfg["arm_tcp_torque_norm_limit_nm"]):
                raise SafetyTrip(f"arm_tcp_torque_norm {torque:.5g} Nm exceeds threshold")
            replan_fingers=[]
            if bool(self.cfg.get("replan_enabled",False)) and self.last_peaks is not None:
                replan_limit=np.asarray(self.cfg["tactile_replan_step_change_limits_raw"],dtype=float)
                events=np.flatnonzero(step>=replan_limit)
                if len(events) and (
                    self.last_replan_time is None
                    or now-self.last_replan_time>=float(self.cfg["replan_min_interval_s"])
                ):
                    replan_fingers=[int(i) for i in events]
                    self.last_replan_time=now
                    self.replan_events+=1
            self.last_time=now
            self.last_peaks=peaks.copy()
            return {"tactile_finger_peaks_raw":peaks.tolist(),
                    "tactile_finger_step_change_raw":step.tolist(),
                    "arm_tcp_force_norm_n":force,"arm_tcp_torque_norm_nm":torque,
                    "replan_recommended":bool(replan_fingers),
                    "replan_triggered_fingers":replan_fingers,
                    "replan_events_so_far":self.replan_events}
        except SafetyTrip as exc:
            self.tripped=str(exc)
            raise
        except (ValueError,TypeError,KeyError) as exc:
            self.tripped=f"Safety sensor failure: {exc}"
            raise SafetyTrip(self.tripped) from exc
