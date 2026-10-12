"""Causal relative-grasp labels and action schedules. No torch/Isaac imports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

VERSION = 'cw-slip-v1'
KINDS = ('continue', 'slow', 'pause_resume', 'hold', 'retreat_resume',
         'x_plus', 'x_minus', 'y_plus', 'y_minus', 'pitch_plus', 'pitch_minus')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def save(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grasp(ee_pos, ee_quat, plug_pos, plug_quat):
    """USB root pose in the moving gripper frame, mm and xyzw quaternion."""
    inv = Rotation.from_quat(ee_quat).inv()
    return inv.apply(np.asarray(plug_pos) - ee_pos) * 1000, (inv * Rotation.from_quat(plug_quat)).as_quat()


def rotation_distance(q1, q2):
    return np.rad2deg((Rotation.from_quat(q1).inv() * Rotation.from_quat(q2)).magnitude())


def labels(position, quaternion, mm=.5, degrees=1., window=2):
    """Causal confirmation: two high-motion frames enter; three quiet frames exit.

    Thresholds are operational relative-motion thresholds, not contact-level
    frictional slip ground truth. Active/confirmation use only past and present;
    retrospective event_start is returned separately for forecast scoring.
    """
    n = len(position)
    prev = np.maximum(0, np.arange(n) - window)
    dp = np.linalg.norm(position - position[prev], axis=1)
    dr = rotation_distance(quaternion[prev], quaternion)
    movement = (dp >= mm) | (dr >= degrees)
    active = np.zeros(n, bool)
    high = quiet = 0
    state = False
    for t in range(n):
        high = high + 1 if movement[t] else 0
        quiet = quiet + 1 if not movement[t] else 0
        if high >= 2:
            state = True
        if quiet >= 3:
            state = False
        active[t] = state
    onset = active & ~np.r_[False, active[:-1]]
    # Retrospective event start is the first high-motion frame of a confirmed pair.
    # It is a scoring label only; causal detector confirmation remains one step later.
    event_start = np.r_[onset[1:], False]
    return dict(dp=dp, dr=dr, movement=movement, active=active, onset=onset,event_start=event_start)


def targets(position, quaternion, mm=.5, degrees=1.):
    lab = labels(position, quaternion, mm, degrees)
    n = len(position)
    result = {'detect': lab['active'].astype(np.float32)}
    valid = {'detect': np.arange(n) >= 4}
    quiet = np.array([not lab['movement'][max(0, t-1):t+1].any() for t in range(n)])
    for h in (1, 2, 4):
        result[f'forecast{h}'] = np.array([lab['event_start'][t+1:min(n, t+h+1)].any() for t in range(n)], np.float32)
        valid[f'forecast{h}'] = ((np.arange(n) >= 4) & (np.arange(n)+h+1 < n) &
                               ~lab['active'] & ~lab['event_start'])
        # Stricter subset cannot be explained by motion already above threshold.
        valid[f'quiet{h}'] = valid[f'forecast{h}'] & quiet
    future = np.minimum(np.arange(n)+4, n-1)
    result['worsen4'] = ((np.linalg.norm(position[future]-position, axis=1) >= 2*mm) |
                        (rotation_distance(quaternion, quaternion[future]) >= 2*degrees)).astype(np.float32)
    valid['worsen4'] = (np.arange(n) >= 4) & (np.arange(n)+4 < n) & lab['active']
    return lab, result, valid


def history(x, t, lags=(0, 1, 2, 4)):
    return np.concatenate([x[np.maximum(0, t-lag)] for lag in lags], axis=-1)


def causal_baseline(force, state, actions):
    """First observable quiet 4-frame window, frozen once found, no object GT."""
    baseline = np.zeros_like(force)
    valid = np.zeros(len(force), np.float32)
    found = None
    for t in range(4, len(force)):
        if found is None:
            f = force[t-3:t+1]
            scale = max(float(np.abs(f).mean()), 1e-7)
            if (np.abs(state[t-3:t+1, 9:]).max() < .08 and
                    np.abs(actions[t-4:t]).max() < .12 and
                    np.abs(np.diff(f, axis=0)).mean()/scale < .15):
                found = f.mean(0)
        if found is not None:
            baseline[t] = found
            valid[t] = 1
    return baseline, valid


def action_schedule(kind, planned, socket_quat, ps, rs, horizon=16):
    """First four commands are a known policy chunk; repeat that plan openly.

    All schedules have equal elapsed time. Resume means returning to this
    fixed command template, NOT querying/recovering a closed-loop policy.
    """
    if kind not in KINDS:
        raise ValueError(kind)
    planned = np.asarray(planned, np.float32)
    if planned.shape != (4, 6):
        raise ValueError('Branch must start at a replan step with four known actions')
    actions = np.tile(planned, (int(np.ceil(horizon/4)), 1))[:horizon].copy()
    rot = Rotation.from_quat(socket_quat)
    if kind == 'slow':
        actions *= .5
    elif kind == 'hold':
        actions[:] = 0
    elif kind == 'pause_resume':
        actions[:4] = 0
    elif kind != 'continue':
        actions[:4] = 0
        vec = np.zeros(3)
        ang = np.zeros(3)
        if kind == 'retreat_resume':
            vec[2] = .0015 / 2
        elif kind.startswith('pitch'):
            ang[0] = (1 if kind.endswith('plus') else -1) * np.deg2rad(2) / 2
        else:
            vec[0 if kind.startswith('x') else 1] = (1 if kind.endswith('plus') else -1) * .0015 / 2
        actions[:2, :3] = rot.apply(vec) / np.asarray(ps)
        actions[:2, 3:] = rot.apply(ang) / np.asarray(rs)
    return actions


def branch_effect(z):
    p, q = grasp(z['ee_pos'], z['ee_quat'], z['plug_pos'], z['plug_quat'])
    delta = np.linalg.norm(p-p[0], axis=1)
    angle = rotation_distance(np.broadcast_to(q[0], q.shape), q)
    rel = Rotation.from_quat(z['socket_quat']).inv().apply(z['plug_pos']-z['socket_pos']) * 1000
    return dict(slip_mm=float(delta.max()), rotation_deg=float(angle.max()),
                progress_mm=float(rel[0, 2]-rel[-1, 2]), xy_final_mm=float(np.linalg.norm(rel[-1, :2])),
                late_slip_mm=float(np.linalg.norm(p[4:]-p[4], axis=1).max()),
                final_success=bool(z['success'][-1]),
                curve_mm=delta.tolist(), angle_curve_deg=angle.tolist(),
                progress_curve_mm=(rel[0, 2]-rel[:, 2]).tolist())
