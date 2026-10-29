#!/usr/bin/env python3
"""Compare saved 18-DOF Pi0 open-loop actions on the EXACT SAME processed frames.

First generate each model's actions using scripts/deploy/run_policy_dry_run.py,
pointing every model to the same --frames-dir and using equal --frame-stride.
This script NEVER connects to the robot or evaluates closed-loop success.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np


ARM_NAMES = tuple(f"arm_joint_{i}.pos" for i in range(6))
HAND_NAMES = tuple(f"hand_joint_{i}.pos" for i in range(12))
JOINT_NAMES = ARM_NAMES + HAND_NAMES


def _read_runs(specs: list[str], chunk_steps: int) -> dict[str, dict[int, np.ndarray]]:
    runs: dict[str, dict[int, np.ndarray]] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected NAME=RESULTS_DIR, got {spec!r}")
        name, directory = spec.split("=", 1)
        if not name or name in runs:
            raise ValueError(f"Empty or duplicate model name: {name!r}")
        root = Path(directory).expanduser().resolve()
        if not (root / "summary.json").is_file():
            raise FileNotFoundError(f"{name}: missing dry-run summary.json in {root}")
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        if summary.get("dry_run") is not True or summary.get("status") != "pass":
            raise ValueError(f"{name}: result directory is not a completed dry run")
        predictions: dict[int, np.ndarray] = {}
        for path in sorted(root.glob("actions_*.npz")):
            number = path.stem.removeprefix("actions_")
            if not number.isdigit():
                continue
            frame = int(number)
            with np.load(path, allow_pickle=False) as stored:
                action = np.asarray(stored["actions"], dtype=np.float64)
            if action.ndim != 2 or action.shape[1] != 18 or action.shape[0] < chunk_steps:
                raise ValueError(f"{path}: action shape {action.shape}, expected [T>=chunk_steps,18]")
            if not np.isfinite(action).all():
                raise ValueError(f"{path}: nonfinite actions")
            predictions[frame] = action[:chunk_steps]
        if not predictions:
            raise ValueError(f"{name}: no actions_*.npz files in {root}")
        runs[name] = predictions
    if len(runs) < 2:
        raise ValueError("Compare at least 2 models")
    model_frames = [set(r) for r in runs.values()]
    if not all(m == model_frames[0] for m in model_frames[1:]):
        raise ValueError(
            "Model result frame sets differ; replay all models with the same "
            f"--frames-dir, --frame-stride, --max-frames. Sets: "
            f"{ {name: sorted(d) for name, d in runs.items()} }"
        )
    return runs


def _measured_joint_positions(frames_dir: Path, frame: int) -> np.ndarray:
    path = frames_dir / f"frame_{frame:06d}.npz"
    with np.load(path, allow_pickle=False) as data:
        raw = np.asarray(data["robot_state"], dtype=np.float64)
        saved_frame = int(np.asarray(data["frame_index"]).reshape(()))
    if saved_frame != frame or raw.shape != (1972,) or not np.isfinite(raw).all():
        raise ValueError(f"{path}: incorrect frame index, malformed 1972-state or nonfinite values")
    # Canonical 1972 state: arm positions [0:6]; arm velocities [6:12];
    # 4x4 ee pose [12:28]; XHand positions [28:40]; then other fields.
    return np.concatenate((raw[:6], raw[28:40]))


def _cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return None if na <= 1e-8 or nb <= 1e-8 else float(np.dot(a, b) / (na * nb))


def analyze(
    runs: dict[str, dict[int, np.ndarray]],
    frames_dir: Path,
    chunk_steps: int,
) -> tuple[list[dict], list[dict], dict]:
    names = list(runs)
    frames = sorted(next(iter(runs.values())))
    measured = {f: _measured_joint_positions(frames_dir, f) for f in frames}
    joint_rows, pair_rows = [], []
    per_model = {m: {"arm_first_abs_delta": [], "hand_first_abs_delta": [],
                     "arm_chunk_path": [], "hand_chunk_path": []} for m in names}
    per_pair = {}
    for frame in frames:
        q = measured[frame]
        for name in names:
            a = runs[name][frame]
            delta = a[0] - q
            per_model[name]["arm_first_abs_delta"].append(float(np.mean(np.abs(delta[:6]))))
            per_model[name]["hand_first_abs_delta"].append(float(np.mean(np.abs(delta[6:]))))
            per_model[name]["arm_chunk_path"].append(float(np.abs(np.diff(a[:, :6], axis=0)).sum(axis=0).mean()))
            per_model[name]["hand_chunk_path"].append(float(np.abs(np.diff(a[:, 6:], axis=0)).sum(axis=0).mean()))
            for idx, joint in enumerate(JOINT_NAMES):
                joint_rows.append({
                    "frame": frame, "model": name, "joint": joint, "joint_index": idx,
                    "observed_rad": float(q[idx]), "first_predicted_rad": float(a[0, idx]),
                    "delta_from_observed_rad": float(delta[idx]),
                    "last_chunk_target_rad": float(a[-1, idx]),
                })
        for name1, name2 in itertools.combinations(names, 2):
            a, b = runs[name1][frame], runs[name2][frame]
            d1, d2 = a[0] - q, b[0] - q
            key = (name1, name2)
            mae = float(np.mean(np.abs(a[:, :] - b[:, :])))
            per_pair.setdefault(key, []).append(mae)
            valid_dir = (np.abs(d1) > 0.01) & (np.abs(d2) > 0.01)
            pair_rows.append({
                "frame": frame, "model_a": name1, "model_b": name2,
                "first_arm6_mae_rad": float(np.mean(np.abs(a[0, :6] - b[0, :6]))),
                "first_hand12_mae_rad": float(np.mean(np.abs(a[0, 6:] - b[0, 6:]))),
                "chunk_all18_mae_rad": mae,
                "arm6_intention_cosine": _cosine(d1[:6], d2[:6]),
                "hand12_intention_cosine": _cosine(d1[6:], d2[6:]),
                "sign_agreement_fraction": (
                    float(np.mean(np.sign(d1[valid_dir]) == np.sign(d2[valid_dir])))
                    if np.any(valid_dir) else None
                ),
            })
    summary = {
        "description": "Identical recorded input, open-loop action comparison; NOT a success or safety metric.",
        "source_frames_dir": str(frames_dir),
        "frames": frames,
        "model_names": names,
        "chunk_steps_compared": chunk_steps,
        "models": {
            name: {k + "_mean_rad": float(np.mean(v)) for k, v in values.items()}
            for name, values in per_model.items()
        },
        "pairs": {
            f"{a}__vs__{b}": {
                "mean_chunk_all18_mae_rad": float(np.mean(values)),
                "worst_frame": int(frames[int(np.argmax(values))]),
                "worst_frame_chunk_all18_mae_rad": float(np.max(values)),
            }
            for (a, b), values in per_pair.items()
        },
    }
    return joint_rows, pair_rows, summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No comparison rows")
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    runs: dict[str, dict[int, np.ndarray]],
    measured: dict[int, np.ndarray],
    focus_frame: int,
    out: Path,
) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; comparison CSV/JSON still available")
        return False
    names = list(runs)
    frames = sorted(next(iter(runs.values())))
    for label, span, rows, cols in (
        ("arm6", range(6), 2, 3),
        ("hand12", range(6, 18), 3, 4),
    ):
        for kind in ("first_target", "delta_from_observed"):
            fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3 * rows), sharex=True)
            for ax, idx in zip(axes.flat, span, strict=True):
                for name in names:
                    y = np.asarray([runs[name][f][0, idx] for f in frames])
                    if kind == "delta_from_observed":
                        y = y - np.asarray([measured[f][idx] for f in frames])
                    ax.plot(frames, y, marker=".", linewidth=1.1, label=name)
                if kind == "delta_from_observed":
                    ax.axhline(0, linewidth=0.7)
                ax.set_title(JOINT_NAMES[idx], fontsize=9)
                ax.set_ylabel("rad")
            axes.flat[0].legend(fontsize=8)
            fig.supxlabel("recorded frame index")
            fig.suptitle(f"{label}: {kind.replace('_', ' ')}")
            fig.tight_layout()
            fig.savefig(out / f"{label}_{kind}.png", dpi=145)
            plt.close(fig)
        fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3 * rows), sharex=True)
        for ax, idx in zip(axes.flat, span, strict=True):
            for name in names:
                a = runs[name][focus_frame]
                ax.plot(np.arange(len(a)), a[:, idx], marker=".", linewidth=1.0, label=name)
            ax.set_title(JOINT_NAMES[idx], fontsize=9)
            ax.set_ylabel("target rad")
        axes.flat[0].legend(fontsize=8)
        fig.supxlabel("chunk step (0-based)")
        fig.suptitle(f"{label}: full predicted chunk at recorded frame {focus_frame}")
        fig.tight_layout()
        fig.savefig(out / f"{label}_chunk_frame_{focus_frame:06d}.png", dpi=145)
        plt.close(fig)
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames-dir", type=Path, required=True)
    p.add_argument("--run", action="append", required=True, metavar="NAME=RESULTS_DIR",
                   help="Repeat for each completed run_policy_dry_run.py output")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--chunk-steps", type=int, default=16)
    p.add_argument("--focus-frame", type=int, default=None,
                   help="One saved frame to plot all models' 16-step chunks")
    a = p.parse_args()
    if a.chunk_steps < 1 or a.chunk_steps > 50:
        raise ValueError("chunk-steps must be 1..50")
    source = a.frames_dir.expanduser().resolve()
    out = a.output_dir.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {out}")
    runs = _read_runs(a.run, a.chunk_steps)
    frames = sorted(next(iter(runs.values())))
    focus = frames[0] if a.focus_frame is None else a.focus_frame
    if focus not in frames:
        raise ValueError(f"--focus-frame {focus} not among {frames}")
    joint_rows, pair_rows, summary = analyze(runs, source, a.chunk_steps)
    measured = {f: _measured_joint_positions(source, f) for f in frames}
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "first_actions_by_joint.csv", joint_rows)
    _write_csv(out / "pairwise_by_frame.csv", pair_rows)
    summary["focus_frame"] = focus
    summary["plots_created"] = _plot(runs, measured, focus, out)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved comparison under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
