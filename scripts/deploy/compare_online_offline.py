#!/usr/bin/env python3
"""Compare saved online geometry packets with the training Sidecar records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def _flatten_sample(sample) -> dict[str, np.ndarray]:
    values = sample.as_dict()
    result = {key: np.asarray(value) for key, value in values.items() if key not in {"episode_index", "frame_index"}}
    hand = result.pop("hand_scene", None)
    if hand is not None:
        for key, value in hand.items():
            if key == "causal_tactile":
                for tactile_key, tactile_value in value.items():
                    result[f"hand_scene__causal_tactile__{tactile_key}"] = np.asarray(tactile_value)
            else:
                result[f"hand_scene__{key}"] = np.asarray(value)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--online-frames", type=Path, required=True)
    parser.add_argument("--sidecar-root", type=Path, default=None)
    parser.add_argument("--version", default="v1_hand_scene_q2")
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()
    root = args.repo_root.expanduser().resolve()
    sidecar_root = (args.sidecar_root or (root / "data/1018_peg_all")).expanduser().resolve()
    online_frames = args.online_frames.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not (sidecar_root / "object_geometry" / args.version).is_dir():
        report = {
            "status": "NOT_TESTED",
            "reason": f"training sidecar not present: {sidecar_root / 'object_geometry' / args.version}",
            "online_frames": str(online_frames),
            "version": args.version,
            "episode": args.episode,
        }
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 2

    import sys

    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    from openpi.object_geometry_dataset.derived import ObjectGeometryDerivedDataset
    from run_policy_dry_run import _geometry_from_npz

    dataset = ObjectGeometryDerivedDataset(sidecar_root, version=args.version)
    frame_paths = sorted(online_frames.glob("frame_*.npz"))
    if not frame_paths:
        raise FileNotFoundError(f"No online frames in {online_frames}")
    comparisons = []
    for path in frame_paths:
        with np.load(path, allow_pickle=False) as packet:
            frame = int(np.asarray(packet["frame_index"]).reshape(()))
            online = _geometry_from_npz(packet)
            online_flat = {key: np.asarray(value) for key, value in online.items() if key != "hand_scene"}
            hand = online["hand_scene"]
            online_flat.update({f"hand_scene__{key}": np.asarray(value) for key, value in hand.items() if key != "causal_tactile"})
            if "causal_tactile" in hand:
                online_flat.update(
                    {f"hand_scene__causal_tactile__{key}": np.asarray(value) for key, value in hand["causal_tactile"].items()}
                )
        offline = _flatten_sample(dataset.get(episode_index=args.episode, frame_index=frame))
        if set(online_flat) != set(offline):
            raise ValueError(f"Field mismatch at frame {frame}: online-only={sorted(set(online_flat)-set(offline))}, offline-only={sorted(set(offline)-set(online_flat))}")
        field_rows = {}
        for key in sorted(online_flat):
            left, right = online_flat[key], offline[key]
            if left.shape != right.shape:
                raise ValueError(f"Shape mismatch at frame {frame} field {key}: {left.shape} != {right.shape}")
            exact = bool(np.array_equal(left, right))
            if np.issubdtype(left.dtype, np.number) and np.issubdtype(right.dtype, np.number):
                delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
                max_abs = float(delta.max(initial=0.0))
                passed = bool(np.allclose(left, right, atol=args.atol, rtol=0))
            else:
                max_abs = 0.0 if exact else None
                passed = exact
            field_rows[key] = {"pass": passed, "exact": exact, "max_abs_error": max_abs}
            if not passed:
                raise AssertionError(f"Online/offline mismatch at frame {frame}, {key}: max_abs={max_abs}")
        comparisons.append({"frame_index": frame, "pass": True, "fields": field_rows})
    report = {
        "status": "PASS",
        "sidecar_root": str(sidecar_root),
        "version": args.version,
        "episode": args.episode,
        "frames_compared": len(comparisons),
        "atol": args.atol,
        "comparisons": comparisons,
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "comparisons"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
