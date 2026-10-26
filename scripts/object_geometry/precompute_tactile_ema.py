#!/usr/bin/env python3
"""Precompute causal DTCR tactile EMA state from an existing Spatial sidecar.

This only reads ``spatial/<version>`` and writes small per-frame state files;
it never modifies the original Spatial sidecar or an existing object-geometry
version.  The output is consumed by ``export_hand_scene_sidecar.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from openpi.spatial.tactile_ema import CausalTactileEMA


def precompute(
    *,
    dataset_root: Path,
    spatial_version: str,
    output_root: Path,
    episodes: set[int] | None,
    tau_s: float,
    resume: bool,
) -> dict[str, object]:
    from openpi.spatial_dataset.derived import SpatialDerivedDataset

    source = SpatialDerivedDataset(dataset_root, version=spatial_version)
    selected = set(source.episode_ids) if episodes is None else set(episodes)
    unknown = selected - set(source.episode_ids)
    if unknown:
        raise ValueError(f"Episodes absent from Spatial sidecar: {sorted(unknown)}")
    output_root.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    for episode in sorted(selected):
        filter_state = CausalTactileEMA(tau_s=tau_s, taxel_count=source.tactile_points_per_frame)
        frame_ids = sorted(frame for ep, frame in source._records if int(ep) == int(episode))  # noqa: SLF001
        for frame in frame_ids:
            sample = source.get(episode_index=episode, frame_index=frame)
            state_path = output_root / f"e{episode:03d}" / f"f{frame:06d}.npz"
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state = filter_state.update(
                timestamp_s=sample.timestamp_s,
                force_norm=sample.tactile_force_norm,
            )
            if resume and state_path.is_file():
                # Still advance the filter so subsequent frames retain the
                # exact causal state; only the file write is skipped.
                skipped += 1
                continue
            np.savez_compressed(
                state_path,
                causal_max_source_frame=np.asarray(frame, dtype=np.int64),
                dtcr_ema_before_force_norm=state.ema_before_force_norm,
                dtcr_ema_after_force_norm=state.ema_after_force_norm,
                dtcr_current_minus_ema_force_norm=state.current_minus_ema_force_norm,
                dtcr_current_minus_previous_force_norm=state.current_minus_previous_force_norm,
                dtcr_reference_valid=state.reference_valid,
                dtcr_previous_valid=state.previous_valid,
                dtcr_sample_valid=state.sample_valid,
                dtcr_alpha=np.asarray(state.alpha, dtype=np.float32),
            )
            written += 1
    return {"episodes": sorted(selected), "written": written, "skipped": skipped, "tau_s": float(tau_s)}


def _parse_episodes(value: str | None) -> set[int] | None:
    if value is None:
        return None
    selected: set[int] = set()
    for token in value.split(","):
        bounds = token.split("-", 1)
        if len(bounds) == 1:
            selected.add(int(bounds[0]))
        else:
            selected.update(range(int(bounds[0]), int(bounds[1]) + 1))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--spatial-version", default="v1_front_candidate")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episodes", help="Episode list/ranges, e.g. 23,38,65 or 0-69")
    parser.add_argument("--tau-s", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = precompute(
        dataset_root=args.dataset_root.expanduser().resolve(),
        spatial_version=args.spatial_version,
        output_root=args.output_root.expanduser().resolve(),
        episodes=_parse_episodes(args.episodes),
        tau_s=args.tau_s,
        resume=args.resume,
    )
    print(result)


if __name__ == "__main__":
    main()
