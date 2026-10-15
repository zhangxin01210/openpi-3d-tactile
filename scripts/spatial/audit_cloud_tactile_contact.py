#!/usr/bin/env python3
"""Audit front-only RGB-D cloud against XHand tactile contact points.

This is an offline QA tool.  It deliberately does not modify the canonical
spatial preprocessing path or write a derived spatial sidecar.

The primary mode matches ``pi0_xhand_spatial_structured_suffix_front``:

    camera_roles = ("front",)
    front_calibration_profile = configs/ur7e_xhand/front_calibration_candidate.json

For every source frame the script writes a contact-state table.  For selected
visual frames it additionally writes baseline/candidate distance metrics and a
Plotly page containing:

    dense RGB-D cloud
    CAD/self-filtered object cloud
    final 4096 model cloud
    all tactile taxels and active tactile taxels
    CAD silhouette
    nearest-pair lines

Click any two point markers in a frame page to measure an arbitrary 3-D
distance.  The selected pair is shown in base_link coordinates.

Examples
--------
Smoke test on several contact frames:

    PYTHONPATH=src:scripts/spatial python scripts/spatial/audit_cloud_tactile_contact.py \
        --dataset data/press_button_0 --episodes 0 --frames 140,150,160 \
        --output diagnostics/cloud_tactile_ep0_smoke

All contact-event context frames in all episodes:

    PYTHONPATH=src:scripts/spatial python scripts/spatial/audit_cloud_tactile_contact.py \
        --dataset data/press_button_0 --episodes all --frames contact_events \
        --output diagnostics/cloud_tactile_press_button_0

``contact_states.csv`` always contains every frame/finger, even when only a
subset is selected for expensive depth/CAD analysis.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from openpi.spatial.calibration_profile import apply_front_calibration_profile
from openpi.spatial.config import make_baseline_config
from openpi.spatial.geometry import apply_diagnostic_overrides
from openpi.spatial.geometry import build_camera_cache
from openpi.spatial.geometry import colorize_from_source_cameras
from openpi.spatial.geometry import depth_to_base_roi
from openpi.spatial.preprocess import SpatialPreprocessor
from openpi.spatial.tactile import extract_raw_tactile_force
from openpi.spatial_dataset.source import RawSpatialDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATE_PROFILE = Path(
    "configs/ur7e_xhand/front_calibration_candidate.json"
)
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
FINGER_COLORS = (
    "rgb(230,60,60)",
    "rgb(55,105,230)",
    "rgb(50,180,85)",
    "rgb(175,80,210)",
    "rgb(235,155,45)",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--episodes",
        default="all",
        help="Episode ids, e.g. 0,2,5, or all.",
    )
    parser.add_argument(
        "--frames",
        default="contact_events",
        help=(
            "all, contact_events, or comma-separated frame ids. "
            "Contact states are always computed for all frames."
        ),
    )
    parser.add_argument(
        "--contact-context",
        type=int,
        default=2,
        help="Extra frames before/after each contact event.",
    )
    parser.add_argument(
        "--max-visual-frames-per-episode",
        type=int,
        default=30,
        help="Safety cap for HTML/depth analysis; 0 means no cap.",
    )
    parser.add_argument(
        "--contact-min-force",
        type=float,
        default=10.0,
        help="Minimum dataset-native max-taxel force for contact detection.",
    )
    parser.add_argument(
        "--contact-mad-scale",
        type=float,
        default=5.0,
        help="Quiet-window MAD multiplier used per finger.",
    )
    parser.add_argument(
        "--quiet-frames",
        type=int,
        default=30,
        help="Initial frames used to estimate each finger's quiet force.",
    )
    parser.add_argument(
        "--self-tolerance-mm",
        type=float,
        default=15.0,
        help="Depth/CAD z-buffer tolerance for robot-self points.",
    )
    parser.add_argument(
        "--false-overlap-mm",
        type=float,
        default=10.0,
        help="Object-cloud distance treated as a no-contact overlap candidate.",
    )
    parser.add_argument(
        "--max-display-points",
        type=int,
        default=12000,
        help="Maximum points per 3-D trace; metrics always use all points.",
    )
    parser.add_argument(
        "--pair-topk",
        type=int,
        default=20,
        help="Nearest/worst tactile-object pairs drawn on each page.",
    )
    parser.add_argument(
        "--candidate-profile",
        type=Path,
        default=DEFAULT_CANDIDATE_PROFILE,
        help="Front profile used by the training helper's candidate mode.",
    )
    parser.add_argument("--mesh-urdf", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--open", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path, *, root: Path = REPO_ROOT) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def parse_episode_ids(spec: str, available: Iterable[int]) -> tuple[int, ...]:
    available_set = set(int(value) for value in available)
    if spec.strip().lower() == "all":
        return tuple(sorted(available_set))
    values = tuple(sorted({int(part.strip()) for part in spec.split(",") if part.strip()}))
    if not values or set(values) - available_set:
        raise ValueError(
            f"Unknown episode ids {sorted(set(values) - available_set)}; "
            f"available={sorted(available_set)}"
        )
    return values


def quiet_thresholds(
    force_max_by_frame_finger: np.ndarray,
    *,
    quiet_frames: int,
    minimum: float,
    mad_scale: float,
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    """Estimate conservative per-finger contact thresholds from episode start."""
    if force_max_by_frame_finger.ndim != 2 or force_max_by_frame_finger.shape[1] != 5:
        raise ValueError("force_max_by_frame_finger must have shape [T,5]")
    quiet = force_max_by_frame_finger[: max(1, min(quiet_frames, len(force_max_by_frame_finger)))]
    median = np.median(quiet, axis=0)
    mad = np.median(np.abs(quiet - median[None, :]), axis=0)
    thresholds = np.maximum(
        float(minimum),
        median + float(mad_scale) * np.maximum(mad, 0.5),
    ).astype(np.float32)
    detail = {
        FINGER_NAMES[index]: {
            "quiet_median": float(median[index]),
            "quiet_mad": float(mad[index]),
            "threshold": float(thresholds[index]),
        }
        for index in range(5)
    }
    return thresholds, detail


def contact_document(
    dataset: RawSpatialDataset,
    *,
    quiet_frames: int,
    minimum: float,
    mad_scale: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Read all states for one episode and return contact rows and force arrays."""
    states_doc = dataset.states()
    frame_ids = np.asarray(states_doc["frame_index"], dtype=np.int64)
    timestamps = np.asarray(states_doc["timestamp"], dtype=np.float64)
    states = np.asarray(states_doc["observation.state"], dtype=np.float32)
    if len(frame_ids) != len(states) or len(frame_ids) != len(timestamps):
        raise ValueError(f"Episode {dataset.episode}: inconsistent state lengths")

    tactile_config = make_baseline_config().tactile
    force_max = np.empty((len(states), 5), dtype=np.float32)
    for index, state in enumerate(states):
        raw, _ = extract_raw_tactile_force(state, config=tactile_config)
        force_max[index] = np.linalg.norm(raw, axis=-1).max(axis=-1)
    thresholds, threshold_detail = quiet_thresholds(
        force_max,
        quiet_frames=quiet_frames,
        minimum=minimum,
        mad_scale=mad_scale,
    )
    contact = force_max >= thresholds[None, :]

    rows: list[dict[str, Any]] = []
    for row_index, frame in enumerate(frame_ids):
        for finger in range(5):
            before = bool(contact[row_index - 1, finger]) if row_index else False
            after = bool(contact[row_index + 1, finger]) if row_index + 1 < len(contact) else False
            rows.append(
                {
                    "episode": int(dataset.episode),
                    "frame": int(frame),
                    "timestamp_s": float(timestamps[row_index]),
                    "finger_id": finger,
                    "finger": FINGER_NAMES[finger],
                    "force_max": float(force_max[row_index, finger]),
                    "threshold": float(thresholds[finger]),
                    "contact": bool(contact[row_index, finger]),
                    "contact_onset": bool(contact[row_index, finger] and not before),
                    "contact_offset": bool(not contact[row_index, finger] and before),
                    "contact_neighbor": bool(before or after),
                }
            )

    document = {
        "episode": int(dataset.episode),
        "frame_count": int(len(frame_ids)),
        "thresholds": threshold_detail,
    }
    return document, rows, force_max, contact


def parse_frame_ids(
    spec: str,
    *,
    frame_ids: np.ndarray,
    contact: np.ndarray,
    context: int,
    max_count: int,
) -> list[int]:
    available = {int(value) for value in frame_ids}
    if spec.strip().lower() == "all":
        selected = set(available)
    elif spec.strip().lower() in {"contact", "contact_events"}:
        selected: set[int] = set()
        any_contact = contact.any(axis=1)
        starts = np.flatnonzero(any_contact & ~np.r_[False, any_contact[:-1]])
        ends = np.flatnonzero(any_contact & ~np.r_[any_contact[1:], False])
        for start, end in zip(starts, ends, strict=True):
            start_i = max(0, int(start) - context)
            end_i = min(len(frame_ids) - 1, int(end) + context)
            selected.update(int(value) for value in frame_ids[start_i : end_i + 1])
            peak = start + int(np.argmax(np.max(contact[start : end + 1] * 1.0, axis=1)))
            selected.add(int(frame_ids[peak]))
    else:
        selected = {int(part.strip()) for part in spec.split(",") if part.strip()}
        if not selected or selected - available:
            raise ValueError(f"Requested frames not available: {sorted(selected - available)}")

    ordered = sorted(selected)
    if max_count > 0 and len(ordered) > max_count:
        # Preserve event boundaries and then fill the remaining budget evenly.
        priority = sorted(
            {
                int(frame_ids[index])
                for index in np.flatnonzero(
                    contact.any(axis=1)
                    & (~np.r_[False, contact.any(axis=1)[:-1]] | ~np.r_[contact.any(axis=1)[1:], False])
                )
            }
        )
        keep = list(dict.fromkeys(priority))
        remaining = [value for value in ordered if value not in keep]
        budget = max(0, max_count - len(keep))
        if budget < len(remaining):
            positions = np.linspace(0, len(remaining) - 1, budget, dtype=np.int64) if budget else np.array([], dtype=np.int64)
            keep.extend(remaining[position] for position in positions)
        else:
            keep.extend(remaining)
        ordered = sorted(set(keep))
    return ordered


def make_mode_configs(candidate_profile: Path) -> dict[str, tuple[Any, Path | None]]:
    """Return the exact front-only baseline/candidate configs used by the QA."""
    baseline = make_baseline_config().with_camera_roles("front")
    candidate = apply_front_calibration_profile(baseline, candidate_profile)
    return {
        "baseline": (baseline, None),
        "candidate": (candidate, candidate_profile),
    }


def calibration_effects(config: Any) -> dict[str, Any]:
    """Make the actual diagnostic override explicit in report.json."""
    diagnostics = config.diagnostics
    color = diagnostics.front_color_k1
    return {
        "front_extrinsic_correction_enabled": bool(
            diagnostics.enable_front_extrinsic_correction
        ),
        "front_extrinsic_translation_base_m": list(
            diagnostics.front_extrinsic_translation_base_m
        ),
        "front_extrinsic_translation_applied_base_m": list(
            diagnostics.front_extrinsic_translation_base_m
            if diagnostics.enable_front_extrinsic_correction
            else (0.0, 0.0, 0.0)
        ),
        "front_color_intrinsic_k1_enabled": bool(
            diagnostics.enable_front_intrinsic_k1
        ),
        "front_color_intrinsics": {
            "fx": float(color.fx),
            "fy": float(color.fy),
            "cx": float(color.cx),
            "cy": float(color.cy),
        },
        "front_color_intrinsics_applied": (
            {
                "fx": float(color.fx),
                "fy": float(color.fy),
                "cx": float(color.cx),
                "cy": float(color.cy),
            }
            if diagnostics.enable_front_intrinsic_k1
            else None
        ),
        "front_depth_intrinsics_override": False,
        "note": (
            "The candidate changes base-frame extrinsic translation and color K1; "
            "it does not change depth intrinsics."
        ),
    }


def load_cad_assets(mesh_urdf: Path | None):
    """Load the verified visual CAD and the depth z-buffer renderer lazily."""
    try:
        from audit_front_depth_cad import mesh_groups, render_cad_depth
        from root_to_tip_mesh import load_visual_meshes, resolve_mesh_urdf
    except ImportError as exc:
        raise RuntimeError(
            "CAD self-filtering requires scripts/spatial on PYTHONPATH and "
            "the offline dependencies (trimesh, pycollada, scipy, opencv)."
        ) from exc
    selected_urdf = resolve_mesh_urdf(REPO_ROOT, mesh_urdf)
    meshes = load_visual_meshes(selected_urdf, include_fingers=True)
    return meshes, mesh_groups(meshes), render_cad_depth


def state_joint_positions(preprocessor: SpatialPreprocessor, state: np.ndarray) -> dict[str, float]:
    return {
        name: float(state[index])
        for name, index in preprocessor.state_mapping.items()
        if name in preprocessor.kinematics.measured_movable
    }


def build_dense_mode(
    *,
    preprocessor: SpatialPreprocessor,
    state: np.ndarray,
    depth: np.ndarray,
    rgb: np.ndarray,
    meshes: dict[str, Any],
    group_defs: tuple,
    render_cad_depth,
    self_tolerance_m: float,
) -> dict[str, Any]:
    """Build dense front cloud and self/object masks in the raw depth grid."""
    # ``calibration_bundle`` stores the source calibration.  The requested
    # front candidate profile is applied through diagnostic overrides, which
    # is also what the canonical visual preprocessor uses.
    camera = apply_diagnostic_overrides(
        preprocessor.calibration_bundle.cameras,
        preprocessor.config.diagnostics,
    )["front"]
    cache = build_camera_cache(
        camera,
        image_height=depth.shape[0],
        image_width=depth.shape[1],
    )
    xyz, source_pixel = depth_to_base_roi(
        depth,
        cache,
        roi=preprocessor.config.roi,
        min_depth_m=preprocessor.config.visual.min_depth_m,
        max_depth_m=preprocessor.config.visual.max_depth_m,
    )
    source_camera = np.zeros(len(xyz), dtype=np.int16)
    colors, rgb_valid = colorize_from_source_cameras(
        xyz,
        source_camera,
        camera_roles=("front",),
        rgb_by_role={"front": rgb},
        calibrations={"front": camera},
    )

    transforms = preprocessor.kinematics.compute(
        state_joint_positions(preprocessor, state),
        root="base_link",
    )
    cad_depth, cad_groups = render_cad_depth(
        meshes,
        transforms,
        camera,
        depth.shape,
        group_defs,
    )
    height, width = depth.shape
    y = source_pixel // width
    x = source_pixel % width
    observed_z = depth[y, x].astype(np.float32) * np.float32(camera.depth_scale_m_per_unit)
    predicted_z = cad_depth[y, x]
    self_mask = (
        (cad_groups[y, x] >= 0)
        & np.isfinite(predicted_z)
        & (np.abs(observed_z - predicted_z) <= float(self_tolerance_m))
    )
    return {
        "xyz": np.asarray(xyz, dtype=np.float32),
        "rgb": np.asarray(colors, dtype=np.uint8),
        "rgb_valid": np.asarray(rgb_valid, dtype=bool),
        "source_pixel": np.asarray(source_pixel, dtype=np.int64),
        "self_mask": np.asarray(self_mask, dtype=bool),
        "cad_group": np.asarray(cad_groups[y, x], dtype=np.int16),
        "cad_depth": np.asarray(predicted_z, dtype=np.float32),
        "observed_depth": np.asarray(observed_z, dtype=np.float32),
        "object_xyz": np.asarray(xyz[~self_mask], dtype=np.float32),
        "object_source_pixel": np.asarray(source_pixel[~self_mask], dtype=np.int64),
        "self_xyz": np.asarray(xyz[self_mask], dtype=np.float32),
        "self_source_pixel": np.asarray(source_pixel[self_mask], dtype=np.int64),
        "final_xyz": np.asarray(
            preprocessor.preprocess_visual(
                depth_by_role={"front": depth},
                rgb_by_role={"front": rgb},
            ).xyz_m,
            dtype=np.float32,
        ),
    }


def finite_stats(values_mm: np.ndarray) -> dict[str, Any]:
    values_mm = np.asarray(values_mm, dtype=np.float64)
    values_mm = values_mm[np.isfinite(values_mm)]
    if not len(values_mm):
        return {
            "count": 0,
            "min_mm": None,
            "median_mm": None,
            "p90_mm": None,
            "max_mm": None,
            "within_10mm": 0,
            "within_20mm": 0,
            "within_30mm": 0,
        }
    return {
        "count": int(len(values_mm)),
        "min_mm": float(np.min(values_mm)),
        "median_mm": float(np.median(values_mm)),
        "p90_mm": float(np.percentile(values_mm, 90)),
        "max_mm": float(np.max(values_mm)),
        "within_10mm": int(np.count_nonzero(values_mm <= 10.0)),
        "within_20mm": int(np.count_nonzero(values_mm <= 20.0)),
        "within_30mm": int(np.count_nonzero(values_mm <= 30.0)),
    }


def distance_metrics(
    *,
    mode: str,
    episode: int,
    frame: int,
    timestamp_s: float,
    dense: dict[str, Any],
    observation: Any,
    thresholds: np.ndarray,
    frame_contact: np.ndarray,
    false_overlap_m: float,
    pair_topk: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError("Distance metrics require scipy") from exc

    object_xyz = dense["object_xyz"]
    tactile_xyz = np.asarray(observation.tactile_xyz_m, dtype=np.float32)
    force_norm = np.asarray(observation.tactile_force_norm, dtype=np.float32)
    finger_id = np.asarray(observation.finger_id, dtype=np.int8)
    tree = cKDTree(object_xyz) if len(object_xyz) else None
    summary_rows: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []

    for finger in range(5):
        finger_mask = finger_id == finger
        active_mask = finger_mask & (force_norm >= float(thresholds[finger]))
        contact = bool(frame_contact[finger])
        query_mask = active_mask if contact else finger_mask
        if tree is None or not np.any(query_mask):
            distances = np.empty(0, dtype=np.float64)
            object_indices = np.empty(0, dtype=np.int64)
        else:
            distances, object_indices = tree.query(tactile_xyz[query_mask], k=1)
            distances = np.asarray(distances, dtype=np.float64)
            object_indices = np.asarray(object_indices, dtype=np.int64)
        distances_mm = distances * 1000.0
        stats = finite_stats(distances_mm)
        overlap = bool(np.any(distances <= false_overlap_m)) if not contact else False
        relation = "contact_distance" if contact else "false_overlap_probe"

        if len(distances):
            order = np.argsort(distances)
            if contact:
                order = order[::-1]
            for local_index in order[: max(0, pair_topk)]:
                tactile_index = np.flatnonzero(query_mask)[local_index]
                object_index = int(object_indices[local_index])
                pairs.append(
                    {
                        "mode": mode,
                        "episode": episode,
                        "frame": frame,
                        "finger": finger,
                        "finger_name": FINGER_NAMES[finger],
                        "relation": relation,
                        "distance_mm": float(distances_mm[local_index]),
                        "tactile_index": int(tactile_index),
                        "object_index": object_index,
                        "tactile_xyz_m": tactile_xyz[tactile_index].tolist(),
                        "object_xyz_m": object_xyz[object_index].tolist(),
                        "force_norm": float(force_norm[tactile_index]),
                    }
                )

        summary_rows.append(
            {
                "episode": episode,
                "frame": frame,
                "timestamp_s": timestamp_s,
                "mode": mode,
                "finger_id": finger,
                "finger": FINGER_NAMES[finger],
                "contact": contact,
                "force_max": float(force_norm[finger_mask].max()) if np.any(finger_mask) else 0.0,
                "threshold": float(thresholds[finger]),
                "active_taxels": int(active_mask.sum()),
                "object_points": int(len(object_xyz)),
                "self_points": int(len(dense["self_xyz"])),
                "dense_points": int(len(dense["xyz"])),
                "distance_role": relation,
                "false_overlap": overlap,
                "false_overlap_radius_mm": float(false_overlap_m * 1000.0),
                **stats,
            }
        )
    return summary_rows, pairs


def thin_indices(count: int, maximum: int, mandatory: Iterable[int] = ()) -> np.ndarray:
    if maximum <= 0 or count <= maximum:
        return np.arange(count, dtype=np.int64)
    required = sorted({int(index) for index in mandatory if 0 <= int(index) < count})
    remaining = max(0, maximum - len(required))
    candidates = np.array([index for index in range(count) if index not in set(required)], dtype=np.int64)
    if remaining >= len(candidates):
        extra = candidates
    elif remaining:
        extra = candidates[np.linspace(0, len(candidates) - 1, remaining, dtype=np.int64)]
    else:
        extra = np.empty(0, dtype=np.int64)
    return np.asarray(sorted(set(required).union(int(value) for value in extra)), dtype=np.int64)


def plotly_colors(rgb: np.ndarray, valid: np.ndarray) -> list[str]:
    values = np.asarray(rgb, dtype=np.uint8).copy()
    values[~np.asarray(valid, dtype=bool)] = np.array([140, 140, 140], dtype=np.uint8)
    return [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in values]


def add_cloud_trace(go, fig, *, name: str, xyz: np.ndarray, color: Any, customdata: np.ndarray, visible: Any, size: float, symbol: str = "circle") -> None:
    fig.add_trace(
        go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
            mode="markers", name=name, visible=visible,
            marker={"size": size, "color": color, "opacity": 0.85, "symbol": symbol},
            customdata=customdata,
            hovertemplate=(
                "mode=%{customdata[0]}<br>kind=%{customdata[1]}<br>"
                "index=%{customdata[2]}<br>x=%{x:.5f} m<br>"
                "y=%{y:.5f} m<br>z=%{z:.5f} m<extra></extra>"
            ),
        )
    )


def frame_html(
    *,
    frame_path: Path,
    episode: int,
    frame: int,
    timestamp_s: float,
    mode_data: dict[str, dict[str, Any]],
    observation: Any,
    pairs: list[dict[str, Any]],
    meshes: dict[str, Any] | None,
    group_defs: tuple | None,
    max_display: int,
) -> dict[str, Any]:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError("HTML output requires plotly") from exc

    fig = go.Figure()
    pair_object_indices = {
        mode: [int(pair["object_index"]) for pair in pairs if pair["mode"] == mode]
        for mode in mode_data
    }
    pair_trace_indices: list[int] = []

    for mode_index, (mode, data) in enumerate(mode_data.items()):
        dense = data["dense"]
        visible = True if mode == "candidate" else "legendonly"
        all_indices = thin_indices(len(dense["xyz"]), max_display)
        object_global = np.flatnonzero(~dense["self_mask"])
        self_global = np.flatnonzero(dense["self_mask"])
        object_indices = thin_indices(
            len(dense["object_xyz"]),
            max_display,
            mandatory=pair_object_indices[mode],
        )
        self_indices = thin_indices(len(dense["self_xyz"]), max_display)

        mode_labels = np.full((len(all_indices), 3), "", dtype=object)
        mode_labels[:, 0] = mode
        mode_labels[:, 1] = "dense"
        mode_labels[:, 2] = all_indices
        add_cloud_trace(
            go, fig,
            name=f"{mode} | dense RGB-D ({len(all_indices)}/{len(dense['xyz'])})",
            xyz=dense["xyz"][all_indices],
            color=plotly_colors(dense["rgb"][all_indices], dense["rgb_valid"][all_indices]),
            customdata=mode_labels,
            visible=visible,
            size=1.7,
        )

        object_labels = np.column_stack([
            np.full(len(object_indices), mode, dtype=object),
            np.full(len(object_indices), "object", dtype=object),
            object_indices,
        ])
        add_cloud_trace(
            go, fig,
            name=f"{mode} | SELF-FILTERED object ({len(object_indices)}/{len(dense['object_xyz'])})",
            xyz=dense["object_xyz"][object_indices],
            color="rgb(35,190,90)",
            customdata=object_labels,
            visible=visible,
            size=2.5,
        )

        self_labels = np.column_stack([
            np.full(len(self_indices), mode, dtype=object),
            np.full(len(self_indices), "robot_self", dtype=object),
            self_indices,
        ])
        add_cloud_trace(
            go, fig,
            name=f"{mode} | robot-self ({len(self_indices)}/{len(dense['self_xyz'])})",
            xyz=dense["self_xyz"][self_indices],
            color="rgb(240,130,30)",
            customdata=self_labels,
            visible="legendonly",
            size=2.0,
        )

        final_xyz = dense["final_xyz"]
        final_indices = thin_indices(len(final_xyz), max_display)
        final_labels = np.column_stack([
            np.full(len(final_indices), mode, dtype=object),
            np.full(len(final_indices), "final_4096", dtype=object),
            final_indices,
        ])
        add_cloud_trace(
            go, fig,
            name=f"{mode} | FINAL model visual ({len(final_xyz)})",
            xyz=final_xyz[final_indices],
            color="rgb(35,100,230)",
            customdata=final_labels,
            visible="legendonly",
            size=3.4,
            symbol="diamond",
        )

        # Pair traces are appended after point traces.  Each segment has three
        # points (start, end, None), so the hover/click handler can ignore them.
        mode_pairs = [pair for pair in pairs if pair["mode"] == mode]
        px: list[float | None] = []
        py: list[float | None] = []
        pz: list[float | None] = []
        for pair in mode_pairs:
            start = np.asarray(pair["tactile_xyz_m"], dtype=float)
            end = np.asarray(pair["object_xyz_m"], dtype=float)
            px.extend([float(start[0]), float(end[0]), None])
            py.extend([float(start[1]), float(end[1]), None])
            pz.extend([float(start[2]), float(end[2]), None])
        fig.add_trace(
            go.Scatter3d(
                x=px or [None], y=py or [None], z=pz or [None],
                mode="lines", name=f"{mode} | nearest pair lines ({len(mode_pairs)})",
                visible=visible,
                line={"width": 5, "color": "rgb(20,20,20)"}, hoverinfo="skip",
            )
        )
        pair_trace_indices.append(len(fig.data) - 1)

    tactile_xyz = np.asarray(observation.tactile_xyz_m, dtype=np.float32)
    tactile_force = np.asarray(observation.tactile_force_norm, dtype=np.float32)
    tactile_custom = np.column_stack([
        np.full(len(tactile_xyz), "shared", dtype=object),
        np.full(len(tactile_xyz), "tactile", dtype=object),
        np.arange(len(tactile_xyz), dtype=np.int64),
        np.asarray(observation.finger_id, dtype=np.int16),
        np.asarray(observation.taxel_id, dtype=np.int16),
        tactile_force,
    ])
    fig.add_trace(
        go.Scatter3d(
            x=tactile_xyz[:, 0], y=tactile_xyz[:, 1], z=tactile_xyz[:, 2],
            mode="markers", name="tactile 600", visible=True,
            marker={"size": 4.2, "color": [FINGER_COLORS[int(value)] for value in observation.finger_id]},
            customdata=tactile_custom,
            hovertemplate=(
                "kind=tactile<br>index=%{customdata[2]}<br>"
                "finger=%{customdata[3]}<br>taxel=%{customdata[4]}<br>"
                "force=%{customdata[5]:.4f}<br>x=%{x:.5f} m<br>"
                "y=%{y:.5f} m<br>z=%{z:.5f} m<extra></extra>"
            ),
        )
    )

    shared_thresholds = np.asarray(
        next(iter(mode_data.values()))["thresholds"],
        dtype=np.float32,
    )
    for finger in range(5):
        active = np.flatnonzero(
            (np.asarray(observation.finger_id) == finger)
            & (tactile_force >= shared_thresholds[finger])
        )
        if not len(active):
            continue
        custom = np.column_stack([
            np.full(len(active), "shared", dtype=object),
            np.full(len(active), f"active_{FINGER_NAMES[finger]}", dtype=object),
            active,
        ])
        fig.add_trace(
            go.Scatter3d(
                x=tactile_xyz[active, 0], y=tactile_xyz[active, 1], z=tactile_xyz[active, 2],
                mode="markers", name=f"active tactile | {FINGER_NAMES[finger]}",
                marker={"size": 7.0, "color": FINGER_COLORS[finger], "symbol": "diamond"},
                customdata=custom,
                hovertemplate="kind=%{customdata[1]}<br>index=%{customdata[2]}<br>x=%{x:.5f} m<br>y=%{y:.5f} m<br>z=%{z:.5f} m<extra></extra>",
            )
        )

    # CAD silhouette is useful context but not used as the object cloud.
    if meshes is not None and group_defs is not None:
        try:
            from visualize_spatial_web import cad_root_to_tip_traces
            for mode, data in mode_data.items():
                camera = apply_diagnostic_overrides(
                    data["preprocessor"].calibration_bundle.cameras,
                    data["preprocessor"].config.diagnostics,
                )["front"]
                traces = cad_root_to_tip_traces(
                    go=go,
                    preprocessor=data["preprocessor"],
                    state=data["state"],
                    camera=camera,
                    mesh_urdf=data["mesh_urdf"],
                    max_edges=1500,
                )
                for trace in traces:
                    trace.name = f"{mode} | {trace.name}"
                    trace.visible = mode == "candidate"
                    fig.add_trace(trace)
        except Exception as exc:  # CAD lines should not hide cloud QA output.
            print(f"warning: CAD silhouette omitted for episode={episode} frame={frame}: {exc}")

    pair_trace_index = len(fig.data)
    fig.add_trace(
        go.Scatter3d(
            x=[None, None, None], y=[None, None, None], z=[None, None, None],
            mode="lines+markers", name="selected arbitrary pair", visible=True,
            line={"width": 10, "color": "rgb(255,0,255)"},
            marker={"size": 6, "color": "rgb(255,0,255)"}, hoverinfo="skip",
        )
    )
    fig.update_layout(
        title=(
            f"Front cloud+tactile QA | episode={episode} frame={frame} "
            f"t={timestamp_s:.3f}s | click any two point markers"
        ),
        scene={
            "xaxis_title": "base_link X [m]",
            "yaxis_title": "base_link Y [m]",
            "zaxis_title": "base_link Z [m]",
            "aspectmode": "data",
            "dragmode": "orbit",
        },
        height=920,
        margin={"l": 0, "r": 260, "b": 0, "t": 90},
        legend={"itemsizing": "constant"},
    )

    plot_id = f"cloud_tactile_plot_{episode}_{frame}"
    plot_html = fig.to_html(
        full_html=False,
        include_plotlyjs="inline",
        div_id=plot_id,
    )
    pair_trace_index_json = json.dumps(pair_trace_index)
    script = f"""
<script>
(function() {{
  const gd = document.getElementById({json.dumps(plot_id)});
  const status = document.getElementById('pair-status');
  const selected = [];
  const pairTrace = {pair_trace_index_json};
  gd.on('plotly_click', function(event) {{
    if (!event.points || !event.points.length) return;
    const p = event.points[0];
    if (p.curveNumber === pairTrace || p.x === undefined || p.x === null) return;
    const point = {{x: Number(p.x), y: Number(p.y), z: Number(p.z),
                    label: (p.data.name || 'point') + ' / index=' +
                           (p.customdata && p.customdata[2] !== undefined ? p.customdata[2] : '?')}};
    if (selected.length >= 2) selected.length = 0;
    selected.push(point);
    if (selected.length === 1) {{
      status.textContent = 'First point selected: ' + point.label + '. Click a second point.';
      return;
    }}
    const a = selected[0], b = selected[1];
    const dx=b.x-a.x, dy=b.y-a.y, dz=b.z-a.z;
    const distance=Math.sqrt(dx*dx+dy*dy+dz*dz);
    status.textContent = 'distance=' + (distance*1000).toFixed(2) + ' mm | ' + a.label + ' ↔ ' + b.label;
    Plotly.restyle(gd, {{x:[[a.x,b.x,null]], y:[[a.y,b.y,null]], z:[[a.z,b.z,null]]}}, [pairTrace]);
  }});
}})();
</script>
"""
    mode_summary = {mode: data["summary_rows"] for mode, data in mode_data.items()}
    summary_text = html.escape(json.dumps(mode_summary, indent=2, ensure_ascii=False))
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>cloud tactile QA ep{episode} f{frame}</title>
<style>body{{font-family:system-ui,sans-serif;margin:18px}}#pair-status{{padding:10px;background:#fff3cd;border:1px solid #e0b85c;white-space:pre-wrap}}details{{margin:12px 0}}pre{{max-height:360px;overflow:auto}}</style>
</head><body><h1>Front-only cloud/tactile QA — episode {episode}, frame {frame}</h1>
<div id="pair-status">Click any two point markers to measure arbitrary 3-D distance.</div>
<p>Green = self-filtered object cloud; orange = robot-self points; blue diamonds = final model visual points; magenta = selected pair.</p>
{plot_html}
<details><summary>Per-finger metrics</summary><pre>{summary_text}</pre></details>
{script}</body></html>"""
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    frame_path.write_text(document, encoding="utf-8")
    return {"path": str(frame_path), "pair_trace_index": pair_trace_index}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_cloud_npz(path: Path, *, mode: str, dense: dict[str, Any]) -> None:
    """Persist the full-resolution cloud used for metrics, not just HTML points."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        mode=np.asarray(mode),
        dense_xyz_m=np.asarray(dense["xyz"], dtype=np.float32),
        dense_rgb=np.asarray(dense["rgb"], dtype=np.uint8),
        dense_rgb_valid=np.asarray(dense["rgb_valid"], dtype=bool),
        dense_source_pixel=np.asarray(dense["source_pixel"], dtype=np.int64),
        dense_self_mask=np.asarray(dense["self_mask"], dtype=bool),
        dense_cad_group=np.asarray(dense["cad_group"], dtype=np.int16),
        dense_cad_depth_m=np.asarray(dense["cad_depth"], dtype=np.float32),
        dense_observed_depth_m=np.asarray(dense["observed_depth"], dtype=np.float32),
        object_xyz_m=np.asarray(dense["object_xyz"], dtype=np.float32),
        object_source_pixel=np.asarray(dense["object_source_pixel"], dtype=np.int64),
        self_xyz_m=np.asarray(dense["self_xyz"], dtype=np.float32),
        self_source_pixel=np.asarray(dense["self_source_pixel"], dtype=np.int64),
        final_xyz_m=np.asarray(dense["final_xyz"], dtype=np.float32),
    )


def aggregate_false_overlap(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize no-contact nearest-neighbor probes by mode and finger."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if str(row["contact"]).lower() == "true":
            continue
        grouped.setdefault((str(row["mode"]), str(row["finger"])), []).append(row)

    output: list[dict[str, Any]] = []
    for (mode, finger), values in sorted(grouped.items()):
        distances = np.asarray(
            [float(row["min_mm"]) for row in values if row["min_mm"] not in (None, "")],
            dtype=np.float64,
        )
        overlap_count = sum(
            str(row["false_overlap"]).lower() == "true" for row in values
        )
        stats = finite_stats(distances)
        output.append(
            {
                "mode": mode,
                "finger": finger,
                "no_contact_frames": len(values),
                "false_overlap_frames": overlap_count,
                "false_overlap_rate": float(overlap_count / len(values)) if values else 0.0,
                "overlap_radius_mm": values[0]["false_overlap_radius_mm"] if values else None,
                "nearest_min_mm": stats["min_mm"],
                "nearest_median_mm": stats["median_mm"],
                "nearest_p90_mm": stats["p90_mm"],
                "nearest_max_mm": stats["max_mm"],
            }
        )
    return output


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def main() -> None:
    args = parse_args()
    if args.contact_context < 0 or args.max_visual_frames_per_episode < 0:
        raise ValueError("contact-context and max-visual-frames-per-episode must be >= 0")
    if args.contact_min_force < 0 or args.contact_mad_scale < 0:
        raise ValueError("contact force parameters must be >= 0")
    if args.self_tolerance_mm <= 0 or args.false_overlap_mm <= 0:
        raise ValueError("distance tolerances must be > 0")
    if args.max_display_points < 0 or args.pair_topk < 0:
        raise ValueError("display and pair limits must be >= 0")

    dataset_root = resolve_path(args.dataset)
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    dataset_ids = RawSpatialDataset.discover_episode_ids(dataset_root)
    episode_ids = parse_episode_ids(args.episodes, dataset_ids)
    candidate_profile = resolve_path(args.candidate_profile)
    if not candidate_profile.is_file():
        raise FileNotFoundError(candidate_profile)
    output = resolve_path(args.output)
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {output}")
    output.mkdir(parents=True)

    mode_configs = make_mode_configs(candidate_profile)
    preprocessors = {
        mode: SpatialPreprocessor.from_repo_root(repo_root=REPO_ROOT, config=config)
        for mode, (config, _) in mode_configs.items()
    }
    meshes, group_defs, render_cad_depth = load_cad_assets(args.mesh_urdf)
    mesh_urdf = resolve_path(args.mesh_urdf) if args.mesh_urdf else None
    if mesh_urdf is None:
        from root_to_tip_mesh import resolve_mesh_urdf
        mesh_urdf = resolve_mesh_urdf(REPO_ROOT, None)

    contact_rows: list[dict[str, Any]] = []
    distance_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    episode_reports: list[dict[str, Any]] = []
    frame_links: list[dict[str, Any]] = []

    for episode in episode_ids:
        dataset = RawSpatialDataset(dataset_root, episode=episode)
        contact_doc, rows, force_max, contact = contact_document(
            dataset,
            quiet_frames=args.quiet_frames,
            minimum=args.contact_min_force,
            mad_scale=args.contact_mad_scale,
        )
        thresholds, _ = quiet_thresholds(
            force_max,
            quiet_frames=args.quiet_frames,
            minimum=args.contact_min_force,
            mad_scale=args.contact_mad_scale,
        )
        contact_rows.extend(rows)
        states_doc = dataset.states()
        frame_ids = np.asarray(states_doc["frame_index"], dtype=np.int64)
        timestamps = np.asarray(states_doc["timestamp"], dtype=np.float64)
        states = np.asarray(states_doc["observation.state"], dtype=np.float32)
        frame_to_row = {int(frame): index for index, frame in enumerate(frame_ids)}
        selected_frames = parse_frame_ids(
            args.frames,
            frame_ids=frame_ids,
            contact=contact,
            context=args.contact_context,
            max_count=args.max_visual_frames_per_episode,
        )
        selected_set = set(selected_frames)
        raw_rows = {
            int(row["frame_index"]): row
            for row in dataset.iter_rows(selected=selected_set, depth_roles=("front",))
        }
        rgb_by_frame = dataset.load_video_frames("front", selected=selected_set)

        episode_reports.append({
            **contact_doc,
            "selected_visual_frames": selected_frames,
            "contact_frames": int(contact.any(axis=1).sum()),
        })
        print(
            f"episode {episode}: {len(frame_ids)} frames, "
            f"{int(contact.any(axis=1).sum())} contact frames, "
            f"{len(selected_frames)} visual frames",
            flush=True,
        )

        for frame in selected_frames:
            row = raw_rows[frame]
            state = np.asarray(row["observation.state"], dtype=np.float32)
            depth = np.asarray(row["observation.depths.cam_front"])
            rgb = np.asarray(rgb_by_frame[frame], dtype=np.uint8)
            timestamp_s = float(row["timestamp"])
            mode_data: dict[str, dict[str, Any]] = {}
            all_pairs: list[dict[str, Any]] = []
            for mode, preprocessor in preprocessors.items():
                dense = build_dense_mode(
                    preprocessor=preprocessor,
                    state=state,
                    depth=depth,
                    rgb=rgb,
                    meshes=meshes,
                    group_defs=group_defs,
                    render_cad_depth=render_cad_depth,
                    self_tolerance_m=args.self_tolerance_mm / 1000.0,
                )
                observation = preprocessor.preprocess(
                    frame_index=frame,
                    timestamp_s=timestamp_s,
                    state=state,
                    depth_by_role={"front": depth},
                    rgb_by_role={"front": rgb},
                )
                summary, pairs = distance_metrics(
                    mode=mode,
                    episode=episode,
                    frame=frame,
                    timestamp_s=timestamp_s,
                    dense=dense,
                    observation=observation,
                    thresholds=thresholds,
                    frame_contact=contact[frame_to_row[frame]],
                    false_overlap_m=args.false_overlap_mm / 1000.0,
                    pair_topk=args.pair_topk,
                )
                distance_rows.extend(summary)
                pair_rows.extend(pairs)
                all_pairs.extend(pairs)
                cloud_relative = (
                    Path("clouds")
                    / f"episode_{episode:06d}"
                    / f"frame_{frame:06d}_{mode}.npz"
                )
                save_cloud_npz(output / cloud_relative, mode=mode, dense=dense)
                mode_data[mode] = {
                    "dense": dense,
                    "preprocessor": preprocessor,
                    "state": state,
                    "mesh_urdf": mesh_urdf,
                    "thresholds": thresholds,
                    "summary_rows": summary,
                    "observation": observation,
                    "cloud_npz": str(cloud_relative),
                }
            frame_relative = Path("frames") / f"episode_{episode:06d}" / f"frame_{frame:06d}_compare.html"
            frame_abs = output / frame_relative
            html_info = frame_html(
                frame_path=frame_abs,
                episode=episode,
                frame=frame,
                timestamp_s=timestamp_s,
                mode_data=mode_data,
                observation=mode_data["candidate"]["observation"],
                pairs=all_pairs,
                meshes=meshes,
                group_defs=group_defs,
                max_display=args.max_display_points,
            )
            frame_report = {
                "episode": episode,
                "frame": frame,
                "timestamp_s": timestamp_s,
                "html": str(frame_relative),
                "clouds": {
                    mode: data["cloud_npz"] for mode, data in mode_data.items()
                },
                "pairs": all_pairs,
                "metrics": {mode: data["summary_rows"] for mode, data in mode_data.items()},
            }
            frame_abs.with_suffix(".json").write_text(
                json.dumps(jsonable(frame_report), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            frame_links.append({
                "episode": episode,
                "frame": frame,
                "timestamp_s": timestamp_s,
                "html": str(frame_relative),
                "json": str(frame_relative.with_suffix(".json")),
                "contact_any": bool(contact[frame_to_row[frame]].any()),
                "pair_count": len(all_pairs),
                "pair_trace_index": html_info["pair_trace_index"],
            })

    write_csv(output / "contact_states.csv", contact_rows)
    write_csv(output / "distance_summary.csv", distance_rows)
    write_csv(output / "pair_summary.csv", pair_rows)
    false_overlap_rows = aggregate_false_overlap(distance_rows)
    write_csv(output / "false_overlap_summary.csv", false_overlap_rows)
    report = {
        "dataset": str(dataset_root),
        "episodes": episode_ids,
        "mode": "front_only_structured_suffix",
        "training_config_equivalent": {
            "camera_roles": ["front"],
            "sidecar_version": "v1_front_candidate",
            "candidate_profile": str(candidate_profile),
            "use_visual": True,
            "use_tactile": True,
        },
        "modes": {
            "baseline": {
                "profile": None,
                "diagnostics": calibration_effects(mode_configs["baseline"][0]),
            },
            "candidate": {
                "profile": str(candidate_profile),
                "diagnostics": calibration_effects(mode_configs["candidate"][0]),
            },
        },
        "contact_detection": {
            "min_force": args.contact_min_force,
            "mad_scale": args.contact_mad_scale,
            "quiet_frames": args.quiet_frames,
        },
        "self_filter": {
            "method": "FK visual CAD depth z-buffer in raw front depth pixels",
            "tolerance_mm": args.self_tolerance_mm,
            "object_cloud_definition": "dense ROI points not matching visible CAD depth within tolerance",
        },
        "false_overlap": {
            "radius_mm": args.false_overlap_mm,
            "note": "No-contact proximity is a candidate for inspection, not automatic physical contact ground truth.",
        },
        "visual_frames": frame_links,
        "false_overlap_summary": false_overlap_rows,
        "episodes": episode_reports,
    }
    (output / "report.json").write_text(
        json.dumps(jsonable(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    table_rows = "\n".join(
        f"<tr><td>{item['episode']}</td><td>{item['frame']}</td>"
        f"<td>{item['timestamp_s']:.3f}</td><td>{item['contact_any']}</td>"
        f"<td>{item['pair_count']}</td><td><a href='{html.escape(item['html'])}'>open 3D QA</a></td>"
        f"<td><a href='{html.escape(item['json'])}'>metrics JSON</a></td></tr>"
        for item in frame_links
    )
    index_html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>front cloud tactile audit</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px}}table{{border-collapse:collapse}}th,td{{padding:8px 12px;border:1px solid #bbb}}code{{word-break:break-all}}</style>
</head><body><h1>Front-only cloud/tactile audit</h1>
<p>Dataset: <code>{html.escape(str(dataset_root))}</code></p>
<p>Candidate profile: <code>{html.escape(str(candidate_profile))}</code></p>
<p>Every frame/finger contact state is in <a href="contact_states.csv">contact_states.csv</a>.
Visual metrics are in <a href="distance_summary.csv">distance_summary.csv</a>, no-contact aggregates in <a href="false_overlap_summary.csv">false_overlap_summary.csv</a>,
arbitrary pair records in <a href="pair_summary.csv">pair_summary.csv</a>,
and full-resolution baseline/candidate cloud arrays are under <code>clouds/</code>.</p>
<table><thead><tr><th>episode</th><th>frame</th><th>time [s]</th><th>any contact</th><th>pair records</th><th>3-D page</th><th>JSON</th></tr></thead>
<tbody>{table_rows}</tbody></table></body></html>"""
    (output / "index.html").write_text(index_html, encoding="utf-8")
    print(f"wrote {output / 'index.html'}")

    if args.open:
        import webbrowser
        webbrowser.open((output / "index.html").as_uri())


if __name__ == "__main__":
    main()
