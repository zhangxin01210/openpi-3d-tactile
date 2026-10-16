#!/usr/bin/env python3
"""Render a recorded UR7e + XHand PI0 run into adaptive 2-D and 3-D diagnostics.

The renderer detects the raw arrays present in the recording instead of trusting
the selected model name: RGB-only runs render RGB/state/action diagnostics,
while spatial runs additionally rebuild the recorded RGB-D cloud and write
self-contained interactive Plotly pages under ``report/cloud_index.html``.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "src").is_dir() else SCRIPT_DIR
sys.path.insert(0, str(REPO_ROOT / "src"))


FINGER_COLORS = ("#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render PI0 deployment diagnostic recordings")
    parser.add_argument("run_dir", type=Path, help="One recorded run directory containing manifest.json")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Repository root containing spatial assets")
    parser.add_argument("--video", action="store_true", help="Also create diagnostic.mp4 from query figures")
    parser.add_argument("--video-fps", type=float, default=2.0)
    parser.add_argument("--max-summary-keyframes", type=int, default=3)
    parser.add_argument(
        "--max-cloud-keyframes",
        type=int,
        default=3,
        help="Number of representative spatial queries exported as interactive cloud HTML pages; 0 exports all.",
    )
    parser.add_argument(
        "--max-cloud-points",
        type=int,
        default=35000,
        help="Maximum dense RGB-D points shown per camera in each interactive HTML page; 0 keeps all.",
    )
    parser.add_argument(
        "--no-cloud-html",
        action="store_true",
        help="Do not write interactive Plotly cloud pages for spatial recordings.",
    )
    parser.add_argument("--skip-query-images", action="store_true")
    parser.add_argument("--front-calibration-profile", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def load_control(run_dir: Path) -> dict[str, np.ndarray]:
    blocks = [load_npz(path) for path in sorted((run_dir / "control").glob("block_*.npz"))]
    if not blocks:
        return {}
    common_keys = set.intersection(*(set(block) for block in blocks))
    return {key: np.concatenate([block[key] for block in blocks], axis=0) for key in sorted(common_keys)}


def load_queries(run_dir: Path) -> list[dict]:
    queries = []
    for input_path in sorted((run_dir / "inference").glob("query_*_input.npz")):
        stem = input_path.name.removesuffix("_input.npz")
        output_path = input_path.with_name(f"{stem}_output.npz")
        queries.append(
            {
                "name": stem,
                "input_path": input_path,
                "output_path": output_path if output_path.exists() else None,
                "input": load_npz(input_path),
                "output": load_npz(output_path) if output_path.exists() else None,
            }
        )
    return queries


def scalar(value, default=np.nan):
    if value is None:
        return default
    array = np.asarray(value)
    return array.reshape(-1)[0].item() if array.size else default


def selected_indices(count: int, maximum: int) -> list[int]:
    if count <= 0 or maximum <= 0:
        return []
    return sorted(set(np.linspace(0, count - 1, min(count, maximum), dtype=int).tolist()))


def query_depth_roles(query_input: dict[str, np.ndarray]) -> tuple[str, ...]:
    """Return camera roles actually recorded in one query, in a stable order."""
    prefix = "depth_cam_"
    return tuple(sorted(key.removeprefix(prefix) for key in query_input if key.startswith(prefix)))


def recorded_depth_roles(manifest: dict, queries: list[dict]) -> tuple[str, ...]:
    """Prefer manifest order, but trust the recorded arrays over stale metadata."""
    observed = {role for query in queries for role in query_depth_roles(query["input"])}
    configured = tuple(str(role).removeprefix("cam_") for role in manifest.get("spatial_camera_roles", ()))
    return tuple(role for role in configured if role in observed) + tuple(sorted(observed - set(configured)))


def recorded_rgb_cameras(queries: list[dict]) -> tuple[str, ...]:
    prefix = "rgb_"
    return tuple(sorted({key.removeprefix(prefix) for query in queries for key in query["input"] if key.startswith(prefix)}))


def has_recorded_tactile(queries: list[dict]) -> bool:
    return any("tactile_raw_force_norm" in query["input"] for query in queries)


def image_for_display(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[-1] == 3:
        return array
    return np.squeeze(array)


def show_depth(ax, depth: np.ndarray, title: str) -> None:
    values = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(values) & (values > 0)
    display = np.full(values.shape, np.nan, dtype=np.float32)
    if np.any(valid):
        low, high = np.percentile(values[valid], [1, 99])
        display[valid] = np.clip(values[valid], low, high)
        title += f"\nvalid={valid.mean() * 100:.1f}% p50={np.median(values[valid]):.1f}"
    else:
        title += "\nno valid depth"
    ax.imshow(display, cmap="turbo")
    ax.set_title(title)
    ax.axis("off")


def named_indices(names: list[str], prefix: str, suffix: str = ".pos") -> list[int]:
    return [index for index, name in enumerate(names) if name.startswith(prefix) and name.endswith(suffix)]


def plot_channels(ax, time_s: np.ndarray, values: np.ndarray, names: list[str], title: str) -> None:
    if values.size == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
    else:
        for index in range(values.shape[1]):
            label = names[index] if index < len(names) else str(index)
            ax.plot(time_s, values[:, index], linewidth=1.0, label=label)
        if values.shape[1] <= 12:
            ax.legend(fontsize=6, ncol=min(4, values.shape[1]), loc="best")
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.grid(alpha=0.25)


def make_preprocessor(repo_root: Path, roles: tuple[str, ...], profile: Path | None = None):
    from openpi.spatial.config import make_baseline_config
    from openpi.spatial.preprocess import SpatialPreprocessor

    config = make_baseline_config().with_camera_roles(*(roles or ("front",)))
    if profile is not None:
        from openpi.spatial.calibration_profile import apply_front_calibration_profile

        config = apply_front_calibration_profile(config, profile)
    return SpatialPreprocessor.from_repo_root(repo_root=repo_root.expanduser().resolve(), config=config)


def reconstruct_spatial(
    preprocessor,
    query_input: dict[str, np.ndarray],
    roles: tuple[str, ...],
    *,
    include_tactile: bool,
):
    """Rebuild only the modalities that were actually recorded for this query."""
    visual = None
    tactile = None
    if roles:
        visual = preprocessor.preprocess_visual(
            depth_by_role={role: query_input[f"depth_cam_{role}"] for role in roles},
            rgb_by_role={role: query_input[f"rgb_cam_{role}"] for role in roles},
        )
    if include_tactile:
        tactile = preprocessor.preprocess_tactile(np.asarray(query_input["state"], dtype=np.float32))
    return visual, tactile


def display_indices(count: int, maximum: int) -> np.ndarray:
    """Evenly thin only the HTML display copy, never the reconstructed cloud."""
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    if maximum < 0:
        raise ValueError("maximum must be non-negative")
    if maximum == 0 or count <= maximum:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, num=maximum, dtype=np.int64)


def reconstruct_dense_clouds(preprocessor, query_input: dict[str, np.ndarray], roles: tuple[str, ...]) -> dict[str, dict]:
    """Reconstruct dense, per-camera ROI clouds for display/provenance only."""
    from openpi.spatial.geometry import apply_diagnostic_overrides
    from openpi.spatial.geometry import build_camera_cache
    from openpi.spatial.geometry import colorize_from_source_cameras
    from openpi.spatial.geometry import depth_to_base_roi

    calibrations = apply_diagnostic_overrides(
        preprocessor.calibration_bundle.cameras,
        preprocessor.config.diagnostics,
    )
    rgb_by_role = {role: query_input[f"rgb_cam_{role}"] for role in roles}
    output = {}
    for camera_index, role in enumerate(roles):
        depth = np.asarray(query_input[f"depth_cam_{role}"])
        cache = build_camera_cache(
            calibrations[role],
            image_height=depth.shape[0],
            image_width=depth.shape[1],
        )
        xyz, _ = depth_to_base_roi(
            depth,
            cache,
            roi=preprocessor.config.roi,
            min_depth_m=preprocessor.config.visual.min_depth_m,
            max_depth_m=preprocessor.config.visual.max_depth_m,
        )
        source_camera = np.full(len(xyz), camera_index, dtype=np.int16)
        rgb, rgb_valid = colorize_from_source_cameras(
            xyz,
            source_camera,
            camera_roles=roles,
            rgb_by_role=rgb_by_role,
            calibrations=calibrations,
        )
        output[role] = {"xyz": xyz, "rgb": rgb, "rgb_valid": rgb_valid}
    return output


def query_times(queries: list[dict]) -> np.ndarray:
    return np.asarray(
        [
            float(
                scalar(
                    query["input"].get("client_elapsed_s", query["input"].get("timestamp_s")),
                    query_index,
                )
            )
            for query_index, query in enumerate(queries)
        ]
    )


def render_summary(
    run_dir: Path,
    report_dir: Path,
    manifest: dict,
    control: dict,
    queries: list[dict],
    max_keyframes: int,
) -> Path:
    import matplotlib.pyplot as plt

    rgb_cameras = recorded_rgb_cameras(queries)
    depth_roles = recorded_depth_roles(manifest, queries)
    record_tactile = has_recorded_tactile(queries)
    figure = plt.figure(figsize=(21, 24), constrained_layout=True)
    grid = figure.add_gridspec(5, 3, height_ratios=(1.0, 1.0, 1.0, 0.9, 0.55))
    keyframes = selected_indices(len(queries), max_keyframes)
    for column in range(3):
        ax = figure.add_subplot(grid[0, column])
        if column < len(keyframes):
            query = queries[keyframes[column]]["input"]
            image = query.get("rgb_cam_front")
            if image is None and rgb_cameras:
                image = query.get(f"rgb_{rgb_cameras[0]}")
            if image is not None:
                ax.imshow(image_for_display(image))
            ax.set_title(
                f"RGB: query {keyframes[column]} / frame {int(scalar(query.get('frame_index'), -1))}"
            )
        else:
            ax.text(0.5, 0.5, "no keyframe", ha="center", va="center")
        ax.axis("off")

    time_s = np.asarray(control.get("elapsed_s", np.empty(0)), dtype=np.float64)
    state = np.asarray(control.get("state", np.empty((0, 0))), dtype=np.float32)
    command = np.asarray(control.get("command_action", np.empty((0, 0))), dtype=np.float32)
    state_names = list(manifest.get("state_names", []))
    action_names = list(manifest.get("action_names", []))
    arm_state_idx = named_indices(state_names, "arm_joint_")[:6]
    hand_state_idx = named_indices(state_names, "hand_joint_")[:12]
    arm_action_idx = named_indices(action_names, "arm_joint_")[:6]
    hand_action_idx = named_indices(action_names, "hand_joint_")[:12]

    plot_channels(
        figure.add_subplot(grid[1, 0]),
        time_s,
        state[:, arm_state_idx] if state.size and arm_state_idx else np.empty((0, 0)),
        [state_names[index] for index in arm_state_idx],
        "Observed arm joint positions",
    )
    plot_channels(
        figure.add_subplot(grid[1, 1]),
        time_s,
        state[:, hand_state_idx] if state.size and hand_state_idx else np.empty((0, 0)),
        [state_names[index] for index in hand_state_idx],
        "Observed hand joint positions",
    )
    timing_ax = figure.add_subplot(grid[1, 2])
    if time_s.size:
        timing_ax.plot(time_s, control["loop_ms"], label="control loop", linewidth=1.0)
    q_time = query_times(queries)
    roundtrip = np.asarray(
        [float(scalar(query["output"].get("client_roundtrip_ms"))) if query["output"] else np.nan for query in queries]
    )
    if q_time.size:
        timing_ax.scatter(q_time, roundtrip, label="inference roundtrip", s=22, color="#d62728")
    timing_ax.set_title("Latency")
    timing_ax.set_xlabel("time (s)")
    timing_ax.set_ylabel("ms")
    timing_ax.grid(alpha=0.25)
    timing_ax.legend(fontsize=7)

    plot_channels(
        figure.add_subplot(grid[2, 0]),
        time_s,
        command[:, arm_action_idx] if command.size and arm_action_idx else np.empty((0, 0)),
        [action_names[index] for index in arm_action_idx],
        "Commanded arm actions",
    )
    plot_channels(
        figure.add_subplot(grid[2, 1]),
        time_s,
        command[:, hand_action_idx] if command.size and hand_action_idx else np.empty((0, 0)),
        [action_names[index] for index in hand_action_idx],
        "Commanded hand actions",
    )
    tactile_ax = figure.add_subplot(grid[2, 2])
    finger_names = manifest.get("tactile", {}).get("finger_names", [f"finger_{i}" for i in range(5)])
    if record_tactile:
        tactile_max = np.asarray(
            [np.max(query["input"]["tactile_raw_force_norm"], axis=1) for query in queries], dtype=np.float32
        )
        for finger_index in range(tactile_max.shape[1]):
            tactile_ax.plot(
                q_time,
                tactile_max[:, finger_index],
                marker=".",
                color=FINGER_COLORS[finger_index],
                label=finger_names[finger_index],
            )
    tactile_ax.set_title("Per-finger max raw tactile norm" if record_tactile else "Tactile not recorded")
    tactile_ax.set_xlabel("time (s)")
    tactile_ax.grid(alpha=0.25)
    tactile_ax.legend(fontsize=7)

    depth_ax = figure.add_subplot(grid[3, 0])
    for role, color in zip(depth_roles, ("#1f77b4", "#ff7f0e", "#2ca02c"), strict=False):
        ratios = []
        for query in queries:
            depth = np.asarray(query["input"].get(f"depth_cam_{role}", np.empty(0)))
            ratios.append(float(np.mean(np.isfinite(depth) & (depth > 0))) if depth.size else np.nan)
        if ratios:
            depth_ax.plot(q_time, np.asarray(ratios) * 100.0, marker=".", color=color, label=role)
    depth_ax.set_title("Valid depth pixels" if depth_roles else "Depth not recorded")
    depth_ax.set_xlabel("time (s)")
    depth_ax.set_ylabel("percent")
    depth_ax.grid(alpha=0.25)
    depth_ax.legend(fontsize=7)

    chunk_ax = figure.add_subplot(grid[3, 1])
    if time_s.size:
        chunk_ax.step(time_s, control["query_id"], where="post", label="query id")
        chunk_ax.plot(time_s, control["chunk_step"], linewidth=0.8, label="chunk step")
    chunk_ax.set_title("Action chunk execution")
    chunk_ax.set_xlabel("time (s)")
    chunk_ax.grid(alpha=0.25)
    chunk_ax.legend(fontsize=7)

    fallback_ax = figure.add_subplot(grid[3, 2])
    if time_s.size:
        fallback_ax.fill_between(time_s, 0, np.asarray(control["fallback"], dtype=np.float32), step="post", alpha=0.5)
    fallback_ax.set_title("Inference fallback / hold")
    fallback_ax.set_xlabel("time (s)")
    fallback_ax.set_ylim(-0.05, 1.05)
    fallback_ax.grid(alpha=0.25)

    info_ax = figure.add_subplot(grid[4, :])
    success_count = sum(bool(scalar(query["output"].get("success"), False)) for query in queries if query["output"])
    info = (
        f"run={manifest.get('run_index')}  status={manifest.get('status')}  end={manifest.get('end_reason')}  "
        f"frames={manifest.get('frame_count')}  queries={len(queries)}  successful={success_count}\n"
        f"dropped_records={manifest.get('dropped_records', 0)}  "
        f"writer_errors={len(manifest.get('writer_errors', []))}  "
        f"rgb={','.join(rgb_cameras) or 'none'}  depth={','.join(depth_roles) or 'none'}  "
        f"tactile={'yes' if record_tactile else 'no'}\n"
        f"prompt={manifest.get('args', {}).get('prompt') or manifest.get('args', {}).get('task')}\n"
        "Raw input, output chunks, per-frame state, raw action, and final commanded action remain in the run directory."
    )
    info_ax.text(0.01, 0.65, info, va="center", family="monospace", fontsize=11)
    info_ax.axis("off")
    figure.suptitle(f"PI0 diagnostic summary: {run_dir.name}", fontsize=18)
    output_path = report_dir / "summary.png"
    figure.savefig(output_path, dpi=140)
    plt.close(figure)
    return output_path


def scatter_spatial(ax, xyz: np.ndarray, color: np.ndarray | str, title: str) -> None:
    points = np.asarray(xyz)
    if not points.size:
        ax.text(0.5, 0.5, "no points", ha="center", va="center")
    else:
        stride = max(1, math.ceil(points.shape[0] / 12000))
        selected = points[::stride]
        selected_color = color[::stride] if isinstance(color, np.ndarray) else color
        ax.scatter(selected[:, 0], selected[:, 1], c=selected_color, s=1.2, alpha=0.75)
        ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=0.2)


def plotly_colors(rgb: np.ndarray, rgb_valid: np.ndarray) -> list[str]:
    values = np.asarray(rgb, dtype=np.uint8).copy()
    values[~np.asarray(rgb_valid, dtype=np.bool_)] = np.array((140, 140, 140), dtype=np.uint8)
    return [f"rgb({int(red)},{int(green)},{int(blue)})" for red, green, blue in values]


def add_cloud_trace(go, figure, *, name: str, xyz: np.ndarray, color, customdata: np.ndarray, visible, size: float, symbol: str = "circle") -> None:
    figure.add_trace(
        go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers", name=name, visible=visible,
            marker={"size": size, "color": color, "opacity": 0.86, "symbol": symbol}, customdata=customdata,
            hovertemplate=(
                "kind=%{customdata[0]}<br>index=%{customdata[1]}<br>"
                "x=%{x:.5f} m<br>y=%{y:.5f} m<br>z=%{z:.5f} m<extra></extra>"
            ),
        )
    )


def render_cloud_html(
    report_dir: Path,
    query_index: int,
    query: dict,
    preprocessor,
    roles: tuple[str, ...],
    *,
    include_tactile: bool,
    max_dense_points: int,
) -> Path:
    """Write a self-contained interactive base_link cloud viewer for one query."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError(
            "Interactive cloud output requires plotly. Install it with `pip install plotly`, then rerun this command."
        ) from exc

    query_input = query["input"]
    visual, tactile = reconstruct_spatial(
        preprocessor, query_input, roles, include_tactile=include_tactile
    )
    dense_clouds = reconstruct_dense_clouds(preprocessor, query_input, roles)
    figure = go.Figure()
    summary = {"dense_roi": {}, "final_model_visual": 0, "tactile": 0}

    for role in roles:
        dense = dense_clouds[role]
        indices = display_indices(len(dense["xyz"]), max_dense_points)
        points = dense["xyz"][indices]
        summary["dense_roi"][role] = {"total": int(len(dense["xyz"])), "shown": int(len(points))}
        if len(points):
            customdata = np.column_stack((np.full(len(indices), f"dense_roi_{role}", dtype=object), indices))
            add_cloud_trace(
                go, figure, name=f"dense ROI {role} ({len(points):,}/{len(dense['xyz']):,})", xyz=points,
                color=plotly_colors(dense["rgb"][indices], dense["rgb_valid"][indices]), customdata=customdata,
                visible=True, size=1.6,
            )

    if visual is not None:
        indices = display_indices(len(visual.xyz_m), max_dense_points)
        points = visual.xyz_m[indices]
        summary["final_model_visual"] = int(len(visual.xyz_m))
        customdata = np.column_stack((np.full(len(indices), "final_model_visual", dtype=object), indices))
        add_cloud_trace(
            go, figure, name=f"final model visual ({len(visual.xyz_m):,})", xyz=points,
            color="rgb(30,100,235)", customdata=customdata, visible="legendonly", size=3.5, symbol="diamond",
        )

    if tactile is not None:
        summary["tactile"] = int(len(tactile.xyz_m))
        customdata = np.column_stack((np.full(len(tactile.xyz_m), "tactile", dtype=object), np.arange(len(tactile.xyz_m))))
        finger_colors = [FINGER_COLORS[int(finger)] for finger in tactile.finger_id]
        add_cloud_trace(
            go, figure, name=f"tactile taxels ({len(tactile.xyz_m):,})", xyz=tactile.xyz_m,
            color=finger_colors, customdata=customdata, visible=True, size=4.0, symbol="diamond",
        )

    selected_pair_trace = len(figure.data)
    figure.add_trace(
        go.Scatter3d(
            x=[None, None, None], y=[None, None, None], z=[None, None, None], mode="lines+markers",
            name="selected distance pair", line={"width": 8, "color": "rgb(255,0,255)"},
            marker={"size": 6, "color": "rgb(255,0,255)"}, hoverinfo="skip",
        )
    )
    frame_index = int(scalar(query_input.get("frame_index"), -1))
    timestamp_s = float(scalar(query_input.get("timestamp_s"), np.nan))
    figure.update_layout(
        title=(
            f"PI0 deployment cloud | query={query_index}, frame={frame_index}, t={timestamp_s:.3f}s "
            "| click any two points to measure distance"
        ),
        scene={
            "xaxis_title": "base_link X [m]", "yaxis_title": "base_link Y [m]", "zaxis_title": "base_link Z [m]",
            "aspectmode": "data", "dragmode": "orbit",
        },
        height=920, margin={"l": 0, "r": 260, "b": 0, "t": 80}, legend={"itemsizing": "constant"},
    )
    plot_id = f"deployment_cloud_{query_index}"
    plot_html = figure.to_html(full_html=False, include_plotlyjs="inline", div_id=plot_id)
    output_path = report_dir / f"{query['name']}_cloud.html"
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>PI0 cloud {query['name']}</title>
<style>body{{font-family:system-ui,sans-serif;margin:18px}}#pair-status{{padding:10px;background:#fff3cd;border:1px solid #e0b85c;white-space:pre-wrap}}button{{margin:10px 0;padding:6px 10px}}pre{{max-height:260px;overflow:auto}}</style>
</head><body><h1>PI0 deployment cloud — {html.escape(query['name'])}</h1>
<div id="pair-status">Click any two visible point markers to measure their 3-D distance.</div>
<button id="clear-selection">Clear selected pair</button>
<p>Dense clouds are raw depth points after workspace ROI; blue diamonds are the final voxel/sample model cloud; tactile colors identify fingers. Coordinates use <code>base_link</code> in metres.</p>
{plot_html}
<details><summary>Cloud counts</summary><pre>{html.escape(json.dumps(summary, indent=2))}</pre></details>
<script>
(function() {{
  const gd = document.getElementById({json.dumps(plot_id)});
  const status = document.getElementById('pair-status');
  const selected = [];
  const pairTrace = {selected_pair_trace};
  const clear = () => {{
    selected.length = 0;
    status.textContent = 'Click any two visible point markers to measure their 3-D distance.';
    Plotly.restyle(gd, {{x: [[null, null, null]], y: [[null, null, null]], z: [[null, null, null]]}}, [pairTrace]);
  }};
  document.getElementById('clear-selection').addEventListener('click', clear);
  gd.on('plotly_click', function(event) {{
    if (!event.points || !event.points.length) return;
    const point = event.points[0];
    if (point.curveNumber === pairTrace || point.x === null || point.x === undefined) return;
    const label = (point.data.name || 'point') + ' / index=' + point.pointNumber;
    const picked = {{x: Number(point.x), y: Number(point.y), z: Number(point.z), label}};
    if (selected.length >= 2) selected.length = 0;
    selected.push(picked);
    if (selected.length === 1) {{
      status.textContent = 'First point selected: ' + picked.label + '. Click a second point.';
      return;
    }}
    const a = selected[0], b = selected[1];
    const dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
    const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
    status.textContent = 'distance = ' + (distance * 1000).toFixed(2) + ' mm | ' + a.label + ' ↔ ' + b.label;
    Plotly.restyle(gd, {{x: [[a.x, b.x, null]], y: [[a.y, b.y, null]], z: [[a.z, b.z, null]]}}, [pairTrace]);
  }});
}})();
</script></body></html>"""
    output_path.write_text(document, encoding="utf-8")
    return output_path


def render_query(
    report_dir: Path,
    query_index: int,
    query: dict,
    manifest: dict,
    control: dict,
    preprocessor,
    roles: tuple[str, ...],
    *,
    include_tactile: bool,
) -> Path:
    import matplotlib.pyplot as plt

    query_input = query["input"]
    query_output = query["output"]
    visual = None
    tactile = None
    spatial_error = None
    if preprocessor is not None and (roles or include_tactile):
        try:
            visual, tactile = reconstruct_spatial(
                preprocessor,
                query_input,
                roles,
                include_tactile=include_tactile,
            )
        except Exception as exc:
            spatial_error = f"{type(exc).__name__}: {exc}"

    figure, axes = plt.subplots(4, 3, figsize=(21, 24), constrained_layout=True)
    rgb_cameras = recorded_rgb_cameras([query])
    for column in range(3):
        camera = rgb_cameras[column] if column < len(rgb_cameras) else None
        image = query_input.get(f"rgb_{camera}") if camera is not None else None
        if image is not None:
            axes[0, column].imshow(image_for_display(image))
        else:
            axes[0, column].text(0.5, 0.5, "missing", ha="center", va="center")
        axes[0, column].set_title(f"RGB {camera}" if camera is not None else "RGB not recorded")
        axes[0, column].axis("off")

    for column in range(2):
        role = roles[column] if column < len(roles) else None
        if role is None:
            axes[1, column].text(0.5, 0.5, "Depth not recorded", ha="center", va="center")
            axes[1, column].axis("off")
        else:
            show_depth(axes[1, column], query_input[f"depth_cam_{role}"], f"depth {role}")
    if visual is not None:
        rgb = np.asarray(visual.rgb, dtype=np.float32) / 255.0
        scatter_spatial(axes[1, 2], visual.xyz_m, rgb, f"final visual cloud ({len(visual.xyz_m)})")
    else:
        message = (
            "Spatial RGB-D input not recorded for this policy"
            if not roles
            else f"spatial reconstruction unavailable\n{spatial_error or 'not recorded'}"
        )
        axes[1, 2].text(0.02, 0.5, message, va="center")
        axes[1, 2].axis("off")

    if include_tactile:
        raw_norm = np.asarray(query_input["tactile_raw_force_norm"], dtype=np.float32)
        tactile_image = axes[2, 0].imshow(raw_norm, aspect="auto", cmap="magma")
        axes[2, 0].set_yticks(range(raw_norm.shape[0]))
        axes[2, 0].set_yticklabels(manifest.get("tactile", {}).get("finger_names", range(raw_norm.shape[0])))
        axes[2, 0].set_xlabel("taxel id")
        axes[2, 0].set_title("Per-finger raw tactile force norm")
        figure.colorbar(tactile_image, ax=axes[2, 0], fraction=0.025)
    else:
        axes[2, 0].text(0.5, 0.5, "Tactile not recorded", ha="center", va="center")
        axes[2, 0].axis("off")

    if tactile is not None:
        tactile_color = np.asarray(tactile.force_norm, dtype=np.float32)
        scatter = axes[2, 1].scatter(
            tactile.xyz_m[:, 0],
            tactile.xyz_m[:, 1],
            c=tactile_color,
            s=7,
            cmap="magma",
        )
        axes[2, 1].set_aspect("equal", adjustable="box")
        axes[2, 1].set_title("Tactile spatial points, colored by force norm")
        axes[2, 1].set_xlabel("x (m)")
        axes[2, 1].set_ylabel("y (m)")
        figure.colorbar(scatter, ax=axes[2, 1], fraction=0.025)
    else:
        axes[2, 1].text(0.5, 0.5, "no reconstructed tactile points", ha="center", va="center")
        axes[2, 1].axis("off")

    metadata_ax = axes[2, 2]
    success = bool(scalar(query_output.get("success"), False)) if query_output else False
    error = str(scalar(query_output.get("error"), "missing output")) if query_output else "missing output"
    used_chunk_shape = tuple(query_output["action_chunk"].shape) if query_output else ()
    server_chunk_shape = tuple(query_output.get("server_action_chunk", np.empty((0, 0))).shape) if query_output else ()
    text = (
        f"query: {query_index}\n"
        f"frame: {int(scalar(query_input.get('frame_index'), -1))}\n"
        f"timestamp_s: {float(scalar(query_input.get('timestamp_s'), np.nan)):.3f}\n"
        f"success: {success}\n"
        "roundtrip_ms: "
        f"{float(scalar(query_output.get('client_roundtrip_ms'), np.nan)) if query_output else np.nan:.1f}\n"
        f"policy_ms: {float(scalar(query_output.get('policy_infer_ms'), np.nan)) if query_output else np.nan:.1f}\n"
        f"server_ms: {float(scalar(query_output.get('server_infer_ms'), np.nan)) if query_output else np.nan:.1f}\n"
        f"used chunk: {used_chunk_shape}\n"
        f"server chunk: {server_chunk_shape}\n"
        f"error: {error or '-'}"
    )
    metadata_ax.text(0.02, 0.98, text, va="top", family="monospace")
    metadata_ax.axis("off")

    action_names = list(manifest.get("action_names", []))
    action_chunk = (
        np.asarray(query_output["action_chunk"], dtype=np.float32)
        if query_output is not None and "action_chunk" in query_output
        else np.empty((0, 0), dtype=np.float32)
    )
    chunk_time = np.arange(action_chunk.shape[0])
    arm_idx = named_indices(action_names, "arm_joint_")[:6]
    hand_idx = named_indices(action_names, "hand_joint_")[:12]
    plot_channels(
        axes[3, 0],
        chunk_time,
        action_chunk[:, arm_idx] if action_chunk.size and arm_idx else np.empty((0, 0)),
        [action_names[index] for index in arm_idx],
        "Predicted arm action chunk",
    )
    axes[3, 0].set_xlabel("chunk step")
    plot_channels(
        axes[3, 1],
        chunk_time,
        action_chunk[:, hand_idx] if action_chunk.size and hand_idx else np.empty((0, 0)),
        [action_names[index] for index in hand_idx],
        "Predicted hand action chunk",
    )
    axes[3, 1].set_xlabel("chunk step")

    context_ax = axes[3, 2]
    if control:
        target_query_id = int(scalar(query_input.get("query_id"), query_index))
        mask = np.asarray(control["query_id"]) == target_query_id
        if np.any(mask):
            indices = np.flatnonzero(mask)
            start = max(0, int(indices[0]) - 15)
            end = min(len(control["elapsed_s"]), int(indices[-1]) + 16)
            context_ax.plot(control["elapsed_s"][start:end], control["loop_ms"][start:end], label="loop ms")
            context_ax.axvspan(
                control["elapsed_s"][indices[0]], control["elapsed_s"][indices[-1]], color="#ff7f0e", alpha=0.2
            )
    context_ax.set_title("Control timing around this chunk")
    context_ax.set_xlabel("time (s)")
    context_ax.set_ylabel("ms")
    context_ax.grid(alpha=0.25)
    figure.suptitle(f"PI0 query diagnostic: {query['name']}", fontsize=18)
    output_path = report_dir / f"{query['name']}.png"
    figure.savefig(output_path, dpi=120)
    plt.close(figure)
    return output_path


def render_video(image_paths: list[Path], output_path: Path, fps: float) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for --video") from exc
    if not image_paths:
        raise ValueError("No query images are available for video rendering")
    first = cv2.imread(str(image_paths[0]))
    if first is None:
        raise ValueError(f"Cannot read {image_paths[0]}")
    target_width = 1600
    target_height = int(round(first.shape[0] * target_width / first.shape[1]))
    if target_height % 2:
        target_height += 1
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (target_width, target_height),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create the MP4 writer")
    try:
        repeat = max(1, int(round(fps)))
        for path in image_paths:
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            for _ in range(repeat):
                writer.write(frame)
    finally:
        writer.release()


def main() -> int:
    args = parse_args()
    if args.max_cloud_keyframes < 0:
        raise ValueError("--max-cloud-keyframes must be non-negative")
    if args.max_cloud_points < 0:
        raise ValueError("--max-cloud-points must be non-negative")
    run_dir = args.run_dir.expanduser().resolve()
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError as exc:
        raise ImportError("Diagnostic rendering requires matplotlib") from exc

    manifest = load_json(manifest_path)
    control = load_control(run_dir)
    queries = load_queries(run_dir)
    report_dir = run_dir / "report"
    report_dir.mkdir(exist_ok=True)
    roles = recorded_depth_roles(manifest, queries)
    include_tactile = has_recorded_tactile(queries)

    preprocessor = None
    if roles or include_tactile:
        try:
            if roles and any(set(roles) - set(query_depth_roles(query["input"])) for query in queries):
                raise ValueError("Depth camera roles are inconsistent across recorded queries")
            profile = args.front_calibration_profile
            if profile is None:
                profile = manifest.get("args", {}).get("server_metadata", {}).get("front_calibration_profile")
            if profile is not None:
                profile = Path(profile)
                if not profile.is_absolute():
                    profile = args.repo_root / profile
            preprocessor = make_preprocessor(args.repo_root, roles, profile)
        except Exception as exc:
            print(f"Warning: spatial reconstruction disabled: {type(exc).__name__}: {exc}", file=sys.stderr)

    summary_path = render_summary(
        run_dir,
        report_dir,
        manifest,
        control,
        queries,
        max_keyframes=args.max_summary_keyframes,
    )
    print(f"Wrote {summary_path}")

    query_images = []
    if not args.skip_query_images:
        for query_index, query in enumerate(queries):
            path = render_query(
                report_dir,
                query_index,
                query,
                manifest,
                control,
                preprocessor,
                roles,
                include_tactile=include_tactile,
            )
            query_images.append(path)
            print(f"Wrote {path}")

    if roles and not args.no_cloud_html:
        if preprocessor is None:
            print("Skipping interactive cloud output because spatial reconstruction is unavailable.", file=sys.stderr)
        else:
            cloud_paths = []
            cloud_indices = (
                list(range(len(queries)))
                if args.max_cloud_keyframes == 0
                else selected_indices(len(queries), args.max_cloud_keyframes)
            )
            for query_index in cloud_indices:
                try:
                    path = render_cloud_html(
                        report_dir,
                        query_index,
                        queries[query_index],
                        preprocessor,
                        roles,
                        include_tactile=include_tactile,
                        max_dense_points=args.max_cloud_points,
                    )
                except Exception as exc:
                    print(
                        f"Warning: cloud export failed for query {query_index}: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    continue
                cloud_paths.append(path)
                print(f"Wrote {path}")
            if cloud_paths:
                index_path = report_dir / "cloud_index.html"
                links = "\n".join(
                    f'<li><a href="{html.escape(path.name)}">{html.escape(path.name)}</a></li>' for path in cloud_paths
                )
                index_path.write_text(
                    "<!doctype html><meta charset=\"utf-8\"><title>PI0 cloud reports</title>"
                    "<h1>PI0 interactive cloud reports</h1><p>Open a frame page, orbit/zoom the cloud, "
                    "and click any two points to measure their distance.</p><ul>" + links + "</ul>",
                    encoding="utf-8",
                )
                print(f"Wrote {index_path}")

    if args.video:
        video_path = report_dir / "diagnostic.mp4"
        render_video(query_images, video_path, args.video_fps)
        print(f"Wrote {video_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
