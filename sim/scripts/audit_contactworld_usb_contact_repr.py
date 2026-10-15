#!/usr/bin/env python3
"""Render a human review page for the contact-centred USB representation.

The page deliberately shows both retained and excluded local scene points.  It
is a review tool for the tactile-facing half-space filter, not evidence of a
full robot/object segmentation.
"""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from openpi.spatial_dataset.contactworld_contact_repr import (  # noqa: E402
    BACKSIDE_TOLERANCE_M, RELATION_RADIUS_M, SELF_SURFACE_MARGIN_M, build_frame,
)


def png_base64(figure) -> str:
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(figure)
    return base64.b64encode(buffer.getvalue()).decode()


def frame_from_video(dataset: Path, role: str, episode: int, frame: int) -> np.ndarray:
    path = (dataset / ("videos/chunk-%03d" % (episode // 1000)) /
            f"observation.images.{role}" / f"episode_{episode:06d}.mp4")
    reader = cv2.VideoCapture(str(path))
    reader.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, image = reader.read()
    reader.release()
    if not ok:
        return np.zeros((256, 256, 3), np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def force_canvas(local: np.ndarray, selected: np.ndarray, centers_uv: np.ndarray, anchors: np.ndarray) -> str:
    scale = max(float(np.abs(local).max()), 1e-6)
    figure, axes = plt.subplots(2, 3, figsize=(9, 5.7), constrained_layout=True)
    for finger in range(2):
        for axis in range(3):
            ax = axes[finger, axis]
            ax.imshow(local[finger, :, :, axis], cmap="coolwarm", vmin=-scale, vmax=scale, origin="upper")
            for anchor in range(2):
                if not anchors[finger, anchor]:
                    continue
                picks = selected[finger, anchor]
                # Convert normalized [u, v] back to grid coordinates.
                col = (picks[:, 0] + 1) * 0.5 * 13
                row = (picks[:, 1] + 1) * 0.5 * 9
                ax.scatter(col, row, s=8, facecolors="none", edgecolors="lime", linewidths=.45)
                center = centers_uv[finger, anchor]
                ax.scatter((center[0] + 1) * .5 * 13, (center[1] + 1) * .5 * 9,
                           s=32, marker="x", color="black", linewidths=1.2)
            ax.set_title(f"{'L' if finger == 0 else 'R'} F{'xyz'[axis]}")
            ax.set_xticks([]); ax.set_yticks([])
    figure.suptitle("Raw local TacFF; green=retained taxels, black ×=contact-component centre", fontsize=11)
    return png_base64(figure)


def relation_canvas(cloud: np.ndarray, taxel_xyz: np.ndarray, derived: dict[str, np.ndarray]) -> str:
    pad = taxel_xyz.mean(axis=1)
    query = derived["query_xyz_base"]
    # Display nearby points rejected by the half-space filter as grey context.
    nearby = []
    retained = []
    for finger in range(2):
        distance = np.linalg.norm(cloud - query[finger], axis=-1)
        context = cloud[distance <= RELATION_RADIUS_M]
        nearby.append(context)
        retained.append(derived["candidate_xyz_base"][finger][derived["candidate_mask"][finger]])
    figure = plt.figure(figsize=(8.8, 6.2), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    context = np.concatenate(nearby) if any(len(row) for row in nearby) else np.zeros((0, 3))
    if len(context):
        axis.scatter(context[:, 0], context[:, 1], context[:, 2], s=2, c="#a8afb9", alpha=.20,
                     label="nearby cloud before tactile-facing filter")
    colours = ("#46d3ff", "#ffbf3f")
    for finger in range(2):
        axis.scatter(taxel_xyz[finger, :, 0], taxel_xyz[finger, :, 1], taxel_xyz[finger, :, 2],
                     s=5, c=colours[finger], alpha=.7, label=f"{'left' if finger == 0 else 'right'} pad taxels")
        axis.scatter(query[finger, 0], query[finger, 1], query[finger, 2], s=65, marker="x", c=colours[finger])
        if len(retained[finger]):
            axis.scatter(retained[finger][:, 0], retained[finger][:, 1], retained[finger][:, 2],
                         s=14, c=colours[finger], edgecolors="black", linewidths=.2,
                         label=f"{'left' if finger == 0 else 'right'} retained candidates")
        inward = pad[1 - finger] - pad[finger]
        inward /= max(np.linalg.norm(inward), 1e-8)
        axis.quiver(query[finger, 0], query[finger, 1], query[finger, 2],
                    inward[0], inward[1], inward[2], length=.05, color=colours[finger])
    axis.set_xlabel("base X (m)"); axis.set_ylabel("base Y (m)"); axis.set_zlabel("base Z (m)")
    axis.set_title("Local cloud relation candidates; arrows point inward across the two pads")
    axis.legend(loc="upper left", fontsize=7)
    return png_base64(figure)


def chosen_frames(force: np.ndarray, length: int) -> list[int]:
    magnitude = np.linalg.norm(force, axis=-1).sum(axis=(1, 2))
    values = {0, length // 2, length - 1, int(np.argmax(magnitude)), int(np.argmin(magnitude))}
    return sorted(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "data/contactworld_usb_v3_bilateral")
    parser.add_argument("--repr-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--max-episodes", type=int, default=5)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    repr_root = (args.repr_root or dataset / "spatial/contact_repr_v1").resolve()
    manifest = json.loads((dataset / "spatial/manifest.json").read_text())
    all_rows = manifest["episodes"]
    if args.episodes:
        wanted = set(args.episodes)
        rows = [row for row in all_rows if int(row["episode_index"]) in wanted]
    else:
        indices = np.rint(np.linspace(0, len(all_rows) - 1, args.max_episodes)).astype(int)
        rows = [all_rows[index] for index in sorted(set(indices))]
    args.output.mkdir(parents=True, exist_ok=True)
    cards = []
    for row in rows:
        episode = int(row["episode_index"])
        src = dataset / "spatial/episodes" / f"episode_{episode:06d}"
        cloud = np.load(src / "pointcloud_fused_xyz.npy", mmap_mode="r")
        xyz = np.stack((np.load(src / "tactile_xyz_left_base.npy", mmap_mode="r"),
                        np.load(src / "tactile_xyz_right_base.npy", mmap_mode="r")), axis=1)
        force = np.stack((np.load(src / "tactile_force_left_base.npy", mmap_mode="r"),
                          np.load(src / "tactile_force_right_base.npy", mmap_mode="r")), axis=1)
        local = np.stack((np.load(src / "force_grid_left_local.npy", mmap_mode="r"),
                          np.load(src / "force_grid_right_local.npy", mmap_mode="r")), axis=1)
        with np.load(repr_root / "episodes" / f"episode_{episode:06d}.npz") as saved:
            derived = {key: saved[key] for key in saved.files}
        for frame in chosen_frames(force, len(force)):
            # Recompute once and compare enough fields to catch a stale sidecar.
            recomputed = build_frame(cloud[frame], xyz[frame], force[frame])
            if not np.array_equal(recomputed["candidate_mask"], derived["candidate_mask"][frame]):
                raise ValueError("Contact representation sidecar does not match source frame")
            front = frame_from_video(dataset, "front", episode, frame)
            wrist = frame_from_video(dataset, "wrist", episode, frame)
            image = np.concatenate((front, wrist), axis=1)
            ok, packed = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                                      [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                raise RuntimeError("JPEG encoding failed")
            selected = derived["patch_xyz_local"][frame] + derived["anchor_uv"][frame, :, :, None, :]
            local_force = local[frame]
            force_plot = force_canvas(local_force, selected, derived["anchor_uv"][frame],
                                      derived["anchor_mask"][frame])
            relation_plot = relation_canvas(cloud[frame], xyz[frame],
                                            {key: derived[key][frame] for key in (
                                                "query_xyz_base", "candidate_xyz_base", "candidate_mask")})
            counts = derived["candidate_mask"][frame].sum(axis=-1)
            valid = derived["finger_valid"][frame].tolist()
            cards.append(f'''<details class="case"><summary>episode {episode} · frame {frame} · left/right candidate count {counts.tolist()} · contact valid {valid}</summary>
<div class="grid"><section><h3>Front / wrist RGB</h3><img src="data:image/jpeg;base64,{base64.b64encode(packed.tobytes()).decode()}"></section>
<section><h3>Contact-centred sparse tactile</h3><img src="data:image/png;base64,{force_plot}"></section>
<section class="wide"><h3>Tactile-to-scene candidates</h3><img src="data:image/png;base64,{relation_plot}"></section></div>
<p>Grey points are every cloud point within {float(RELATION_RADIUS_M):.3f} m of a tactile query. Coloured points pass the tactile-facing half-space and self-surface-margin filters. This rejects clear pad-backside geometry; it does not claim a full robot or object segmentation.</p></details>''')
    page = f'''<!doctype html><meta charset="utf-8"><title>ContactWorld USB contact representation audit</title>
<style>body{{margin:0 auto;max-width:1500px;padding:24px;background:#101722;color:#e8effa;font:16px system-ui;line-height:1.5}}h1{{margin-top:0}}details{{border:1px solid #40536d;border-radius:8px;margin:16px 0;background:#172334}}summary{{padding:12px;cursor:pointer;font-weight:650}}.case>div{{padding:0 12px 12px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}.wide{{grid-column:1/-1}}img{{max-width:100%;display:block;background:#081018}}p{{color:#bccce1}}.notice{{border-left:4px solid #67b7ff;background:#182a40;padding:12px}}</style>
<h1>USB V3: contact-centred tactile and local tactile–scene relation input audit</h1>
<p class="notice">This page must be reviewed before training 16–19. It uses the same deterministic sidecar that training and online policy evaluation read. The contact centre is a finger-pad force centre, not a USB–socket collision point. The relation filter is tactile-facing local geometry, not simulator object identity or a full robot segmentation.</p>
<p>Each finger has up to two force-connected patches × 16 retained taxels; the model receives the two patch tokens plus one full-pad summary token. Relation models receive three learned relation tokens per finger over the coloured candidates.</p>
{''.join(cards)}</html>'''
    (args.output / "index.html").write_text(page)
    (args.output / "audit.json").write_text(json.dumps({"episodes": [int(r["episode_index"]) for r in rows],
        "contact_repr_root": str(repr_root), "status": "passed"}, indent=2) + "\n")
    print(str(args.output / "index.html"))


if __name__ == "__main__":
    main()
