#!/usr/bin/env python3
"""Generate compact automated Level-1 review artifacts without modifying HOI."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import json
import math

import numpy as np


GROUPS = ((0, 15), (16, 30), (31, 45), (46, 53), (54, 61), (62, 69))


def episodes(spec: str) -> list[int]:
    result = []
    for token in spec.split(","):
        if "-" in token:
            start, stop = map(int, token.split("-", 1)); result.extend(range(start, stop + 1))
        else:
            result.append(int(token))
    return list(dict.fromkeys(result))


def _stat(points: np.ndarray) -> dict:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    finite = np.isfinite(points).all(axis=1)
    valid = points[finite]
    if not len(valid):
        return {"count": 0, "finite_count": int(finite.sum()), "nan_inf_count": int((~finite).sum()), "centroid": None, "extent": None}
    return {
        "count": int(len(points)),
        "finite_count": int(len(valid)),
        "nan_inf_count": int((~finite).sum()),
        "centroid": valid.mean(axis=0).round(6).tolist(),
        "extent": (valid.max(axis=0) - valid.min(axis=0)).round(6).tolist(),
    }


def _contact_sheet(hoi_root: Path, output: Path, episode_ids: list[int]) -> None:
    import cv2
    tiles = []
    for episode in episode_ids:
        path = hoi_root / f"level1_sequence_e{episode}" / "mask_preview_frame0.jpg"
        if path.is_file():
            image = cv2.imread(str(path))
            image = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
        else:
            image = np.full((240, 320, 3), 235, dtype=np.uint8)
            cv2.putText(image, "MISSING PREVIEW", (28, 120), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 180), 2)
        cv2.rectangle(image, (0, 0), (319, 30), (25, 25, 25), -1)
        cv2.putText(image, f"e{episode:02d}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
        tiles.append(image)
    cols = 5
    rows = max(1, math.ceil(len(tiles) / cols))
    canvas = np.full((rows * 240, cols * 320, 3), 235, dtype=np.uint8)
    for index, image in enumerate(tiles):
        row, col = divmod(index, cols)
        canvas[row * 240 : (row + 1) * 240, col * 320 : (col + 1) * 320] = image
    cv2.imwrite(str(output / "all70_frame0_contact_sheet.jpg"), canvas)


def _tracking_sheets(hoi_root: Path, output: Path, episode_ids: list[int]) -> None:
    import cv2
    for group_index, (start, stop) in enumerate(GROUPS, 1):
        tiles = []
        for episode in [x for x in episode_ids if start <= x <= stop]:
            path = hoi_root / f"level1_sequence_e{episode}" / "tracking_contact_sheet.jpg"
            if path.is_file():
                image = cv2.imread(str(path)); image = cv2.resize(image, (480, 360), interpolation=cv2.INTER_AREA)
            else:
                image = np.full((360, 480, 3), 235, dtype=np.uint8)
                cv2.putText(image, "TRACKING NOT AVAILABLE", (45, 180), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 180), 2)
            cv2.rectangle(image, (0, 0), (479, 30), (25, 25, 25), -1)
            cv2.putText(image, f"e{episode:02d}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
            tiles.append(image)
        cols = 2
        rows = max(1, math.ceil(len(tiles) / cols))
        canvas = np.full((rows * 360, cols * 480, 3), 235, dtype=np.uint8)
        for index, image in enumerate(tiles):
            row, col = divmod(index, cols)
            canvas[row * 360 : (row + 1) * 360, col * 480 : (col + 1) * 480] = image
        cv2.imwrite(str(output / f"group{group_index}_tracking_contact_sheet.jpg"), canvas)


def _geometry_rows(hoi_root: Path, episode: int) -> tuple[list[dict], list[str]]:
    root = hoi_root / f"level1_sequence_e{episode}"
    cloud_root, memory_root = root / "cloud_frames", root / "abc_v1" / "frames"
    cloud_paths = sorted(cloud_root.glob("*.npz"))
    memory_paths = {p.name: p for p in memory_root.glob("*.npz")}
    rows, flags = [], []
    previous = {}
    for path in cloud_paths:
        try:
            with np.load(path, allow_pickle=False) as cloud:
                frame = int(np.asarray(cloud["frame_index"]).item())
                row = {"episode_index": episode, "frame_index": frame, "episode_status": "geometry_available"}
                for object_name, current_key in (("manipulated", "funnel_xyz_base_m"), ("target", "target_flask_xyz_base_m")):
                    current = np.asarray(cloud[current_key], dtype=np.float32)
                    row[f"{object_name}_current_count"] = len(current)
                    stats = _stat(current)
                    row[f"{object_name}_centroid"] = stats["centroid"]
                    row[f"{object_name}_extent"] = stats["extent"]
                    row[f"{object_name}_nan_inf_count"] = stats["nan_inf_count"]
                    if stats["nan_inf_count"]:
                        flags.append(f"e{episode}f{frame}:{object_name}:nan_inf")
                    if stats["centroid"] is not None and previous.get(object_name) is not None:
                        jump = float(np.linalg.norm(np.asarray(stats["centroid"]) - previous[object_name]))
                        row[f"{object_name}_centroid_jump_m"] = round(jump, 6)
                        if jump > 0.25:
                            flags.append(f"e{episode}f{frame}:{object_name}:centroid_jump_{jump:.3f}m")
                    previous[object_name] = np.asarray(stats["centroid"]) if stats["centroid"] is not None else None
                sequence = root / "sequence_report.json"
                if sequence.is_file():
                    row["valid_depth_ratio"] = None
                else:
                    row["valid_depth_ratio"] = None
            memory = memory_paths.get(path.name)
            if memory is None:
                row["episode_status"] = "missing_history"
                flags.append(f"e{episode}f{frame}:missing_history")
            else:
                with np.load(memory, allow_pickle=False) as history:
                    row["manipulated_history_count"] = int(len(history["B_funnel_supported_history_xyz_base_m"]))
                    row["target_history_count"] = int(len(history["B_flask_static_history_xyz_base_m"]))
                    row["manipulated_source_ratio"] = 0.0 if row["manipulated_current_count"] + row["manipulated_history_count"] == 0 else round(row["manipulated_history_count"] / (row["manipulated_current_count"] + row["manipulated_history_count"]), 6)
                    row["target_source_ratio"] = 0.0 if row["target_current_count"] + row["target_history_count"] == 0 else round(row["target_history_count"] / (row["target_current_count"] + row["target_history_count"]), 6)
                    row["age_statistics"] = {"current_age": 0, "history_age": 1}
                    row["confidence_statistics"] = {"current": 1.0, "history": "support-derived in future sidecar export"}
            rows.append(row)
        except Exception as exc:
            flags.append(f"e{episode}:{path.name}:read_error:{type(exc).__name__}")
    if not cloud_paths:
        flags.append(f"e{episode}:no_geometry")
    return rows, flags


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hoi-output-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episodes", default="0-69")
    args = parser.parse_args()
    episode_ids = episodes(args.episodes)
    args.output_root.mkdir(parents=True, exist_ok=True)
    _contact_sheet(args.hoi_output_root, args.output_root, episode_ids)
    _tracking_sheets(args.hoi_output_root, args.output_root, episode_ids)
    all_rows, flags = [], []
    for episode in episode_ids:
        rows, episode_flags = _geometry_rows(args.hoi_output_root, episode)
        all_rows.extend(rows); flags.extend(episode_flags)
    statuses = {
        "episodes_requested": episode_ids,
        "episodes_with_tracking": [e for e in episode_ids if (args.hoi_output_root / f"level1_sequence_e{e}" / "masks" / "masks_report.json").is_file()],
        "episodes_with_review": [e for e in episode_ids if (args.hoi_output_root / f"level1_sequence_e{e}" / "tracking_reviewed.json").is_file()],
        "episodes_with_preview": [e for e in episode_ids if (args.hoi_output_root / f"level1_sequence_e{e}" / "mask_preview_frame0.jpg").is_file()],
        "flags": flags,
        "status": "AUTO_QA_INCOMPLETE_HUMAN_REVIEW_PENDING" if len(flags) or len(episode_ids) != len([e for e in episode_ids if (args.hoi_output_root / f"level1_sequence_e{e}" / "tracking_reviewed.json").is_file()]) else "AUTO_QA_PASS_HUMAN_REVIEW_PENDING",
        "human_approval": False,
    }
    (args.output_root / "level1_auto_qa.json").write_text(json.dumps({"summary": statuses, "frames": all_rows}, indent=2), encoding="utf8")
    (args.output_root / "level1_auto_qa.md").write_text(
        "# Level-1 automated QA\n\n"
        f"Status: `{statuses['status']}`\n\n"
        f"Preview episodes: {len(statuses['episodes_with_preview'])}/{len(episode_ids)}\n\n"
        f"Full tracking episodes: {len(statuses['episodes_with_tracking'])}/{len(episode_ids)}\n\n"
        f"Explicit tracking review records: {len(statuses['episodes_with_review'])}/{len(episode_ids)}\n\n"
        "Human approval: **pending**\n\n"
        "## Flags\n\n" + "\n".join(f"- `{flag}`" for flag in flags[:500]) + "\n",
        encoding="utf8",
    )
    with (args.output_root / "flagged_frames.csv").open("w", newline="", encoding="utf8") as handle:
        writer = csv.writer(handle); writer.writerow(["flag"]); writer.writerows([[flag] for flag in flags])
    print(json.dumps(statuses, indent=2))


if __name__ == "__main__":
    main()
