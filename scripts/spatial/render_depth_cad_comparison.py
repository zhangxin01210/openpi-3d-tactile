#!/usr/bin/env python3
"""Make pixel-aligned baseline/corrected comparisons from an existing depth-CAD audit."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import cv2
import numpy as np


def split_sheet(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sheet = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if sheet is None or sheet.ndim != 3 or sheet.shape[0] % 2 or sheet.shape[1] % 2:
        raise ValueError(f"Invalid depth-CAD PNG: {path}")
    h, w = sheet.shape[0] // 2 - 32, sheet.shape[1] // 2
    if h <= 0:
        raise ValueError(f"Invalid depth-CAD panel size: {path}")
    raw = sheet[32:32 + h, :w].copy()
    predicted = sheet[32:32 + h, w:2 * w]
    residual = sheet[h + 64:2 * (h + 32), :w].copy()
    return raw, np.any(predicted != 0, axis=2).astype(np.uint8), residual


def draw_contours(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = image.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2, cv2.LINE_AA)
    return out


def label(image: np.ndarray, title: str) -> np.ndarray:
    h, w = image.shape[:2]
    out = np.zeros((h + 36, w, 3), np.uint8)
    out[36:] = image
    cv2.putText(out, title, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.57,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def make_comparison(baseline: tuple, corrected: tuple, baseline_stats: dict, corrected_stats: dict) -> np.ndarray:
    raw_b, mask_b, residual_b = baseline
    raw_c, mask_c, residual_c = corrected
    if raw_b.shape != raw_c.shape or mask_b.shape != mask_c.shape:
        raise ValueError("Baseline/corrected PNGs must have the same pixel grid")
    # The same raw depth should have been used for both modes within one audit.
    if not np.array_equal(raw_b, raw_c):
        raise ValueError("Baseline/corrected raw-depth panels differ; cannot make an aligned comparison")
    both = draw_contours(draw_contours(raw_b, mask_b, (0, 0, 255)), mask_c, (0, 255, 0))
    footprint = np.zeros_like(raw_b)
    footprint[(mask_b != 0) & (mask_c == 0)] = (0, 0, 255)
    footprint[(mask_b == 0) & (mask_c != 0)] = (0, 255, 0)
    footprint[(mask_b != 0) & (mask_c != 0)] = (0, 210, 255)

    def metric(stats):
        value = stats["near_surface"]["signed_median_mm"]
        count = stats["near_surface"]["count"]
        return f"median {value:+.1f} mm / n={count}" if value is not None else "no near-surface pixels"

    panels = (
        label(both, "Raw depth + CAD: baseline red / corrected green"),
        label(draw_contours(raw_b, mask_b, (0, 0, 255)), "Baseline CAD on raw depth"),
        label(draw_contours(raw_c, mask_c, (0, 255, 0)), "Corrected CAD on raw depth"),
        label(residual_b, f"Baseline residual: {metric(baseline_stats)}"),
        label(residual_c, f"Corrected residual: {metric(corrected_stats)}"),
        label(footprint, "CAD footprint: red only / green only / yellow both"),
    )
    return np.vstack((np.hstack(panels[:3]), np.hstack(panels[3:])))


def render(input_dir: Path, output_dir: Path) -> None:
    report = json.loads((input_dir / "report.json").read_text(encoding="utf-8"))
    stats = {(entry["mode"], int(entry["frame"])): entry for entry in report["results"]}
    frames = [int(frame) for frame in report["frames"]]
    for frame in frames:
        if ("baseline", frame) not in stats or ("corrected", frame) not in stats:
            raise ValueError(f"Frame {frame} has no baseline/corrected pair")
        for mode in ("baseline", "corrected"):
            if not (input_dir / mode / f"frame_{frame:06d}_depth_cad.png").is_file():
                raise FileNotFoundError(input_dir / mode / f"frame_{frame:06d}_depth_cad.png")
    output_dir.mkdir(parents=True, exist_ok=False)
    links = []
    for frame in frames:
        base = split_sheet(input_dir / "baseline" / f"frame_{frame:06d}_depth_cad.png")
        corr = split_sheet(input_dir / "corrected" / f"frame_{frame:06d}_depth_cad.png")
        image = make_comparison(base, corr, stats["baseline", frame], stats["corrected", frame])
        filename = f"frame_{frame:06d}_comparison.png"
        if not cv2.imwrite(str(output_dir / filename), image):
            raise RuntimeError(f"Could not write {output_dir / filename}")
        paired = report.get("paired_common_pixels", {}).get(str(frame))
        note = ""
        if paired and paired["common_pixels"]:
            b = paired["baseline"]["abs_median_mm"]
            c = paired["corrected"]["abs_median_mm"]
            note = (f"<p>Same-pixel absolute median: {b:.1f} mm baseline, "
                    f"{c:.1f} mm corrected; {paired['common_pixels']} pixels.</p>")
        links.append(f"<h2>Frame {frame}</h2>{note}<a href='{filename}'><img src='{filename}'></a>")
        print(output_dir / filename)
    page = ("<!doctype html><meta charset='utf-8'><title>Depth CAD comparison</title>"
            "<style>body{font:15px system-ui;margin:24px}img{max-width:100%}</style>"
            f"<h1>Depth CAD comparison</h1><p>{html.escape(str(report['dataset']))}</p>"
            "<p>Same raw-depth pixel grid. Red: baseline; green: corrected. "
            "Residual panels retain the original audit's color scale; their masks can differ.</p>"
            + "\n".join(links))
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    print(f"Open {output_dir / 'index.html'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Directory containing report.json and mode PNGs")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    render(args.input.expanduser().resolve(), args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
