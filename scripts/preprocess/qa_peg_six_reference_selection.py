#!/usr/bin/env python3
"""Offline, read-only probe: select one of six historical Peg position references.

This script does not change masks, prompts, SAM2 output, or deployment code.
The reference groups are provisional, as recorded by the original six-position
operator guide. Matching is a baseline heuristic, NOT validated localization.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

REFERENCES = (0, 23, 38, 49, 58, 65)
GROUPS = ((0, 15, 0), (16, 30, 23), (31, 45, 38),
          (46, 53, 49), (54, 61, 58), (62, 69, 65))


def expected_reference(ep: int) -> int:
    for a, b, ref in GROUPS:
        if a <= ep <= b:
            return ref
    raise ValueError(ep)


def first_image(hoi: Path, dataset: Path, ep: int) -> tuple[np.ndarray | None, str | None]:
    paths = (
        hoi / f"data/level1_sequence_e{ep}/frames/00000.jpg",
        hoi / f"data/level1_sequence_e{ep}/frames/000000.jpg",
    )
    for p in paths:
        if p.is_file():
            im = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if im is not None:
                return im, str(p)

    # LeRobot v2 video fallback; reads only first video frame.
    candidates = sorted((dataset / "videos").glob(
        f"**/observation.images.cam_front/episode_{ep:06d}.mp4"
    ))
    for p in candidates:
        cap = cv2.VideoCapture(str(p))
        ok, image = cap.read()
        cap.release()
        if ok and image is not None:
            return image, str(p)
    return None, None


def extract_features(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = cv2.resize(bgr, (320, 240), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray = cv2.GaussianBlur(gray, (9, 9), 2.0)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.minimum(cv2.magnitude(sx, sy), 1.0)
    # Lower ROI emphasizes table / targets and largely ignores upper robot arm.
    return gray, edge, gray[85:230, 30:300], edge[85:230, 30:300]


def pair_score(a, b, mode: str) -> float:
    if mode == "whole":
        g1, e1 = a[0], a[1]
        g2, e2 = b[0], b[1]
    elif mode == "table_roi":
        g1, e1 = a[2], a[3]
        g2, e2 = b[2], b[3]
    else:
        raise ValueError(mode)
    return float(0.65 * np.mean(np.abs(g1 - g2)) +
                 0.35 * np.mean(np.abs(e1 - e2)))


def make_sheet(items: list[dict], pictures: dict[int, np.ndarray], out: Path,
               *, mode: str, columns: int = 7) -> None:
    w, h, caption_h = 180, 135, 38
    rows = (len(items) + columns - 1) // columns
    canvas = np.full((rows * (h + caption_h), columns * w, 3), 246, np.uint8)
    for j, row in enumerate(items):
        y, x = divmod(j, columns)
        x0, y0 = x * w, y * (h + caption_h)
        image = pictures.get(row["episode"])
        if image is None:
            cv2.putText(canvas, f'e{row["episode"]} MISSING', (x0 + 4, y0 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 160), 1)
            continue
        resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        canvas[y0:y0+h, x0:x0+w] = resized
        good = row[f"{mode}_selected"] == row["expected_ref"]
        border = (0, 170, 0) if good else (0, 0, 225)
        cv2.rectangle(canvas, (x0, y0), (x0+w-1, y0+h-1), border, 3)
        cv2.putText(canvas, f"e{row['episode']:02d} expected={row['expected_ref']}",
                    (x0 + 4, y0 + h + 14), cv2.FONT_HERSHEY_SIMPLEX, .4,
                    (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"pick={row[f'{mode}_selected']} gap={row[f'{mode}_margin']:.3f}",
                    (x0 + 4, y0 + h + 30), cv2.FONT_HERSHEY_SIMPLEX, .4,
                    (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), canvas, [cv2.IMWRITE_JPEG_QUALITY, 91])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hoi', type=Path,
                        default=Path('/workspace/mnt/sqzhang26/hoi'))
    parser.add_argument('--dataset', type=Path, default=Path('data/1018_peg_all'))
    parser.add_argument('--output', type=Path,
                        default=Path('output/preprocess/1018_peg_all/six_reference_selection_probe'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    pictures: dict[int, np.ndarray] = {}
    paths = {}
    for ep in range(70):
        image, path = first_image(args.hoi, args.dataset, ep)
        if image is not None:
            if image.shape[:2] != (480, 640):
                raise ValueError(f'e{ep}: expected RGB (480,640), found {image.shape}')
            pictures[ep] = image
            paths[str(ep)] = path
        else:
            print(f'[MISSING] e{ep:02d}: no prepared HOI frame or LeRobot front video')
    missing_refs = [ep for ep in REFERENCES if ep not in pictures]
    if missing_refs:
        raise RuntimeError(f'Cannot evaluate missing reference frames: {missing_refs}')
    if not pictures:
        raise RuntimeError('No images available')
    features = {ep: extract_features(image) for ep, image in pictures.items()}
    modes = ('whole', 'table_roi')
    rows = []
    for ep in range(70):
        record = {'episode': ep, 'expected_ref': expected_reference(ep),
                  'has_image': ep in pictures}
        for mode in modes:
            if ep not in pictures:
                record.update({f'{mode}_selected': None, f'{mode}_best': None,
                               f'{mode}_second': None, f'{mode}_margin': None})
                continue
            ranks = sorted(((pair_score(features[ep], features[r], mode), r)
                            for r in REFERENCES), key=lambda z: z[0])
            best, chosen = ranks[0]
            second = ranks[1][0]
            record.update({f'{mode}_selected': chosen,
                           f'{mode}_best': round(best, 6),
                           f'{mode}_second': round(second, 6),
                           f'{mode}_margin': round((second - best) / max(second, 1e-8), 6)})
        rows.append(record)

    with (args.output / 'selection.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    results = {}
    covered = [r for r in rows if r['has_image']]
    nonrefs = [r for r in covered if r['episode'] not in REFERENCES]
    for mode in modes:
        for subset_name, subset in [('all', covered), ('non_reference', nonrefs)]:
            right = sum(r[f'{mode}_selected'] == r['expected_ref'] for r in subset)
            acc = right / len(subset) if subset else None
            results[f'{mode}_{subset_name}'] = {
                'correct': right, 'total': len(subset), 'accuracy': acc,
            }
            print(f'[RESULT] {mode} {subset_name}: {right}/{len(subset)} = {acc}')
        make_sheet(rows, pictures, args.output / f'contact_sheet_{mode}.jpg', mode=mode)
    summary = {
        'note': 'PRELIMINARY: historic six-group boundaries are provisional; this is only RGB nearest-reference probing, not SAM2/mask accuracy.',
        'expected_groups': [{'begin': a, 'end': b, 'reference_episode': r} for a, b, r in GROUPS],
        'references': list(REFERENCES), 'images_found': len(covered),
        'missing_episodes': [r['episode'] for r in rows if not r['has_image']],
        'scores': results, 'source_paths': paths,
    }
    (args.output / 'report.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print('[REPORT]', (args.output / 'report.json').resolve())
    print('[CSV]', (args.output / 'selection.csv').resolve())
    print('[CONTACT SHEETS]', (args.output / 'contact_sheet_whole.jpg').resolve(),
          (args.output / 'contact_sheet_table_roi.jpg').resolve())
    if len(covered) < 70:
        print('[WARNING] Incomplete coverage: no conclusion about all 70 episodes')
    print('[IMPORTANT] Do not connect this unvalidated selector to online deployment.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
