#!/usr/bin/env python3
"""Package reviewed task-specific SAM2 initialization assets for deployment.

Read-only to original HOI and preprocessing directories. Default is dry-run.
Run with --apply to create an immutable v1 asset bundle; never overwrites existing.
No model inference, videos, generated per-frame masks, or large data copied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

PEG_EPISODES = (0, 23, 38, 49, 58, 65)
PEG_OBJECTS = ("funnel", "target_flask")
NEW_TASKS = {
    "1028_press_button": ("button", "box"),
    "1030_yellow_socket": ("yellow", "red"),
}


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def mask_check(path: Path, shape: tuple[int, int]) -> int:
    array = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if array is None or array.shape != shape:
        raise ValueError(f"Mask missing or wrong shape: {path}, expected={shape}")
    count = int((array > 127).sum())
    if not 8 < count < array.size // 2:
        raise ValueError(f"Implausible mask area {count}: {path}")
    return count


def validate_rgb(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unreadable reference RGB: {path}")
    return image.shape[:2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--hoi", type=Path, default=Path("/workspace/mnt/sqzhang26/hoi"))
    parser.add_argument("--output", type=Path,
                        default=Path("assets/perception/sam2_references"))
    parser.add_argument("--apply", action="store_true", help="Write the new asset bundle")
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()
    hoi = args.hoi.expanduser().resolve()
    out_root = (args.output if args.output.is_absolute() else repo / args.output).resolve()
    targets: list[tuple[str, str, list[dict]]] = []

    # Use the *existing* reviewed deployment prompt resolver; e0 has a documented
    # flask/funnel fallback, and copying raw e0 JSON would subtly break it.
    sys.path.insert(0, str(repo))
    from scripts.deploy.position_initialization import resolve_prompts

    peg_entries: list[dict] = []
    for ep in PEG_EPISODES:
        frame = hoi / f"data/level1_sequence_e{ep}/frames/00000.jpg"
        hw = validate_rgb(frame)
        if hw != (480, 640):
            raise ValueError(f"e{ep}: expected 480x640, got {hw}")
        # e23 has a later manually repaired mask package; use it when present.
        mask_root = hoi / f"outputs/level1_sequence_e{ep}"
        if ep == 23 and (hoi / "outputs/level1_sequence_e23_maskfix_v1/masks").is_dir():
            mask_root = hoi / "outputs/level1_sequence_e23_maskfix_v1"
        masks = {obj: mask_root / "masks" / obj / "00000.png"
                 for obj in PEG_OBJECTS}
        counts = {name: mask_check(path, hw) for name, path in masks.items()}
        resolved = resolve_prompts(
            repo_root=repo, prompt_file=None, template_episode=ep,
            template_confirmed=True, first_frame_rgb_shape=(480, 640, 3),
        )
        if set(resolved.get("objects", {})) != set(PEG_OBJECTS):
            raise ValueError(f"e{ep}: unexpected prompt object names")
        for name, prompt in resolved["objects"].items():
            pts = prompt.get("positive", [])
            if not pts:
                raise ValueError(f"e{ep}: missing positive prompts for {name}")
        print(f"[PEG] e{ep:02d} masks={counts} prompts=OK frame={frame}")
        peg_entries.append({"episode": ep, "frame": frame,
                            "masks": masks, "prompts": resolved, "counts": counts})
    targets.append(("1018_peg_all", "multi_reference_points", peg_entries))

    for task, expected_objects in NEW_TASKS.items():
        task_root = repo / "output/preprocess" / task / "sam2"
        ref = task_root / "reference"
        man = read_json(task_root / "manifest.json")
        report = read_json(ref / "candidate_report.json")
        review = read_json(ref / "review.json")
        objects = list(man.get("objects", []))
        if set(objects) != set(expected_objects) or len(objects) != 2:
            raise ValueError(f"{task}: unexpected object list {objects}")
        if review.get("status") != "human_accepted":
            raise ValueError(f"{task}: reference not human-approved")
        if review.get("candidate_report_sha256") != digest(ref / "candidate_report.json"):
            raise ValueError(f"{task}: reference approval is stale")
        if report.get("prompts_sha256") != digest(ref / "prompts.json"):
            raise ValueError(f"{task}: prompt hash mismatch")
        frame = ref / "reference_frame.jpg"
        hw = validate_rgb(frame)
        counts = {}
        masks = {}
        for obj in objects:
            path = ref / "templates" / f"{obj}.png"
            counts[obj] = mask_check(path, hw)
            if digest(path) != report["template_hashes"].get(obj):
                raise ValueError(f"{task}: stale reference template hash for {obj}")
            masks[obj] = path
        if report.get("frame_sha256") != digest(frame):
            raise ValueError(f"{task}: frame hash differs from candidate report")
        print(f"[TASK] {task} masks={counts} reference=human_accepted")
        targets.append((task, "single_reference_mask_match", [{"episode": int(read_json(ref / 'prompts.json').get('reference_episode', 0)),
            "frame": frame, "masks": masks, "prompts_file": ref / 'prompts.json',
            "candidate_report": ref / 'candidate_report.json', "review": ref / 'review.json',
            "counts": counts}]))

    for task, mode, entries in targets:
        destination = out_root / task / "v1"
        print(f"[PLAN] {task}: {len(entries)} reference(s) -> {destination}")
        if not args.apply:
            continue
        if destination.exists():
            raise FileExistsError(f"Refusing overwrite: {destination}")
        stage = destination.with_name(".v1_packaging")
        if stage.exists():
            raise FileExistsError(f"Existing interrupted staging directory: {stage}")
        stage.mkdir(parents=True)
        try:
            hashes = {}
            references = []
            for entry in entries:
                ep = entry["episode"]
                rel_root = Path("references") / f"e{ep:05d}"
                subdir = stage / rel_root
                (subdir / "templates").mkdir(parents=True)
                rgb_dest = subdir / "reference_frame.jpg"
                shutil.copy2(entry["frame"], rgb_dest)
                prompt_dest = subdir / "prompts.json"
                if "prompts_file" in entry:
                    shutil.copy2(entry["prompts_file"], prompt_dest)
                else:
                    prompt_dest.write_text(json.dumps(entry["prompts"], ensure_ascii=False, indent=2) + "\n")
                for name, p in entry["masks"].items():
                    shutil.copy2(p, subdir / "templates" / f"{name}.png")
                for key in ("candidate_report", "review"):
                    if key in entry:
                        shutil.copy2(entry[key], subdir / ("review.json" if key == "review" else "candidate_report.json"))
                for p in subdir.rglob("*"):
                    if p.is_file():
                        hashes[str(p.relative_to(stage))] = digest(p)
                references.append({"reference_episode": ep, "path": str(rel_root),
                                   "mask_pixels": entry["counts"]})
            manifest = {
                "schema_version": 1, "task": task, "version": "v1", "camera": "front",
                "initialization_strategy": mode, "objects": sorted(entries[0]["masks"]),
                "references": references, "sha256": hashes,
                "status": "packaged_from_reviewed_sources",
                "note": ("Packaging validates file integrity and approval metadata when available. "
                         "It does NOT certify online object detection/segmentation or new camera poses."),
            }
            (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
            stage.rename(destination)
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise
        print(f"[PACKAGED] {destination} files={len(hashes)+1}")
    print("[DONE]" if args.apply else "[DRY-RUN ONLY] Run with --apply after review")


if __name__ == "__main__":
    main()