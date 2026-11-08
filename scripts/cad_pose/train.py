#!/usr/bin/env python3
"""Train the standalone pose model; --synthetic is strictly a pipeline smoke."""

import argparse
import json
from pathlib import Path

from openpi.cad_pose.pipeline import DEFAULT_TASKS
from openpi.cad_pose.pipeline import PROFILE
from openpi.cad_pose.pipeline import open_tasks
from openpi.cad_pose.pipeline import splits_for
from openpi.cad_pose.pipeline import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON overrides for seed, lr, patience and validation_batches")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--tasks", nargs="+", choices=DEFAULT_TASKS, default=list(DEFAULT_TASKS))
    parser.add_argument("--profile", default=PROFILE)
    parser.add_argument(
        "--peg-review", type=Path, help="accepted Peg keyframe review JSON bound to geometry and calibration hashes"
    )
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument(
        "--peg-legacy-intrinsics", action="store_true", help="explicitly allow historical Peg color intrinsics"
    )
    parser.add_argument("--full-audit", action="store_true", help="run the expensive full audit before training")
    parser.add_argument("--timeout-seconds", type=float, default=120, help="per-task quick timeout")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text()) if args.config else {}
    unknown = set(settings) - {"seed", "lr", "patience", "validation_batches", "symmetry"}
    if unknown:
        raise ValueError(f"unknown config settings: {unknown}")
    repo = args.repo.resolve()
    tasks, reports = open_tasks(
        repo,
        (args.data_root or repo / "data").resolve(),
        args.tasks,
        args.profile,
        synthetic=args.synthetic,
        quick=not args.full_audit,
        timeout=args.timeout_seconds,
        peg_legacy_intrinsics=args.peg_legacy_intrinsics,
    )
    for name, report in reports.items():
        print(name, report["status"], report.get("errors", []), flush=True)
    if set(tasks) != set(args.tasks):
        raise RuntimeError("selected tasks missing validated pose sources; see per-task audit above")
    peg = reports.get("1018_peg_all", {})
    if not args.synthetic and peg.get("checks", {}).get("adapter") == "legacy_ordered_fit":
        if not args.peg_review:
            raise ValueError("legacy Peg requires --peg-review with accepted keyframe geometry review")
        review = json.loads(args.peg_review.read_text())
        checks = peg["checks"]
        if peg.get("mode") == "quick":
            from scripts.preprocess.export_object_geometry_rgbtrack import sha256

            checks["geometry_manifest_sha256"] = sha256(tasks["1018_peg_all"].root / "object_geometry/v1/manifest.json")
        if (
            review.get("status") != "accepted"
            or not review.get("reviewer")
            or review.get("geometry_manifest_sha256") != checks["geometry_manifest_sha256"]
            or review.get("profile_sha256") != checks["profile_sha256"]
            or len(review.get("reviewed_frames", [])) < 3
        ):
            raise ValueError("Peg review must name reviewer, accept >=3 frames, and match geometry/profile hashes")
    split = splits_for(tasks, settings.get("seed", 42))
    train(
        tasks=tasks,
        splits=split,
        output=args.output_dir,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        steps=args.steps_per_epoch,
        lr=settings.get("lr", 3e-4),
        seed=settings.get("seed", 42),
        resume=args.resume,
        repo=repo,
        profile=args.profile,
        val_batches=settings.get("validation_batches", 100),
        patience=settings.get("patience", 10),
        synthetic=args.synthetic,
        symmetry=settings.get("symmetry", {}),
    )


if __name__ == "__main__":
    main()
