#!/usr/bin/env python3
"""Prepare verified SAM2 / RGBTrack weights without guessing model IDs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
SAM2_RELATIVE_PATH = Path("checkpoints/sam2/sam2.1_hiera_small.pt")
RGBTRACK_RUNS = {
    "scorer": Path("2024-01-11-20-02-45/model_best.pth"),
    "refiner": Path("2023-10-28-18-33-37/model_best.pth"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _direct_opener() -> urllib.request.OpenerDirector:
    # Respect the user's no-proxy-first requirement even if shell proxy
    # variables point to a stale local proxy process.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def download_resumable(url: str, destination: Path) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    opener = _direct_opener()
    head = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "openpi-deploy/1.0"})
    with opener.open(head, timeout=30) as response:
        expected_size = int(response.headers.get("Content-Length", "0"))
    offset = partial.stat().st_size if partial.exists() else 0
    if expected_size and offset > expected_size:
        partial.unlink()
        offset = 0
    headers = {"User-Agent": "openpi-deploy/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        response = opener.open(request, timeout=60)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Download failed: HTTP {exc.code} for {url}") from exc
    with response:
        append = offset > 0 and response.status == 206
        if offset and not append:
            offset = 0
        mode = "ab" if append else "wb"
        with partial.open(mode) as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    actual_size = partial.stat().st_size
    if expected_size and actual_size != expected_size:
        raise RuntimeError(
            f"Incomplete file retained for resume: {partial} has {actual_size} bytes, expected {expected_size}"
        )
    digest = sha256_file(partial)
    partial.replace(destination)
    return {"url": url, "size_bytes": actual_size, "sha256": digest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--download-sam2", action="store_true")
    parser.add_argument(
        "--rgbtrack-source-dir",
        type=Path,
        help="Existing verified computer-4 weights directory; files are copied by the adapter's exact names.",
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = args.repo_root.expanduser().resolve()
    manifest_path = root / "checkpoints/deploy_weights_manifest.json"
    records: dict[str, object] = {}
    if manifest_path.is_file():
        try:
            records = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            records = {}
    records.update(
        {
            "schema_version": 1,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "sam2_model": "SAM2.1 Hiera Small (official checkpoint)",
            "sam2_source_repository": "https://github.com/facebookresearch/sam2",
            "sam2_source_revision": "2b90b9f5ceec907a1c18123530e92e794ad901a4",
            "rgbtrack_source_revision": "23e43f832b82f4813d6ab72773779bf8d2ee4d83",
        }
    )
    if args.download_sam2:
        sam2_path = root / SAM2_RELATIVE_PATH
        if sam2_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing SAM2 checkpoint: {sam2_path}")
        records["sam2_checkpoint"] = {"path": str(sam2_path.relative_to(root)), **download_resumable(SAM2_URL, sam2_path)}
    if args.rgbtrack_source_dir is not None:
        source_root = args.rgbtrack_source_dir.expanduser().resolve()
        weight_records = records.get("rgbtrack_checkpoints", {})
        if not isinstance(weight_records, dict):
            weight_records = {}
        for object_name, relative in RGBTRACK_RUNS.items():
            source = source_root / relative
            target = root / "checkpoints/rgbtrack" / relative
            if not source.is_file():
                raise FileNotFoundError(f"Computer-4 {object_name} weight is missing: {source}")
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite existing RGBTrack checkpoint: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if target.stat().st_size != source.stat().st_size or sha256_file(target) != sha256_file(source):
                raise IOError(f"Copied {object_name} weight did not verify: {target}")
            weight_records[object_name] = {
                "path": str(target.relative_to(root)),
                "copied_from": str(source),
                "size_bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }
        records["rgbtrack_checkpoints"] = weight_records

    files: dict[str, object] = {}
    sam2_path = root / SAM2_RELATIVE_PATH
    files["sam2"] = {"exists": sam2_path.is_file()}
    if sam2_path.is_file():
        files["sam2"].update({"path": str(sam2_path.relative_to(root)), "size_bytes": sam2_path.stat().st_size, "sha256": sha256_file(sam2_path)})
    for name, relative in RGBTRACK_RUNS.items():
        path = root / "checkpoints/rgbtrack" / relative
        files[name] = {"exists": path.is_file()}
        if path.is_file():
            files[name].update({"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    records["files"] = files
    records["ok"] = bool(files["sam2"]["exists"] and files["scorer"]["exists"] and files["refiner"]["exists"])
    if args.download_sam2 or args.rgbtrack_source_dir is not None or manifest_path.exists():
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(records, indent=2, ensure_ascii=False))
    return 0 if records["ok"] else (1 if args.verify_only else 0)


if __name__ == "__main__":
    raise SystemExit(main())
