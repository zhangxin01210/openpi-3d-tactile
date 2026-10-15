#!/usr/bin/env python3
"""Resumable direct downloader for the two ContactWorld cross-task archives.

Large archives are fetched without proxy from the official Hugging Face dataset.
The script uses a .part file and HTTP ranges, then validates the server-reported
size.  It intentionally does not unpack or overwrite an existing source group.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

from contactworld_cross_task_common import TASKS

REPO = "Pokuang/ContactWorld"
REVISION = "main"
HF = "https://hf-mirror.com/datasets/%s/resolve/%s/releases" % (REPO, REVISION)


def remote_info(filename: str) -> tuple[int, str | None]:
    request = urllib.request.Request(
        "%s/%s?download=true" % (HF, filename), method="HEAD")
    with urllib.request.urlopen(request, timeout=45) as reply:
        size = int(reply.headers["Content-Length"])
        return size, reply.headers.get("ETag", "").strip('"') or None


def download(filename: str, output: Path) -> dict:
    target = output / filename
    partial = output / (filename + ".part")
    size, etag = remote_info(filename)
    if target.exists() and target.stat().st_size == size:
        return {"file": str(target), "bytes": size, "etag": etag, "status": "already_complete"}
    start = partial.stat().st_size if partial.exists() else 0
    if start > size:
        partial.unlink()
        start = 0
    url = "%s/%s?download=true" % (HF, filename)
    # Some mirrors close multi-GB HTTP streams with curl exit 18.  Resume from
    # the actual .part length, including when the previous curl call failed.
    # Do not delete partial data unless it exceeds the authoritative size.
    for attempt in range(1, 10):
        start = partial.stat().st_size if partial.exists() else 0
        if start == size:
            break
        print("Downloading %s from byte %d / %d (attempt %d/9)" %
              (filename, start, size, attempt), flush=True)
        command = ["curl", "--noproxy", "*", "--fail", "--location", "--retry", "4",
                   "--retry-all-errors", "--retry-delay", "3", "--continue-at", str(start),
                   "--output", str(partial), url]
        result = subprocess.run(command, check=False)
        if result.returncode == 0 and partial.exists() and partial.stat().st_size == size:
            break
        if partial.exists() and partial.stat().st_size > size:
            raise RuntimeError("Partial file exceeds expected size: " + str(partial))
        if attempt == 9:
            raise RuntimeError("Download did not finish after 9 resumable attempts: %s (curl=%d)" %
                               (filename, result.returncode))
    if partial.stat().st_size != size:
        raise RuntimeError("Incomplete download %s: %d != %d" % (filename, partial.stat().st_size, size))
    partial.replace(target)
    return {"file": str(target), "bytes": size, "etag": etag, "status": "downloaded"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=["barbed_flat", "valve"], choices=TASKS)
    parser.add_argument("--output", type=Path,
                        default=Path("/home/sai/zx/openpi-sim-runtime/downloads/contactworld"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.tasks:
        spec = TASKS[name]
        if spec.archive is None:
            print("%s is already expected at the local source path; no archive download." % name)
            continue
        rows.append({"task": name, **download(spec.archive, args.output)})
    receipt = {"repository": REPO, "revision": REVISION, "proxy": "disabled (--noproxy *)", "files": rows}
    (args.output / "cross_task_download_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
