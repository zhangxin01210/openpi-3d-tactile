#!/usr/bin/env python3
"""Download the pinned USB/Peg archives with resumable ranges and LFS SHA256 checks."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import time

REVISION = "6587189d8e06480e739af5a4004d411c5c07743b"
FILES = {
    "insertion_usb": (6585153409, "e4b0a6b9d7751cea5d43b48916fb2bbca847b814d29242d58ec7eb7dae452f09"),
    "insertion_peg": (5441545984, "2c21d2ba2a19da64da5ec9945db6b50b98b4969e84d47515e41d16a4c13b8d7f"),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--host", default="https://hf-mirror.com")
    p.add_argument("--direct", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    block = 64 * 1024 * 1024
    jobs = []
    for task, (size, _) in FILES.items():
        directory = args.directory / (task + ".parts")
        directory.mkdir(exist_ok=True)
        for i, start in enumerate(range(0, size, block)):
            jobs.append((task, directory / f"{i:04d}", start, min(size, start + block) - 1))

    def download(job):
        task, dest, start, end = job
        if dest.exists() and dest.stat().st_size == end - start + 1:
            return
        url = f"{args.host}/datasets/Pokuang/ContactWorld/resolve/{REVISION}/releases/{task}_dataset.tar.gz?download=true&part={start}"
        temp = dest.with_suffix(".tmp")
        header = dest.with_suffix(".headers")
        for attempt in range(4):
            # Large dataset transfers always bypass environment proxies, including redirects.
            command = ["curl", "--noproxy", "*", "-sS", "-L", "--fail", "--connect-timeout", "20", "--max-time", "240",
                       "-r", f"{start}-{end}", "-D", str(header), "-o", str(temp), url]
            result = subprocess.run(command, capture_output=True, text=True)
            h = header.read_text().lower() if header.exists() else ""
            expected = f"content-range: bytes {start}-{end}/{FILES[task][0]}"
            if result.returncode == 0 and temp.stat().st_size == end-start+1 and expected in h:
                temp.replace(dest)
                return
            if attempt == 3:
                raise RuntimeError(f"Range failed {job}: {result.stderr[-300:]}")
            time.sleep(2)

    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, future in enumerate(concurrent.futures.as_completed([pool.submit(download, j) for j in jobs]), 1):
            future.result()
            print(f"ranges {i}/{len(jobs)}, elapsed {time.monotonic()-started:.0f}s", flush=True)
    manifest = {"revision": REVISION, "download_host": args.host, "proxy_used": False, "files": []}
    for task, (size, expected) in FILES.items():
        dest = args.directory / f"{task}_dataset.tar.gz"
        temp = dest.with_suffix(".assembling")
        sha = hashlib.sha256()
        with temp.open("wb") as out:
            for part in sorted((args.directory / (task + ".parts")).glob("[0-9][0-9][0-9][0-9]")):
                with part.open("rb") as src:
                    while data := src.read(8*1024*1024):
                        out.write(data)
                        sha.update(data)
        if temp.stat().st_size != size or sha.hexdigest() != expected:
            raise RuntimeError(f"Archive validation failed: {task} {sha.hexdigest()}")
        temp.replace(dest)
        manifest["files"].append({"path": str(dest), "bytes": size, "sha256": expected, "verified": True})
        print(f"SHA256 verified: {task}", flush=True)
    (args.directory / "verified_sources.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
