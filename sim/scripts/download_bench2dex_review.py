#!/usr/bin/env python3
"""Small, pinned ModelScope downloads, always direct and SHA256 checked."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

ASSET_REV = "63e6ca0412eefbfaca0bbec0640dfd98ff466b57"
DATA_REV = "4062f0b5c8506375aa04208e149f151b39b152a0"


def opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def list_files(repo, revision, root, recursive=True):
    result, page = [], 1
    while True:
        query = urllib.parse.urlencode(dict(Revision=revision, Root=root,
            Recursive=str(recursive).lower(), PageNumber=page, PageSize=100))
        url = f"https://modelscope.cn/api/v1/datasets/Bench2Dex/{repo}/repo/tree?{query}"
        with opener().open(url, timeout=30) as response:
            data = json.load(response)
        if data.get("Code") != 200:
            raise RuntimeError(data)
        batch = data["Data"]["Files"]
        result.extend(batch)
        if len(result) >= data["Data"]["TotalCount"] or not batch:
            return result
        page += 1


def download(repo, revision, entry, output_root):
    relative = Path(entry["Path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe remote path")
    path = Path(output_root)/relative
    expected = entry["Sha256"]
    if not expected:
        raise ValueError("Missing publisher SHA256")
    if path.exists() and path.stat().st_size == entry["Size"]:
        with path.open("rb") as existing:
            if hashlib.file_digest(existing, "sha256").hexdigest() == expected:
                return path
    path.parent.mkdir(parents=True, exist_ok=True)
    query = urllib.parse.urlencode(dict(Revision=revision, FilePath=entry["Path"]))
    url = f"https://modelscope.cn/api/v1/datasets/Bench2Dex/{repo}/repo?{query}"
    partial = path.with_name(path.name+".partial")
    digest, size = hashlib.sha256(), 0
    with opener().open(url, timeout=60) as response, partial.open("wb") as f:
        while chunk := response.read(1024*1024):
            f.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    if size != entry["Size"] or digest.hexdigest() != expected:
        raise RuntimeError(f"Size/hash mismatch: {path}: {size}, {digest.hexdigest()}")
    partial.replace(path)
    print(f"Verified {entry['Path']} ({size} bytes)", flush=True)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", choices=["Bench2Dex", "teleopdata"], required=True)
    p.add_argument("--revision")
    p.add_argument("--root", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=3)
    p.add_argument("--max-bytes", type=int, default=100_000_000)
    args = p.parse_args()
    revision = args.revision or (ASSET_REV if args.repo == "Bench2Dex" else DATA_REV)
    files = [x for x in list_files(args.repo, revision, args.root) if x["Type"] == "blob"]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"listing.json").write_text(json.dumps(files, indent=2)+"\n")
    # Releases may contain exact duplicate *_1 files; do not count them as
    # independent demonstrations. Spread the sample across unique file order.
    unique = list({x["Sha256"]: x for x in reversed(files)}.values())[::-1]
    count = min(args.limit, len(unique))
    indices = [round(i*(len(unique)-1)/max(count-1, 1)) for i in range(count)]
    selected = [unique[i] for i in indices]
    if sum(x["Size"] for x in selected) > args.max_bytes:
        raise SystemExit("Download would exceed byte budget; inspect listing.json first")
    for entry in selected:
        download(args.repo, revision, entry, args.output)
    (args.output/"download_receipt.json").write_text(json.dumps(dict(
        repo=args.repo, revision=revision, proxy_used=False, files=selected), indent=2)+"\n")


if __name__ == "__main__":
    main()
