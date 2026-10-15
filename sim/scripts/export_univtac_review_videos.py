#!/usr/bin/env python3
"""Export every available UniVTAC insert_hole HDF5 episode for visual review."""

from __future__ import annotations

import argparse
import html
import json
import subprocess
from pathlib import Path

import h5py

from visualize_univtac_episode import export


DEFAULT_PUBLISHED = Path("/home/sai/zx/openpi-sim-runtime/data/isaac51/insert_hole/hdf5")
DEFAULT_RECAPTURED = Path(
    "/home/sai/zx/openpi-sim-runtime/data_depth/insert_hole/univtac_insert_hole_depth/hdf5"
)
DEFAULT_OUTPUT = Path("/home/sai/zx/openpi-sim-runtime/visualizations/univtac_insert_hole")


def probe_video(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,pix_fmt,width,height,nb_read_frames,r_frame_rate",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)["streams"][0]


def verify_decode(path: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-xerror", "-i", str(path),
         "-f", "null", "-"],
        check=True, capture_output=True, text=True,
    )


def make_poster(video: Path, poster: Path, duration_s: float) -> None:
    if poster.is_file() and poster.stat().st_mtime_ns >= video.stat().st_mtime_ns:
        return
    poster.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss",
         str(duration_s / 2), "-i", str(video), "-frames:v", "1", "-vf", "scale=480:-2",
         str(poster)],
        check=True, capture_output=True, text=True,
    )


def write_index(output: Path, items: list[dict]) -> None:
    cards = []
    for item in items:
        rel = Path(item["output"]).relative_to(output).as_posix()
        poster = Path(item["poster"]).relative_to(output).as_posix()
        label = html.escape(item["label"])
        cards.append(
            f'<article><video controls preload="none" playsinline poster="{html.escape(poster)}" '
            f'src="{html.escape(rel)}"></video>'
            f'<div><a href="{html.escape(rel)}">{label}</a> · '
            f'{item["video_frames"]} 帧 · {item["duration_s"]:.2f} 秒</div></article>'
        )
    document = """<!doctype html><html lang="zh"><meta charset="utf-8">
<title>UniVTAC insert_hole 视频总览</title>
<style>body{font:16px system-ui;margin:24px;background:#141820;color:#eef}a{color:#8bd}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:20px}
article{background:#222a35;padding:10px;border-radius:8px}video{width:100%;aspect-ratio:16/13}
small{color:#bac5d0}</style><h1>UniVTAC insert_hole 视频总览</h1>
<p>公开数据 100 条仅有 head/wrist RGB 和双侧 GelSight RGB/压入量；本机重采 3 条还展示世界相机深度。
这些都是筛选后保存的成功示范，不能从视频数量估算任务成功率。</p>
<p><small>每格为同步多模态拼图；按需点击播放，原始路径和逐条校验结果见 manifest.json。</small></p>
<main>""" + "\n".join(cards) + "</main></html>\n"
    (output / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published-dir", type=Path, default=DEFAULT_PUBLISHED)
    parser.add_argument("--recaptured-dir", type=Path, default=DEFAULT_RECAPTURED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--encoder", choices=("av1_nvenc", "libsvtav1", "libx264"),
                        default="av1_nvenc")
    parser.add_argument("--stride", type=int, default=1, help="1 retains all 60 Hz frames")
    parser.add_argument("--limit", type=int, help="Only export first N inputs for a smoke check")
    args = parser.parse_args()
    if args.stride < 1 or (args.limit is not None and args.limit < 1):
        parser.error("stride and limit must be positive")
    sources = []
    for kind, directory in (("published", args.published_dir), ("recaptured", args.recaptured_dir)):
        if not directory.is_dir():
            parser.error(f"Missing {kind} HDF5 directory: {directory}")
        files = sorted(directory.glob("*.hdf5"), key=lambda path: int(path.stem))
        if not files:
            parser.error(f"No HDF5 episodes in {directory}")
        sources.extend((kind, path) for path in files)
    if args.limit is not None:
        sources = sources[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for number, (kind, source) in enumerate(sources, start=1):
        target = args.output_dir / kind / f"{int(source.stem):06d}.mp4"
        with h5py.File(source, "r") as root:
            source_frames = len(root["step"])
        expected_frames = len(range(0, source_frames, args.stride))
        expected_codec = "h264" if args.encoder == "libx264" else "av1"
        expected_height = 1080 if kind == "recaptured" else 810
        reusable = False
        if target.is_file():
            try:
                stream = probe_video(target)
                reusable = (
                    stream.get("codec_name") == expected_codec
                    and stream.get("pix_fmt") == "yuv420p"
                    and int(stream.get("nb_read_frames", -1)) == expected_frames
                    and (stream.get("width"), stream.get("height")) == (960, expected_height)
                )
            except (subprocess.CalledProcessError, KeyError, IndexError, ValueError):
                reusable = False
        if reusable:
            result = {"source": str(source), "output": str(target),
                      "source_frames": source_frames, "video_frames": expected_frames,
                      "fps": 60 / args.stride, "duration_s": expected_frames / (60 / args.stride),
                      "codec": expected_codec, "camera_depth_included": kind == "recaptured",
                      "bytes": target.stat().st_size}
        else:
            result = export(source, target, 60, args.stride, None,
                            kind == "recaptured", args.encoder)
        verify_decode(target)
        poster = args.output_dir / "posters" / kind / f"{int(source.stem):06d}.jpg"
        make_poster(target, poster, result["duration_s"])
        result["label"] = f"{kind} / {int(source.stem):03d}"
        result["poster"] = str(poster)
        result["full_decode_pass"] = True
        items.append(result)
        (args.output_dir / "manifest.json").write_text(
            json.dumps({"count": len(items), "items": items}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[{number}/{len(sources)}] {result['label']} · {expected_frames} frames · "
              f"{'reused' if reusable else 'encoded'} · decode OK", flush=True)
    write_index(args.output_dir, items)
    print(f"Open {args.output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
