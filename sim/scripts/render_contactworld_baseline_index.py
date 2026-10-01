#!/usr/bin/env python3
"""Local browser page for the fixed-seed joint-state baseline pilot."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    data = json.loads((args.run / "baseline_summary.json").read_text())
    sections = []
    for task, label in (("insertion_usb", "USB"), ("insertion_peg", "Peg")):
        entry = data["online"][task]
        video = args.run / ("eval_" + label.lower()) / "seed_10000/front_wrist.mp4"
        if not video.is_file():
            raise FileNotFoundError(video)
        sections.append(f"""<section><h2>{label}</h2>
<p>20 个随机初态；源码判据 {entry['source_ever_count']}/20；
最小关键点误差的中位数 {entry['minimum_keypoint_error_mm']['median']:.1f} mm，
最接近的一次 {entry['minimum_keypoint_error_mm']['min']:.1f} mm；源码阈值约 8.0 mm。</p>
<video controls preload="metadata" src="eval_{label.lower()}/seed_10000/front_wrist.mp4"></video>
<p>录像为种子 10000：左前视，右腕视；每个任务的其余种子和动作在
<a href="eval_{label.lower()}/summary.json">完整评测结果</a>中。</p></section>""")
    output = args.run / "index.html"
    output.write_text("""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>ContactWorld 关节状态基线：闭环试验</title>
<style>body{font:16px system-ui;max-width:1000px;margin:auto;padding:24px;background:#f5f6f8;color:#17202a}
section{background:white;margin:18px 0;padding:18px;border-radius:10px}
video{width:100%;max-width:850px}p{line-height:1.5}</style></head><body>
<h1>ContactWorld 关节状态＋步数基线</h1>
<p>这是从全新随机初态执行的策略录像。策略只读 9 维关节位置、9 维速度和步数；
不读取目标位姿、示范、图像、触觉或点云。录像 10 fps 只是播放速度。</p>
""" + "\n".join(sections) + "</body></html>\n")
    print(output)


if __name__ == "__main__":
    main()
