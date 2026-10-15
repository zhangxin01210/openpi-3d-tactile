#!/usr/bin/env python3
"""Build a local browser index for the fixed ContactWorld replay audit."""
import argparse
import html
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    args = parser.parse_args()
    result = json.loads((args.batch / "batch_result.json").read_text())
    cards = []
    for row in result["results"]:
        task, episode = row["task"], row["episode"]
        name = f"{task}_episode_{episode:03d}"
        video = args.batch / name / "recorded_left_replay_right.mp4"
        if not video.is_file():
            raise FileNotFoundError(video)
        cards.append(f"""<article><h2>{html.escape(task)} #{episode:03d}</h2>
<p>{row['frames']} 帧 · 源码终点判据：{row['final_source_success']} ·
插件终点位姿误差：{row['final_plug_error_mm']:.3f} mm</p>
<video controls preload="metadata" src="{name}/recorded_left_replay_right.mp4"></video>
<p><a href="{name}/replay.json">逐帧指标与执行动作</a></p></article>""")
    output = args.batch / "index.html"
    output.write_text("""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>ContactWorld USB / Peg 固定示范重放</title>
<style>body{font:16px system-ui;max-width:1100px;margin:auto;padding:24px;background:#f5f6f8;color:#17202a}
article{background:white;margin:18px 0;padding:18px;border-radius:10px}video{width:100%;max-width:850px}
p{line-height:1.5}small{color:#52606d}</style></head><body>
<h1>ContactWorld USB / Peg 固定示范重放</h1>
<p>左侧：发布示范；右侧：本机在线重放。上：前视；下：腕视。以记录的末端控制目标进行反馈跟踪；这是重放验证，不是策略评测。</p>
<p>8 个编号在执行前固定。胶垫与安装座仅做视觉颜色匹配；完整指标见每条 replay.json。</p>
""" + "\n".join(cards) + "</body></html>\n")
    print(output)


if __name__ == "__main__":
    main()
