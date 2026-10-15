#!/usr/bin/env python3
"""Render a reviewed first cross-task LeRobot episode, preserving all sensor views."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from contactworld_cross_task_common import TASKS, get_task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    spec = get_task(args.task)
    manifest = json.loads((args.dataset / "spatial/manifest.json").read_text())
    if manifest.get("task") != spec.key:
        raise ValueError("Dataset task does not match --task")
    # The mature USB V3 renderer reads only the generic V3 sidecar contract.
    # Keep this wrapper so its invocation/receipt is task-specific.
    renderer = Path(__file__).with_name("render_contactworld_v3_dataset_review.py")
    subprocess.run([sys.executable, str(renderer), str(args.dataset), str(args.output)], check=True)
    index = args.output / "index.html"
    html = index.read_text()
    html = html.replace("ContactWorld USB：转好的 LeRobot 单条审核", "ContactWorld %s：转好的 LeRobot 单条审核" % spec.key)
    html = html.replace("<p>以下内容直接读取", "<p>任务指令：<code>%s</code>；动作维度：%d。</p><p>以下内容直接读取" % (spec.prompt, spec.action_dim), 1)
    index.write_text(html)
    (args.output / "review_contract.json").write_text(json.dumps({"task": spec.key, "dataset": str(args.dataset),
        "checks": ["front/wrist RGB", "4096 base-frame point clouds", "left/right Fx/Fy/Fz previews",
                   "taxel geometry and force vectors", "executed action dimensionality"]}, indent=2) + "\n")
    print(index)


if __name__ == "__main__":
    main()
