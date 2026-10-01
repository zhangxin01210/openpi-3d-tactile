"""Launch the eleven ContactWorld USB configs, one process per selected GPU.

Example: uv run --no-sync python sim/scripts/launch_contactworld_matrix.py --label first --gpus 0,1,2,3
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import time

CONFIGS = (
    "pi0_cw_usb_01_rgb", "pi0_cw_usb_02_rgb_pc", "pi0_cw_usb_03_rgb_ff",
    "pi0_cw_usb_04_rgb_pc_ff", "pi0_cw_usb_05_ff_summary", "pi0_cw_usb_06_ff_ee3d_proxy",
    "pi0_cw_usb_07_tacrgb", "pi0_cw_usb_08_tacdepth", "pi0_cw_usb_09_prefix",
    "pi0_cw_usb_10_split", "pi0_cw_usb_11_pcnoise",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="A fresh experiment label shared by the matrix")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--configs", nargs="*", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Validate files and config names without training")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.label):
        parser.error("Label may contain only letters, digits, dot, underscore, hyphen")
    gpus = [x.strip() for x in args.gpus.split(",")]
    if not gpus or len(set(gpus)) != len(gpus) or any(not x.isdigit() for x in gpus):
        parser.error("--gpus must be a comma-separated list of distinct GPU indices")
    repo = Path(__file__).resolve().parents[2]
    data = repo / "data/contactworld_usb_positive_all"
    stats = data / "assets/norm/norm_stats.json"
    candidates = (repo / "checkpoints/pi0_base/params",
                  Path("/workspace/mnt/sqzhang26/hf_weight/pi0_base/params"))
    configured = os.environ.get("OPENPI_CONTACTWORLD_BASE_WEIGHTS")
    weights = Path(configured) if configured else next((p for p in candidates if p.exists()), candidates[0])
    if not weights.is_absolute():
        weights = repo / weights
    for required in (data / "meta/info.json", data / "spatial/manifest.json", stats, weights):
        if not required.exists():
            parser.error(f"Missing required training input: {required}")
    subprocess.run([sys.executable, "sim/scripts/verify_contactworld_lerobot.py", str(data)],
                   cwd=repo, check=True)
    log_dir = repo / "runs/contactworld_usb_train" / args.label
    if log_dir.exists():
        parser.error(f"Log directory exists: {log_dir}; choose a new --label")
    for name in args.configs:
        checkpoint = repo / "checkpoints" / name / args.label
        if checkpoint.exists():
            parser.error(f"Checkpoint directory exists: {checkpoint}; choose a new --label")
    if args.check_only:
        print(f"Preflight passed: {len(args.configs)} configs, weights {weights}, dataset {data}")
        return
    log_dir.mkdir(parents=True)
    pending = list(args.configs)
    active: dict[str, tuple[subprocess.Popen, object, str]] = {}
    failed = []
    print(f"Launching {len(pending)} configs on GPUs {gpus}; logs: {log_dir}", flush=True)
    while pending or active:
        for gpu in gpus:
            if not pending or gpu in active:
                continue
            name = pending.pop(0)
            log = (log_dir / f"{name}.log").open("w")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.88")
            command = [sys.executable, "scripts/train.py", name, "--exp-name", args.label]
            if not args.wandb:
                command.append("--no-wandb-enabled")
            process = subprocess.Popen(command, cwd=repo, env=env, stdout=log,
                                       stderr=subprocess.STDOUT)
            active[gpu] = process, log, name
            print(f"GPU {gpu}: {name}, pid {process.pid}", flush=True)
        time.sleep(5)
        for gpu, (process, log, name) in list(active.items()):
            rc = process.poll()
            if rc is None:
                continue
            log.close()
            print(f"GPU {gpu}: {name} exited {rc}", flush=True)
            if rc != 0:
                failed.append(name)
            del active[gpu]
    if failed:
        raise SystemExit(f"Failed configs: {', '.join(failed)}. Read their logs in {log_dir}")
    print("All ContactWorld configs finished successfully.")


if __name__ == "__main__":
    main()
