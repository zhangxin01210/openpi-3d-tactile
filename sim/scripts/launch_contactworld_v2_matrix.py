#!/usr/bin/env python3
"""Launch isolated ContactWorld USB replay-data configs on distinct GPUs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import time

CONFIGS_V2 = (
    "pi0_cw2_usb_01_rgb", "pi0_cw2_usb_02_front_pc",
    "pi0_cw2_usb_03_fused_pc", "pi0_cw2_usb_04_base_ff",
    "pi0_cw2_usb_05_front_pc_base_ff", "pi0_cw2_usb_06_fused_pc_base_ff",
    "pi0_cw2_usb_07_fused_pc_local_ff",
    "pi0_cw2_usb_08_fused_pc_ff_summary",
    "pi0_cw2_usb_09_fused_pc_basepos_localvec",
)
CONFIGS_V3 = (
    "pi0_cw3_usb_01_rgb", "pi0_cw3_usb_02_fused_pc",
    "pi0_cw3_usb_03_rgb_local_map", "pi0_cw3_usb_04_fused_pc_local_map",
    "pi0_cw3_usb_05_fused_pc_local_summary", "pi0_cw3_usb_06_fused_pc_base_structured",
    "pi0_cw3_usb_07_fused_pc_base_pooled", "pi0_cw3_usb_08_fused_pc_base_map",
    "pi0_cw3_usb_09_fused_pc_base_map_split", "pi0_cw3_usb_10_fused_pc_base_map_matched",
    "pi0_cw3_usb_11_fused_pc_base_structured_prefix",
    "pi0_cw3_usb_12_fused_pc_base_structured_split_route",
    "pi0_cw3_usb_13_fused_pc_local_map_3tok",
    "pi0_cw3_usb_14_base_pooled_no_pc",
    "pi0_cw3_usb_15_base_map_matched_no_pc",
    "pi0_cw3_usb_16_contact_centered_point",
    "pi0_cw3_usb_17_contact_centered_map",
    "pi0_cw3_usb_18_base_pooled_relation",
    "pi0_cw3_usb_19_base_map_matched_relation",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=("v2", "v3"), default="v2")
    parser.add_argument("--label", required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--configs", nargs="*", default=None)
    parser.add_argument("--exclude", nargs="+", default=[], metavar="ID_OR_NAME",
                        help="Skip configs by two-digit ID (e.g. 01) or full config name")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    configs = CONFIGS_V3 if args.version == "v3" else CONFIGS_V2
    selected = list(args.configs) if args.configs is not None else list(configs)
    if not selected or any(name not in configs for name in selected) or len(set(selected)) != len(selected):
        parser.error("--configs must be unique names from the selected version: " + ", ".join(configs))
    config_by_id = {name.split("_")[3]: name for name in configs}
    excluded = []
    for value in args.exclude:
        normalized_id = value.zfill(2) if value.isdigit() else value
        name = config_by_id.get(normalized_id, value)
        if name not in configs or name in excluded:
            parser.error("--exclude must contain unique IDs or names from the selected version: " + value)
        excluded.append(name)
    selected = [name for name in selected if name not in excluded]
    if not selected:
        parser.error("No configurations remain after --exclude")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.label):
        parser.error("Invalid label")
    gpus = [value.strip() for value in args.gpus.split(",")]
    if not gpus or len(gpus) != len(set(gpus)) or any(not value.isdigit() for value in gpus):
        parser.error("--gpus must list distinct GPU indices")
    repo = Path(__file__).resolve().parents[2]
    data = repo / ("data/contactworld_usb_v3_bilateral" if args.version == "v3" else
                   "data/contactworld_usb_v2")
    configured = os.environ.get("OPENPI_CONTACTWORLD_BASE_WEIGHTS")
    candidates = (repo / "checkpoints/pi0_base/params",
                  Path("/workspace/mnt/sqzhang26/hf_weight/pi0_base/params"))
    weights = Path(configured) if configured else next(
        (path for path in candidates if path.exists()), candidates[0])
    if not weights.is_absolute():
        weights = repo / weights
    for path in (data / "meta/info.json", data / "spatial/manifest.json",
                 data / "assets/norm/norm_stats.json", weights / "_METADATA"):
        if not path.exists():
            parser.error("Missing training input: " + str(path))
    subprocess.run([sys.executable, "sim/scripts/verify_contactworld_v2_lerobot.py", str(data)],
                   cwd=repo, check=True)
    if args.version == "v3":
        representation_configs = {
            "pi0_cw3_usb_16_contact_centered_point",
            "pi0_cw3_usb_17_contact_centered_map",
            "pi0_cw3_usb_18_base_pooled_relation",
            "pi0_cw3_usb_19_base_map_matched_relation",
        }
        if representation_configs.intersection(selected):
            representation_manifest = data / "spatial/contact_repr_v1/manifest.json"
            if not representation_manifest.is_file():
                parser.error("Missing reviewed contact representation sidecar: " + str(representation_manifest))
            subprocess.run([sys.executable, "sim/scripts/verify_contactworld_usb_contact_repr.py",
                            "--dataset", str(data), "--repr-root", str(representation_manifest.parent)],
                           cwd=repo, check=True)
        # The audit reads only local LeRobot files, but datasets may otherwise
        # try to create a cache under the invoking account's home directory.
        # Keep that write isolated and make the preflight work under both a
        # workstation user and a root-owned training pod.
        audit_env = dict(os.environ, JAX_PLATFORM_NAME="cpu")
        audit_env.setdefault("HF_DATASETS_CACHE", "/tmp/openpi-hf-datasets")
        subprocess.run([sys.executable, "sim/scripts/audit_contactworld_v3_inputs.py", str(data),
                        "--configs", *selected],
                       cwd=repo, env=audit_env, check=True)
    log_dir = repo / ("runs/contactworld_usb_v3_train" if args.version == "v3" else
                       "runs/contactworld_usb_v2_train") / args.label
    if log_dir.exists():
        parser.error("Log directory exists: " + str(log_dir))
    for name in selected:
        checkpoint = repo / "checkpoints" / name / args.label
        if checkpoint.exists():
            parser.error("Checkpoint directory exists: " + str(checkpoint))
    if args.check_only:
        print(f"Preflight passed: {len(selected)} configs, weights={weights}, data={data}")
        print("Selected configs: " + ", ".join(selected))
        return
    log_dir.mkdir(parents=True)
    pending = selected
    active = {}
    failed = []
    while pending or active:
        for gpu in gpus:
            if not pending or gpu in active:
                continue
            name = pending.pop(0)
            log = (log_dir / (name + ".log")).open("w")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.88")
            command = [sys.executable, "scripts/train.py", name,
                       "--exp-name", args.label]
            if not args.wandb:
                command.append("--no-wandb-enabled")
            process = subprocess.Popen(command, cwd=repo, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
            active[gpu] = (process, log, name)
            print(f"GPU {gpu}: {name} pid={process.pid}", flush=True)
        time.sleep(5)
        for gpu, (process, log, name) in list(active.items()):
            rc = process.poll()
            if rc is None:
                continue
            log.close()
            print(f"GPU {gpu}: {name} exit={rc}", flush=True)
            if rc:
                failed.append(name)
            del active[gpu]
    if failed:
        raise SystemExit("Failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
