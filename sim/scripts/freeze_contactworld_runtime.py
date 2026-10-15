#!/usr/bin/env python3
"""Record the exact isolated runtime, collision assets, and replay entry points."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def command(*args):
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=30, check=True)
    return result.stdout.strip()


def hash_file(path):
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    runtime = args.runtime.resolve()
    project = Path(__file__).resolve().parents[2]
    source = runtime / "third_party/ContactWorld-sm120"
    gymenvs = source / "thirdparty/manifeel-isaacgymenvs"
    manifeel = source / "thirdparty/manifeel"
    files = [
        gymenvs / "assets/industreal/urdf/USB_plug.urdf",
        gymenvs / "assets/industreal/urdf/USB_socket.urdf",
        gymenvs / "assets/industreal/mesh/TVB_mesh/manifeel_usb_plug.obj",
        gymenvs / "assets/industreal/mesh/TVB_mesh/manifeel_usb_socket.obj",
        gymenvs / "assets/industreal/urdf/industreal_rectangular_peg_16mm.urdf",
        gymenvs / "assets/industreal/urdf/industreal_rectangular_hole_16mm.urdf",
        gymenvs / "assets/industreal/mesh/industreal_pegs/industreal_rectangular_peg_16mm.obj",
        gymenvs / "assets/industreal/mesh/industreal_pegs/industreal_tray_insert_rectangular_peg_16mm.obj",
        manifeel / "manifeel/config/task/TacSLTaskUSB.yaml",
        manifeel / "manifeel/config/task/TacSLTaskPeg.yaml",
        manifeel / "manifeel/config/isaacgym_config_usb.yaml",
        manifeel / "manifeel/config/isaacgym_config_peg.yaml",
        manifeel / "assets/tacsl/yaml/tacsl_asset_info_power.yaml",
        manifeel / "assets/industreal/yaml/industreal_asset_info_pegs.yaml",
        gymenvs / "isaacgymenvs/tasks/tacsl/tacsl_task_usb.py",
        gymenvs / "isaacgymenvs/tasks/tacsl/tacsl_task_peg.py",
        gymenvs / "isaacgymenvs/tasks/tacsl/tacsl_env_insertion.py",
        gymenvs / "isaacgymenvs/tacsl_sensors/tacsl_sensors.py",
        project / "sim/scripts/probe_contactworld_usb.py",
        project / "sim/scripts/replay_contactworld_episode.py",
        project / "sim/scripts/run_contactworld_sm120.sh",
        project / "sim/scripts/export_contactworld_replay_subset.py",
        project / "sim/scripts/run_contactworld_replay_batch.py",
        project / "sim/scripts/render_contactworld_replay_batch.py",
        project / "sim/scripts/audit_contactworld_episodes.py",
        project / "sim/scripts/contactworld_proprio_policy.py",
        project / "sim/scripts/train_contactworld_proprio_baseline.py",
        project / "sim/scripts/eval_contactworld_proprio_baseline.py",
        project / "sim/scripts/summarize_contactworld_baseline.py",
        project / "sim/scripts/render_contactworld_baseline_index.py",
        project / "sim/scripts/freeze_contactworld_runtime.py",
        project / "sim/scripts/audit_contactworld_training_contract.py",
        project / "src/openpi/spatial_dataset/contactworld.py",
        project / "sim/reports/2026-10-01/contactworld/runtime_geometry.json",
    ]
    assert all(path.is_file() for path in files)
    conda_env = runtime / "envs/contactworld-build-sm120"
    manifest = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "role": "isolated ContactWorld runtime and replay; not historical data-collection provenance",
        "source_git_commit": command("git", "-C", str(source), "rev-parse", "HEAD"),
        "pytorch_git_commit": command("git", "-C", str(runtime / "third_party/pytorch-cw-sm120"), "rev-parse", "HEAD"),
        "runtime_prefix": str(conda_env),
        "python_version": command(str(conda_env / "bin/python"), "--version"),
        "pip_freeze": command(str(conda_env / "bin/python"), "-m", "pip", "freeze").splitlines(),
        "gpu": command("nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"),
        "files": [hash_file(path) for path in files],
        "notes": ["USB uses the released manifeel_usb meshes; original patched USB meshes are different",
                  "Peg 16 mm collision OBJ and URDF now match released assets",
                  "Replay optional white-gel and white-mounts flags match pale released visuals without altering physics",
                  "Original legacy and Isaac Sim environments are separate; this script performs no installs"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.output, len(files), "hashed files")


if __name__ == "__main__":
    main()
