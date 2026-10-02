#!/usr/bin/env python3
"""Measure action sensitivity to stale cloud/TacFF with fixed diffusion noise.

This is an offline input intervention, not an insertion success experiment.
Use it to select cases for later closed-loop and human review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_contactworld_pi0_matrix import CONFIGS
from eval_contactworld_pi0_matrix import policy_input
import numpy as np

from openpi.policies.policy_config import create_trained_policy
from openpi.training.config import get_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("config", choices=CONFIGS)
    parser.add_argument("--checkpoints", type=Path, default=Path("/home/sai/zsq/FactileLDM/1002_ckpt"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[10013, 10021, 10025, 10030, 10043])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checkpoint = args.checkpoints / args.config / "cw_usb_4way/19999"
    policy = create_trained_policy(get_config(args.config), checkpoint, sample_kwargs={"num_steps": 10})
    noise = np.random.default_rng(4242).standard_normal((16, 32)).astype(np.float32)
    cloud_enabled, force_enabled = CONFIGS[args.config]
    results = []
    for seed in args.seeds:
        path = args.root / args.config / f"seed_{seed}" / "inputs.npz"
        trajectory = json.loads((path.parent / "trajectory.json").read_text())["rows"]
        with np.load(path) as archive:
            data = {key: archive[key] for key in archive.files}
        contact = [r["step"] for r in trajectory if
                   np.linalg.norm(r["plug_socket_contact_force_xyz"]) > 0.1]
        picks = {0, len(data["steps"]) // 2, len(data["steps"]) - 1}
        if contact:
            picks.add(int(np.argmin(abs(data["steps"] - contact[0]))))
        for index in sorted(picks):
            snap = {"front": data["front"][index], "wrist": data["wrist"][index],
                    "state": data["state"][index], "cloud_xyz": data["cloud_xyz"][index],
                    "force_grid": data["force_grid"][index]}
            original = policy_input(snap, cloud=cloud_enabled, force=force_enabled)
            baseline = np.asarray(policy.infer(original, noise=noise)["actions"][:4], dtype=np.float32)
            again = np.asarray(policy.infer(original, noise=noise)["actions"][:4], dtype=np.float32)
            if not np.allclose(baseline, again, rtol=1e-5, atol=1e-5):
                raise ValueError("Fixed-noise inference is not repeatable")
            row = {"seed": seed, "step": int(data["steps"][index]), "config": args.config,
                   "baseline_first_action": baseline[0].tolist()}
            for modality, enabled, key in (("cloud", cloud_enabled, "cloud_xyz"),
                                           ("force", force_enabled, "force_grid")):
                if not enabled:
                    continue
                altered = dict(snap)
                altered[key] = data[key][0]
                request = policy_input(altered, cloud=cloud_enabled, force=force_enabled)
                changed = np.asarray(policy.infer(request, noise=noise)["actions"][:4], dtype=np.float32)
                row[f"{modality}_stale_action_first"] = changed[0].tolist()
                row[f"{modality}_stale_first4_l2"] = float(np.linalg.norm(changed - baseline))
                row[f"{modality}_stale_first4_meanabs"] = float(np.abs(changed - baseline).mean())
            results.append(row)
            print(seed, row["step"], {k: round(v, 4) for k, v in row.items() if k.endswith("meanabs")}, flush=True)
    output = args.output or args.root / f"sensitivity_{args.config}.json"
    output.write_text(json.dumps({"checkpoint": str(checkpoint), "diffusion_noise_seed": 4242,
                                  "intervention": "replace one live modality with its first-frame value",
                                  "warning": "Offline action sensitivity only; stale values can be out of distribution",
                                  "rows": results}, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
