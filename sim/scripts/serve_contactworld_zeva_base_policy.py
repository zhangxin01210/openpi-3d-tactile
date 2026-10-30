#!/usr/bin/env python3
"""Serve a task-specific Pi0.5 baseline export without Zeva memory."""

from __future__ import annotations

import argparse
import json
from multiprocessing.connection import Listener
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from serve_contactworld_zeva_policy import TASKS, prep_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--num-steps", type=int, default=10)
    args = parser.parse_args()
    if args.socket.exists():
        parser.error(f"Socket already exists: {args.socket}")
    root = args.root.resolve()
    runtime_root = (args.runtime_root or root).resolve()
    task_dir = root / "tasks" / args.task
    action_dim, expected_prompt, _ = TASKS[args.task]

    from pi0_zeva.runtime import configure_openpi, PromptTokenizer, make_observation
    configure_openpi(str(runtime_root / "code/openpi_src"))
    from pi0_zeva.model import build_policy, load_pretrained_backbone
    from pi0_zeva.contactworld import ContactWorldNormalizer
    from pi0_zeva import checkpoint as checkpoint_io

    config = json.loads((task_dir / "policy/config.json").read_text())
    required = {
        "pi05": True, "horizon": 16,
        "real_action_dim": action_dim, "real_state_dim": 18,
        "camera_contract": "contactworld_bilateral_v1",
    }
    mismatches = {
        key: (config.get(key), value) for key, value in required.items()
        if config.get(key) != value
    }
    if mismatches:
        parser.error(f"Export config mismatch: {mismatches}")
    if config.get("mode") not in ("baseline", "zeva"):
        parser.error(f"Unsupported source mode for base inference: {config.get('mode')!r}")

    print("LOAD Stage-1 base policy", flush=True)
    policy = build_policy(
        mode="baseline", pi05=True, horizon=16, real_action_dim=action_dim,
        backbone_dtype=config["backbone_dtype"],
    )
    if config["mode"] == "baseline":
        backbone = checkpoint_io.backbone_path(
            task_dir / "policy/checkpoints/latest.json")
    else:
        backbone = task_dir / "policy/base_backbone.safetensors"
    load_pretrained_backbone(policy, backbone)
    policy.to("cuda").eval().requires_grad_(False)
    tokenizer = PromptTokenizer(
        runtime_root / "assets/paligemma_tokenizer.model", pi05=True)
    normalizer = ContactWorldNormalizer.load(
        task_dir / "policy/norm_stats.json", action_dim)

    request_index = 0
    with Listener(str(args.socket), family="AF_UNIX", authkey=b"contactworld-ttt") as listener:
        args.socket.chmod(0o600)
        print(f"READY {args.socket}", flush=True)
        while True:
            with listener.accept() as connection:
                while True:
                    try:
                        request = connection.recv()
                    except EOFError:
                        break
                    if request == "shutdown":
                        return
                    try:
                        start = time.monotonic()
                        reset = bool(request.get("reset_zeva_state", False))
                        front = np.asarray(request["observation/front_rgb"])
                        wrist = np.asarray(request["observation/wrist_rgb"])
                        state = np.asarray(
                            request["observation/state"], dtype=np.float32)
                        if state.shape != (18,) or not np.isfinite(state).all():
                            raise ValueError(f"Invalid 18D state: {state.shape}")
                        prompt = request.get("prompt", expected_prompt)
                        if prompt != expected_prompt:
                            raise ValueError(f"Prompt mismatch: {prompt!r}")

                        front_pi, wrist_pi = prep_image(front), prep_image(wrist)
                        state18 = normalizer.normalize_state(
                            torch.from_numpy(state.copy()))
                        state32 = F.pad(state18, (0, 14))[None]
                        observation = make_observation({
                            "images": {
                                "base_0_rgb": front_pi,
                                "left_wrist_0_rgb": wrist_pi,
                                "right_wrist_0_rgb": torch.zeros_like(front_pi),
                            },
                            "image_masks": {
                                "base_0_rgb": torch.tensor([True]),
                                "left_wrist_0_rgb": torch.tensor([True]),
                                "right_wrist_0_rgb": torch.tensor([False]),
                            },
                            "state": state32,
                            "prompt": [prompt],
                        }, tokenizer, "cuda", real_state_dim=18)

                        fixed_seed = request.get("__fixed_noise_seed__")
                        if fixed_seed is None:
                            noise = torch.randn((1, 16, 32), device="cuda")
                        else:
                            values = np.random.default_rng(int(fixed_seed)).normal(
                                size=(16, 32)).astype(np.float32)
                            noise = torch.from_numpy(values)[None].to("cuda")
                        prediction = policy.sample_actions(
                            observation, noise=noise, num_steps=args.num_steps)
                        actions = normalizer.denormalize_actions(prediction)[
                            0, :, :action_dim].float().cpu().numpy()
                        if actions.shape != (16, action_dim) or not np.isfinite(actions).all():
                            raise ValueError(f"Invalid base actions: {actions.shape}")
                        connection.send({
                            "actions": actions.astype(np.float32, copy=False),
                            "infer_ms": (time.monotonic() - start) * 1000,
                            "request_index": request_index,
                            "ttt_runtime": "zeva_base",
                            "zeva_reset": reset,
                        })
                        request_index += 1
                    except Exception:
                        connection.send({"error": traceback.format_exc()})


if __name__ == "__main__":
    main()
