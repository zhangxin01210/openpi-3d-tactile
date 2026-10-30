#!/usr/bin/env python3
"""Serve one exported ContactWorld Pi0.5+Zeva policy to the simulator."""

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


TASKS = {
    "peg": (6, "Insert the peg into the matching hole.", "contactworld_peg_v3_bilateral"),
    "usb": (6, "Insert the USB plug into the socket.", "contactworld_usb_v3_bilateral"),
    "barbed_flat": (7, "Remove the flat barbed plug from its socket.",
                    "contactworld_barbed_flat_v3_bilateral"),
    "valve": (7, "Turn the valve to complete the screwing task.",
              "contactworld_valve_v3_bilateral"),
}


def prep_image(value: np.ndarray) -> torch.Tensor:
    value = np.asarray(value)
    if value.shape != (256, 256, 3) or value.dtype != np.uint8:
        raise ValueError(f"Expected uint8 HWC RGB image, got {value.shape} {value.dtype}")
    image = torch.from_numpy(value.copy()).permute(2, 0, 1)[None].float() / 255.0
    image = F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
    return image.clamp(0, 1) * 2 - 1


class OnlineCTE:
    """Episode-local causal feature state at one latent per four controls."""

    def __init__(self, model, cfg, action_dim: int, window_latents: int = 9):
        self.model = model
        self.cfg = cfg
        self.action_dim = action_dim
        self.window_latents = window_latents
        self.latents: list[torch.Tensor] = []
        self.actions: list[np.ndarray] = []

    def push_actions(self, actions) -> None:
        block = np.asarray(actions, dtype=np.float32)
        if block.shape != (4, self.action_dim) or not np.isfinite(block).all():
            raise ValueError(
                f"Expected one raw four-action transition block, got {block.shape}")
        self.actions.extend(row.copy() for row in block)

    @torch.no_grad()
    def query(self, latent: torch.Tensor):
        expected = (48, 30, 52)
        if tuple(latent.shape) != expected or not torch.isfinite(latent).all():
            raise ValueError(f"Invalid Wan latent {tuple(latent.shape)}, expected {expected}")
        self.latents.append(latent)
        self.latents = self.latents[-self.window_latents:]
        count = len(self.latents)
        frames = torch.stack(self.latents, dim=0).unsqueeze(0)
        if count > 1:
            needed = (count - 1) * 4
            if len(self.actions) < needed:
                raise ValueError("Online CTE is missing executed transition actions")
            transitions = torch.from_numpy(np.stack(self.actions[-needed:])).reshape(
                1, count - 1, 4, self.action_dim).to("cuda")
        else:
            transitions = torch.zeros(
                (1, 0, 4, self.action_dim), dtype=torch.float32, device="cuda")
        valid = torch.ones((1, count), dtype=torch.bool, device="cuda")
        transition_valid = torch.ones(
            transitions.shape[:-1], dtype=torch.bool, device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = self.model(frames, transitions, valid, transition_valid)
        phase = output["phase"][0, -1].float()
        completed = output["effect_post"][0][output["effect_complete"][0]].float()
        effect = torch.zeros((4, 128), dtype=torch.float32, device="cuda")
        effect_valid = torch.zeros(4, dtype=torch.bool, device="cuda")
        take = min(4, completed.shape[0])
        if take:
            effect[-take:] = completed[-take:]
            effect_valid[-take:] = True
        return phase, effect, effect_valid


@torch.no_grad()
def wan_latent(vae, front: np.ndarray, wrist: np.ndarray) -> torch.Tensor:
    views = []
    for value in (front, wrist):
        image = torch.from_numpy(np.asarray(value).copy()).permute(2, 0, 1)[None].float()
        views.append(F.interpolate(
            image, size=(256, 256), mode="bilinear", align_corners=False)[0])
    composite = torch.cat(views, dim=2)
    video = F.interpolate(
        composite[None], size=(480, 832), mode="bilinear", align_corners=False)
    video = (video / 127.5 - 1.0).unsqueeze(2).to("cuda")
    return vae.encode(video)[0, :, 0].float()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--num-steps", type=int, default=10)
    args = parser.parse_args()
    if args.socket.exists():
        parser.error(f"Socket already exists: {args.socket}")
    root = args.root.resolve()
    task_dir = root / "tasks" / args.task
    action_dim, expected_prompt, task_name = TASKS[args.task]

    from pi0_zeva.runtime import configure_openpi, PromptTokenizer, make_observation
    configure_openpi(str(root / "code/openpi_src"))
    from pi0_zeva.model import build_policy, load_pretrained_backbone
    from pi0_zeva import checkpoint as checkpoint_io
    from pi0_zeva.contactworld import ContactWorldNormalizer
    from pi0_zeva.data import _task_context
    from cosmos_framework.zeva_training.cte_features import load_cte
    from cosmos_framework.model.zeva.demonstration_memory import demonstration_memory
    from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import (
        Wan2pt2VAEInterface,
    )

    config = json.loads((task_dir / "policy/config.json").read_text())
    required = {
        "mode": "zeva", "pi05": True, "horizon": 16,
        "real_action_dim": action_dim, "real_state_dim": 18,
        "camera_contract": "contactworld_bilateral_v1", "pim_top_k": 4,
    }
    mismatches = {key: (config.get(key), value) for key, value in required.items()
                  if config.get(key) != value}
    if mismatches:
        parser.error(f"Export config mismatch: {mismatches}")

    print("LOAD policy", flush=True)
    policy = build_policy(
        mode="zeva", pi05=True, horizon=16, real_action_dim=action_dim,
        prior_loss_weight=config["prior_loss_weight"],
        prior_dropout_rate=config["prior_dropout_rate"],
        prior_inference_guidance_scale=config["prior_inference_guidance_scale"],
        backbone_dtype=config["backbone_dtype"],
    )
    load_pretrained_backbone(policy, task_dir / "policy/base_backbone.safetensors")
    step_dir = checkpoint_io.resolve_checkpoint(
        task_dir / "policy/checkpoints/latest.json")
    checkpoint_io.load_memory(policy, step_dir)
    policy.to("cuda").eval().requires_grad_(False)
    tokenizer = PromptTokenizer(root / "assets/paligemma_tokenizer.model", pi05=True)
    normalizer = ContactWorldNormalizer.load(
        task_dir / "policy/norm_stats.json", action_dim)

    print("LOAD CTE", flush=True)
    cte, cte_cfg = load_cte(
        task_dir / "cte/cte_step_001000.pt", "cuda",
        camera_contract="contactworld_bilateral_v1")
    if int(cte_cfg.action_dim) != action_dim:
        raise ValueError("CTE action dimension does not match policy")
    print("LOAD PIM", flush=True)
    with np.load(task_dir / "cte_features/features_000000.npz", allow_pickle=False) as data:
        support = {key: data[key].copy() for key in
                   ("phase", "effect", "effect_valid", "boundary_frame")}
    memory = demonstration_memory(support, task_name, top_k=4)
    behavior_global = _task_context(task_name)
    print("LOAD Wan VAE", flush=True)
    vae = Wan2pt2VAEInterface(
        vae_path=str(root / "assets/Wan2.2_VAE.pth"),
        spatial_compression_factor=16, temporal_compression_factor=4, causal=True)

    online: OnlineCTE | None = None
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
                        if reset:
                            online = OnlineCTE(cte, cte_cfg, action_dim)
                        if online is None:
                            raise ValueError("First request of an episode must reset Zeva state")
                        executed = request.get("executed_actions")
                        if reset:
                            if executed is not None and np.asarray(executed).size:
                                raise ValueError("Reset request cannot include executed actions")
                        else:
                            online.push_actions(executed)

                        front = np.asarray(request["observation/front_rgb"])
                        wrist = np.asarray(request["observation/wrist_rgb"])
                        state = np.asarray(request["observation/state"], dtype=np.float32)
                        if state.shape != (18,) or not np.isfinite(state).all():
                            raise ValueError(f"Invalid 18D state: {state.shape}")
                        prompt = request.get("prompt", expected_prompt)
                        if prompt != expected_prompt:
                            raise ValueError(f"Prompt mismatch: {prompt!r}")

                        front_pi, wrist_pi = prep_image(front), prep_image(wrist)
                        state18 = normalizer.normalize_state(torch.from_numpy(state.copy()))
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

                        phase, effect, effect_valid = online.query(
                            wan_latent(vae, front, wrist))
                        pim_phase, pim_effect, pim_valid, _ = memory.query_tensors(
                            phase, top_k=4)
                        behavior = {
                            "global": behavior_global.unsqueeze(0),
                            "phase": phase.unsqueeze(0),
                            "effect": effect.unsqueeze(0),
                            "effect_valid": effect_valid.unsqueeze(0),
                            "pim_phase": pim_phase.unsqueeze(0),
                            "pim_effect": pim_effect.unsqueeze(0),
                            "pim_valid": pim_valid.unsqueeze(0),
                        }
                        fixed_seed = request.get("__fixed_noise_seed__")
                        if fixed_seed is None:
                            noise = torch.randn((1, 16, 32), device="cuda")
                        else:
                            values = np.random.default_rng(int(fixed_seed)).normal(
                                size=(16, 32)).astype(np.float32)
                            noise = torch.from_numpy(values)[None].to("cuda")
                        prediction = policy.sample_actions(
                            observation, behavior=behavior, noise=noise,
                            num_steps=args.num_steps)
                        actions = normalizer.denormalize_actions(prediction)[
                            0, :, :action_dim].float().cpu().numpy()
                        if actions.shape != (16, action_dim) or not np.isfinite(actions).all():
                            raise ValueError(f"Invalid Zeva actions: {actions.shape}")
                        connection.send({
                            "actions": actions.astype(np.float32, copy=False),
                            "infer_ms": (time.monotonic() - start) * 1000,
                            "request_index": request_index,
                            "ttt_runtime": "zeva",
                            "zeva_reset": reset,
                            "cte_latents": len(online.latents),
                            "cte_raw_actions": len(online.actions),
                        })
                        request_index += 1
                    except Exception:
                        connection.send({"error": traceback.format_exc()})


if __name__ == "__main__":
    main()
