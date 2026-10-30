#!/usr/bin/env python3
"""Serve one 1025 ContactWorld Pi0.5+Zeva force-tactile policy."""

from __future__ import annotations

import argparse
from collections import deque
import importlib
import importlib.util
import json
from multiprocessing.connection import Listener
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from serve_contactworld_zeva_policy import TASKS, OnlineCTE, prep_image, wan_latent


def load_override(name: str, path: Path):
    """Load one 1025 increment under the package name used by the 1024 base."""
    importlib.import_module(name.rpartition(".")[0])
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load override module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ForceMemory:
    """Episode-local rolling buffer of raw bilateral 10x14x3 force grids."""

    def __init__(self, memory_steps: int = 30):
        self.memory_steps = memory_steps
        self.frames: deque[np.ndarray] = deque(maxlen=memory_steps)

    def push(self, force: np.ndarray) -> None:
        value = np.asarray(force, dtype=np.float32)
        if value.shape != (2, 10, 14, 3) or not np.isfinite(value).all():
            raise ValueError(f"Invalid bilateral tactile force frame: {value.shape}")
        self.frames.append(np.concatenate((value[0].reshape(-1), value[1].reshape(-1))))

    def push_block(self, block) -> None:
        value = np.asarray(block, dtype=np.float32)
        if value.shape != (4, 2, 10, 14, 3) or not np.isfinite(value).all():
            raise ValueError(f"Expected four tactile frames, got {value.shape}")
        for frame in value:
            self.push(frame)

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.frames:
            raise ValueError("Tactile force memory is empty")
        history = torch.from_numpy(np.stack(self.frames))
        pad = self.memory_steps - history.shape[0]
        state = F.pad(history, (0, 0, pad, 0))[None].to("cuda")
        valid = (torch.arange(self.memory_steps) >= pad)[None].to("cuda")
        return state, valid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True,
                        help="Complete 1024 package providing shared weights/assets/code")
    parser.add_argument("--increment-root", type=Path, required=True,
                        help="1025 tactile increment package")
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--num-steps", type=int, default=10)
    args = parser.parse_args()
    if args.socket.exists():
        parser.error(f"Socket already exists: {args.socket}")
    base = args.base_root.resolve()
    increment = args.increment_root.resolve()
    base_task = base / "tasks" / args.task
    tactile_task = increment / "tasks" / args.task
    action_dim, expected_prompt, task_name = TASKS[args.task]

    from pi0_zeva.runtime import configure_openpi, PromptTokenizer, make_observation
    configure_openpi(str(base / "code/openpi_src"))
    load_override(
        "cosmos_framework.model.zeva.contactworld_tactile_encoder",
        increment / "code/cosmos-framework/cosmos_framework/model/zeva/contactworld_tactile_encoder.py")
    model_module = load_override(
        "pi0_zeva.model", increment / "code/pi0_zeva/model.py")
    from pi0_zeva import checkpoint as checkpoint_io
    from pi0_zeva.contactworld import ContactWorldNormalizer
    from pi0_zeva.data import _task_context
    from cosmos_framework.zeva_training.cte_features import load_cte
    from cosmos_framework.model.zeva.demonstration_memory import demonstration_memory
    from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import (
        Wan2pt2VAEInterface,
    )

    config = json.loads((tactile_task / "policy/config.json").read_text())
    required = {
        "mode": "zeva_tactile", "pi05": True, "horizon": 16,
        "real_action_dim": action_dim, "real_state_dim": 18,
        "camera_contract": "contactworld_bilateral_v1", "pim_top_k": 4,
        "tactile_architecture": "contactworld_force_v1", "tactile_memory_steps": 30,
    }
    mismatches = {key: (config.get(key), value) for key, value in required.items()
                  if config.get(key) != value}
    if mismatches:
        parser.error(f"1025 export config mismatch: {mismatches}")

    print("LOAD tactile policy", flush=True)
    policy = model_module.build_policy(
        mode="zeva_tactile", pi05=True, horizon=16, real_action_dim=action_dim,
        prior_loss_weight=config["prior_loss_weight"],
        prior_dropout_rate=config["prior_dropout_rate"],
        prior_inference_guidance_scale=config["prior_inference_guidance_scale"],
        backbone_dtype=config["backbone_dtype"],
        tactile_architecture="contactworld_force_v1",
        tactile_force_stats=config["tactile_force_stats"],
    )
    model_module.load_pretrained_backbone(
        policy, base_task / "policy/base_backbone.safetensors")
    step_dir = checkpoint_io.resolve_checkpoint(
        tactile_task / "policy/checkpoints/latest.json")
    checkpoint_io.load_memory(policy, step_dir)
    policy.to("cuda").eval().requires_grad_(False)
    tokenizer = PromptTokenizer(base / "assets/paligemma_tokenizer.model", pi05=True)
    normalizer = ContactWorldNormalizer.load(
        base_task / "policy/norm_stats.json", action_dim)

    print("LOAD shared CTE/PIM", flush=True)
    cte, cte_cfg = load_cte(
        base_task / "cte/cte_step_001000.pt", "cuda",
        camera_contract="contactworld_bilateral_v1")
    if int(cte_cfg.action_dim) != action_dim:
        raise ValueError("CTE action dimension does not match policy")
    with np.load(base_task / "cte_features/features_000000.npz", allow_pickle=False) as data:
        support = {key: data[key].copy() for key in
                   ("phase", "effect", "effect_valid", "boundary_frame")}
    memory = demonstration_memory(support, task_name, top_k=4)
    behavior_global = _task_context(task_name)
    print("LOAD shared Wan VAE", flush=True)
    vae = Wan2pt2VAEInterface(
        vae_path=str(base / "assets/Wan2.2_VAE.pth"),
        spatial_compression_factor=16, temporal_compression_factor=4, causal=True)

    online: OnlineCTE | None = None
    force_memory: ForceMemory | None = None
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
                        current_force = np.asarray(
                            request["observation/tactile_force"], dtype=np.float32)
                        if reset:
                            online = OnlineCTE(cte, cte_cfg, action_dim)
                            force_memory = ForceMemory(30)
                            force_memory.push(current_force)
                            if request.get("executed_actions") is not None:
                                raise ValueError("Reset request cannot include executed actions")
                            if request.get("tactile_force_frames") is not None:
                                raise ValueError("Reset request cannot include tactile history block")
                        else:
                            if online is None or force_memory is None:
                                raise ValueError("First request of an episode must reset state")
                            online.push_actions(request.get("executed_actions"))
                            force_block = np.asarray(
                                request.get("tactile_force_frames"), dtype=np.float32)
                            force_memory.push_block(force_block)
                            if not np.allclose(force_block[-1], current_force, rtol=0, atol=1e-7):
                                raise ValueError("Latest tactile history frame differs from current observation")
                        assert online is not None and force_memory is not None

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
                            "state": F.pad(state18, (0, 14))[None],
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
                        tactile_state, tactile_valid = force_memory.tensors()
                        fixed_seed = request.get("__fixed_noise_seed__")
                        if fixed_seed is None:
                            noise = torch.randn((1, 16, 32), device="cuda")
                        else:
                            values = np.random.default_rng(int(fixed_seed)).normal(
                                size=(16, 32)).astype(np.float32)
                            noise = torch.from_numpy(values)[None].to("cuda")
                        prediction = policy.sample_actions(
                            observation, behavior=behavior,
                            tactile_state=tactile_state, tactile_valid=tactile_valid,
                            noise=noise, num_steps=args.num_steps)
                        actions = normalizer.denormalize_actions(prediction)[
                            0, :, :action_dim].float().cpu().numpy()
                        if actions.shape != (16, action_dim) or not np.isfinite(actions).all():
                            raise ValueError(f"Invalid tactile Zeva actions: {actions.shape}")
                        connection.send({
                            "actions": actions.astype(np.float32, copy=False),
                            "infer_ms": (time.monotonic() - start) * 1000,
                            "request_index": request_index,
                            "ttt_runtime": "zeva_tactile",
                            "zeva_reset": reset,
                            "cte_latents": len(online.latents),
                            "cte_raw_actions": len(online.actions),
                            "tactile_frames": len(force_memory.frames),
                            "tactile_valid": int(tactile_valid.sum().item()),
                        })
                        request_index += 1
                    except Exception:
                        connection.send({"error": traceback.format_exc()})


if __name__ == "__main__":
    main()
