"""Serve one ContactWorld policy to the separate Isaac Gym Python runtime."""

from __future__ import annotations

import argparse
from multiprocessing.connection import Listener
from pathlib import Path
import time
import traceback

import numpy as np

from openpi.policies.policy_config import create_trained_policy
from openpi.training.config import get_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi0_cw_usb_01_rgb")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--num-steps", type=int, default=10, help="Diffusion sampling steps")
    args = parser.parse_args()
    if args.socket.exists():
        parser.error(f"Socket already exists: {args.socket}")
    config = get_config(args.config)
    # ContactWorld trains with the shared 32D Pi0 action/state padding, then
    # ContactWorldOutputs truncates the action back to the task's 6D or 7D
    # control vector.  The socket protocol carries the post-transform shape.
    output_action_dim = int(getattr(config.data, "action_dim", config.model.action_dim))
    policy = create_trained_policy(
        config, args.checkpoint,
        sample_kwargs={"num_steps": args.num_steps},
    )
    with Listener(str(args.socket), family="AF_UNIX", authkey=b"contactworld-rgb") as listener:
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
                        if isinstance(request, dict) and "__fixed_noise_seed__" in request:
                            request = dict(request)
                            noise_seed = int(request.pop("__fixed_noise_seed__"))
                            noise = np.random.default_rng(noise_seed).normal(
                                size=(config.model.action_horizon, config.model.action_dim),
                            ).astype(np.float32)
                            result = policy.infer(request, noise=noise)
                        else:
                            result = policy.infer(request)
                        actions = np.asarray(result["actions"], dtype=np.float32)
                        expected_shape = (config.model.action_horizon, output_action_dim)
                        if actions.shape != expected_shape or not np.isfinite(actions).all():
                            raise ValueError(f"Invalid policy actions: {actions.shape}")
                        connection.send({"actions": actions, "infer_ms": (time.monotonic() - start) * 1000})
                    except Exception:
                        connection.send({"error": traceback.format_exc()})


if __name__ == "__main__":
    main()
