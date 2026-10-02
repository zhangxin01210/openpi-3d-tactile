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
    policy = create_trained_policy(
        get_config(args.config), args.checkpoint,
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
                        result = policy.infer(request)
                        actions = np.asarray(result["actions"], dtype=np.float32)
                        if actions.shape != (16, 6) or not np.isfinite(actions).all():
                            raise ValueError(f"Invalid policy actions: {actions.shape}")
                        connection.send({"actions": actions, "infer_ms": (time.monotonic() - start) * 1000})
                    except Exception:
                        connection.send({"error": traceback.format_exc()})


if __name__ == "__main__":
    main()
