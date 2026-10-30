#!/usr/bin/env python3
"""Serve one ContactWorld Pi0.5 RGB/state baseline to the simulator."""

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
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--num-steps", type=int, default=10)
    args = parser.parse_args()
    if args.socket.exists():
        parser.error(f"Socket already exists: {args.socket}")
    config = get_config(args.config)
    if bool(getattr(config.model, "tactile_ttt_enabled", False)):
        parser.error("Selected config is a TactileTTT model, not a baseline")
    output_action_dim = int(config.data.action_dim)
    policy = create_trained_policy(
        config,
        args.checkpoint,
        sample_kwargs={"num_steps": args.num_steps},
    )
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
                        result = policy.infer(request)
                        actions = np.asarray(result["actions"], dtype=np.float32)
                        expected = (config.model.action_horizon, output_action_dim)
                        if actions.shape != expected or not np.isfinite(actions).all():
                            raise ValueError(f"Invalid policy actions: {actions.shape}, expected {expected}")
                        connection.send({
                            "actions": actions,
                            "infer_ms": (time.monotonic() - start) * 1000,
                            "request_index": request_index,
                            "ttt_runtime": "baseline",
                        })
                        request_index += 1
                    except Exception:
                        connection.send({"error": traceback.format_exc()})


if __name__ == "__main__":
    main()
