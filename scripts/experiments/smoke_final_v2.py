#!/usr/bin/env python3
"""Run read-only preflight, then real-weight model or isolated short training smoke."""
from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from preflight_final_v2 import inspect
from preflight_final_v2 import names


def model_check(cfg):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from flax import nnx
    import jax
    import jax.numpy as jnp

    from openpi.training import data_loader
    from openpi.training import sharding
    import scripts.train as train

    if cfg.batch_size % jax.device_count():
        raise ValueError(f"batch_size={cfg.batch_size} must be divisible by device count {jax.device_count()}")
    mesh = sharding.make_mesh(cfg.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    batch = next(iter(data_loader.create_data_loader(cfg, sharding=data_sharding, shuffle=False)))
    state, _ = train.init_train_state(cfg, jax.random.key(cfg.seed), mesh, resume=False)
    model = nnx.merge(state.model_def, state.params)
    with sharding.set_mesh(mesh):
        if hasattr(cfg.model, "encoder"):
            encoded = model.spatial_router.encoder(batch[0].spatial)
            routed = model._encode_spatial_conditioning(batch[0])  # noqa: SLF001
            expected = cfg.model.encoder.default_token_count
            if encoded.tokens.shape[1:] != (expected, 128) or routed.suffix_tokens.shape[1] != expected:
                raise ValueError("encoder or suffix adapter token shape mismatch")
            if encoded.token_mask.shape != encoded.tokens.shape[:2] or routed.suffix_mask.shape != encoded.token_mask.shape:
                raise ValueError("encoder or adapter token mask shape mismatch")
            if not bool(jnp.isfinite(encoded.tokens).all() & jnp.isfinite(routed.suffix_tokens).all()):
                raise ValueError("nonfinite encoder/adapter output")
            print(f"[SPATIAL OK] {encoded.tokens.shape}; suffix={routed.suffix_tokens.shape}", flush=True)
        loss = model.compute_loss(jax.random.key(cfg.seed), *batch, train=False)
        if not bool(jnp.isfinite(loss).all()):
            raise ValueError("nonfinite model loss")
        print(f"[MODEL OK] loss shape={loss.shape}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", choices=names())
    group.add_argument("--all-configs", action="store_true")
    parser.add_argument("--check", choices=("data", "model", "train", "all"), default="all")
    parser.add_argument("--steps", type=int, choices=range(2, 6), default=3)
    parser.add_argument("--checkpoint-base-dir", type=Path)
    args = parser.parse_args()
    selected = names() if args.all_configs else [args.config]
    failures = []
    for name in selected:
        report = inspect(name)
        if not report.get("ready"):
            failures.append(name)
            print(f"[PREFLIGHT FAILED] {name}: {report['errors']}", flush=True)
            continue
        print(f"[DATA OK] {name}", flush=True)
        if args.check == "data":
            continue
        try:
            from openpi.training.config import get_config
            cfg = dataclasses.replace(get_config(name), exp_name="final_v2_smoke")
            if args.check == "model":
                model_check(cfg)
                continue
            # The ordinary trainer loads the real base once, runs encoder/router
            # checks, loss, backward and optimizer, and saves isolated checkpoints.
            out = args.checkpoint_base_dir or Path(tempfile.mkdtemp(prefix="openpi-final-v2-smoke-"))
            env = dict(os.environ, OPENPI_FINAL_V2_SMOKE_CHECKS="1")
            command = [sys.executable, "scripts/train.py", name, "--exp-name", "final_v2_smoke",
                       "--checkpoint-base-dir", str(out), "--num-train-steps", str(args.steps),
                       "--save-interval", "1", "--no-wandb-enabled"]
            print(f"[TRAIN SMOKE] checkpoint root={out}", flush=True)
            subprocess.run(command, env=env, check=True)
            print(f"[TRAIN/CHECKPOINT OK] {name}: {out / name / 'final_v2_smoke'}", flush=True)
        except Exception as exc:
            failures.append(name)
            print(f"[SMOKE FAILED] {name}: {type(exc).__name__}: {exc}", flush=True)
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
