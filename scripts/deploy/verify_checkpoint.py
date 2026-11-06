#!/usr/bin/env python3
"""Restore one checkpoint and strictly verify it against its named config."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import jax.numpy as jnp

from openpi.models import model as model_lib
from openpi.training import checkpoints
from scripts.serve_policy import resolve_checkpoint_config

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, dest="config_name")
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()

    os.environ.setdefault("OPENPI_REPO_ROOT", str(ROOT))
    checkpoint = args.checkpoint.expanduser().resolve()
    cfg = resolve_checkpoint_config(args.config_name, checkpoint)
    params = model_lib.restore_params(checkpoint, dtype=jnp.bfloat16)
    cfg.model.load(params, remove_extra_params=False)
    data = cfg.data.create(cfg.assets_dirs, cfg.model)
    if data.asset_id is None:
        raise ValueError(f"Config {cfg.name!r} does not define a normalization asset id")
    checkpoints.load_norm_stats(checkpoint, data.asset_id)
    print(f"PASS config={cfg.name} checkpoint={checkpoint}")


if __name__ == "__main__":
    main()
