from __future__ import annotations

import argparse
import dataclasses
import math

from flax import nnx
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.training import config as _config
from openpi.training import data_loader
from openpi.training import optimizer as _optimizer


def _flatten_state(state) -> dict[str, object]:
    flat = flax.traverse_util.flatten_dict(
        state.to_pure_dict(),
    )
    return {"/".join(str(part) for part in path): value for path, value in flat.items()}


def _spatial_param_snapshot(model) -> tuple[str, np.ndarray]:
    flat = _flatten_state(nnx.state(model, nnx.Param))
    for path, value in flat.items():
        if path.startswith("spatial_router/"):
            return path, np.asarray(value).copy()
    raise AssertionError("No spatial parameter found")


def _load_model(train_config: _config.TrainConfig):
    model = train_config.model.create(jax.random.key(0))
    params = nnx.state(model, nnx.Param)
    loaded = train_config.weight_loader.load(params.to_pure_dict())
    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(loaded)
    return nnx.merge(graphdef, state)


def _make_loader(train_config: _config.TrainConfig, *, steps: int):
    tiny_config = dataclasses.replace(
        train_config,
        batch_size=1,
        num_workers=0,
    )
    return data_loader.create_data_loader(
        tiny_config,
        shuffle=True,
        num_batches=steps,
        skip_norm_stats=False,
        framework="jax",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="pi0_xhand_spatial_joint_pointnet_suffix")
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()

    train_config = _config.get_config(args.config_name)
    model = _load_model(train_config)
    model.train()

    tx = _optimizer.create_optimizer(
        train_config.optimizer,
        train_config.lr_schedule,
        weight_decay_mask=None,
    )
    full_state = nnx.state(model)
    trainable_params = full_state.filter(train_config.trainable_filter)
    opt_state = tx.init(trainable_params)

    first_path, first_before = _spatial_param_snapshot(model)
    loader = _make_loader(train_config, steps=args.steps)

    def loss_fn(model, rng, observation, actions):
        losses = model.compute_loss(
            rng,
            observation,
            actions,
            train=True,
        )
        return jnp.mean(losses)

    diff_state = nnx.DiffState(0, train_config.trainable_filter)
    rng = jax.random.key(123)
    losses = []

    for step, batch in enumerate(loader):
        rng, step_rng = jax.random.split(rng)
        observation, actions = batch
        loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
            model,
            step_rng,
            observation,
            actions,
        )
        loss_value = float(loss)
        grad_norm = float(optax.global_norm(grads))
        if not math.isfinite(loss_value):
            raise AssertionError(f"loss is not finite at step {step}: {loss_value}")
        if not math.isfinite(grad_norm):
            raise AssertionError(f"grad norm is not finite at step {step}: {grad_norm}")

        params = nnx.state(model).filter(train_config.trainable_filter)
        updates, opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        nnx.update(model, new_params)
        losses.append(loss_value)
        print(f"step={step + 1} loss={loss_value:.6f} grad_norm={grad_norm:.6e}")

    flat_after = _flatten_state(nnx.state(model, nnx.Param))
    delta = float(np.max(np.abs(np.asarray(flat_after[first_path]) - first_before)))
    if delta <= 0.0:
        raise AssertionError("spatial parameter did not change after training")

    print(
        f"TRAIN_SMOKE_PASS steps={args.steps} "
        f"loss_first={losses[0]:.6f} loss_last={losses[-1]:.6f} "
        f"spatial_update_path={first_path} max_abs_delta={delta:.6e}"
    )


if __name__ == "__main__":
    main()
