from __future__ import annotations

import argparse
import dataclasses
import math

import flax
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.models import gemma as _gemma
from openpi.models import pi0 as _pi0
from openpi.models import model as _model
from openpi.models import spatial_pi0_config
from openpi.training import config as _config
from openpi.training import data_loader
from openpi.training import optimizer as _optimizer
from openpi.training import spatial_weight_loaders


_CONFIG_NAMES = (
    "pi0_xhand_spatial_joint_pointnet_prefix",
    "pi0_xhand_spatial_joint_pointnet_suffix",
    "pi0_xhand_spatial_joint_pointnet_both",
    "pi0_xhand_spatial_structured_prefix",
    "pi0_xhand_spatial_structured_suffix",
    "pi0_xhand_spatial_structured_both",
)


def _flatten_state(state) -> dict[str, object]:
    flat = flax.traverse_util.flatten_dict(
        state.to_pure_dict(),
    )
    return {"/".join(str(part) for part in path): value for path, value in flat.items()}


def _summarize_paths(paths: list[str], *, limit: int = 20) -> str:
    shown = "\n".join(f"  - {path}" for path in paths[:limit])
    if len(paths) > limit:
        shown += f"\n  ... and {len(paths) - limit} more"
    return shown


def _global_norm_for_prefix(flat_tree: dict[str, object], prefix: str) -> float:
    leaves = [
        value
        for path, value in flat_tree.items()
        if path.startswith(prefix)
    ]
    if not leaves:
        return 0.0
    return float(optax.global_norm(leaves))


def _max_delta_for_prefix(
    before: dict[str, object],
    after: dict[str, object],
    *,
    prefix: str,
) -> tuple[str, float]:
    best_path = None
    best_delta = 0.0

    for path, before_value in before.items():
        if not path.startswith(prefix):
            continue
        if path not in after:
            raise AssertionError(f"{path} disappeared after optimizer update")

        delta = float(
            np.max(
                np.abs(
                    np.asarray(after[path])
                    - np.asarray(before_value)
                )
            )
        )

        if best_path is None or delta > best_delta:
            best_path = path
            best_delta = delta

    if best_path is None:
        raise AssertionError(f"No parameter found with prefix {prefix!r}")

    return best_path, best_delta


def _assert_lora_training_contract(
    *,
    flat_params: dict[str, object],
    frozen_params: dict[str, object],
    trainable_params: dict[str, object],
) -> None:
    lora_paths = sorted(path for path in flat_params if "lora" in path)
    if not lora_paths:
        raise AssertionError("No LoRA parameters found")

    trainable_lora = sorted(path for path in trainable_params if "lora" in path)
    if not trainable_lora:
        raise AssertionError("No trainable LoRA parameters found")

    frozen_lora = sorted(path for path in frozen_params if "lora" in path)
    if frozen_lora:
        print("frozen LoRA params:\n" + _summarize_paths(frozen_lora, limit=30))
        raise AssertionError("LoRA parameters are unexpectedly frozen")

    frozen_dense_llm = sorted(
        path
        for path in frozen_params
        if "llm" in path and "lora" not in path
    )
    if not frozen_dense_llm:
        raise AssertionError("No pretrained dense LLM parameters are frozen")

    print(
        "LoRA audit: "
        f"lora_total={len(lora_paths)} "
        f"lora_trainable={len(trainable_lora)} "
        f"dense_llm_frozen={len(frozen_dense_llm)}"
    )


def _assert_finite(name: str, value) -> None:
    array = np.asarray(value)
    if not np.all(np.isfinite(array)):
        raise AssertionError(f"{name} is not finite")


def _make_batch(train_config: _config.TrainConfig):
    tiny_config = dataclasses.replace(
        train_config,
        batch_size=1,
        num_workers=0,
    )
    loader = data_loader.create_data_loader(
        tiny_config,
        shuffle=False,
        num_batches=1,
        skip_norm_stats=False,
        framework="jax",
    )
    return next(iter(loader))


def _inspect_sequences(model, observation: _model.Observation, actions: _model.Actions) -> None:
    conditioned = model._encode_spatial_conditioning(observation)
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(
        observation,
        conditioned,
    )
    timestep = jnp.full((actions.shape[0],), 0.5, dtype=jnp.float32)
    suffix_tokens, suffix_mask, suffix_ar_mask, _ = model.embed_suffix(
        observation,
        actions,
        timestep,
        conditioned,
    )
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask = _pi0.make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1

    print(f"prefix_tokens={prefix_tokens.shape} prefix_mask={prefix_mask.shape} prefix_ar={prefix_ar_mask.shape}")
    print(f"suffix_tokens={suffix_tokens.shape} suffix_mask={suffix_mask.shape} suffix_ar={suffix_ar_mask.shape}")
    print(f"attn_mask={attn_mask.shape} ar_mask={ar_mask.shape} positions={positions.shape}")
    _assert_finite("positions", positions)


def run_config(config_name: str, *, run_loss: bool) -> None:
    print(f"\n===== {config_name} =====")
    train_config = _config.get_config(config_name)
    if not isinstance(train_config.model, spatial_pi0_config.SpatialPi0Config):
        raise AssertionError(f"{config_name} is not a spatial Pi0 config")

    data_config = train_config.data.create(
        train_config.assets_dirs,
        train_config.model,
    )
    if data_config.norm_stats is None:
        raise AssertionError(f"{config_name} did not load norm stats")
    if set(data_config.norm_stats) != {"state", "actions"}:
        raise AssertionError(f"Unexpected norm stats keys: {sorted(data_config.norm_stats)!r}")

    paligemma_width = _gemma.get_config(train_config.model.paligemma_variant).width
    action_width = _gemma.get_config(train_config.model.action_expert_variant).width
    print(
        f"target={train_config.model.conditioning.target.value} "
        f"paligemma_width={paligemma_width} action_expert_width={action_width}"
    )

    observation, actions = _make_batch(train_config)
    print(f"batch state={observation.state.shape} actions={actions.shape}")
    print(
        "spatial visual="
        f"{observation.spatial.visual.xyz_m.shape if observation.spatial and observation.spatial.visual else None} "
        "tactile="
        f"{observation.spatial.tactile.xyz_m.shape if observation.spatial and observation.spatial.tactile else None}"
    )
    _assert_finite("normalized state", observation.state)
    _assert_finite("normalized actions", actions)

    model = train_config.model.create(jax.random.key(0))
    param_state = nnx.state(model, nnx.Param)
    flat_params = _flatten_state(param_state)
    spatial_paths = sorted(path for path in flat_params if path.startswith("spatial_router/"))
    if not spatial_paths:
        raise AssertionError("spatial_router params are missing from nnx.state(..., nnx.Param)")
    spatial_param_count = sum(np.asarray(flat_params[path]).size for path in spatial_paths)
    print(f"spatial_param_paths={len(spatial_paths)} spatial_param_count={spatial_param_count}")
    print(_summarize_paths(spatial_paths, limit=12))

    if not isinstance(train_config.weight_loader, spatial_weight_loaders.SpatialCheckpointWeightLoader):
        raise AssertionError("Expected SpatialCheckpointWeightLoader")

    audit = train_config.weight_loader.audit(param_state.to_pure_dict())
    print(
        "checkpoint audit: "
        f"loaded={len(audit.loaded)} "
        f"allowed_missing={len(audit.allowed_missing)} "
        f"unexpected_missing={len(audit.unexpected_missing)} "
        f"unexpected_loaded={len(audit.unexpected_loaded)} "
        f"shape_mismatch={len(audit.shape_mismatches)}"
    )
    if audit.unexpected_missing or audit.shape_mismatches:
        print("unexpected missing:\n" + _summarize_paths(list(audit.unexpected_missing), limit=30))
        print("shape mismatches:\n" + _summarize_paths(list(audit.shape_mismatches), limit=30))
        raise AssertionError("Checkpoint audit failed")
    if not all(path.startswith("spatial_router/") or "lora" in path for path in audit.allowed_missing):
        raise AssertionError("Allowed missing parameters are not limited to spatial/lora")

    loaded_params = train_config.weight_loader.load(param_state.to_pure_dict())
    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(loaded_params)
    model = nnx.merge(graphdef, state)
    print("checkpoint merge: PASS")

    _inspect_sequences(model, observation, actions)

    if not run_loss:
        print("FULL_PI0_INIT_PASS")
        return

    model.train()

    def loss_fn(model, rng, obs, target_actions):
        losses = model.compute_loss(
            rng,
            obs,
            target_actions,
            train=True,
        )
        return jnp.mean(losses)

    diff_state = nnx.DiffState(0, train_config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
        model,
        jax.random.key(1),
        observation,
        actions,
    )
    loss_value = float(loss)
    if not math.isfinite(loss_value):
        raise AssertionError(f"loss is not finite: {loss_value}")

    flat_grads = _flatten_state(grads)
    encoder_grad = _global_norm_for_prefix(flat_grads, "spatial_router/encoder/")
    prefix_grad = _global_norm_for_prefix(flat_grads, "spatial_router/prefix_adapter/")
    suffix_grad = _global_norm_for_prefix(flat_grads, "spatial_router/suffix_adapter/")
    print(
        f"loss={loss_value:.6f} "
        f"spatial_encoder_grad_norm={encoder_grad:.6e} "
        f"prefix_grad_norm={prefix_grad:.6e} "
        f"suffix_grad_norm={suffix_grad:.6e}"
    )
    if encoder_grad <= 0.0:
        raise AssertionError("spatial encoder grad norm is zero")
    if train_config.model.conditioning.use_prefix and prefix_grad <= 0.0:
        raise AssertionError("prefix projection grad norm is zero")
    if train_config.model.conditioning.use_suffix and suffix_grad <= 0.0:
        raise AssertionError("suffix projection grad norm is zero")

    tx = _optimizer.create_optimizer(
        train_config.optimizer,
        train_config.lr_schedule,
        weight_decay_mask=None,
    )
    full_state = nnx.state(model)
    trainable_params = full_state.filter(train_config.trainable_filter)
    trainable_flat = _flatten_state(trainable_params)
    frozen_params = _flatten_state(full_state.filter(train_config.freeze_filter))
    _assert_lora_training_contract(
        flat_params=_flatten_state(nnx.state(model, nnx.Param)),
        frozen_params=frozen_params,
        trainable_params=trainable_flat,
    )

    params_before = _flatten_state(nnx.state(model, nnx.Param))
    opt_state = tx.init(trainable_params)
    updates, _ = tx.update(grads, opt_state, trainable_params)
    new_params = optax.apply_updates(trainable_params, updates)
    nnx.update(model, new_params)

    params_after = _flatten_state(nnx.state(model, nnx.Param))
    delta_path, delta = _max_delta_for_prefix(
        params_before,
        params_after,
        prefix="spatial_router/",
    )
    print(f"optimizer_update path={delta_path} max_abs_delta={delta:.6e}")
    if delta <= 0.0:
        raise AssertionError("no spatial parameter updated")

    print(
        f"freeze audit: frozen_paths={len(frozen_params)} "
        f"trainable_paths={len(trainable_flat)} "
        f"spatial_trainable_paths={len(spatial_paths)}"
    )
    if any(path.startswith("spatial_router/") for path in frozen_params):
        raise AssertionError("spatial_router is unexpectedly frozen")

    print("FULL_PI0_FORWARD_GRAD_UPDATE_PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", choices=(*_CONFIG_NAMES, "all"), default="all")
    parser.add_argument("--skip-loss", action="store_true")
    args = parser.parse_args()

    configs = _CONFIG_NAMES if args.config_name == "all" else (args.config_name,)
    for config_name in configs:
        run_config(config_name, run_loss=not args.skip_loss)


if __name__ == "__main__":
    main()
