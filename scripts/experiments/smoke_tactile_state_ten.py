#!/usr/bin/env python3
# ruff: noqa: SLF001
"""One real-data optimizer step, no checkpoint or W&B writes; run on an idle GPU."""
import argparse
import dataclasses
import json
from pathlib import Path
import runpy

from flax import nnx
import jax
import numpy as np

from openpi import transforms
from openpi.training import config
from openpi.training import data_loader
from openpi.training import sharding


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config_name")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--audit-only", action="store_true")
    args = ap.parse_args()
    cfg = dataclasses.replace(config.get_config(args.config_name), num_workers=0)
    if not args.audit_only and (jax.default_backend() != "gpu" or len(jax.devices()) != 1):
        raise RuntimeError("smoke test requires exactly one visible GPU")
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    raw = data_loader.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
    if dc.norm_stats is None:
        raise ValueError("task norm stats are missing")
    expected_dim = 1818 if cfg.data.raw_tactile_state else 18
    for name, size in (("state", expected_dim), ("actions", 18)):
        stats = dc.norm_stats[name]
        if np.asarray(stats.mean).shape != (size,) or np.asarray(stats.std).shape != (size,):
            raise ValueError(f"incorrect {name} normalization shape")
        if cfg.model.pi05 and (stats.q01 is None or stats.q99 is None):
            raise ValueError("pi05 requires quantile normalization")
        for item in (stats.mean, stats.std, stats.q01, stats.q99):
            if item is not None and not np.isfinite(item).all():
                raise ValueError("non-finite normalization statistics")
    # Samples cover the full dataset index range; record contact classes and
    # episodes rather than claiming an unobserved class was tested.
    samples, contact_types, episodes = [], set(), set()
    for index in np.unique(np.linspace(0, len(raw)-1, min(64, len(raw)), dtype=int)):
        sample = raw[int(index)]
        source = np.asarray(sample["observation.state"], dtype=np.float32)
        if source.shape != (1972,):
            raise ValueError(f"raw observation state must be 1972, got {source.shape}")
        from openpi.policies.xhand_raw_tactile import extract_raw_tactile_state
        tactile = extract_raw_tactile_state(source)[18:]
        contact_types.add("no_contact" if not np.any(tactile) else "nonzero")
        episodes.add(int(np.asarray(sample["episode_index"]).item()))
        prepared = transforms.compose([*dc.repack_transforms.inputs, *dc.data_transforms.inputs,
            transforms.Normalize(dc.norm_stats, use_quantiles=dc.use_quantile_norm)])(sample)
        state = np.asarray(prepared["state"])
        if state.shape != (expected_dim,) or not np.isfinite(state).all():
            raise ValueError(f"bad normalized state: {state.shape}")
        record = {"index": int(index), "episode": int(np.asarray(sample["episode_index"]).item()),
                  "raw_state_dim": 1972, "tactile_dim": 1800 if cfg.data.raw_tactile_state else 0,
                  "state_dim": expected_dim, "tactile_max_abs": float(np.max(np.abs(tactile)))}
        if cfg.model.pi05:
            tok = next(x.tokenizer for x in dc.model_transforms.inputs if isinstance(x, transforms.TokenizePrompt))
            text = str(prepared["prompt"]).strip().replace("_", " ").replace("\n", " ")
            discrete = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
            prompt = f"Task: {text}, State: {' '.join(map(str, discrete))};\nAction: "
            length = len(tok._tokenizer.encode(prompt, add_bos=True))
            encoded, mask = tok.tokenize(str(prepared["prompt"]), state)
            record.update(prompt_length_before_padding=length, max_token_len=cfg.model.max_token_len,
                          truncated=length > cfg.model.max_token_len, state_values_encoded=len(discrete),
                          action_marker_preserved=tok._tokenizer.decode(encoded[mask].tolist()).endswith(";\nAction: "))
            if record["truncated"] or not record["action_marker_preserved"]:
                raise ValueError("state prompt was truncated or lost Action marker")
        samples.append(record)
    if cfg.data.raw_tactile_state and (len(episodes) < 2 or contact_types != {"no_contact", "nonzero"}):
        raise ValueError(f"representative sample coverage incomplete: episodes={len(episodes)} contacts={contact_types}")
    if args.audit_only:
        result = {"status": "audited", "config": cfg.name, "samples": samples,
                  "contact_types": sorted(contact_types), "episodes_sampled": len(episodes)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"status": "audited", "config": cfg.name, "samples": len(samples)}), flush=True)
        return
    mesh = sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    loader = data_loader.create_data_loader(cfg, sharding=data_sharding, num_batches=1, shuffle=False)
    batch = next(iter(loader))
    if batch[0].state.shape[-1] != (1818 if cfg.data.raw_tactile_state else 32):
        raise ValueError(f"model received wrong state shape {batch[0].state.shape}")
    if batch[1].shape[-1] != 32:
        raise ValueError("model action width changed")
    train = runpy.run_path(str(Path(__file__).resolve().parents[1] / "train.py"))
    rng = jax.random.key(cfg.seed)
    state, state_sharding = train["init_train_state"](cfg, rng, mesh, resume=False)
    model = nnx.merge(state.model_def, state.params)
    prefix = jax.eval_shape(lambda obs: model.embed_prefix(obs), batch[0])[0]
    image_token_count = prefix.shape[1] - cfg.model.max_token_len
    suffix_count = cfg.model.action_horizon + (0 if cfg.model.pi05 else 1)
    context_tokens = prefix.shape[1] + suffix_count
    with sharding.set_mesh(mesh):
        step = jax.jit(lambda r, st, batch: train["train_step"](cfg, r, st, batch),
            in_shardings=(jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()), state_sharding, data_sharding))
        updated, info = step(rng, state, batch)
        jax.block_until_ready(info)
    loss, grad = float(info["loss"]), float(info["grad_norm"])
    if not np.isfinite(loss) or not np.isfinite(grad) or grad <= 0 or int(updated.step) != 1:
        raise ValueError(f"failed real optimizer step: loss={loss} grad_norm={grad}")
    result = {"status": "passed", "config": cfg.name, "samples": samples, "contact_types": sorted(contact_types),
        "episodes_sampled": len(episodes), "loss": loss, "gradient_norm": grad, "optimizer_steps": 1,
        "model_state_dim": batch[0].state.shape[-1], "action_dim": batch[1].shape[-1],
        "action_horizon": cfg.model.action_horizon, "batch_size": cfg.batch_size,
        "pretrained_checkpoint": str(cfg.weight_loader.params_path),
        "image_token_count": image_token_count, "padded_prompt_tokens": cfg.model.max_token_len,
        "suffix_token_count": suffix_count, "total_context_tokens": context_tokens,
        "model_context_note": "local RoPE implementation exposes no explicit maximum; pretrained context quality is unverified",
        "device_memory": jax.devices()[0].memory_stats()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"config": cfg.name, "loss": loss, "grad_norm": grad, "status": "passed"}), flush=True)


if __name__ == "__main__":
    main()
