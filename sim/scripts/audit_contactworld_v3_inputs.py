#!/usr/bin/env python3
"""Check a bilateral LeRobot episode through the exact OpenPI training transforms."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from flax import nnx
import jax
import numpy as np

from openpi.models.model import Observation
from openpi.models.spatial_encoders.router import SpatialConditioningRouter
from openpi.training import config as training_config
from openpi.training import data_loader

CONFIGS = (
    "pi0_cw3_usb_01_rgb", "pi0_cw3_usb_02_fused_pc",
    "pi0_cw3_usb_03_rgb_local_map", "pi0_cw3_usb_04_fused_pc_local_map",
    "pi0_cw3_usb_05_fused_pc_local_summary", "pi0_cw3_usb_06_fused_pc_base_structured",
    "pi0_cw3_usb_07_fused_pc_base_pooled", "pi0_cw3_usb_08_fused_pc_base_map",
    "pi0_cw3_usb_09_fused_pc_base_map_split", "pi0_cw3_usb_10_fused_pc_base_map_matched",
    "pi0_cw3_usb_11_fused_pc_base_structured_prefix",
    "pi0_cw3_usb_12_fused_pc_base_structured_split_route",
    "pi0_cw3_usb_13_fused_pc_local_map_3tok",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.dataset.resolve()
    manifest = json.loads((root / "spatial/manifest.json").read_text())
    if manifest["format"] != "contactworld-spatial-v3-bilateral":
        raise ValueError("Expected bilateral V3")
    if not manifest["episodes"] or manifest["episodes"][0]["split"] != "train":
        raise ValueError("The audit needs a train episode at index zero")
    frame = min(20, int(manifest["episodes"][0]["length"]) - 16)
    if frame < 0:
        raise ValueError("Episode shorter than the action horizon")
    expected_train_samples = sum(
        max(0, int(episode["length"]) - 16 + 1)
        for episode in manifest["episodes"] if episode["split"] == "train")
    allowed = []
    offset = 0
    for episode in manifest["episodes"]:
        length = int(episode["length"])
        if episode["split"] == "train":
            allowed.extend(range(offset, offset + max(0, length - 16 + 1)))
        offset += length
    spatial_dir = root / "spatial/episodes/episode_000000"
    expected = {
        side: {
            "xyz": np.load(spatial_dir / f"tactile_xyz_{side}_base.npy")[frame],
            "force": np.load(spatial_dir / f"tactile_force_{side}_base.npy")[frame],
            "local": np.load(spatial_dir / f"force_grid_{side}_local.npy")[frame].reshape(140, 3),
        } for side in ("left", "right")
    }
    results = []
    structured_reference = None
    split_map_parameters = None
    for name in CONFIGS:
        config = training_config.get_config(name)
        factory = dataclasses.replace(
            config.data, repo_id=str(root),
            sidecar=dataclasses.replace(config.data.sidecar, root=str(root)),
        )
        data_config = factory.create(config.assets_dirs, config.model)
        raw = data_loader.create_torch_dataset(
            data_config, config.model.action_horizon, config.model)
        if len(raw) != expected_train_samples:
            raise ValueError(f"{name}: train split or action-horizon selection differs")
        if raw.sample_indices != allowed:
            raise ValueError(f"{name}: non-train or boundary-crossing samples selected")
        # Exercise the same normalization path used by scripts/train.py.
        transformed = data_loader.transform_dataset(
            raw, data_config, skip_norm_stats=False)
        item = transformed[frame]
        if name == "pi0_cw3_usb_01_rgb":
            native = raw[frame]
            native_state = np.asarray(native["observation.state"], dtype=np.float32)
            native_actions = np.asarray(native["action"], dtype=np.float32)
            state_stats = data_config.norm_stats["state"]
            action_stats = data_config.norm_stats["actions"]
            state_expected = (native_state - state_stats.mean) / (state_stats.std + 1e-6)
            action_expected = (native_actions - action_stats.mean) / (action_stats.std + 1e-6)
            if (native_state.shape != (18,) or native_actions.shape != (16, 6) or
                    not np.allclose(item["state"][:18], state_expected, atol=1e-6) or
                    not np.allclose(item["actions"][:, :6], action_expected, atol=1e-6) or
                    not np.all(item["state"][18:] == 0) or
                    not np.all(item["actions"][:, 6:] == 0)):
                raise ValueError("Robot state/action normalization, order, or padding differs")
        if (not item["image_mask"]["base_0_rgb"] or
                not item["image_mask"]["left_wrist_0_rgb"] or
                item["image_mask"]["right_wrist_0_rgb"]):
            raise ValueError(f"{name}: diagnostic tactile preview entered image slot")
        if set(item["image"]) != {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}:
            raise ValueError(f"{name}: unexpected RGB roles")
        if np.any(item["image"]["right_wrist_0_rgb"]):
            raise ValueError(f"{name}: diagnostic preview populated the masked image slot")
        spatial = item.get("spatial")
        mode = factory.sidecar.force_mode
        use_tactile = bool(getattr(config.model, "use_tactile", False))
        use_visual = bool(getattr(config.model, "use_visual", False))
        if use_tactile:
            tactile = spatial["tactile"]
            sides = ("left", "right")
            for position, side in enumerate(sides):
                part = slice(140 * position, 140 * (position + 1))
                if not np.array_equal(tactile["xyz_m"][part], expected[side]["xyz"]):
                    raise ValueError(f"{name}: {side} taxel XYZ differs from LeRobot sidecar")
                force_key = "local" if mode == "both_local_grid" else "force"
                if not np.array_equal(tactile["force"][part], expected[side][force_key]):
                    raise ValueError(f"{name}: {side} force differs from LeRobot sidecar")
                if not np.all(tactile["finger_id"][part] == position):
                    raise ValueError(f"{name}: {side} finger ID is wrong")
            if not np.any(np.linalg.norm(tactile["force"], axis=-1) > 0):
                raise ValueError(f"{name}: all tactile force is zero")
        elif spatial is not None and spatial["tactile"] is not None:
            raise ValueError(f"{name}: unintended tactile input")
        if use_visual:
            key = "pointcloud_front_xyz" if factory.sidecar.cloud_mode == "front" else "pointcloud_fused_xyz"
            stored = np.load(spatial_dir / (key + ".npy"))[frame]
            if not np.array_equal(spatial["visual"]["xyz_m"], stored):
                raise ValueError(f"{name}: model point cloud differs from LeRobot sidecar")
        elif spatial is not None and spatial["visual"] is not None:
            raise ValueError(f"{name}: unintended cloud input")
        row = {"config": name, "frame": frame,
               "cloud_points": 4096 if use_visual else 0,
               "left_taxels": 140 if use_tactile else 0,
               "right_taxels": 140 if use_tactile else 0,
               "preview_image_masked": True}
        if use_tactile or use_visual:
            batched = jax.tree.map(lambda value: np.expand_dims(value, 0), item)
            observation = Observation.from_dict(batched)
            spec, _ = config.model.inputs_spec(batch_size=1)
            if use_visual and observation.spatial.visual.xyz_m.shape != spec.spatial.visual.xyz_m.shape:
                raise ValueError(f"{name}: visual tensor differs from model input spec")
            if use_tactile and observation.spatial.tactile.force.shape != spec.spatial.tactile.force.shape:
                raise ValueError(f"{name}: tactile tensor differs from model input spec")
            encoder = config.model.create_spatial_encoder(rngs=nnx.Rngs(0))
            encoded = encoder(observation.spatial)
            tokens = np.asarray(encoded.tokens)
            mask = np.asarray(encoded.token_mask)
            representation = config.model.tactile_representation if use_tactile else "none"
            split_map = representation in {"map_split", "map_split_local"}
            matched_map = representation == "map_coupled_matched"
            multiquery_map = representation == "map_local_multiquery"
            map_mode = representation in {"map_coupled", "map_split", "map_local", "map_split_local",
                                          "map_coupled_matched", "map_local_multiquery"}
            visual_tokens = (config.model.visual_num_centers + 1) if use_visual else 0
            tactile_tokens = (6 if split_map or matched_map or multiquery_map or (use_tactile and not map_mode and
                              representation != "summary") else 2 if use_tactile else 0)
            expected_tokens = visual_tokens + tactile_tokens
            expected_mask = np.ones((1, expected_tokens), dtype=bool)
            if tokens.shape != (1, expected_tokens, config.model.token_dim) or not np.array_equal(mask, expected_mask):
                raise ValueError(f"{name}: spatial encoder token/mask shape differs")
            if not np.isfinite(tokens).all():
                raise ValueError(f"{name}: nonfinite encoded token")
            row["encoder_tokens"] = expected_tokens
            row["active_encoder_tokens"] = int(mask.sum())
            if name.endswith("06_fused_pc_base_structured"):
                structured_reference = tokens.copy()
            if name.endswith(("11_fused_pc_base_structured_prefix",
                              "12_fused_pc_base_structured_split_route")) and not np.array_equal(
                    tokens, structured_reference):
                raise ValueError(f"{name}: routing config changed the encoded input")
            if name.endswith(("06_fused_pc_base_structured",
                              "11_fused_pc_base_structured_prefix",
                              "12_fused_pc_base_structured_split_route")):
                prefix_indices = config.model.spatial_prefix_token_indices
                suffix_indices = config.model.spatial_suffix_token_indices
                router = SpatialConditioningRouter(
                    encoder=encoder, encoder_token_dim=config.model.token_dim,
                    conditioning=config.model.conditioning,
                    prefix_dim=32 if config.model.conditioning.use_prefix else None,
                    suffix_dim=32 if config.model.conditioning.use_suffix else None,
                    prefix_token_indices=prefix_indices, suffix_token_indices=suffix_indices,
                    compact_token_selection=config.model.spatial_route == "visual_prefix_tactile_suffix",
                    rngs=nnx.Rngs(1))
                routed = router(observation.spatial)
                expected_route = {
                    "06": (None, (1, 23, 32)),
                    "11": ((1, 23, 32), None),
                    "12": ((1, 17, 32), (1, 6, 32)),
                }[name.split("_")[3]]
                actual_route = (None if routed.prefix_tokens is None else routed.prefix_tokens.shape,
                                None if routed.suffix_tokens is None else routed.suffix_tokens.shape)
                if actual_route != expected_route:
                    raise ValueError(f"{name}: actual spatial route is {actual_route}, expected {expected_route}")
                row["route_shapes"] = [list(shape) if shape is not None else None for shape in actual_route]
            if use_tactile and config.model.tactile_pool_per_finger:
                original_force = np.asarray(observation.spatial.tactile.force)
                original_norm = np.asarray(observation.spatial.tactile.force_norm)
                altered = dataclasses.replace(observation.spatial,
                    tactile=dataclasses.replace(observation.spatial.tactile,
                        force=original_force * np.where(np.arange(280) < 140, 0.0, 1.0)[None, :, None],
                        force_norm=original_norm * np.where(np.arange(280) < 140, 0.0, 1.0)[None, :]))
                changed = np.asarray(encoder(altered).tokens)
                left_token = visual_tokens
                finger_tokens = 3 if split_map or matched_map or multiquery_map or (
                    not map_mode and representation != "summary") else 1
                if (np.array_equal(changed[:, left_token:left_token + finger_tokens],
                                   tokens[:, left_token:left_token + finger_tokens]) or
                        not np.allclose(changed[:, left_token + finger_tokens:],
                                        tokens[:, left_token + finger_tokens:], atol=1e-6, rtol=1e-6)):
                    raise ValueError(f"{name}: left/right token routing failed")
                row["independent_finger_tokens"] = True
                if representation == "summary":
                    shifted_xyz = np.asarray(observation.spatial.tactile.xyz_m).copy()
                    shifted_xyz[:, :140, 0] += 0.01
                    shifted = dataclasses.replace(observation.spatial,
                        tactile=dataclasses.replace(observation.spatial.tactile,
                                                    xyz_m=shifted_xyz))
                    shifted_tokens = np.asarray(encoder(shifted).tokens)
                    if (np.array_equal(shifted_tokens[:, left_token], tokens[:, left_token]) or
                            not np.allclose(shifted_tokens[:, left_token + 1:],
                                            tokens[:, left_token + 1:], atol=1e-6, rtol=1e-6)):
                        raise ValueError(f"{name}: summary pad centroid is not localized")
                    row["summary_pad_position_encoded"] = True
                if split_map:
                    selective_force = np.asarray(observation.spatial.tactile.force).copy()
                    selective_force[:, :140, 0] = 0
                    selective = dataclasses.replace(observation.spatial,
                        tactile=dataclasses.replace(observation.spatial.tactile,
                                                    force=selective_force,
                                                    force_norm=np.linalg.norm(selective_force, axis=-1)))
                    selective_tokens = np.asarray(encoder(selective).tokens)
                    if (np.array_equal(selective_tokens[:, left_token], tokens[:, left_token]) or
                            not np.allclose(selective_tokens[:, left_token + 1:],
                                            tokens[:, left_token + 1:], atol=1e-6, rtol=1e-6)):
                        raise ValueError(f"{name}: channel-specific token routing failed")
                    row["independent_force_channels"] = True
            if name.endswith("07_fused_pc_base_pooled"):
                if (encoder.structured_encoder.config.tactile_local_tokens_per_finger != 4 or
                        encoder.structured_encoder.config.tactile_summary_tokens_per_finger != 2):
                    raise ValueError(f"{name}: pooled query budget changed")
                left_force = np.asarray(observation.spatial.tactile.force).copy()
                left_force[:, :140] = 0
                pooled_input = dataclasses.replace(observation.spatial,
                    tactile=dataclasses.replace(observation.spatial.tactile, force=left_force,
                                                force_norm=np.linalg.norm(left_force, axis=-1)))
                if np.array_equal(np.asarray(encoder(pooled_input).tokens)[:, visual_tokens:],
                                  tokens[:, visual_tokens:]):
                    raise ValueError(f"{name}: pooled tokens ignore left force")
                row["six_pooled_tokens"] = True
            if name.endswith(("09_fused_pc_base_map_split", "10_fused_pc_base_map_matched")):
                parameter_count = sum(value.size for value in jax.tree.leaves(
                    nnx.state(encoder.map_encoders, nnx.Param)))
                row["tactile_vit_parameters"] = parameter_count
                if name.endswith("09_fused_pc_base_map_split"):
                    split_map_parameters = parameter_count
                elif split_map_parameters is None or abs(parameter_count / split_map_parameters - 1) > 0.05:
                    raise ValueError(f"{name}: capacity control differs from split ViT by over 5%")
            if name.endswith("13_fused_pc_local_map_3tok"):
                if (mode != "both_local_grid" or len(encoder.map_encoders) != 1 or
                        encoder.map_encoders[0].num_outputs != 3 or
                        encoder.map_encoders[0].patch.out_features != 128 or
                        config.model.conditioning.use_prefix or not config.model.conditioning.use_suffix):
                    raise ValueError(f"{name}: local-map token-budget control changed more than intended")
                row["three_queries_per_finger"] = True
            if name.endswith("12_fused_pc_base_structured_split_route"):
                prefix = config.model.spatial_prefix_token_indices
                suffix = config.model.spatial_suffix_token_indices
                if prefix != tuple(range(17)) or suffix != tuple(range(17, 23)) or set(prefix) & set(suffix):
                    raise ValueError(f"{name}: modality routing token ranges are wrong")
                if (not np.asarray(routed.prefix_mask).all() or
                        not np.asarray(routed.suffix_mask).all()):
                    raise ValueError(f"{name}: actual spatial routing failed")
                row["visual_prefix_tactile_suffix"] = True
        results.append(row)
        print(name, row, flush=True)
    result = {"status": "passed", "dataset": str(root), "configs": results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "configs_checked": len(results)}))


if __name__ == "__main__":
    main()
