#!/usr/bin/env python3
"""Fail fast when a front-only ablation config or derived sidecar is stale."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
VARIANTS = {
    "pi0_xhand_spatial_structured_suffix_front": (True, True),
    "pi0_xhand_spatial_structured_suffix_front_visual": (True, False),
    "pi0_xhand_spatial_structured_suffix_front_tactile": (False, True),
}


def check_manifest(manifest: dict, info: dict, profile: dict, *, version: str) -> None:
    if manifest.get("status") != "complete" or manifest.get("spatial_version") != version:
        raise ValueError("Spatial sidecar is incomplete or has the wrong version")
    contract = manifest.get("spatial_contract", {})
    if (contract.get("camera_roles") != ["front"]
            or contract.get("visual_points_per_frame") != 4096
            or contract.get("tactile_points_per_frame") != 600):
        raise ValueError("Sidecar must contain front-only 4096 visual and 600 tactile points")
    diagnostic = manifest.get("preprocess_config", {}).get("diagnostics", {})
    if manifest.get("preprocess_config", {}).get("visual", {}).get("camera_roles") != ["front"]:
        raise ValueError("Sidecar preprocessor itself was not front-only")
    if not diagnostic.get("enable_front_extrinsic_correction") or not diagnostic.get("enable_front_intrinsic_k1"):
        raise ValueError("Sidecar did not apply both front calibration corrections")
    expected_translation = profile["front_extrinsic_translation_base_m"]
    actual_translation = diagnostic.get("front_extrinsic_translation_base_m", [])
    if (len(actual_translation) != 3 or any(
        not math.isclose(float(actual), float(expected), abs_tol=1e-8)
        for actual, expected in zip(actual_translation, expected_translation, strict=True)
    )):
        raise ValueError("Sidecar front extrinsic translation differs from the profile")
    actual_k = diagnostic.get("front_color_k1", {})
    expected_k = profile["front_color_intrinsics"]
    if any(not math.isclose(float(actual_k.get(key, float("nan"))), float(expected_k[key]), abs_tol=1e-6)
           for key in ("fx", "fy", "cx", "cy")):
        raise ValueError("Sidecar front color intrinsics differ from the profile")
    episodes = manifest.get("storage", {}).get("episodes", [])
    if (len(episodes) != info["total_episodes"]
            or sum(item["selected_frame_count"] for item in episodes) != info["total_frames"]
            or any(item["selected_frame_count"] != item["source_frame_count"] for item in episodes)):
        raise ValueError("Sidecar does not cover every source frame of every episode")


def check_asset_provenance(manifest: dict, *, repo_root: Path) -> None:
    expected_assets = {
        "camera_internal_config", "camera_external_config", "robot_urdf",
        "state_mapping", "tactile_t16_transformed", "tactile_t30_right_transformed",
    }
    assets = manifest.get("asset_provenance", {})
    if set(assets) != expected_assets:
        raise ValueError("Sidecar is missing numerical spatial asset provenance")
    for name, record in assets.items():
        path = repo_root / record["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"Sidecar {name} does not match the current repository asset: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    from openpi.training import config as training_config

    configs = {name: training_config.get_config(name) for name in VARIANTS}
    reference = configs[next(iter(VARIANTS))]
    spatial = reference.data.spatial
    if spatial is None or spatial.camera_roles != ("front",) or spatial.front_calibration_profile is None:
        raise ValueError("Front ablation config lacks front-only camera or candidate profile")
    for name, expected in VARIANTS.items():
        cfg = configs[name]
        if (cfg.model.use_visual, cfg.model.use_tactile) != expected:
            raise ValueError(f"{name}: incorrect visual/tactile ablation flags")
        if (cfg.data != reference.data
                or cfg.seed != reference.seed or cfg.batch_size != reference.batch_size
                or cfg.num_train_steps != reference.num_train_steps
                or cfg.weight_loader != reference.weight_loader):
            raise ValueError(f"{name}: training setup differs from the two other ablations")
        if (cfg.model.conditioning.target != "suffix"
                or cfg.model.encoder != reference.model.encoder
                or cfg.model.paligemma_variant != reference.model.paligemma_variant
                or cfg.model.action_expert_variant != reference.model.action_expert_variant
                or cfg.model.visual_points != 4096 or cfg.model.tactile_points != 600):
            raise ValueError(f"{name}: spatial encoder or base Pi0 architecture differs")
        expected_roles = ["front"] if expected[0] else []
        if cfg.policy_metadata != {
            "spatial_camera_roles": expected_roles,
            "spatial_use_visual": expected[0],
            "spatial_use_tactile": expected[1],
            "front_calibration_profile": spatial.front_calibration_profile,
        }:
            raise ValueError(f"{name}: deployment policy metadata differs from training modalities")
    dataset_root = Path(spatial.dataset_root).expanduser().resolve()
    if Path(reference.data.repo_id).expanduser().resolve() != dataset_root:
        raise ValueError("Training repo_id and spatial dataset_root refer to different datasets")
    profile_path = Path(spatial.front_calibration_profile)
    if not profile_path.is_absolute():
        profile_path = repo_root / profile_path
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if not {"front_extrinsic_translation_base_m", "front_color_intrinsics"} <= profile.keys():
        raise ValueError("Candidate profile must provide both front corrections")
    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    manifest_path = dataset_root / "spatial" / spatial.version / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    check_manifest(manifest, info, profile, version=spatial.version)
    check_asset_provenance(manifest, repo_root=repo_root)
    print(f"READY: {dataset_root}")
    print(f"sidecar: {manifest_path}")
    print(f"profile: {profile_path}")
    for name, (visual, tactile) in VARIANTS.items():
        print(f"{name}: visual={visual}, tactile={tactile}, camera=front, seed={configs[name].seed}")


if __name__ == "__main__":
    main()
