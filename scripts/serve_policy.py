import dataclasses
import enum
import hashlib
import json
import logging
import os
import pathlib
import socket
import tempfile

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def _contains_path_parts(path_parts: tuple[str, ...], expected: tuple[str, ...]) -> bool:
    """Return whether ``expected`` occurs contiguously in ``path_parts``."""
    if not expected:
        return False
    width = len(expected)
    return any(path_parts[index : index + width] == expected for index in range(len(path_parts) - width + 1))


def resolve_checkpoint_config(config_name: str, checkpoint_dir: str | os.PathLike[str]) -> _config.TrainConfig:
    """Resolve either an exact config name or a config family for a checkpoint.

    Family names are accepted only when the task can be identified without
    ambiguity from the checkpoint path and/or its copied normalization asset.
    """
    config_name = config_name.rstrip("/")
    try:
        return _config.get_config(config_name)
    except ValueError as exact_error:
        prefix = f"{config_name}/"
        candidates = {config.name: config for config in _config.get_configs_by_prefix(prefix)}
        if not candidates:
            raise exact_error

    checkpoint = pathlib.Path(checkpoint_dir).expanduser()
    path_parts = checkpoint.parts
    path_matches = {
        name
        for name in candidates
        if _contains_path_parts(path_parts, tuple(name.removeprefix(prefix).split("/")))
    }

    norm_asset_ids = {
        norm_stats.parent.name for norm_stats in checkpoint.glob("*/norm_stats.json")
    }
    expected_asset_ids = {
        name: getattr(getattr(config.data, "assets", None), "asset_id", None)
        for name, config in candidates.items()
    }
    asset_matches = {
        name for name, asset_id in expected_asset_ids.items() if asset_id in norm_asset_ids
    }

    if path_matches and norm_asset_ids and not asset_matches:
        raise ValueError(
            f"Checkpoint task path matches {sorted(path_matches)}, but normalization assets "
            f"{sorted(norm_asset_ids)} do not match any config in family '{config_name}'."
        )
    if path_matches and asset_matches:
        matches = path_matches & asset_matches
        if not matches:
            raise ValueError(
                f"Checkpoint path implies {sorted(path_matches)}, but normalization assets imply "
                f"{sorted(asset_matches)}. Refusing to load a mismatched task config."
            )
    elif path_matches:
        matches = path_matches
    elif asset_matches:
        matches = asset_matches
    elif len(candidates) == 1 and not norm_asset_ids:
        matches = set(candidates)
    else:
        matches = set()

    if len(matches) != 1:
        evidence = "no task evidence was found" if not matches else f"matches were {sorted(matches)}"
        raise ValueError(
            f"Config family '{config_name}' could not be resolved uniquely from checkpoint "
            f"'{checkpoint}': {evidence}. Use one of these full config names: "
            f"{', '.join(sorted(candidates))}."
        )

    resolved_name = matches.pop()
    logging.info(
        "Resolved config family %r to %r from checkpoint %s",
        config_name,
        resolved_name,
        checkpoint,
    )
    return candidates[resolved_name]


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            train_config = resolve_checkpoint_config(args.policy.config, args.policy.dir)
            anchor_surface_hashes = None
            object_settings = getattr(train_config.data, "object_geometry", None)
            anchor_asset_root = getattr(object_settings, "anchor_asset_root", None)
            if anchor_asset_root is not None:
                # Fail before restoring a multi-GB checkpoint when required HO
                # V4 provenance assets are absent or stale.
                from openpi.object_geometry.ho_v4_anchors import load_assets

                root = pathlib.Path(
                    os.environ.get("OPENPI_REPO_ROOT", pathlib.Path(__file__).resolve().parents[1])
                ).resolve()
                dataset_root = pathlib.Path(object_settings.dataset_root)
                if not dataset_root.is_absolute():
                    dataset_root = root / dataset_root
                anchor_assets = load_assets(
                    dataset_root.parent.parent,
                    dataset_root.name,
                    anchor_asset_root,
                )
                anchor_surface_hashes = {
                    role: value["surface_points_sha256"] for role, value in anchor_assets.items()
                }
            policy = _policy_config.create_trained_policy(
                train_config, args.policy.dir, default_prompt=args.default_prompt
            )
            if getattr(train_config.data, "object_geometry", None) is not None:
                # Reuse the existing causal perception bridge and WebSocket server.
                from scripts.deploy.run_online_perception import _task_geometry_inputs
                from scripts.deploy.websocket_hand_scene_bridge import HandScenePerceptionBridge

                root = pathlib.Path(os.environ.get("OPENPI_REPO_ROOT", pathlib.Path(__file__).resolve().parents[1])).resolve()
                dataset_root = pathlib.Path(train_config.data.object_geometry.dataset_root)
                task = dataset_root.name
                geometry = root / "output/preprocess" / task / "geometry/approved/manifest.json"
                sidecar = dataset_root / "object_geometry" / train_config.data.object_geometry.version / "manifest.json"
                if not sidecar.is_absolute():
                    sidecar = root / sidecar
                if not sidecar.is_file():
                    raise FileNotFoundError(f"Training object geometry manifest missing: {sidecar}")
                sidecar_manifest = json.loads(sidecar.read_text())
                if sidecar_manifest.get("status") != "complete":
                    raise ValueError(f"Training object geometry manifest incomplete: {sidecar}")
                contract = sidecar_manifest.get("object_geometry_contract", {})
                coordinate_frame = (sidecar_manifest.get("coordinate_frame") or contract.get("coordinate_frame")
                                    or (sidecar_manifest.get("calibration") or {}).get("coordinate_frame"))
                if contract.get("point_count") != 512 or coordinate_frame != "base_link":
                    raise ValueError("Training geometry must be 512 points in base_link")
                legacy = sidecar_manifest.get("source", {}).get("mode") == "geometry_only_drop_hand_scene_extras_keep_raw_q2"
                if legacy:
                    geometry = root / "configs/object_geometry/datasets" / task / "config.json"
                    if not geometry.is_file():
                        raise FileNotFoundError(f"Legacy training geometry config missing: {geometry}")
                else:
                    spec = _task_geometry_inputs(geometry, root)  # fail before accepting any client
                    from scripts.preprocess.export_object_geometry_rgbtrack import SURFACE_SAMPLING_SEED
                    def sha(path):
                        return hashlib.sha256(path.read_bytes()).hexdigest()
                    if (sidecar_manifest.get("source_release_sha256") != sha(spec["release_path"])
                            or sidecar_manifest.get("approved_geometry_sha256") != sha(geometry)
                            or sidecar_manifest.get("role_to_object") != spec["roles"]
                            or sidecar_manifest.get("surface_sampling_seed") != SURFACE_SAMPLING_SEED
                            or sidecar_manifest.get("approved_meshes_sha256") !=
                               {name: sha(path) for name, path in spec["meshes"].items()}):
                        raise ValueError("Task release, meshes, role or sampling disagree with training geometry manifest")
                    settings = train_config.data.object_geometry
                    relation_mode = getattr(getattr(train_config.model, "encoder", None), "object_relation_mode", None)
                    if getattr(settings, "m2_state_from_rgbtrack_v1", False):
                        expected_roles = {
                            "1028_press_button": {"manipulated": "button", "target": "box"},
                            "1030_yellow_socket": {"manipulated": "red", "target": "yellow"},
                        }
                        if (relation_mode != "hand_scene_oo_v1" or settings.version != "v1"
                                or settings.missing != "error"
                                or not getattr(train_config.model, "use_tactile", False)
                                or getattr(train_config.model.encoder, "hand_scene_enable_funnel_mask_encoder", False)
                                or sidecar_manifest.get("producer") != "scripts/preprocess/export_object_geometry_rgbtrack.py"
                                or sidecar_manifest.get("dataset") != task
                                or sidecar_manifest.get("hand_scene_contract") is not None
                                or sidecar_manifest.get("length_unit") != "m"
                                or sidecar_manifest.get("role_to_object") != expected_roles.get(task)
                                or task not in expected_roles):
                            raise ValueError("Online M2 state requires the accepted task RGBTrack v1 training contract")
                    elif getattr(settings, "require_hand_scene", False) or relation_mode in {
                        "hand_scene_v0", "hand_scene_oo_v1", "dtcr_v1", "dtcr_independent_v1"
                    }:
                        raise ValueError("This task's online geometry worker does not produce the HandScene state "
                                         "required by the training configuration")
                profile = train_config.data.spatial.front_calibration_profile
                if not profile:
                    raise ValueError("Online geometry requires the shared Front calibration profile")
                profile = pathlib.Path(profile).expanduser()
                if not profile.is_absolute():
                    profile = root / profile
                if not profile.is_file():
                    raise FileNotFoundError(f"Training calibration profile missing: {profile}")
                from openpi.training.data_loader import validate_object_geometry_calibration
                validate_object_geometry_calibration(train_config.data, train_config.model, sidecar_manifest)
                python = os.environ.get("OPENPI_PERCEPTION_PYTHON")
                if not python:
                    raise ValueError("Set OPENPI_PERCEPTION_PYTHON to the existing SAM2/RGBTrack environment")
                spool = pathlib.Path(tempfile.mkdtemp(prefix="openpi-perception-"))
                scales = geometry.parent / "metric_scales_pilot.json" if legacy else None
                if legacy and not scales.is_file():
                    raise FileNotFoundError(f"Legacy training mesh scales missing: {scales}")
                sam2_source = os.environ.get("OPENPI_SAM2_SOURCE") or os.environ.get("OPENPI_SAM2_ROOT")
                sam2_checkpoint = os.environ.get("OPENPI_SAM2_CHECKPOINT")
                rgbtrack_weights = os.environ.get("OPENPI_RGBTRACK_WEIGHTS")
                policy = HandScenePerceptionBridge(policy, repo_root=root, perception_python=pathlib.Path(python),
                    workdir=spool, geometry_config=geometry, calibration_profile=profile, scale_manifest=scales,
                    sam2_source=pathlib.Path(sam2_source) if sam2_source else None,
                    sam2_checkpoint=pathlib.Path(sam2_checkpoint) if sam2_checkpoint else None,
                    rgbtrack_weights=pathlib.Path(rgbtrack_weights) if rgbtrack_weights else None,
                    surface_points_sha256=anchor_surface_hashes)
            return policy
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
        infer_threaded=bool(policy_metadata.get("hand_scene_geometry_service")),
    )
    try:
        server.serve_forever()
    finally:
        close = getattr(policy, "close", None)
        if close is not None:
            close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
