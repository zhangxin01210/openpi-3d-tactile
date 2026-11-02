import dataclasses
import enum
import json
import hashlib
import logging
import os
import pathlib
import tempfile
import socket

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


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            train_config = _config.get_config(args.policy.config)
            policy = _policy_config.create_trained_policy(
                train_config, args.policy.dir, default_prompt=args.default_prompt
            )
            if (getattr(getattr(train_config.model, "encoder", None), "object_relation_mode", None) == "final_v2"
                    and getattr(train_config.data, "object_geometry", None) is not None):
                # Reuse the existing causal perception bridge and WebSocket server.
                from scripts.deploy.websocket_hand_scene_bridge import HandScenePerceptionBridge
                from scripts.deploy.run_online_perception import _task_geometry_inputs

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
                profile = train_config.data.spatial.front_calibration_profile
                if not profile:
                    raise ValueError("Training config requires an explicit front calibration profile for online geometry")
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
                policy = HandScenePerceptionBridge(policy, repo_root=root, perception_python=pathlib.Path(python),
                    workdir=spool, geometry_config=geometry, calibration_profile=profile, scale_manifest=scales)
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
