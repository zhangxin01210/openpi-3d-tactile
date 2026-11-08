"""Tracker-free online pose bridge; uses only the current RGB-D frame."""

import hashlib
import json
from pathlib import Path

import numpy as np

from openpi.cad_pose.runtime import PoseRuntime
from scripts.cad_pose.preflight import ROLES
from scripts.preprocess.export_object_geometry_rgbtrack import SURFACE_SAMPLING_SEED
from scripts.preprocess.export_object_geometry_rgbtrack import load_surface_points
from scripts.preprocess.export_object_geometry_rgbtrack import sha256


def load_ordered_cad(repo, dataset):
    """Recreate verified local CAD points without opening pose or GT geometry."""
    repo = Path(repo).resolve()
    if dataset == "1018_peg_all":
        from openpi.object_geometry.production import load_mesh_surface_points

        cfg = json.loads((repo / "configs/object_geometry/datasets/1018_peg_all/config.json").read_text())
        points = []
        for role in ("manipulated", "target"):
            item = cfg["assets"][ROLES[dataset][role]]
            mesh = repo / item["mesh"]
            if sha256(mesh) != item["sha256"]:
                raise ValueError("legacy CAD hash mismatch")
            points.append(
                load_mesh_surface_points(mesh, scale_m_per_unit=item["scale_m_per_unit"], seed=item["sampling_seed"])[0]
            )
        return np.stack(points)
    prep = repo / "output/preprocess" / dataset
    release = json.loads((prep / "rgbtrack/release_v1.json").read_text())
    approval = json.loads((prep / "geometry/approved/manifest.json").read_text())
    if release.get("dataset") != dataset or approval.get("status") != "human_accepted":
        raise ValueError("CAD release or approval mismatch")
    objects = release["objects"]
    if set(objects) != set(ROLES[dataset].values()) or len(objects) != 2:
        raise ValueError("CAD object roles differ")
    points = {}
    for i, name in enumerate(objects):
        path = prep / "geometry/approved" / f"{name}.glb"
        asset = approval["assets"][name]
        if sha256(path) != asset["sha256"] or float(asset["scale_m_per_unit"]) != 1:
            raise ValueError("CAD hash/scale mismatch")
        points[name] = load_surface_points(path, SURFACE_SAMPLING_SEED + i)
    return np.stack([points[ROLES[dataset][role]] for role in ("manipulated", "target")])


class PoseConditionedPolicy:
    def __init__(self, policy, runtime):
        self.policy, self.runtime = policy, runtime

    @property
    def metadata(self):
        return self.policy.metadata

    def infer(self, observation, **kwargs):
        if (
            "object_geometry" in observation
            or "object_set" in observation
            or (
                "spatial" in observation
                and ("object_geometry" in observation["spatial"] or "object_set" in observation["spatial"])
            )
        ):
            raise ValueError("pose policy refuses externally supplied geometry")
        rgb = observation.get("observation.images.cam_front")
        depth = observation.get("observation.depths.cam_front")
        if rgb is None or depth is None:
            raise ValueError("pose policy needs the current front RGB and raw depth in the same observation")
        _, geometry = self.runtime.infer(np.asarray(rgb), np.asarray(depth))
        return self.policy.infer({**observation, "object_geometry": geometry}, **kwargs)


def make_pose_runtime(settings, profile, *, repo, provenance_path, device="cpu"):
    if not settings.pose_checkpoint_path:
        raise ValueError("Set OPENPI_CAD_POSE_CHECKPOINT to the accepted pose checkpoint")
    dataset = Path(settings.dataset_root).name
    checkpoint = Path(settings.pose_checkpoint_path)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = json.loads(Path(provenance_path).read_text())
    accepted = manifest.get("acceptance") or {}
    if (
        manifest.get("dataset") != dataset
        or manifest.get("producer") != "openpi.cad_pose.sidecar"
        or manifest.get("pose_checkpoint_sha256") != digest
        or accepted.get("status") != "accepted"
        or accepted.get("checkpoint_sha256") != digest
        or not accepted.get("reviewer")
    ):
        raise ValueError("online pose checkpoint must match accepted training prediction-sidecar provenance")
    cad = load_ordered_cad(repo, dataset)
    from openpi.cad_pose.pipeline import camera_hashes
    from openpi.cad_pose.pipeline import points_sha256

    if (
        manifest.get("ordered_cad_sha256") != points_sha256(cad)
        or manifest.get("camera_hashes") != camera_hashes(Path(repo), profile)
        or manifest.get("model_version") != "xhand_cad_pose_transformer_v1"
        or manifest.get("preprocess_version") != "front_rgbd_spatial_v1"
    ):
        raise ValueError("policy provenance CAD, camera, model or preprocessing mismatch")
    return PoseRuntime.from_checkpoint(checkpoint, Path(repo), profile, cad, dataset=dataset, device=device)
