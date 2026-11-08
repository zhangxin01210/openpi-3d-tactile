"""Tracker-free model runtime shared by future offline and online callers."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from openpi.cad_pose.geometry import predicted_geometry_packet
from openpi.cad_pose.model import XHandCADPoseTransformerV1
from openpi.cad_pose.preprocess import FrontRGBDPreprocessor


def points_sha256(points: np.ndarray) -> str:
    array = np.ascontiguousarray(points, dtype=np.float32)
    return hashlib.sha256(array.tobytes()).hexdigest()


class PoseRuntime:
    """One cached model and two ordered CAD point sets; no tracker or GT fallback.

    Finite network output is not a calibrated detection confidence. Callers
    must not interpret a true output mask as evidence of pose accuracy.
    """

    def __init__(self, model: XHandCADPoseTransformerV1, preprocessor, cad: np.ndarray, *, device: str = "cpu"):
        self.cad = np.asarray(cad, dtype=np.float32).copy()
        if self.cad.shape != (2, 512, 3) or not np.isfinite(self.cad).all():
            raise ValueError("ordered CAD must contain two finite [512,3] arrays")
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.preprocess = preprocessor
        self._cad_tensor = torch.from_numpy(self.cad[None]).to(self.device)

    @classmethod
    def from_checkpoint(cls, checkpoint: Path, repo: Path, profile: str, cad: np.ndarray, *,
                        dataset: str, device: str = "cpu"):
        """Load only a matching model, ordered CAD and front profile; no fallback."""
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        root = Path(repo).resolve()
        profile_path = Path(profile)
        if not profile_path.is_absolute():
            profile_path = root / profile_path
        files = {"profile": profile_path,
                 "camera": root / "configs/ur7e_xhand/cameras_0903.json",
                 "extrinsic": root / "configs/ur7e_xhand/extrinsic_0909.yaml"}
        hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (not isinstance(saved, dict) or saved.get("model_version") != "xhand_cad_pose_transformer_v1"
                or not isinstance(saved.get("ordered_cad_sha256_by_dataset"), dict)
                or saved["ordered_cad_sha256_by_dataset"].get(dataset) != points_sha256(cad)
                or saved.get("camera_hashes") != hashes):
            raise ValueError("checkpoint version, CAD or camera calibration provenance mismatch")
        model = XHandCADPoseTransformerV1()
        model.load_state_dict(saved["model_state_dict"], strict=True)
        return cls(model, FrontRGBDPreprocessor(root, profile), cad, device=device)

    @torch.inference_mode()
    def infer(self, rgb: np.ndarray, raw_depth: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        scene, mask = self.preprocess(rgb, raw_depth)
        output = self.model(torch.as_tensor(scene[None], device=self.device),
                            torch.as_tensor(mask[None], device=self.device), self._cad_tensor)
        rotation = output["rotation"][0].cpu().numpy()
        translation = output["translation"][0].cpu().numpy()
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise ValueError("pose network produced a non-finite prediction")
        transforms = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
        transforms[:, :3, :3] = rotation
        transforms[:, :3, 3] = translation
        geometry = predicted_geometry_packet(self.cad, transforms, np.ones(2, bool))
        return transforms, geometry
