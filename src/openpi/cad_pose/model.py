"""Shared single-frame CAD-conditioned RGB-D pose network, in base_link metres."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional


def rotation_6d_to_matrix(rotation: torch.Tensor) -> torch.Tensor:
    """Continuous 6D columns to SO(3), with a finite orthogonal fallback."""
    if rotation.shape[-1] != 6:
        raise ValueError("rotation_6d must end in 6")
    first, second = rotation[..., :3], rotation[..., 3:]
    fallback = torch.zeros_like(first)
    fallback[..., 0] = 1
    first = torch.where(first.norm(dim=-1, keepdim=True) > 1e-7, first, fallback)
    x = functional.normalize(first, dim=-1, eps=1e-8)
    y = second - (x * second).sum(dim=-1, keepdim=True) * x
    # Choose the canonical direction most orthogonal to x when 6D is degenerate.
    axis = functional.one_hot(x.abs().argmin(dim=-1), 3).to(x.dtype)
    orthogonal = axis - (axis * x).sum(dim=-1, keepdim=True) * x
    y = torch.where(y.norm(dim=-1, keepdim=True) > 1e-7, y, orthogonal)
    y = functional.normalize(y, dim=-1, eps=1e-8)
    z = torch.linalg.cross(x, y, dim=-1)
    return torch.stack((x, y, z), dim=-1)


def _ordered_indices(count: int, selected: int, device: torch.device) -> torch.Tensor:
    if count < selected:
        raise ValueError(f"expected at least {selected} points, got {count}")
    return torch.div(torch.arange(selected, device=device) * count, selected, rounding_mode="floor")


class XHandCADPoseTransformerV1(nn.Module):
    """256 scene / 32 CAD tokens, shared 3-layer 128D cross-attention decoder.

    Scene is ordered metric XYZ + RGB in base_link, CAD is ordered metric XYZ in
    each object's local frame. Both roles are processed with the same weights.
    """

    def __init__(self):
        super().__init__()
        self.scene_encoder = nn.Sequential(nn.Linear(6, 128), nn.GELU(), nn.Linear(128, 128))
        self.cad_encoder = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, 128))
        layer = nn.TransformerDecoderLayer(d_model=128, nhead=4, dim_feedforward=256,
                                           dropout=0.0, batch_first=True, activation="gelu")
        self.decoder = nn.TransformerDecoder(layer, num_layers=3)
        self.pose_head = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 9))

    def forward(self, scene: torch.Tensor, scene_mask: torch.Tensor, cad: torch.Tensor) -> dict[str, torch.Tensor]:
        if scene.ndim != 3 or scene.shape[-1] != 6:
            raise ValueError("scene must be [B,P,6] metric XYZ and RGB")
        if scene_mask.shape != scene.shape[:2] or cad.shape != (scene.shape[0], 2, 512, 3):
            raise ValueError("expected scene mask [B,P] and CAD [B,2,512,3]")
        if not torch.isfinite(scene).all() or not torch.isfinite(cad).all():
            raise ValueError("scene and CAD must be finite")
        index = _ordered_indices(scene.shape[1], 256, scene.device)
        selected_mask = scene_mask[:, index].bool()
        if not selected_mask.any(dim=1).all():
            raise ValueError("each scene requires at least one valid sampled RGB-D point")
        selected_scene = scene[:, index]
        scene_tokens = self.scene_encoder(selected_scene)
        cad_tokens = self.cad_encoder(cad[:, :, _ordered_indices(512, 32, cad.device)])
        b = scene.shape[0]
        decoded = self.decoder(cad_tokens.reshape(b * 2, 32, 128),
                               scene_tokens[:, None].expand(-1, 2, -1, -1).reshape(b * 2, 256, 128),
                               memory_key_padding_mask=(~selected_mask)[:, None].expand(-1, 2, -1).reshape(b * 2, 256))
        prediction = self.pose_head(decoded.mean(dim=1)).reshape(b, 2, 9)
        return {"translation": prediction[..., :3], "rotation_6d": prediction[..., 3:],
                "rotation": rotation_6d_to_matrix(prediction[..., 3:])}


def pose_loss(prediction: dict[str, torch.Tensor], target_transform: torch.Tensor, cad: torch.Tensor,
              valid: torch.Tensor, *, cad_weight: float = 1.0,
              symmetry_rotations: torch.Tensor | None = None,
              symmetry_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Masked translation + CAD loss; only explicitly supplied symmetries are equivalent.

    Symmetry arrays, when supplied, are [B,2,S,3,3] and [B,2,S]. They
    represent local object-frame rotations, not an implicit all-object ADD-S.
    """
    if target_transform.shape != (*valid.shape, 4, 4) or cad.shape != (*valid.shape, 512, 3):
        raise ValueError("target, CAD and validity shapes disagree")
    valid = valid.bool()
    if not valid.any():
        raise ValueError("batch has no valid pose labels")
    gt = target_transform[valid]
    pts = cad[valid]
    pred_r = prediction["rotation"][valid]
    pred_t = prediction["translation"][valid]
    if not torch.isfinite(gt).all() or not torch.isfinite(pts).all():
        raise ValueError("valid GT/CAD must be finite")
    pred_points = torch.einsum("nij,npj->npi", pred_r, pts) + pred_t[:, None]
    if symmetry_rotations is None:
        gt_points = torch.einsum("nij,npj->npi", gt[:, :3, :3], pts) + gt[:, None, :3, 3]
        geometric = (pred_points - gt_points).norm(dim=-1).mean(dim=-1)
    else:
        if symmetry_mask is None or symmetry_rotations.shape[:2] != valid.shape or symmetry_mask.shape != symmetry_rotations.shape[:3]:
            raise ValueError("symmetry rotations and mask must match valid object slots")
        rotations = symmetry_rotations[valid]
        mask = symmetry_mask[valid].bool()
        if not mask.any(dim=-1).all():
            raise ValueError("each valid object needs an explicit symmetry")
        rotated = torch.einsum("nsij,npj->nspi", gt[:, None, :3, :3] @ rotations, pts) + gt[:, None, None, :3, 3]
        errors = (pred_points[:, None] - rotated).norm(dim=-1).mean(dim=-1)
        geometric = errors.masked_fill(~mask, torch.inf).min(dim=-1).values
    translation = (pred_t - gt[:, :3, 3]).norm(dim=-1)
    return (translation + cad_weight * geometric).mean()
