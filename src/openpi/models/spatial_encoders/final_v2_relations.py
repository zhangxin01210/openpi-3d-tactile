"""Geometry-only V2 relations based on the exported XHand FK anchors.

The palm reference is a hand-level frame origin, not a finger landmark. It is
queried along with the other anchors but excluded from finger token pooling.
"""
from __future__ import annotations

from flax import nnx
import jax
import jax.numpy as jnp

from openpi.models.spatial_encoders.object_relation_encoder import _local_anchors
from openpi.models.spatial_encoders.object_relation_encoder import _masked_softmax
from openpi.models.spatial_encoders.structured_spatial_encoder import _batched_take
from openpi.models.spatial_encoders.structured_spatial_encoder import _masked_max
from openpi.models.spatial_encoders.structured_spatial_encoder import _masked_mean


def _distance(delta):
    squared = jnp.sum(delta * delta, axis=-1, keepdims=True)
    return jnp.where(squared > 0, jnp.sqrt(jnp.where(squared > 0, squared, 1.0)), 0.0)


def keypoint_membership(names: list[str], groups: list[str]) -> tuple[tuple[bool, ...], ...]:
    """Validate exporter ordering; pool only the landmarks belonging to each finger."""
    fingers = ("thumb", "index", "middle", "ring", "pinky")
    expected_groups = ("palm", "thumb", "thumb", "thumb", "thumb", "index", "index", "index", "index",
                       "middle", "middle", "middle", "ring", "ring", "ring", "pinky", "pinky", "pinky")
    expected_names = (
        "palm/frame_origin", "right_hand_thumb_bend_joint", "right_hand_thumb_rota_joint1",
        "right_hand_thumb_rota_joint2", "right_hand_thumb_rota_tip", "right_hand_index_bend_joint",
        "right_hand_index_joint1", "right_hand_index_joint2", "right_hand_index_rota_tip",
        "right_hand_mid_joint1", "right_hand_mid_joint2", "right_hand_mid_tip",
        "right_hand_ring_joint1", "right_hand_ring_joint2", "right_hand_ring_tip",
        "right_hand_pinky_joint1", "right_hand_pinky_joint2", "right_hand_pinky_tip",
    )
    if tuple(names) != expected_names or tuple(groups) != expected_groups:
        raise ValueError("hand keypoint manifest names/groups do not match the verified 18-point FK exporter")
    return tuple(tuple(group == finger for group in groups) for finger in fingers)


def _hand_object(encoder, object_xyz, object_mask, h_xyz, h_mask, membership, role):
    distances = jnp.sum((object_xyz[:, None, :, :] - h_xyz[:, :, None, :]) ** 2, axis=-1)
    _, indices = jax.lax.top_k(-jnp.where(object_mask[:, None, :], distances, jnp.inf), 16)
    neighbor_mask = _batched_take(object_mask, indices) & h_mask[..., None]
    delta = _batched_take(object_xyz, indices) - h_xyz[:, :, None, :]
    features = jnp.concatenate((delta, _distance(delta)), axis=-1)
    embedded = encoder.ho(features)
    point_features = encoder.ho_neighbor_pool(jnp.concatenate((
        _masked_mean(embedded, neighbor_mask, axis=2), _masked_max(embedded, neighbor_mask, axis=2),
    ), axis=-1))
    point_mask = jnp.any(neighbor_mask, axis=-1)
    finger_masks = point_mask[:, None, :] & membership[None, :, :].astype(jnp.bool_)
    finger_features = jnp.broadcast_to(point_features[:, None, :, :], (*finger_masks.shape, point_features.shape[-1]))
    ho = encoder.ho_finger_pool(jnp.concatenate((
        _masked_mean(finger_features, finger_masks, axis=2), _masked_max(finger_features, finger_masks, axis=2),
    ), axis=-1)) + encoder.ho_finger.value[None, :, :] + role
    ho_mask = jnp.any(finger_masks, axis=-1)

    return jnp.where(ho_mask[..., None], ho, 0.0), ho_mask


class FinalV2RelationEncoder(nnx.Module):
    """Five hand/manipulated and five manipulated/target relation tokens."""

    def __init__(self, *, rngs: nnx.Rngs, dim: int = 128):
        self.ho = nnx.Linear(4, dim, rngs=rngs)
        self.ho_neighbor_pool = nnx.Linear(2 * dim, dim, rngs=rngs)
        self.ho_finger_pool = nnx.Linear(2 * dim, dim, rngs=rngs)
        self.ho_finger = nnx.Param(jax.random.normal(rngs.params(), (5, dim)) * 0.02)
        self.ho_role = nnx.Param(jax.random.normal(rngs.params(), (dim,)) * 0.02)
        # Preserve M2's two-layer GELU relation MLP; only its input width changes.
        self.oo_in = nnx.Linear(4, dim, rngs=rngs)
        self.oo_out = nnx.Linear(dim, dim, rngs=rngs)
        self.oo_queries = nnx.Param(jax.random.normal(rngs.params(), (5, dim)) * 0.02)
        self.oo_role = nnx.Param(jax.random.normal(rngs.params(), (dim,)) * 0.02)

    def __call__(self, manipulated_xyz, manipulated_mask, target_xyz, target_mask, hand_xyz, hand_mask, membership):
        if hand_xyz.shape[-2:] != (18, 3) or membership.shape != (5, 18):
            raise ValueError("expected 18 hand keypoints and a validated (5,18) finger membership")
        if manipulated_xyz.shape[-2:] != (512, 3) or target_xyz.shape[-2:] != (512, 3):
            raise ValueError("expected 512 points per object")
        m_mask = manipulated_mask.astype(jnp.bool_) & jnp.isfinite(manipulated_xyz).all(axis=-1)
        t_mask = target_mask.astype(jnp.bool_) & jnp.isfinite(target_xyz).all(axis=-1)
        h_mask = hand_mask.astype(jnp.bool_) & jnp.isfinite(hand_xyz).all(axis=-1)
        m_xyz = jnp.where(m_mask[..., None], manipulated_xyz, 0.0)
        t_xyz = jnp.where(t_mask[..., None], target_xyz, 0.0)
        h_xyz = jnp.where(h_mask[..., None], hand_xyz, 0.0)

        ho, ho_mask = _hand_object(self, m_xyz, m_mask, h_xyz, h_mask, membership, self.ho_role.value)

        def anchors(xyz, mask):
            zeros = jnp.zeros(mask.shape, dtype=xyz.dtype)
            centers, valid, *_ = _local_anchors((xyz, mask, zeros, zeros, zeros), 8)
            return centers, valid

        m_center, m_valid = anchors(m_xyz, m_mask)
        t_center, t_valid = anchors(t_xyz, t_mask)
        delta = t_center[:, None, :, :] - m_center[:, :, None, :]
        pair_features = jnp.concatenate((delta, _distance(delta)), axis=-1)
        pair_emb = self.oo_out(jax.nn.gelu(self.oo_in(pair_features))).reshape(m_xyz.shape[0], 64, -1)
        pair_mask = (m_valid[:, :, None] & t_valid[:, None, :]).reshape(m_xyz.shape[0], 64)
        logits = jnp.einsum("bnd,qd->bqn", pair_emb, self.oo_queries.value)
        weights = _masked_softmax(logits, pair_mask[:, None, :])
        oo = jnp.einsum("bqn,bnd->bqd", weights, pair_emb) + self.oo_role.value
        oo_mask = jnp.broadcast_to(jnp.any(pair_mask, axis=-1)[:, None], (m_xyz.shape[0], 5))
        return (jnp.where(ho_mask[..., None], ho, 0.0), ho_mask,
                jnp.where(oo_mask[..., None], oo, 0.0), oo_mask)


def dual_geometry_to_object_set(geometry, slots=8):
    """Map the existing ordered two-object contract to padded generic slots."""
    from openpi.models.spatial_encoders.types import ObjectSetInput

    if slots < 2:
        raise ValueError("dual-object adaptation needs at least two slots")
    xyz = jnp.stack((geometry.manipulated_xyz_m, geometry.target_xyz_m), axis=1)
    mask = jnp.stack((geometry.manipulated_mask, geometry.target_mask), axis=1).astype(jnp.bool_)
    return ObjectSetInput(
        object_xyz_m=jnp.pad(xyz, ((0, 0), (0, slots - 2), (0, 0), (0, 0))),
        object_mask=jnp.pad(mask, ((0, 0), (0, slots - 2), (0, 0))),
        object_valid=jnp.pad(jnp.any(mask, axis=-1), ((0, 0), (0, slots - 2))),
    )


class RelationSetV1Encoder(FinalV2RelationEncoder):
    """V2 geometry layers with a role-free ordered set of object candidates."""

    def __init__(self, *, rngs: nnx.Rngs, dim: int = 128):
        super().__init__(rngs=rngs, dim=dim)
        # V2's two role vectors are relation-type offsets, not object roles.

    def __call__(self, objects, hand_xyz, hand_mask, membership):
        xyz, mask, valid = objects.object_xyz_m, objects.object_mask, objects.object_valid
        if xyz.ndim != 4 or xyz.shape[-2:] != (512, 3):
            raise ValueError("object set must have shape [B,N,512,3]")
        if mask.shape != xyz.shape[:-1] or valid.shape != xyz.shape[:2]:
            raise ValueError("object set mask/valid shapes do not match object slots")
        if hand_xyz.shape[-2:] != (18, 3) or membership.shape != (5, 18):
            raise ValueError("expected 18 hand keypoints and (5,18) membership")
        if xyz.shape[1] < 2:
            padding = 2 - xyz.shape[1]
            xyz = jnp.pad(xyz, ((0, 0), (0, padding), (0, 0), (0, 0)))
            mask = jnp.pad(mask, ((0, 0), (0, padding), (0, 0)))
            valid = jnp.pad(valid, ((0, 0), (0, padding)))
        batch, slots = xyz.shape[:2]
        mask = mask.astype(jnp.bool_) & valid[..., None].astype(jnp.bool_) & jnp.isfinite(xyz).all(axis=-1)
        xyz = jnp.where(mask[..., None], xyz, 0.0)
        h_mask = hand_mask.astype(jnp.bool_) & jnp.isfinite(hand_xyz).all(axis=-1)
        h_xyz = jnp.where(h_mask[..., None], hand_xyz, 0.0)
        ho, ho_mask = _hand_object(self, xyz.reshape(batch, slots * 512, 3),
                                   mask.reshape(batch, slots * 512), h_xyz, h_mask, membership, self.ho_role.value)
        flat_xyz, flat_mask = xyz.reshape(batch * slots, 512, 3), mask.reshape(batch * slots, 512)
        zeros = jnp.zeros(flat_mask.shape, dtype=xyz.dtype)
        centers, anchor_mask, *_ = _local_anchors((flat_xyz, flat_mask, zeros, zeros, zeros), 8)
        centers, anchor_mask = centers.reshape(batch, slots, 8, 3), anchor_mask.reshape(batch, slots, 8)
        # Stable slots define direction: every unordered pair is evaluated once.
        left, right = jnp.triu_indices(slots, k=1)
        delta = centers[:, right, None, :, :] - centers[:, left, :, None, :]
        pair_mask = anchor_mask[:, left, :, None] & anchor_mask[:, right, None, :]
        features = jnp.concatenate((delta, _distance(delta)), axis=-1)
        embedded = self.oo_out(jax.nn.gelu(self.oo_in(features))).reshape(batch, -1, self.oo_queries.value.shape[-1])
        pair_mask = pair_mask.reshape(batch, -1)
        logits = jnp.einsum("bnd,qd->bqn", embedded, self.oo_queries.value)
        weights = _masked_softmax(logits, pair_mask[:, None, :])
        oo = jnp.einsum("bqn,bnd->bqd", weights, embedded) + self.oo_role.value
        oo_mask = jnp.broadcast_to(jnp.any(pair_mask, axis=-1)[:, None], (batch, 5))
        return ho, ho_mask, jnp.where(oo_mask[..., None], oo, 0.0), oo_mask


class HandObjectV3Encoder(nnx.Module):
    """Shared Final V2 hand-object calculation for each object role."""

    def __init__(self, *, rngs: nnx.Rngs, dim: int = 128):
        self.ho = nnx.Linear(4, dim, rngs=rngs)
        self.ho_neighbor_pool = nnx.Linear(2 * dim, dim, rngs=rngs)
        self.ho_finger_pool = nnx.Linear(2 * dim, dim, rngs=rngs)
        self.ho_finger = nnx.Param(jax.random.normal(rngs.params(), (5, dim)) * 0.02)
        self.ho_role = nnx.Param(jax.random.normal(rngs.params(), (2, dim)) * 0.02)

    def __call__(self, manipulated_xyz, manipulated_mask, target_xyz, target_mask, hand_xyz, hand_mask, membership):
        if hand_xyz.shape[-2:] != (18, 3) or membership.shape != (5, 18):
            raise ValueError("expected 18 hand keypoints and a validated (5,18) finger membership")
        h_mask = hand_mask.astype(jnp.bool_) & jnp.isfinite(hand_xyz).all(axis=-1)
        h_xyz = jnp.where(h_mask[..., None], hand_xyz, 0.0)
        outputs = []
        for role, (xyz, mask) in enumerate(((manipulated_xyz, manipulated_mask), (target_xyz, target_mask))):
            if xyz.shape[-2:] != (512, 3):
                raise ValueError("expected 512 points per object")
            valid = mask.astype(jnp.bool_) & jnp.isfinite(xyz).all(axis=-1)
            safe_xyz = jnp.where(valid[..., None], xyz, 0.0)
            outputs.append(_hand_object(self, safe_xyz, valid, h_xyz, h_mask, membership, self.ho_role.value[role]))
        return outputs[0][0], outputs[0][1], outputs[1][0], outputs[1][1]
