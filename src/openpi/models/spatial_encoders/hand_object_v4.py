"""Hand-object relations at local-neighborhood and fixed global shape scales."""
from flax import nnx
import jax
import jax.numpy as jnp

from openpi.models.spatial_encoders.final_v2_relations import HandObjectV3Encoder
from openpi.models.spatial_encoders.final_v2_relations import _distance
from openpi.models.spatial_encoders.structured_spatial_encoder import _batched_take
from openpi.models.spatial_encoders.structured_spatial_encoder import _masked_max
from openpi.models.spatial_encoders.structured_spatial_encoder import _masked_mean


def global_relations(hand_xyz, anchors):
    """Ordered relative xyz and distance: [B,18,16,4]."""
    delta = anchors[:, None, :, :] - hand_xyz[:, :, None, :]
    return jnp.concatenate((delta, _distance(delta)), axis=-1)


class HandObjectV4Encoder(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs, dim: int = 128):
        self.local = HandObjectV3Encoder(rngs=rngs, dim=dim)
        self.global_in = nnx.Linear(64, dim, rngs=rngs)
        self.global_out = nnx.Linear(dim, dim, rngs=rngs)
        self.global_finger = nnx.Linear(3 * dim, dim, rngs=rngs)
        self.local_norm = nnx.LayerNorm(dim, rngs=rngs)
        self.global_norm = nnx.LayerNorm(dim, rngs=rngs)
        self.fuse_in = nnx.Linear(2 * dim, dim, rngs=rngs)
        self.fuse_out = nnx.Linear(dim, dim, rngs=rngs)

    def __call__(self, manipulated_xyz, manipulated_mask, target_xyz, target_mask,
                 hand_xyz, hand_mask, membership, manipulated_indices, target_indices):
        local_m, mask_m, local_t, mask_t = self.local(manipulated_xyz, manipulated_mask, target_xyz, target_mask,
                                                       hand_xyz, hand_mask, membership)
        hand_valid = hand_mask.astype(bool) & jnp.isfinite(hand_xyz).all(axis=-1)
        safe_hand = jnp.where(hand_valid[..., None], hand_xyz, 0.0)
        outputs = []
        for xyz, mask, indices, local, local_mask in (
            (manipulated_xyz, manipulated_mask, manipulated_indices, local_m, mask_m),
            (target_xyz, target_mask, target_indices, local_t, mask_t),
        ):
            if indices.shape != (xyz.shape[0], 16):
                raise ValueError("fixed object anchor indices must have shape [batch,16]")
            valid = mask.astype(bool) & jnp.isfinite(xyz).all(axis=-1)
            safe_xyz = jnp.where(valid[..., None], xyz, 0.0)
            anchors = _batched_take(safe_xyz, indices)
            anchor_valid = _batched_take(valid, indices)
            relation = global_relations(safe_hand, anchors)
            relation = jnp.where(anchor_valid[:, None, :, None] & hand_valid[:, :, None, None], relation, 0.0)
            point = self.global_out(jax.nn.gelu(self.global_in(relation.reshape(xyz.shape[0], 18, 64))))
            point = jnp.where(hand_valid[..., None] & jnp.any(anchor_valid, axis=1)[:, None, None], point, 0.0)
            finger_mask = hand_valid[:, None, :] & membership[None, :, :].astype(bool)
            expanded = jnp.broadcast_to(point[:, None, :, :], (*finger_mask.shape, point.shape[-1]))
            palm = jnp.broadcast_to(point[:, None, 0, :], (xyz.shape[0], 5, point.shape[-1]))
            pooled = jnp.concatenate((_masked_mean(expanded, finger_mask, axis=2),
                                      _masked_max(expanded, finger_mask, axis=2), palm), axis=-1)
            global_feature = self.global_finger(pooled)
            fused = self.fuse_out(jax.nn.gelu(self.fuse_in(jnp.concatenate(
                (self.local_norm(local), self.global_norm(global_feature)), axis=-1))))
            outputs.append((jnp.where(local_mask[..., None], fused, 0.0), local_mask))
        return outputs[0][0], outputs[0][1], outputs[1][0], outputs[1][1]
