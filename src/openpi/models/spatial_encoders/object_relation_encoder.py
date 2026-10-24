"""Small optional HO/OO relation-token encoder for Level-1 object geometry."""

from __future__ import annotations

from flax import nnx
import jax
import jax.numpy as jnp

from openpi.models.spatial_encoders.types import ObjectGeometryInput, TactileSpatialInput


class ObjectRelationEncoder(nnx.Module):
    """Produce 5 hand-object and 9 object-object tokens at width ``token_dim``."""

    NUM_FINGERS = 5
    NUM_LOCAL_OBJECT_ANCHORS = 8
    NUM_LOCAL_RELATION_TOKENS = 4

    def __init__(self, *, token_dim: int, hidden_dim: int, candidate_k: int, rngs: nnx.Rngs) -> None:
        self.token_dim = int(token_dim)
        self.candidate_k = int(candidate_k)
        if self.token_dim <= 0 or hidden_dim <= 0 or self.candidate_k <= 0:
            raise ValueError("relation dimensions must be positive")
        self.ho_layers = _make_mlp(7, (hidden_dim, self.token_dim), rngs)
        self.ho_query = nnx.Param(jax.random.normal(rngs.params(), (self.NUM_FINGERS, self.token_dim)) * 0.02)
        self.ho_finger_embedding = nnx.Param(
            jax.random.normal(rngs.params(), (self.NUM_FINGERS, self.token_dim)) * 0.02
        )
        self.oo_layers = _make_mlp(9, (hidden_dim, self.token_dim), rngs)
        self.oo_query = nnx.Param(
            jax.random.normal(
                rngs.params(), (2, self.NUM_LOCAL_RELATION_TOKENS, self.token_dim)
            ) * 0.02
        )
        self.oo_role_embedding = nnx.Param(jax.random.normal(rngs.params(), (2, self.token_dim)) * 0.02)
        self.oo_global_layers = _make_mlp(18, (hidden_dim, self.token_dim), rngs)

    def __call__(
        self,
        geometry: ObjectGeometryInput,
        tactile: TactileSpatialInput | None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        if tactile is None:
            raise ValueError("Object relations require existing tactile geometry for HO anchors")
        manipulated = _finite_geometry(
            geometry.manipulated_xyz_m,
            geometry.manipulated_mask,
            geometry.manipulated_source,
            geometry.manipulated_age,
            geometry.manipulated_confidence,
        )
        target = _finite_geometry(
            geometry.target_xyz_m,
            geometry.target_mask,
            geometry.target_source,
            geometry.target_age,
            geometry.target_confidence,
        )
        ho_tokens, ho_mask, ho_xyz, ho_weights = self._hand_object(manipulated, tactile)
        oo_tokens, oo_mask, oo_xyz, oo_weights = self._object_object(manipulated, target)
        tokens = jnp.concatenate([ho_tokens, oo_tokens], axis=1)
        token_mask = jnp.concatenate([ho_mask, oo_mask], axis=1)
        token_xyz = jnp.concatenate([ho_xyz, oo_xyz], axis=1)
        tokens = jnp.where(token_mask[..., None], tokens, jnp.zeros_like(tokens))
        token_xyz = jnp.where(token_mask[..., None], token_xyz, jnp.zeros_like(token_xyz))
        return tokens, token_mask, token_xyz, {
            "object_relation_token_count": jnp.sum(token_mask, axis=1, dtype=jnp.int32),
            "ho_attention_weights": ho_weights,
            "oo_attention_weights": oo_weights,
        }

    def _hand_object(self, manipulated, tactile):
        xyz, mask, source, age, confidence = manipulated
        tactile_mask = tactile.point_mask.astype(jnp.bool_)
        finger_id = tactile.finger_id.astype(jnp.int32)
        anchor_parts, valid_parts = [], []
        for finger in range(self.NUM_FINGERS):
            valid = tactile_mask & (finger_id == finger)
            count = jnp.sum(valid, axis=1, keepdims=True)
            anchor_parts.append(jnp.sum(tactile.xyz_m * valid[..., None], axis=1) / jnp.maximum(count, 1))
            valid_parts.append(count[:, 0] > 0)
        anchors = jnp.stack(anchor_parts, axis=1)
        anchor_valid = jnp.stack(valid_parts, axis=1)
        k = min(self.candidate_k, xyz.shape[1])
        distances = jnp.sum((anchors[:, :, None, :] - xyz[:, None, :, :]) ** 2, axis=-1)
        distances = jnp.where(mask[:, None, :], distances, jnp.asarray(jnp.inf, distances.dtype))
        _, indices = jax.lax.top_k(-distances, k)
        candidate_xyz = _take_finger_points(xyz, indices)
        candidate_source = _take_finger_values(source, indices)
        candidate_age = _take_finger_values(age, indices)
        candidate_confidence = _take_finger_values(confidence, indices)
        candidate_mask = anchor_valid[..., None] & _take_finger_values(mask, indices)
        delta = candidate_xyz - anchors[:, :, None, :]
        feature = jnp.concatenate(
            [delta, jnp.linalg.norm(delta, axis=-1, keepdims=True), candidate_source[..., None],
             candidate_age[..., None] / 10.0, candidate_confidence[..., None]], axis=-1
        )
        embedded = _apply_mlp(self.ho_layers, feature)
        logits = jnp.einsum("bfkd,fd->bfk", embedded, self.ho_query.value)
        weights = _masked_softmax(logits, candidate_mask)
        tokens = jnp.einsum("bfk,bfkd->bfd", weights, embedded) + self.ho_finger_embedding.value[None, :, :]
        token_mask = jnp.any(candidate_mask, axis=-1)
        token_xyz = jnp.einsum("bfk,bfkd->bfd", weights, candidate_xyz)
        return tokens, token_mask, token_xyz, weights

    def _object_object(self, manipulated, target):
        m = _local_anchors(manipulated, self.NUM_LOCAL_OBJECT_ANCHORS)
        t = _local_anchors(target, self.NUM_LOCAL_OBJECT_ANCHORS)
        m_xyz, m_mask, m_source, m_age, m_conf = m
        t_xyz, t_mask, t_source, t_age, t_conf = t
        delta_mt = t_xyz[:, None, :, :] - m_xyz[:, :, None, :]
        delta_tm = -delta_mt.transpose(0, 2, 1, 3)
        pairs = []
        masks = []
        pair_xyz = []
        for role, (delta, src, smask, sage, sconf, dst, dmask, dage, dconf) in enumerate(
            ((delta_mt, m, m_mask, m_age, m_conf, t_xyz, t_mask, t_age, t_conf),
             (delta_tm, t, t_mask, t_age, t_conf, m_xyz, m_mask, m_age, m_conf))
        ):
            src_xyz = m_xyz if role == 0 else t_xyz
            dst_xyz = t_xyz if role == 0 else m_xyz
            src_source = m_source if role == 0 else t_source
            dst_source = t_source if role == 0 else m_source
            distance = jnp.linalg.norm(delta, axis=-1, keepdims=True)
            features = jnp.concatenate(
                [delta, distance, src_source[:, :, None, None] + jnp.zeros_like(distance),
                 dst_source[:, None, :, None] + jnp.zeros_like(distance),
                 sage[:, :, None, None] / 10.0 + jnp.zeros_like(distance),
                 dage[:, None, :, None] / 10.0 + jnp.zeros_like(distance),
                 sconf[:, :, None, None] + jnp.zeros_like(distance)], axis=-1
            )
            pairs.append(features)
            masks.append(smask[:, :, None] & dmask[:, None, :])
            pair_xyz.append(dst_xyz[:, None, :, :] + jnp.zeros_like(src_xyz[:, :, None, :]))
        features = jnp.stack(pairs, axis=1)
        pair_mask = jnp.stack(masks, axis=1)
        candidate_xyz = jnp.stack(pair_xyz, axis=1)
        embedded = _apply_mlp(self.oo_layers, features)
        flat_emb = embedded.reshape(embedded.shape[0], 2, -1, self.token_dim)
        flat_mask = pair_mask.reshape(pair_mask.shape[0], 2, -1)
        logits = jnp.einsum("brnd,rqd->brqn", flat_emb, self.oo_query.value)
        weights = _masked_softmax(logits, flat_mask[:, :, None, :])
        tokens = jnp.einsum("brqn,brnd->brqd", weights, flat_emb)
        tokens = tokens + self.oo_role_embedding.value[None, :, None, :]
        tokens = tokens.reshape(tokens.shape[0], -1, self.token_dim)
        local_mask = jnp.broadcast_to(
            jnp.any(flat_mask, axis=-1)[:, :, None],
            (flat_mask.shape[0], 2, self.NUM_LOCAL_RELATION_TOKENS),
        ).reshape(flat_mask.shape[0], -1)
        local_xyz = jnp.einsum("brqn,brnd->brqd", weights, candidate_xyz.reshape(candidate_xyz.shape[0], 2, -1, 3))
        local_xyz = local_xyz.reshape(local_xyz.shape[0], -1, 3)
        m_valid = jnp.any(m_mask, axis=1); t_valid = jnp.any(t_mask, axis=1)
        m_center = _masked_mean(m_xyz, m_mask); t_center = _masked_mean(t_xyz, t_mask)
        global_feature = jnp.concatenate(
            [m_center, t_center, t_center - m_center, jnp.linalg.norm(t_center - m_center, axis=-1, keepdims=True),
             jnp.mean(m_source, axis=1, keepdims=True), jnp.mean(t_source, axis=1, keepdims=True),
             jnp.mean(m_age, axis=1, keepdims=True) / 10.0, jnp.mean(t_age, axis=1, keepdims=True) / 10.0,
             jnp.mean(m_conf, axis=1, keepdims=True), jnp.mean(t_conf, axis=1, keepdims=True),
             jnp.mean(m_mask, axis=1, keepdims=True), jnp.mean(t_mask, axis=1, keepdims=True)], axis=-1
        )
        global_token = _apply_mlp(self.oo_global_layers, global_feature)[:, None, :]
        global_mask = (m_valid & t_valid)[:, None]
        global_xyz = ((m_center + t_center) / 2.0)[:, None, :]
        return (
            jnp.concatenate([tokens, global_token], axis=1),
            jnp.concatenate([local_mask, global_mask], axis=1),
            jnp.concatenate([local_xyz, global_xyz], axis=1),
            weights,
        )


def _finite_geometry(xyz, mask, source, age, confidence):
    finite = jnp.isfinite(xyz).all(axis=-1) & jnp.isfinite(confidence)
    return xyz.astype(jnp.float32), mask.astype(jnp.bool_) & finite, source.astype(jnp.float32), age.astype(jnp.float32), confidence.astype(jnp.float32)


def _local_anchors(geometry, count):
    xyz, mask, source, age, confidence = geometry
    n = xyz.shape[1]
    ids = jnp.minimum(jnp.arange(n) * count // n, count - 1)
    parts = []
    for index in range(count):
        valid = mask & (ids[None, :] == index)
        denom = jnp.maximum(jnp.sum(valid, axis=1, keepdims=True), 1)
        parts.append((jnp.sum(xyz * valid[..., None], axis=1) / denom, jnp.any(valid, axis=1)))
    centers = jnp.stack([part[0] for part in parts], axis=1)
    valid = jnp.stack([part[1] for part in parts], axis=1)
    # Deterministic chunk summaries for provenance fields.
    summary = []
    for values in (source, age, confidence):
        summary.append(jnp.stack([_masked_mean(values[:, :, None], mask & (ids[None, :] == index))[:, 0] for index in range(count)], axis=1))
    return centers, valid, *summary


def _masked_mean(values, mask):
    weights = mask.astype(values.dtype)
    return jnp.sum(values * weights[..., None], axis=-2) / jnp.maximum(jnp.sum(weights, axis=-1, keepdims=True), 1.0)


def _take_finger_points(values, indices):
    values = jnp.broadcast_to(values[:, None, :, :], (values.shape[0], indices.shape[1], values.shape[1], values.shape[2]))
    return jnp.take_along_axis(values, indices[..., None], axis=2)


def _take_finger_values(values, indices):
    values = jnp.broadcast_to(values[:, None, :], (values.shape[0], indices.shape[1], values.shape[1]))
    return jnp.take_along_axis(values, indices, axis=2)


def _masked_softmax(logits, mask):
    masked = jnp.where(mask, logits, jnp.asarray(-1e9, logits.dtype))
    weights = jax.nn.softmax(masked, axis=-1)
    weights = jnp.where(mask, weights, 0.0)
    return weights / jnp.maximum(jnp.sum(weights, axis=-1, keepdims=True), 1e-6)


def _make_mlp(input_dim: int, hidden_dims: tuple[int, ...], rngs: nnx.Rngs) -> list[nnx.Linear]:
    dims = (input_dim, *hidden_dims)
    return [nnx.Linear(dims[i], dims[i + 1], rngs=rngs) for i in range(len(dims) - 1)]


def _apply_mlp(layers: list[nnx.Linear], values: jax.Array) -> jax.Array:
    hidden = values
    for index, layer in enumerate(layers):
        hidden = layer(hidden)
        if index != len(layers) - 1:
            hidden = jax.nn.gelu(hidden)
    return hidden
