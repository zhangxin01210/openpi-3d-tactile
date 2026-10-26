"""Optional legacy HO/OO and Hand-Scene relation tokens for Level-1 geometry."""

from __future__ import annotations

from flax import nnx
import jax
import jax.numpy as jnp

from openpi.models.spatial_encoders.types import ObjectGeometryInput
from openpi.models.spatial_encoders.types import TactileSpatialInput


class ObjectRelationEncoder(nnx.Module):
    """Small causal relation-token encoder for Level-1 object geometry.

    ``legacy_ho_oo`` preserves the original five Hand-Object plus nine
    Object-Object token layout.  ``hand_scene_v0`` instead emits five
    Hand-Funnel and five Hand-Target tokens.  Keeping both layouts here makes
    old relational checkpoints and the new data contract independently usable.
    """

    NUM_FINGERS = 5
    NUM_LOCAL_OBJECT_ANCHORS = 8
    NUM_LOCAL_RELATION_TOKENS = 4

    def __init__(
        self,
        *,
        token_dim: int,
        hidden_dim: int,
        candidate_k: int,
        relation_mode: str = "legacy_ho_oo",
        enable_funnel_mask_encoder: bool = False,
        funnel_mask_size: int = 32,
        rngs: nnx.Rngs,
    ) -> None:
        self.token_dim = int(token_dim)
        self.candidate_k = int(candidate_k)
        self.relation_mode = str(relation_mode)
        self.enable_funnel_mask_encoder = bool(enable_funnel_mask_encoder)
        self.funnel_mask_size = int(funnel_mask_size)
        if self.token_dim <= 0 or hidden_dim <= 0 or self.candidate_k <= 0 or self.funnel_mask_size <= 0:
            raise ValueError("relation dimensions must be positive")
        if self.relation_mode not in {
            "legacy_ho_oo",
            "hand_scene_v0",
            "hand_scene_oo_v1",
            "dtcr_v1",
            "dtcr_independent_v1",
        }:
            raise ValueError(f"Unsupported relation_mode: {self.relation_mode!r}")

        # Hand-Scene adds one scalar geometry-quality feature.  The legacy
        # branch deliberately retains its old dimensions and parameter names.
        ho_input_dim = 7 if self.relation_mode == "legacy_ho_oo" else 8
        self.ho_layers = _make_mlp(ho_input_dim, (hidden_dim, self.token_dim), rngs)
        self.ho_query = nnx.Param(jax.random.normal(rngs.params(), (self.NUM_FINGERS, self.token_dim)) * 0.02)
        self.ho_finger_embedding = nnx.Param(
            jax.random.normal(rngs.params(), (self.NUM_FINGERS, self.token_dim)) * 0.02
        )
        if self.relation_mode == "legacy_ho_oo":
            self.oo_layers = _make_mlp(9, (hidden_dim, self.token_dim), rngs)
            self.oo_query = nnx.Param(
                jax.random.normal(
                    rngs.params(), (2, self.NUM_LOCAL_RELATION_TOKENS, self.token_dim)
                ) * 0.02
            )
            self.oo_role_embedding = nnx.Param(jax.random.normal(rngs.params(), (2, self.token_dim)) * 0.02)
            self.oo_global_layers = _make_mlp(18, (hidden_dim, self.token_dim), rngs)
        else:
            self.hand_scene_role_embedding = nnx.Param(
                jax.random.normal(rngs.params(), (2, self.token_dim)) * 0.02
            )
            # [current force magnitude, stable reference, difference, short
            # delta, contact-valid, finger-local xyz].  Magnitudes are
            # coordinate invariant and never compare base-frame force vectors
            # from different hand poses.
            self.grasp_taxel_layers = _make_mlp(8, (hidden_dim, self.token_dim), rngs)
            self.grasp_adapter = nnx.Linear(self.token_dim, self.token_dim, rngs=rngs)
            self.grasp_gate = nnx.Linear(self.token_dim, 1, rngs=rngs)
            if self.relation_mode in {"hand_scene_oo_v1", "dtcr_v1", "dtcr_independent_v1"}:
                self.funnel_target_layers = _make_mlp(10, (hidden_dim, self.token_dim), rngs)
                self.funnel_target_query = nnx.Param(
                    jax.random.normal(rngs.params(), (self.NUM_FINGERS, self.token_dim)) * 0.02
                )
                self.funnel_target_role_embedding = nnx.Param(
                    jax.random.normal(rngs.params(), (self.token_dim,)) * 0.02
                )
            if self.relation_mode in {"dtcr_v1", "dtcr_independent_v1"}:
                # Eight scalar channels plus taxel xyz: current force, EMA
                # before/after, both causal differences, two validity flags,
                # and the current-sample validity flag.
                self.dtcr_hand_taxel_layers = _make_mlp(11, (hidden_dim, self.token_dim), rngs)
                self.dtcr_change_taxel_layers = _make_mlp(10, (hidden_dim, self.token_dim), rngs)
                self.dtcr_hand_adapter = nnx.Linear(self.token_dim, self.token_dim, rngs=rngs)
                self.dtcr_change_adapter = nnx.Linear(self.token_dim, self.token_dim, rngs=rngs)
                self.dtcr_hand_key = nnx.Linear(self.token_dim, self.token_dim, rngs=rngs)
                self.dtcr_hand_value = nnx.Linear(self.token_dim, self.token_dim, rngs=rngs)
                self.dtcr_ft_query = nnx.Linear(self.token_dim, self.token_dim, rngs=rngs)
            if self.enable_funnel_mask_encoder:
                # Eight current-frame shape/hand-offset moments keep this
                # optional branch intentionally lightweight.
                self.funnel_mask_layers = _make_mlp(8, (hidden_dim, self.token_dim), rngs)

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
        if self.relation_mode == "legacy_ho_oo":
            ho_tokens, ho_mask, ho_xyz, ho_weights = self._hand_object(manipulated, tactile)
            oo_tokens, oo_mask, oo_xyz, oo_weights = self._object_object(manipulated, target)
            tokens = jnp.concatenate([ho_tokens, oo_tokens], axis=1)
            token_mask = jnp.concatenate([ho_mask, oo_mask], axis=1)
            token_xyz = jnp.concatenate([ho_xyz, oo_xyz], axis=1)
            aux = {
                "object_relation_token_count": jnp.sum(token_mask, axis=1, dtype=jnp.int32),
                "ho_attention_weights": ho_weights,
                "oo_attention_weights": oo_weights,
            }
        elif self.relation_mode == "hand_scene_v0":
            hand_scene = geometry.hand_scene
            batch_size = geometry.manipulated_xyz_m.shape[0]
            funnel_valid, target_valid, funnel_quality, target_quality = _hand_scene_relation_state(
                hand_scene, batch_size
            )
            funnel_tokens, funnel_mask, funnel_xyz, funnel_weights = self._hand_object(
                manipulated,
                tactile,
                relation_valid=funnel_valid,
                geometry_quality=funnel_quality,
            )
            target_tokens, target_mask, target_xyz, target_weights = self._hand_object(
                target,
                tactile,
                relation_valid=target_valid,
                geometry_quality=target_quality,
            )
            funnel_tokens = funnel_tokens + self.hand_scene_role_embedding.value[None, 0, None, :]
            target_tokens = target_tokens + self.hand_scene_role_embedding.value[None, 1, None, :]

            grasp_descriptor, grasp_gate = self._grasp_descriptor(tactile, hand_scene)
            # This is an adapter rather than a replacement: target geometry
            # remains present when contact is weak or absent, while valid
            # per-finger tactile evidence directly conditions every target
            # relation token.
            target_tokens = target_tokens + grasp_gate[..., None] * self.grasp_adapter(grasp_descriptor)
            funnel_mask_embedding, funnel_mask_feature_valid = self._funnel_mask_embedding(hand_scene, batch_size)
            funnel_tokens = funnel_tokens + funnel_mask_embedding[:, None, :]

            tokens = jnp.concatenate([funnel_tokens, target_tokens], axis=1)
            token_mask = jnp.concatenate([funnel_mask, target_mask], axis=1)
            token_xyz = jnp.concatenate([funnel_xyz, target_xyz], axis=1)
            aux = {
                "object_relation_token_count": jnp.sum(token_mask, axis=1, dtype=jnp.int32),
                "hand_funnel_attention_weights": funnel_weights,
                "hand_target_attention_weights": target_weights,
                "grasp_finger_gate": grasp_gate,
                "funnel_mask_feature_valid": funnel_mask_feature_valid,
            }
        else:
            hand_scene = geometry.hand_scene
            batch_size = geometry.manipulated_xyz_m.shape[0]
            funnel_valid, target_valid, funnel_quality, target_quality = _hand_scene_relation_state(
                hand_scene, batch_size
            )
            funnel_tokens, funnel_mask, funnel_xyz, funnel_weights = self._hand_object(
                manipulated,
                tactile,
                relation_valid=funnel_valid,
                geometry_quality=funnel_quality,
            )
            funnel_tokens = funnel_tokens + self.hand_scene_role_embedding.value[None, 0, None, :]

            target_tokens, target_mask, target_xyz, target_weights = self._funnel_target(
                manipulated,
                target,
                # Manipulated_relation_valid describes the hand-attached
                # relation only.  Once the funnel is lifted that relation
                # becomes invalid, while a valid funnel mesh and the static
                # target prior still define Funnel--Target geometry.
                relation_valid=target_valid,
                funnel_quality=funnel_quality,
                target_quality=target_quality,
            )
            target_tokens = target_tokens + self.funnel_target_role_embedding.value[None, None, :]
            change_descriptor = jnp.zeros_like(funnel_tokens)
            change_valid = jnp.zeros((batch_size, self.NUM_FINGERS), dtype=jnp.bool_)
            if self.relation_mode in {"dtcr_v1", "dtcr_independent_v1"}:
                hand_descriptor, change_descriptor, change_valid = self._dtcr_tactile_descriptors(tactile, hand_scene)
                funnel_tokens = funnel_tokens + self.dtcr_hand_adapter(hand_descriptor) * change_valid[..., None]
                target_tokens = self._condition_funnel_target(
                    target_tokens,
                    target_mask,
                    funnel_tokens,
                    funnel_mask,
                    change_descriptor,
                    change_valid,
                    use_hand=self.relation_mode == "dtcr_v1",
                )

            tokens = jnp.concatenate([funnel_tokens, target_tokens], axis=1)
            token_mask = jnp.concatenate([funnel_mask, target_mask], axis=1)
            token_xyz = jnp.concatenate([funnel_xyz, target_xyz], axis=1)
            aux = {
                "object_relation_token_count": jnp.sum(token_mask, axis=1, dtype=jnp.int32),
                "hand_funnel_attention_weights": funnel_weights,
                "funnel_target_attention_weights": target_weights,
                "dtcr_change_valid": change_valid,
            }
        tokens = jnp.where(token_mask[..., None], tokens, jnp.zeros_like(tokens))
        token_xyz = jnp.where(token_mask[..., None], token_xyz, jnp.zeros_like(token_xyz))
        return tokens, token_mask, token_xyz, aux

    def _hand_object(self, manipulated, tactile, *, relation_valid=None, geometry_quality=None):
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
        feature_parts = [
            delta,
            jnp.linalg.norm(delta, axis=-1, keepdims=True),
            candidate_source[..., None],
            candidate_age[..., None] / 10.0,
            candidate_confidence[..., None],
        ]
        if self.relation_mode in {"hand_scene_v0", "hand_scene_oo_v1", "dtcr_v1", "dtcr_independent_v1"}:
            assert relation_valid is not None
            assert geometry_quality is not None
            finite_quality = jnp.where(jnp.isfinite(geometry_quality), geometry_quality, 0.0)
            finite_quality = jnp.clip(finite_quality.astype(jnp.float32), 0.0, 1.0)
            candidate_mask = candidate_mask & relation_valid[:, None, None] & (finite_quality[:, None, None] > 0.0)
            feature_parts.append(jnp.broadcast_to(finite_quality[:, None, None, None], delta.shape[:-1] + (1,)))
        feature = jnp.concatenate(feature_parts, axis=-1)
        embedded = _apply_mlp(self.ho_layers, feature)
        logits = jnp.einsum("bfkd,fd->bfk", embedded, self.ho_query.value)
        weights = _masked_softmax(logits, candidate_mask)
        tokens = jnp.einsum("bfk,bfkd->bfd", weights, embedded) + self.ho_finger_embedding.value[None, :, :]
        token_mask = jnp.any(candidate_mask, axis=-1)
        token_xyz = jnp.einsum("bfk,bfkd->bfd", weights, candidate_xyz)
        if self.relation_mode == "hand_scene_v0":
            tokens = tokens * finite_quality[:, None, None]
        return tokens, token_mask, token_xyz, weights

    def _funnel_target(self, funnel, target, *, relation_valid, funnel_quality, target_quality):
        """Encode five deterministic Funnel--Target relation tokens."""

        funnel_anchors = _local_anchors(funnel, self.NUM_LOCAL_OBJECT_ANCHORS)
        target_anchors = _local_anchors(target, self.NUM_LOCAL_OBJECT_ANCHORS)
        funnel_xyz, funnel_mask, funnel_source, funnel_age, funnel_conf = funnel_anchors
        target_xyz, target_mask, target_source, target_age, target_conf = target_anchors
        delta = target_xyz[:, :, None, :] - funnel_xyz[:, None, :, :]
        distance = jnp.linalg.norm(delta, axis=-1, keepdims=True)
        features = jnp.concatenate(
            [
                delta,
                distance,
                funnel_source[:, :, None, None] + jnp.zeros_like(distance),
                target_source[:, None, :, None] + jnp.zeros_like(distance),
                funnel_age[:, :, None, None] / 10.0 + jnp.zeros_like(distance),
                target_age[:, None, :, None] / 10.0 + jnp.zeros_like(distance),
                funnel_conf[:, :, None, None] + jnp.zeros_like(distance),
                target_conf[:, None, :, None] + jnp.zeros_like(distance),
            ],
            axis=-1,
        )
        pair_mask = funnel_mask[:, :, None] & target_mask[:, None, :]
        quality = jnp.clip(
            jnp.where(jnp.isfinite(funnel_quality), funnel_quality, 0.0)
            * jnp.where(jnp.isfinite(target_quality), target_quality, 0.0),
            0.0,
            1.0,
        )
        pair_mask = pair_mask & relation_valid[:, None, None] & (quality[:, None, None] > 0.0)
        embedded = _apply_mlp(self.funnel_target_layers, features)
        flat_emb = embedded.reshape(embedded.shape[0], -1, self.token_dim)
        flat_mask = pair_mask.reshape(pair_mask.shape[0], -1)
        logits = jnp.einsum("bnd,qd->bqn", flat_emb, self.funnel_target_query.value)
        weights = _masked_softmax(logits, flat_mask[:, None, :])
        tokens = jnp.einsum("bqn,bnd->bqd", weights, flat_emb)
        token_mask = jnp.broadcast_to(jnp.any(flat_mask, axis=-1)[:, None], (flat_mask.shape[0], self.NUM_FINGERS))
        pair_xyz = target_xyz[:, None, :, :] + jnp.zeros_like(funnel_xyz[:, :, None, :])
        token_xyz = jnp.einsum("bqn,bnd->bqd", weights, pair_xyz.reshape(pair_xyz.shape[0], -1, 3))
        return tokens * quality[:, None, None], token_mask, token_xyz, weights

    def _dtcr_tactile_descriptors(self, tactile, hand_scene):
        """Pool slow and fast causal tactile channels per finger."""

        current = jnp.where(jnp.isfinite(tactile.force_norm), tactile.force_norm, 0.0).astype(jnp.float32)
        if hand_scene is not None and hand_scene.causal_tactile is not None:
            causal = hand_scene.causal_tactile
            ema_before = jnp.where(jnp.isfinite(causal.ema_before_force_norm), causal.ema_before_force_norm, 0.0)
            ema_after = jnp.where(jnp.isfinite(causal.ema_after_force_norm), causal.ema_after_force_norm, 0.0)
            current_minus_ema = jnp.where(
                jnp.isfinite(causal.current_minus_ema_force_norm), causal.current_minus_ema_force_norm, 0.0
            )
            current_minus_previous = jnp.where(
                jnp.isfinite(causal.current_minus_previous_force_norm), causal.current_minus_previous_force_norm, 0.0
            )
            reference_valid = causal.reference_valid.astype(jnp.float32)
            previous_valid = causal.previous_valid.astype(jnp.float32)
            sample_valid = causal.sample_valid.astype(jnp.float32)
        else:
            # Compatibility fallback for a reviewed pre-DTCR sidecar.  DTCR
            # production data should carry the explicit CausalTactileInput.
            ema_before = jnp.where(
                hand_scene is not None and jnp.isfinite(hand_scene.grasp_reference_force_norm),
                hand_scene.grasp_reference_force_norm if hand_scene is not None else 0.0,
                0.0,
            )
            ema_after = ema_before
            current_minus_ema = current - ema_before
            current_minus_previous = (
                hand_scene.grasp_short_delta_force_norm
                if hand_scene is not None
                else jnp.zeros_like(current)
            )
            reference_valid = jnp.broadcast_to(
                (hand_scene is not None) and jnp.ones_like(current, dtype=jnp.bool_), current.shape
            ).astype(jnp.float32)
            previous_valid = reference_valid
            sample_valid = tactile.point_mask.astype(jnp.float32)

        safe_xyz = jnp.where(jnp.isfinite(tactile.xyz_m), tactile.xyz_m, 0.0).astype(jnp.float32)
        finger_id = tactile.finger_id.astype(jnp.int32)
        point_valid = tactile.point_mask.astype(jnp.bool_) & (sample_valid > 0.0)
        hand_descriptors, change_descriptors, valid = [], [], []
        for finger in range(self.NUM_FINGERS):
            finger_mask = point_valid & (finger_id == finger)
            count = jnp.sum(finger_mask, axis=1, keepdims=True).astype(jnp.float32)
            denom = jnp.maximum(count, 1.0)
            xyz_center = jnp.sum(safe_xyz * finger_mask[..., None], axis=1) / denom
            xyz_local = safe_xyz - xyz_center[:, None, :]
            hand_feature = jnp.concatenate(
                [
                    current[..., None],
                    ema_before[..., None],
                    ema_after[..., None],
                    current_minus_ema[..., None],
                    current_minus_previous[..., None],
                    reference_valid[..., None],
                    previous_valid[..., None],
                    sample_valid[..., None],
                    xyz_local,
                ],
                axis=-1,
            )
            change_feature = jnp.concatenate(
                [
                    current[..., None],
                    ema_before[..., None],
                    current_minus_ema[..., None],
                    current_minus_previous[..., None],
                    reference_valid[..., None],
                    previous_valid[..., None],
                    sample_valid[..., None],
                    xyz_local,
                ],
                axis=-1,
            )
            hand_encoded = _apply_mlp(self.dtcr_hand_taxel_layers, hand_feature)
            change_encoded = _apply_mlp(self.dtcr_change_taxel_layers, change_feature)
            hand_descriptors.append(_masked_mean(hand_encoded, finger_mask))
            change_descriptors.append(_masked_mean(change_encoded, finger_mask))
            valid.append(count[:, 0] > 0)
        return jnp.stack(hand_descriptors, axis=1), jnp.stack(change_descriptors, axis=1), jnp.stack(valid, axis=1)

    def _condition_funnel_target(
        self,
        target_tokens,
        target_mask,
        funnel_tokens,
        funnel_mask,
        change_tokens,
        change_valid,
        *,
        use_hand: bool,
    ):
        """Apply relation attention from Hand--Funnel and tactile changes."""

        change_context = _masked_mean(change_tokens, change_valid)
        change_has_valid = jnp.any(change_valid, axis=1)
        target_tokens = target_tokens + self.dtcr_change_adapter(change_context)[:, None, :] * change_has_valid[:, None, None]
        if not use_hand:
            return target_tokens
        query = self.dtcr_ft_query(target_tokens)
        key = self.dtcr_hand_key(funnel_tokens)
        value = self.dtcr_hand_value(funnel_tokens)
        logits = jnp.einsum("bqd,bfd->bqf", query, key) / jnp.sqrt(float(self.token_dim))
        relation_mask = target_mask[:, :, None] & funnel_mask[:, None, :]
        attention = _masked_softmax(logits, relation_mask)
        return target_tokens + jnp.einsum("bqf,bfd->bqd", attention, value)

    def _grasp_descriptor(self, tactile, hand_scene):
        """Causally pool current/reference tactile force per finger."""
        batch_size, _, _ = tactile.xyz_m.shape
        if hand_scene is None:
            zeros = jnp.zeros((batch_size, self.NUM_FINGERS, self.token_dim), dtype=jnp.float32)
            return zeros, jnp.zeros((batch_size, self.NUM_FINGERS), dtype=jnp.float32)

        force = jnp.where(jnp.isfinite(tactile.force), tactile.force, 0.0).astype(jnp.float32)
        current = jnp.linalg.norm(force, axis=-1)
        reference = jnp.where(
            jnp.isfinite(hand_scene.grasp_reference_force_norm), hand_scene.grasp_reference_force_norm, 0.0
        ).astype(jnp.float32)
        short_delta = jnp.where(
            jnp.isfinite(hand_scene.grasp_short_delta_force_norm), hand_scene.grasp_short_delta_force_norm, 0.0
        ).astype(jnp.float32)
        contact_valid = (
            tactile.point_mask.astype(jnp.bool_)
            & hand_scene.grasp_contact_valid.astype(jnp.bool_)
        )
        finger_id = tactile.finger_id.astype(jnp.int32)
        safe_xyz = jnp.where(jnp.isfinite(tactile.xyz_m), tactile.xyz_m, 0.0).astype(jnp.float32)
        descriptors, gates = [], []
        grasp_quality = jnp.where(jnp.isfinite(hand_scene.grasp_quality), hand_scene.grasp_quality, 0.0)
        grasp_quality = jnp.clip(grasp_quality.astype(jnp.float32), 0.0, 1.0)
        for finger in range(self.NUM_FINGERS):
            finger_mask = contact_valid & (finger_id == finger)
            count = jnp.sum(finger_mask, axis=1, keepdims=True).astype(jnp.float32)
            center = jnp.sum(safe_xyz * finger_mask[..., None], axis=1) / jnp.maximum(count, 1.0)
            local_xyz = safe_xyz - center[:, None, :]
            feature = jnp.concatenate(
                [
                    current[..., None],
                    reference[..., None],
                    (current - reference)[..., None],
                    short_delta[..., None],
                    finger_mask.astype(jnp.float32)[..., None],
                    local_xyz,
                ],
                axis=-1,
            )
            encoded = _apply_mlp(self.grasp_taxel_layers, feature)
            descriptor = _masked_mean(encoded, finger_mask)
            has_contact = count[:, 0] > 0
            descriptor = jnp.where(has_contact[:, None], descriptor, 0.0)
            gate = jax.nn.sigmoid(self.grasp_gate(descriptor)[..., 0])
            gates.append(gate * has_contact.astype(jnp.float32) * grasp_quality)
            descriptors.append(descriptor * grasp_quality[:, None])
        return jnp.stack(descriptors, axis=1), jnp.stack(gates, axis=1)

    def _funnel_mask_embedding(self, hand_scene, batch_size):
        """Encode a current-frame mask without making it required for training."""
        zeros = jnp.zeros((batch_size, self.token_dim), dtype=jnp.float32)
        invalid = jnp.zeros((batch_size,), dtype=jnp.bool_)
        if not self.enable_funnel_mask_encoder or hand_scene is None:
            return zeros, invalid
        mask = hand_scene.funnel_mask.astype(jnp.bool_)
        height, width = mask.shape[-2:]
        y = jnp.linspace(-1.0, 1.0, height, dtype=jnp.float32)[None, :, None]
        x = jnp.linspace(-1.0, 1.0, width, dtype=jnp.float32)[None, None, :]
        weights = mask.astype(jnp.float32)
        count = jnp.sum(weights, axis=(1, 2))
        denom = jnp.maximum(count, 1.0)
        mean_x = jnp.sum(weights * x, axis=(1, 2)) / denom
        mean_y = jnp.sum(weights * y, axis=(1, 2)) / denom
        std_x = jnp.sqrt(jnp.sum(weights * (x - mean_x[:, None, None]) ** 2, axis=(1, 2)) / denom)
        std_y = jnp.sqrt(jnp.sum(weights * (y - mean_y[:, None, None]) ** 2, axis=(1, 2)) / denom)
        quality = jnp.where(jnp.isfinite(hand_scene.funnel_mask_quality), hand_scene.funnel_mask_quality, 0.0)
        quality = jnp.clip(quality.astype(jnp.float32), 0.0, 1.0)
        hand_offset = jnp.where(
            jnp.isfinite(hand_scene.funnel_mask_hand_offset_xy), hand_scene.funnel_mask_hand_offset_xy, 0.0
        ).astype(jnp.float32)
        valid = hand_scene.funnel_mask_valid.astype(jnp.bool_) & (count > 0) & (quality > 0.0)
        features = jnp.stack(
            [
                count / float(height * width), mean_x, mean_y, std_x, std_y, quality,
                hand_offset[:, 0], hand_offset[:, 1],
            ],
            axis=-1,
        )
        encoded = _apply_mlp(self.funnel_mask_layers, features) * quality[:, None]
        return jnp.where(valid[:, None], encoded, 0.0), valid

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
        for role, (delta, smask, sage, sconf, dmask, dage) in enumerate(
            ((delta_mt, m_mask, m_age, m_conf, t_mask, t_age),
             (delta_tm, t_mask, t_age, t_conf, m_mask, m_age))
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
        # Keep the global description on the original point sets.  In
        # particular, summaries over local anchors would give each region the
        # same weight rather than each observed point the same weight.
        m_center, m_source_mean, m_age_mean, m_conf_mean, m_coverage, m_valid = _global_statistics(manipulated)
        t_center, t_source_mean, t_age_mean, t_conf_mean, t_coverage, t_valid = _global_statistics(target)
        global_feature = jnp.concatenate(
            [m_center, t_center, t_center - m_center, jnp.linalg.norm(t_center - m_center, axis=-1, keepdims=True),
             m_source_mean, t_source_mean,
             m_age_mean / 10.0, t_age_mean / 10.0,
             m_conf_mean, t_conf_mean, m_coverage, t_coverage], axis=-1
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


def _hand_scene_relation_state(hand_scene, batch_size):
    """Return safe current-frame relation state; absent means explicitly invalid.

    In particular, this never falls back to treating the manipulated cloud as
    hand-attached.  The causal sidecar producer is the only authority that can
    state a Hand-Funnel relation remains valid after a lift.
    """
    if hand_scene is None:
        zeros_bool = jnp.zeros((batch_size,), dtype=jnp.bool_)
        zeros_float = jnp.zeros((batch_size,), dtype=jnp.float32)
        return zeros_bool, zeros_bool, zeros_float, zeros_float
    funnel_quality = jnp.where(
        jnp.isfinite(hand_scene.manipulated_geometry_quality), hand_scene.manipulated_geometry_quality, 0.0
    )
    target_quality = jnp.where(
        jnp.isfinite(hand_scene.target_geometry_quality), hand_scene.target_geometry_quality, 0.0
    )
    return (
        hand_scene.manipulated_relation_valid.astype(jnp.bool_),
        hand_scene.target_relation_valid.astype(jnp.bool_),
        jnp.clip(funnel_quality.astype(jnp.float32), 0.0, 1.0),
        jnp.clip(target_quality.astype(jnp.float32), 0.0, 1.0),
    )


def _finite_geometry(xyz, mask, source, age, confidence):
    xyz = xyz.astype(jnp.float32)
    source = source.astype(jnp.float32)
    age = age.astype(jnp.float32)
    confidence = confidence.astype(jnp.float32)
    finite = (
        jnp.isfinite(xyz).all(axis=-1)
        & jnp.isfinite(source)
        & jnp.isfinite(age)
        & jnp.isfinite(confidence)
    )
    # Multiplication by a false mask does not neutralize NaNs.  Normalize
    # malformed/inactive values here so they cannot poison reductions before
    # their mask is consulted.
    xyz = jnp.where(finite[..., None], xyz, 0.0)
    source = jnp.where(finite, source, 0.0)
    age = jnp.where(finite, age, 0.0)
    confidence = jnp.where(finite, confidence, 0.0)
    return xyz, mask.astype(jnp.bool_) & finite, source, age, confidence


def _local_anchors(geometry, count):
    """Pool valid points into deterministic, spatial FPS regions.

    Array position is deliberately never used as a region identifier: anchors
    are selected with masked farthest-point sampling, then every valid point is
    assigned to its nearest selected anchor.  The first FPS anchor is the
    lexicographically smallest valid point, which gives a stable geometric
    origin instead of depending on input point order.
    """
    xyz, mask, source, age, confidence = geometry
    anchor_xyz, anchor_mask = _masked_fps_anchors(xyz, mask, count)

    distances = jnp.sum((xyz[:, :, None, :] - anchor_xyz[:, None, :, :]) ** 2, axis=-1)
    distances = jnp.where(anchor_mask[:, None, :], distances, jnp.asarray(jnp.inf, distances.dtype))
    nearest_anchor = jnp.argmin(distances, axis=-1)
    regions = (
        mask[:, :, None]
        & anchor_mask[:, None, :]
        & (nearest_anchor[:, :, None] == jnp.arange(count)[None, None, :])
    )
    counts = jnp.sum(regions, axis=1)
    denom = jnp.maximum(counts, 1).astype(jnp.float32)
    weights = regions.astype(jnp.float32)
    centers = jnp.einsum("bnk,bnd->bkd", weights, xyz) / denom[..., None]
    valid = counts > 0
    summaries = tuple(
        jnp.sum(weights * values[:, :, None], axis=1) / denom
        for values in (source, age, confidence)
    )
    return centers, valid, *summaries


def _masked_fps_anchors(xyz, mask, count):
    """Select up to ``count`` unique valid points with order-stable FPS."""
    batch_size, point_count, _ = xyz.shape
    # lexsort keys are listed from least to most significant.  Invalid points
    # sort after any finite valid point.
    sort_xyz = jnp.where(mask[..., None], xyz, jnp.asarray(jnp.inf, xyz.dtype))
    first_indices = jnp.lexsort((sort_xyz[..., 2], sort_xyz[..., 1], sort_xyz[..., 0]), axis=-1)[:, 0]
    first_valid = jnp.any(mask, axis=1)

    def gather(values, indices):
        return jnp.take_along_axis(values, indices[:, None, None], axis=1)[:, 0, :]

    first_anchor = gather(xyz, first_indices)
    selected = jax.nn.one_hot(first_indices, point_count, dtype=jnp.bool_) & first_valid[:, None]
    min_distance = jnp.sum((xyz - first_anchor[:, None, :]) ** 2, axis=-1)
    anchors = [jnp.where(first_valid[:, None], first_anchor, 0.0)]
    valid = [first_valid]

    for _ in range(1, count):
        available = mask & ~selected
        has_candidate = jnp.any(available, axis=1)
        candidate_distances = jnp.where(
            available,
            min_distance,
            jnp.asarray(-jnp.inf, min_distance.dtype),
        )
        indices = jnp.argmax(candidate_distances, axis=1)
        anchor = gather(xyz, indices)
        anchors.append(jnp.where(has_candidate[:, None], anchor, 0.0))
        valid.append(has_candidate)
        selected = selected | (jax.nn.one_hot(indices, point_count, dtype=jnp.bool_) & has_candidate[:, None])
        distance = jnp.sum((xyz - anchor[:, None, :]) ** 2, axis=-1)
        min_distance = jnp.where(has_candidate[:, None], jnp.minimum(min_distance, distance), min_distance)

    return jnp.stack(anchors, axis=1), jnp.stack(valid, axis=1)


def _global_statistics(geometry):
    """Masked point-wise global statistics with finite empty-object output."""
    xyz, mask, source, age, confidence = geometry
    count = jnp.sum(mask, axis=1, keepdims=True).astype(jnp.float32)
    denom = jnp.maximum(count, 1.0)
    weights = mask.astype(jnp.float32)
    center = jnp.sum(xyz * weights[..., None], axis=1) / denom
    source_mean = jnp.sum(source * weights, axis=1, keepdims=True) / denom
    age_mean = jnp.sum(age * weights, axis=1, keepdims=True) / denom
    confidence_mean = jnp.sum(confidence * weights, axis=1, keepdims=True) / denom
    coverage = count / float(mask.shape[1])
    return center, source_mean, age_mean, confidence_mean, coverage, count[:, 0] > 0


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
