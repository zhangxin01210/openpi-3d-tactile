"""Structured RGB-D + tactile spatial encoder.

This encoder is intentionally a plugin beside ``JointPointNetEncoder``.  It
keeps the same ``SpatialEncoderInput -> SpatialEncoderOutput`` contract, but
preserves more structure:

* visual scene points use masked FPS centers and masked KNN local patches;
* tactile taxels are grouped by ``finger_id`` and never globally FPS-sampled;
* tactile local tokens are produced by learned pooling over contextualized
  taxel features, with one additional summary token per finger.

TODO: when a stable canonical finger-local tactile topology is wired into the
model config, use that fixed adjacency for tactile KNN instead of rebuilding
neighborhoods from current base-frame xyz.
"""

from __future__ import annotations

import dataclasses

from flax import nnx
import jax
import jax.numpy as jnp

from openpi.models.spatial_encoders.transforms import SpatialFeatureTransforms
from openpi.models.spatial_encoders.object_relation_encoder import ObjectRelationEncoder
from openpi.models.spatial_encoders.types import SpatialEncoderInput
from openpi.models.spatial_encoders.types import SpatialEncoderOutput


@dataclasses.dataclass(frozen=True)
class StructuredSpatialEncoderConfig:
    token_dim: int = 128

    visual_num_centers: int = 32
    visual_knn: int = 32

    num_fingers: int = 5
    tactile_knn: int = 8
    tactile_local_tokens_per_finger: int = 4
    tactile_summary_tokens_per_finger: int = 1
    # Some experiments use the same learned query bank for every finger and
    # distinguish fingers only through the per-finger embedding.  This keeps
    # the parameter sharing of the simulator's matched-map branch without
    # requiring a fabricated 2-D taxel grid on real hardware.
    share_tactile_queries_across_fingers: bool = False
    # Contact-centred tactile can use this module's reviewed visual branch
    # while supplying its own tactile encoder.  Avoid allocating an unused
    # PointNet-style tactile branch in that case.
    enable_tactile: bool = True
    enable_object_relations: bool = False
    # ``legacy_ho_oo`` is the reviewed 5 Hand-Object + 9 Object-Object
    # experiment.  ``hand_scene_v0`` keeps the base 58 tokens and appends
    # five Hand-Funnel plus five Hand-Target tokens instead.
    object_relation_mode: str = "legacy_ho_oo"
    # ``scene_rgbd`` is the reviewed visual branch.  ``object_centric``
    # replaces only its 33 visual tokens with Hand/Funnel/Flask geometry;
    # RGB images still flow through Pi0's ordinary PaliGemma prefix.
    visual_geometry_mode: str = "scene_rgbd"
    object_centric_centers: tuple[int, int, int] = (8, 12, 12)
    relation_candidate_k: int = 16
    relation_hidden_dim: int = 128
    # The optional funnel silhouette branch is deliberately off for the first
    # Hand-Scene training run.  A missing/occluded mask is always safe; this
    # flag only controls whether valid current-frame masks are consumed.
    hand_scene_enable_funnel_mask_encoder: bool = False
    hand_scene_funnel_mask_size: int = 32

    hidden_dim: int = 128
    transforms: SpatialFeatureTransforms = dataclasses.field(default_factory=SpatialFeatureTransforms)

    def __post_init__(self) -> None:
        for name in (
            "token_dim",
            "visual_num_centers",
            "visual_knn",
            "num_fingers",
            "tactile_knn",
            "tactile_local_tokens_per_finger",
            "hidden_dim",
            "relation_candidate_k",
            "relation_hidden_dim",
            "hand_scene_funnel_mask_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.tactile_summary_tokens_per_finger < 0:
            raise ValueError("tactile_summary_tokens_per_finger must be >= 0")
        if (self.tactile_local_tokens_per_finger + self.tactile_summary_tokens_per_finger) <= 0:
            raise ValueError("At least one tactile token per finger is required")
        if self.visual_geometry_mode not in {"scene_rgbd", "object_centric"}:
            raise ValueError(f"Unsupported visual_geometry_mode: {self.visual_geometry_mode!r}")
        if len(self.object_centric_centers) != 3 or any(center <= 0 for center in self.object_centric_centers):
            raise ValueError("object_centric_centers must contain three positive counts")
        if self.visual_geometry_mode == "object_centric":
            if sum(self.object_centric_centers) != self.visual_num_centers:
                raise ValueError("object_centric_centers must sum to visual_num_centers")
            if not self.enable_tactile or not self.enable_object_relations:
                raise ValueError("object_centric mode requires tactile input and object relations")
        if self.object_relation_mode not in {
            "legacy_ho_oo",
            "hand_scene_v0",
            "hand_scene_oo_v1",
            "dtcr_v1",
            "dtcr_independent_v1",
        }:
            raise ValueError(f"Unsupported object_relation_mode: {self.object_relation_mode!r}")

    def create(self, *, rngs: nnx.Rngs) -> "StructuredSpatialEncoder":
        return StructuredSpatialEncoder(config=self, rngs=rngs)

    @property
    def default_token_count(self) -> int:
        return (
            self.visual_num_centers
            + 1
            + (self.num_fingers * (self.tactile_local_tokens_per_finger + self.tactile_summary_tokens_per_finger)
               if self.enable_tactile else 0)
            + (self.object_relation_token_count if self.enable_object_relations else 0)
        )

    @property
    def object_relation_token_count(self) -> int:
        return 14 if self.object_relation_mode == "legacy_ho_oo" else 10


class StructuredSpatialEncoder(nnx.Module):
    _VISUAL_PATCH_DIM = 7
    _OBJECT_CENTRIC_PATCH_DIM = 7
    _TACTILE_TAXEL_DIM = 7

    def __init__(self, *, config: StructuredSpatialEncoderConfig, rngs: nnx.Rngs) -> None:
        self.config = config

        d = config.token_dim
        h = config.hidden_dim

        self.visual_patch_layers = _make_mlp(self._VISUAL_PATCH_DIM, (h, d), rngs)
        self.visual_global_layers = _make_mlp(self._VISUAL_PATCH_DIM, (h, d), rngs)
        self.visual_pos_layers = _make_mlp(3, (h, d), rngs)

        self.visual_modality_embedding = nnx.Param(
            jax.random.normal(rngs.params(), (d,), dtype=jnp.float32) * 0.02
        )
        if config.visual_geometry_mode == "object_centric":
            # Object points use relative xyz [3], object one-hot [3], and a
            # clipped geometry confidence/validity scalar [1].  These are
            # intentionally separate parameters from the RGB-D MLP because
            # the final four feature dimensions have different semantics.
            self.object_centric_patch_layers = _make_mlp(self._OBJECT_CENTRIC_PATCH_DIM, (h, d), rngs)
            self.object_centric_global_layers = _make_mlp(self._OBJECT_CENTRIC_PATCH_DIM, (h, d), rngs)
            self.object_centric_pos_layers = _make_mlp(3, (h, d), rngs)
            self.object_centric_modality_embedding = nnx.Param(
                jax.random.normal(rngs.params(), (d,), dtype=jnp.float32) * 0.02
            )
        if config.enable_tactile:
            tactile_input_dim = self._TACTILE_TAXEL_DIM + d
            self.tactile_taxel_layers = _make_mlp(tactile_input_dim, (h, d), rngs)
            self.tactile_edge_layers = _make_mlp(d * 2 + 3, (h, d), rngs)
            self.tactile_context_layers = _make_mlp(d * 3, (h, d), rngs)
            self.tactile_key = nnx.Linear(d, d, rngs=rngs)
            self.tactile_value = nnx.Linear(d, d, rngs=rngs)
            self.tactile_pos_layers = _make_mlp(3, (h, d), rngs)
            self.tactile_modality_embedding = nnx.Param(
                jax.random.normal(rngs.params(), (d,), dtype=jnp.float32) * 0.02
            )
            self.finger_embedding = nnx.Param(
                jax.random.normal(rngs.params(), (config.num_fingers, d), dtype=jnp.float32) * 0.02
            )
            query_finger_count = 1 if config.share_tactile_queries_across_fingers else config.num_fingers
            self.tactile_pool_queries = nnx.Param(
                jax.random.normal(
                    rngs.params(),
                    (query_finger_count, config.tactile_local_tokens_per_finger, d),
                    dtype=jnp.float32,
                ) * 0.02
            )
            self.tactile_summary_queries = None
            if config.tactile_summary_tokens_per_finger:
                self.tactile_summary_queries = nnx.Param(
                    jax.random.normal(
                        rngs.params(),
                        (query_finger_count, config.tactile_summary_tokens_per_finger, d),
                        dtype=jnp.float32,
                    ) * 0.02
                )
        if config.enable_object_relations:
            self.object_relation_encoder = ObjectRelationEncoder(
                token_dim=d,
                hidden_dim=config.relation_hidden_dim,
                candidate_k=config.relation_candidate_k,
                relation_mode=config.object_relation_mode,
                enable_funnel_mask_encoder=config.hand_scene_enable_funnel_mask_encoder,
                funnel_mask_size=config.hand_scene_funnel_mask_size,
                rngs=rngs,
            )

    def __call__(self, inputs: SpatialEncoderInput) -> SpatialEncoderOutput:
        token_parts = []
        mask_parts = []
        xyz_parts = []
        aux = {}

        batch_size = _batch_size(inputs)

        if self.config.visual_geometry_mode == "object_centric":
            tokens, mask, xyz, object_aux = self._encode_object_centric(inputs)
            token_parts.append(tokens)
            mask_parts.append(mask)
            xyz_parts.append(xyz)
            aux.update(object_aux)
        elif inputs.visual is not None:
            tokens, mask, xyz, visual_aux = self._encode_visual(inputs.visual)
            token_parts.append(tokens)
            mask_parts.append(mask)
            xyz_parts.append(xyz)
            aux.update(visual_aux)
        else:
            aux["visual_token_count"] = jnp.zeros((batch_size,), dtype=jnp.int32)

        if inputs.tactile is not None:
            if not self.config.enable_tactile:
                raise ValueError("This structured encoder was configured without a tactile branch")
            tokens, mask, xyz, tactile_aux = self._encode_tactile(inputs.tactile)
            token_parts.append(tokens)
            mask_parts.append(mask)
            xyz_parts.append(xyz)
            aux.update(tactile_aux)
        else:
            aux["tactile_token_count"] = jnp.zeros((batch_size,), dtype=jnp.int32)

        if self.config.enable_object_relations:
            if inputs.object_geometry is None:
                raise ValueError("Object relations are enabled but object_geometry input is missing")
            relation_tokens, relation_mask, relation_xyz, relation_aux = self.object_relation_encoder(
                inputs.object_geometry,
                inputs.tactile,
            )
            token_parts.append(relation_tokens)
            mask_parts.append(relation_mask)
            xyz_parts.append(relation_xyz)
            aux.update(relation_aux)

        if not token_parts:
            raise ValueError("SpatialEncoderInput must contain visual and/or tactile input")

        tokens = jnp.concatenate(token_parts, axis=1)
        token_mask = jnp.concatenate(mask_parts, axis=1)
        token_xyz_m = jnp.concatenate(xyz_parts, axis=1)
        tokens = jnp.where(token_mask[..., None], tokens, jnp.zeros_like(tokens))
        token_xyz_m = jnp.where(token_mask[..., None], token_xyz_m, jnp.zeros_like(token_xyz_m))
        aux["total_token_count"] = jnp.sum(token_mask, axis=1, dtype=jnp.int32)

        return SpatialEncoderOutput(
            tokens=tokens,
            token_mask=token_mask,
            token_xyz_m=token_xyz_m,
            aux=aux,
        )

    def _encode_object_centric(
        self, inputs: SpatialEncoderInput
    ) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        """Encode Hand/Funnel/Flask points into the visual 33-token slot.

        The three groups are kept separate during FPS/KNN so points from
        different objects cannot form a semantically mixed local patch.  The
        returned order is 8 Hand tokens, 12 Funnel tokens, 12 Flask tokens,
        followed by one global token.  This branch consumes only current-frame
        base_link metric points; it does not infer geometry from RGB or use
        future poses.
        """

        if inputs.tactile is None or inputs.object_geometry is None:
            raise ValueError("object_centric mode requires tactile and object_geometry inputs")
        geometry = inputs.object_geometry
        hand_xyz, hand_mask, hand_conf = _safe_geometry_points(
            inputs.tactile.xyz_m,
            inputs.tactile.point_mask,
            jnp.ones_like(inputs.tactile.point_mask, dtype=jnp.float32),
        )
        funnel_xyz, funnel_mask, funnel_conf = _safe_geometry_points(
            geometry.manipulated_xyz_m,
            geometry.manipulated_mask,
            geometry.manipulated_confidence,
        )
        flask_xyz, flask_mask, flask_conf = _safe_geometry_points(
            geometry.target_xyz_m,
            geometry.target_mask,
            geometry.target_confidence,
        )
        groups = (
            (hand_xyz, hand_mask, hand_conf, 0, self.config.object_centric_centers[0]),
            (funnel_xyz, funnel_mask, funnel_conf, 1, self.config.object_centric_centers[1]),
            (flask_xyz, flask_mask, flask_conf, 2, self.config.object_centric_centers[2]),
        )
        local_tokens, local_masks, local_xyzs, valid_counts = [], [], [], []
        all_xyz, all_mask, all_conf, all_ids = [], [], [], []
        for xyz, point_mask, confidence, object_id, center_count in groups:
            tokens, token_mask, token_xyz = self._encode_object_group(
                xyz, point_mask, confidence, object_id=object_id, center_count=center_count
            )
            local_tokens.append(tokens)
            local_masks.append(token_mask)
            local_xyzs.append(token_xyz)
            valid_counts.append(jnp.sum(point_mask, axis=1, dtype=jnp.int32))
            all_xyz.append(xyz)
            all_mask.append(point_mask)
            all_conf.append(confidence)
            all_ids.append(
                jnp.broadcast_to(
                    jax.nn.one_hot(object_id, 3, dtype=jnp.float32),
                    (xyz.shape[0], xyz.shape[1], 3),
                )
            )

        xyz = jnp.concatenate(all_xyz, axis=1)
        point_mask = jnp.concatenate(all_mask, axis=1)
        confidence = jnp.concatenate(all_conf, axis=1)
        object_id = jnp.concatenate(all_ids, axis=1)
        global_center = _masked_mean(xyz, point_mask, axis=1)
        global_features = jnp.concatenate(
            [xyz - global_center[:, None, :], object_id, confidence[..., None]], axis=-1
        )
        global_hidden = _apply_mlp(self.object_centric_global_layers, global_features)
        global_token = _masked_mean(global_hidden, point_mask, axis=1)
        global_token = (
            global_token
            + _apply_mlp(self.object_centric_pos_layers, global_center)
            + self.object_centric_modality_embedding.value
        )
        global_mask = jnp.any(point_mask, axis=1)

        tokens = jnp.concatenate([*local_tokens, global_token[:, None, :]], axis=1)
        token_mask = jnp.concatenate([*local_masks, global_mask[:, None]], axis=1)
        token_xyz = jnp.concatenate([*local_xyzs, global_center[:, None, :]], axis=1)
        return (
            tokens,
            token_mask,
            token_xyz,
            {
                "visual_geometry_mode_object_centric": jnp.ones((xyz.shape[0],), dtype=jnp.bool_),
                "visual_token_count": jnp.sum(token_mask, axis=1, dtype=jnp.int32),
                "object_centric_valid_point_count_by_object": jnp.stack(valid_counts, axis=1),
            },
        )

    def _encode_object_group(
        self,
        xyz: jax.Array,
        point_mask: jax.Array,
        confidence: jax.Array,
        *,
        object_id: int,
        center_count: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        center_indices = _masked_fps_indices(xyz, point_mask, center_count)
        center_xyz = _batched_take(xyz, center_indices)
        center_mask = _batched_take(point_mask, center_indices)
        k = min(self.config.visual_knn, xyz.shape[1])
        neighbor_indices, neighbor_mask = _masked_knn_indices(
            query_xyz=center_xyz,
            point_xyz=xyz,
            point_mask=point_mask,
            k=k,
            query_mask=center_mask,
        )
        neighbor_xyz = _batched_take(xyz, neighbor_indices)
        neighbor_confidence = _batched_take(confidence, neighbor_indices)
        one_hot = jnp.broadcast_to(
            jax.nn.one_hot(object_id, 3, dtype=xyz.dtype),
            (xyz.shape[0], xyz.shape[1], 3),
        )
        neighbor_ids = _batched_take(one_hot, neighbor_indices)
        features = jnp.concatenate(
            [neighbor_xyz - center_xyz[:, :, None, :], neighbor_ids, neighbor_confidence[..., None]], axis=-1
        )
        hidden = _apply_mlp(self.object_centric_patch_layers, features)
        tokens = _masked_mean(hidden, neighbor_mask, axis=2)
        tokens = (
            tokens
            + _apply_mlp(self.object_centric_pos_layers, center_xyz)
            + self.object_centric_modality_embedding.value
        )
        tokens = jnp.where(center_mask[..., None], tokens, jnp.zeros_like(tokens))
        return tokens, center_mask, center_xyz

    def _encode_visual(self, visual) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        xyz_feature = self.config.transforms.visual_xyz(visual.xyz_m)
        xyz_raw = visual.xyz_m.astype(jnp.float32)
        rgb = self.config.transforms.rgb(visual.rgb, visual.rgb_valid)
        mask = visual.point_mask.astype(jnp.bool_)
        rgb_valid = visual.rgb_valid.astype(jnp.float32)

        center_indices = _masked_fps_indices(
            xyz_feature,
            mask,
            self.config.visual_num_centers,
        )
        center_xyz_feature = _batched_take(xyz_feature, center_indices)
        center_xyz_raw = _batched_take(xyz_raw, center_indices)
        center_mask = _batched_take(mask, center_indices)

        neighbor_indices, neighbor_mask = _masked_knn_indices(
            query_xyz=center_xyz_feature,
            point_xyz=xyz_feature,
            point_mask=mask,
            k=self.config.visual_knn,
            query_mask=center_mask,
        )

        patch_xyz = _batched_take(xyz_feature, neighbor_indices)
        patch_rgb = _batched_take(rgb, neighbor_indices)
        patch_rgb_valid = _batched_take(rgb_valid, neighbor_indices)[..., None]
        patch_features = jnp.concatenate(
            [
                patch_xyz - center_xyz_feature[:, :, None, :],
                patch_rgb,
                patch_rgb_valid,
            ],
            axis=-1,
        )

        patch_hidden = _apply_mlp(self.visual_patch_layers, patch_features)
        local_tokens = _masked_mean(patch_hidden, neighbor_mask, axis=2)
        local_tokens = (
            local_tokens
            + _apply_mlp(self.visual_pos_layers, center_xyz_feature)
            + self.visual_modality_embedding.value
        )

        global_center = _masked_mean(xyz_feature, mask, axis=1)
        point_features = jnp.concatenate(
            [
                xyz_feature - global_center[:, None, :],
                rgb,
                rgb_valid[..., None],
            ],
            axis=-1,
        )
        point_hidden = _apply_mlp(self.visual_global_layers, point_features)
        global_token = _masked_mean(point_hidden, mask, axis=1)
        global_xyz = _masked_mean(xyz_raw, mask, axis=1)
        global_mask = jnp.any(mask, axis=1)
        global_token = (
            global_token
            + _apply_mlp(self.visual_pos_layers, global_center)
            + self.visual_modality_embedding.value
        )

        tokens = jnp.concatenate([local_tokens, global_token[:, None, :]], axis=1)
        token_mask = jnp.concatenate([center_mask, global_mask[:, None]], axis=1)
        token_xyz = jnp.concatenate([center_xyz_raw, global_xyz[:, None, :]], axis=1)

        return (
            tokens,
            token_mask,
            token_xyz,
            {
                "visual_token_count": jnp.sum(token_mask, axis=1, dtype=jnp.int32),
                "visual_valid_point_count": jnp.sum(mask, axis=1, dtype=jnp.int32),
            },
        )

    def _encode_tactile(self, tactile) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        # SpatialDerivedDataset stores one canonical ``finger_id``/``taxel_id``
        # topology in ``static/`` and reuses it for every frame.  All tactile
        # features and the DTCR per-taxel history are joined in that same
        # order.  Do not sort by taxel_id here: IDs 1..120 intentionally
        # repeat once per finger, and an extra reorder would change the
        # established M0/M1 tensor order while the history branch remains in
        # the static order.  The encoder's masked KNN/attention reductions
        # are permutation-equivariant when an upstream adapter supplies a
        # different but consistently permuted topology.
        xyz_feature = self.config.transforms.tactile_xyz(tactile.xyz_m)
        xyz_raw = tactile.xyz_m.astype(jnp.float32)
        force_output = self.config.transforms.force(
            tactile.force,
            tactile.force_norm,
        )
        force = jnp.asarray(force_output.force, dtype=jnp.float32)
        force_norm = jnp.asarray(force_output.force_norm, dtype=jnp.float32)

        point_mask = tactile.point_mask.astype(jnp.bool_)
        finger_id = tactile.finger_id.astype(jnp.int32)
        clipped_finger_id = jnp.clip(finger_id, 0, self.config.num_fingers - 1)
        per_point_finger_embedding = self.finger_embedding.value[clipped_finger_id]

        taxel_features = jnp.concatenate(
            [
                xyz_feature,
                force,
                force_norm[..., None],
                per_point_finger_embedding,
            ],
            axis=-1,
        )
        taxel_hidden = _apply_mlp(self.tactile_taxel_layers, taxel_features)

        finger_tokens = []
        finger_masks = []
        finger_xyzs = []
        valid_counts = []

        for finger_index in range(self.config.num_fingers):
            finger_mask = point_mask & (finger_id == finger_index)
            contextual = self._contextualize_finger_taxels(
                taxel_hidden=taxel_hidden,
                xyz_feature=xyz_feature,
                finger_mask=finger_mask,
            )

            query_index = 0 if self.config.share_tactile_queries_across_fingers else finger_index
            queries = self.tactile_pool_queries.value[query_index]
            local_tokens, local_xyz, local_mask = self._pool_tactile_queries(
                contextual=contextual,
                xyz_feature=xyz_feature,
                xyz_raw=xyz_raw,
                mask=finger_mask,
                queries=queries,
                finger_index=finger_index,
            )

            if self.config.tactile_summary_tokens_per_finger:
                assert self.tactile_summary_queries is not None
                summary_queries = self.tactile_summary_queries.value[query_index]
                summary_tokens, summary_xyz, summary_mask = self._pool_tactile_queries(
                    contextual=contextual,
                    xyz_feature=xyz_feature,
                    xyz_raw=xyz_raw,
                    mask=finger_mask,
                    queries=summary_queries,
                    finger_index=finger_index,
                )
                finger_tokens.append(jnp.concatenate([local_tokens, summary_tokens], axis=1))
                finger_masks.append(jnp.concatenate([local_mask, summary_mask], axis=1))
                finger_xyzs.append(jnp.concatenate([local_xyz, summary_xyz], axis=1))
            else:
                finger_tokens.append(local_tokens)
                finger_masks.append(local_mask)
                finger_xyzs.append(local_xyz)
            valid_counts.append(jnp.sum(finger_mask, axis=1, dtype=jnp.int32))

        tokens = jnp.concatenate(finger_tokens, axis=1)
        token_mask = jnp.concatenate(finger_masks, axis=1)
        token_xyz = jnp.concatenate(finger_xyzs, axis=1)

        return (
            tokens,
            token_mask,
            token_xyz,
            {
                "tactile_token_count": jnp.sum(token_mask, axis=1, dtype=jnp.int32),
                "tactile_valid_point_count": jnp.sum(point_mask, axis=1, dtype=jnp.int32),
                "tactile_valid_point_count_by_finger": jnp.stack(valid_counts, axis=1),
            },
        )

    def _contextualize_finger_taxels(
        self,
        *,
        taxel_hidden: jax.Array,
        xyz_feature: jax.Array,
        finger_mask: jax.Array,
    ) -> jax.Array:
        neighbor_indices, neighbor_mask = _masked_knn_indices(
            query_xyz=xyz_feature,
            point_xyz=xyz_feature,
            point_mask=finger_mask,
            k=self.config.tactile_knn,
            query_mask=finger_mask,
        )
        neighbor_hidden = _batched_take(taxel_hidden, neighbor_indices)
        neighbor_xyz = _batched_take(xyz_feature, neighbor_indices)

        center_hidden = taxel_hidden[:, :, None, :]
        center_xyz = xyz_feature[:, :, None, :]
        edge_features = jnp.concatenate(
            [
                jnp.broadcast_to(center_hidden, neighbor_hidden.shape),
                neighbor_hidden - center_hidden,
                neighbor_xyz - center_xyz,
            ],
            axis=-1,
        )
        edge_hidden = _apply_mlp(self.tactile_edge_layers, edge_features)
        edge_mean = _masked_mean(edge_hidden, neighbor_mask, axis=2)
        edge_max = _masked_max(edge_hidden, neighbor_mask, axis=2)
        context_input = jnp.concatenate([taxel_hidden, edge_mean, edge_max], axis=-1)
        contextual = _apply_mlp(self.tactile_context_layers, context_input)
        return jnp.where(finger_mask[..., None], contextual, jnp.zeros_like(contextual))

    def _pool_tactile_queries(
        self,
        *,
        contextual: jax.Array,
        xyz_feature: jax.Array,
        xyz_raw: jax.Array,
        mask: jax.Array,
        queries: jax.Array,
        finger_index: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        keys = self.tactile_key(contextual)
        values = self.tactile_value(contextual)
        pooled, weights, token_mask = _masked_attention_pool(
            queries=queries,
            keys=keys,
            values=values,
            mask=mask,
        )
        token_xyz_feature = jnp.einsum("brn,bnd->brd", weights, xyz_feature)
        token_xyz_raw = jnp.einsum("brn,bnd->brd", weights, xyz_raw)
        finger_embedding = self.finger_embedding.value[finger_index]
        tokens = (
            pooled
            + _apply_mlp(self.tactile_pos_layers, token_xyz_feature)
            + finger_embedding
            + self.tactile_modality_embedding.value
        )
        return tokens, token_xyz_raw, token_mask


def _make_mlp(input_dim: int, hidden_dims: tuple[int, ...], rngs: nnx.Rngs) -> list[nnx.Linear]:
    dims = (input_dim, *hidden_dims)
    return [nnx.Linear(dims[i], dims[i + 1], rngs=rngs) for i in range(len(dims) - 1)]


def _safe_geometry_points(
    xyz: jax.Array, point_mask: jax.Array, confidence: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Normalize invalid object points before any distance/reduction operation."""

    xyz = jnp.asarray(xyz, dtype=jnp.float32)
    confidence = jnp.asarray(confidence, dtype=jnp.float32)
    point_mask = point_mask.astype(jnp.bool_)
    finite = jnp.isfinite(xyz).all(axis=-1) & jnp.isfinite(confidence)
    point_mask = point_mask & finite
    xyz = jnp.where(point_mask[..., None], xyz, 0.0)
    confidence = jnp.where(point_mask, jnp.clip(confidence, 0.0, 1.0), 0.0)
    return xyz, point_mask, confidence


def _apply_mlp(layers: list[nnx.Linear], x: jax.Array) -> jax.Array:
    hidden = x
    for index, layer in enumerate(layers):
        hidden = layer(hidden)
        if index != len(layers) - 1:
            hidden = jax.nn.gelu(hidden)
    return jax.nn.gelu(hidden)


def _batch_size(inputs: SpatialEncoderInput) -> int:
    if inputs.visual is not None:
        return int(inputs.visual.xyz_m.shape[0])
    if inputs.tactile is not None:
        return int(inputs.tactile.xyz_m.shape[0])
    raise ValueError("SpatialEncoderInput must contain visual and/or tactile input")


def _batched_take(values: jax.Array, indices: jax.Array) -> jax.Array:
    squeeze_feature_dim = False
    if values.ndim == 2:
        values = values[..., None]
        squeeze_feature_dim = True

    if indices.ndim == 2:
        output = jnp.take_along_axis(values, indices[..., None], axis=1)
    elif indices.ndim == 3:
        expanded_values = values[:, None, :, :]
        expanded_values = jnp.broadcast_to(
            expanded_values,
            (values.shape[0], indices.shape[1], values.shape[1], values.shape[-1]),
        )
        output = jnp.take_along_axis(expanded_values, indices[..., None], axis=2)
    else:
        raise ValueError(f"indices must be rank 2 or 3, got {indices.ndim}")

    if squeeze_feature_dim:
        output = output[..., 0]
    return output


def _masked_fps_indices(xyz: jax.Array, mask: jax.Array, num_centers: int) -> jax.Array:
    batch_size, point_count, _ = xyz.shape
    first_index = jnp.argmax(mask.astype(jnp.int32), axis=1)
    indices = jnp.zeros((batch_size, num_centers), dtype=jnp.int32)
    min_dist = jnp.full((batch_size, point_count), jnp.inf, dtype=xyz.dtype)

    def body(i: int, carry):
        selected, current_min = carry
        current_index = jnp.where(i == 0, first_index, jnp.argmax(current_min, axis=1))
        selected = selected.at[:, i].set(current_index)
        center = _batched_take(xyz, current_index[:, None])[:, 0, :]
        dist = jnp.sum(jnp.square(xyz - center[:, None, :]), axis=-1)
        dist = jnp.where(mask, dist, -jnp.ones((), dtype=xyz.dtype))
        current_min = jnp.minimum(current_min, dist)
        current_min = jnp.where(mask, current_min, -jnp.ones((), dtype=xyz.dtype))
        return selected, current_min

    indices, _ = jax.lax.fori_loop(0, num_centers, body, (indices, min_dist))
    return indices


def _masked_knn_indices(
    *,
    query_xyz: jax.Array,
    point_xyz: jax.Array,
    point_mask: jax.Array,
    k: int,
    query_mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    distances = jnp.sum(jnp.square(query_xyz[:, :, None, :] - point_xyz[:, None, :, :]), axis=-1)
    large = jnp.asarray(1.0e30, dtype=distances.dtype)
    distances = jnp.where(point_mask[:, None, :], distances, large)
    _, indices = jax.lax.top_k(-distances, k)
    neighbor_valid = _batched_take(point_mask[..., None].astype(jnp.float32), indices)[..., 0] > 0.5
    gathered_distances = jnp.take_along_axis(
        distances,
        indices,
        axis=-1,
    )
    neighbor_valid = neighbor_valid & (gathered_distances < large * 0.5)
    if query_mask is not None:
        neighbor_valid = neighbor_valid & query_mask[:, :, None]
    return indices, neighbor_valid


def _masked_mean(values: jax.Array, mask: jax.Array, *, axis: int) -> jax.Array:
    mask_float = mask.astype(values.dtype)
    expanded_mask = jnp.expand_dims(mask_float, axis=-1)
    count = jnp.sum(expanded_mask, axis=axis)
    count = jnp.maximum(count, jnp.asarray(1.0, dtype=values.dtype))
    return jnp.sum(values * expanded_mask, axis=axis) / count


def _masked_max(values: jax.Array, mask: jax.Array, *, axis: int) -> jax.Array:
    neg_inf = jnp.asarray(-1.0e30, dtype=values.dtype)
    expanded_mask = jnp.expand_dims(mask, axis=-1)
    masked = jnp.where(expanded_mask, values, neg_inf)
    output = jnp.max(masked, axis=axis)
    valid = jnp.any(mask, axis=axis)
    return jnp.where(valid[..., None], output, jnp.zeros_like(output))


def _masked_attention_pool(
    *,
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    mask: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    scale = jnp.asarray(keys.shape[-1], dtype=keys.dtype) ** -0.5
    scores = jnp.einsum("rd,bnd->brn", queries, keys) * scale
    large_neg = jnp.asarray(-1.0e30, dtype=scores.dtype)
    scores = jnp.where(mask[:, None, :], scores, large_neg)
    weights = jax.nn.softmax(scores, axis=-1)
    weights = jnp.where(mask[:, None, :], weights, jnp.zeros_like(weights))
    normalizer = jnp.sum(weights, axis=-1, keepdims=True)
    weights = weights / jnp.maximum(normalizer, jnp.asarray(1.0e-6, dtype=weights.dtype))
    pooled = jnp.einsum("brn,bnd->brd", weights, values)
    token_mask = jnp.broadcast_to(jnp.any(mask, axis=1)[:, None], pooled.shape[:2])
    pooled = jnp.where(token_mask[..., None], pooled, jnp.zeros_like(pooled))
    weights = jnp.where(token_mask[..., None], weights, jnp.zeros_like(weights))
    return pooled, weights, token_mask
