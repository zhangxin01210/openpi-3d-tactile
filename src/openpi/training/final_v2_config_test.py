import dataclasses
from types import SimpleNamespace

from flax import nnx
import jax
import jax.numpy as jnp

from openpi.models import model
from openpi.models.pi0 import Pi0
from openpi.models.spatial_encoders.router import SpatialConditioningRouter
from openpi.models.spatial_encoders.structured_spatial_encoder_test import _object_centric_input
from openpi.models.spatial_encoders.structured_spatial_encoder_test import _synthetic_input
from openpi.training import config


def test_ten_registered_configs_and_isolation():
    expected = [("1028_press_button", "rgb"), ("1028_press_button", "pi05"),
                ("1028_press_button", "st"), ("1028_press_button", "full"),
                ("1030_yellow_socket", "rgb"), ("1030_yellow_socket", "pi05"),
                ("1030_yellow_socket", "st"), ("1030_yellow_socket", "full"),
                ("1018_peg_all", "st"), ("1018_peg_all", "full")]
    configs = []
    for dataset, mode in expected:
        prefix = "pi05" if mode == "pi05" else "pi0"
        name = f"{prefix}_xhand_{dataset}_{'rgb' if mode == 'pi05' else mode}_v2"
        cfg = config.get_config(name)
        configs.append(cfg)
        assert cfg.data.repo_id == f"data/{dataset}"
        assert cfg.model.pi05 == (mode == "pi05")
        assert (cfg.data.spatial is not None) == (mode in {"full", "st"})
        assert (cfg.data.object_geometry is not None) == (mode == "full")
        assert cfg.batch_size == 8
        assert cfg.num_train_steps == 20_000
        if mode in {"full", "st"}:
            assert cfg.model.encoder.visual_hierarchy_v2
            assert cfg.model.encoder.tactile_regions is not None
            assert cfg.model.conditioning.target == "suffix"
            assert cfg.data.spatial.version == "v1"
            if mode == "full":
                assert cfg.data.object_geometry.version == "v1"
            assert cfg.model.encoder.default_token_count == (68 if mode == "full" else 58)
            assert cfg.model.encoder.enable_object_relations == (mode == "full")
            observation_spec, _ = cfg.model.inputs_spec(batch_size=1)
            assert (observation_spec.spatial.hand_keypoints_xyz_m is not None) == (mode == "full")
        ckpt = dataclasses.replace(cfg, exp_name="same_name").checkpoint_dir
        assert ckpt.parent.name == name
    assert len({c.name for c in configs}) == 10
    for dataset in ("1028_press_button", "1030_yellow_socket", "1018_peg_all"):
        st = config.get_config(f"pi0_xhand_{dataset}_st_v2").model.encoder
        full = config.get_config(f"pi0_xhand_{dataset}_full_v2").model.encoder
        assert dataclasses.replace(full, enable_object_relations=False) == st


def test_observation_to_full_encoder_shape_contract():
    cfg = config.get_config("pi0_xhand_1028_press_button_full_v2")
    inp = _synthetic_input(batch_size=1, visual_points=256, tactile_points=600)
    geom = _object_centric_input().object_geometry
    geom = jax.tree.map(lambda x: x[:1], geom)
    spatial = {"visual": dataclasses.asdict(inp.visual), "tactile": dataclasses.asdict(inp.tactile),
               "object_geometry": dataclasses.asdict(geom),
               "hand_keypoints": {"xyz_m": jnp.zeros((1, 18, 3)), "point_mask": jnp.ones((1, 18), bool)}}
    obs = model.Observation.from_dict({"image": {}, "image_mask": {}, "state": jnp.zeros((1, 32)),
                                      "spatial": spatial})
    encoder = cfg.model.encoder.create(rngs=nnx.Rngs(1))
    output = jax.eval_shape(encoder, obs.spatial)
    assert output.tokens.shape == (1, 68, 128)
    assert output.token_mask.shape == (1, 68)
    assert obs.spatial.hand_keypoints_xyz_m.shape == (1, 18, 3)
    restored = model.Observation.from_dict(obs.to_dict())
    assert restored.spatial.hand_keypoints_xyz_m.shape == (1, 18, 3)
    router = SpatialConditioningRouter(encoder=encoder, encoder_token_dim=128,
        conditioning=cfg.model.conditioning, prefix_dim=None, suffix_dim=64, rngs=nnx.Rngs(2))
    pi0_path = SimpleNamespace(spatial_router=router, spatial_use_visual=True, spatial_use_tactile=True)
    def route(ob):
        routed = Pi0._encode_spatial_conditioning(pi0_path, ob)  # noqa: SLF001
        return routed.suffix_tokens, routed.suffix_mask
    tokens, mask = jax.eval_shape(route, obs)
    assert tokens.shape == (1, 68, 64)
    assert mask.shape == (1, 68)
