# ruff: noqa: SLF001
"""Local tests for the opt-in raw tactile state pathway."""
import dataclasses
import sys

from flax import nnx
import jax
import numpy as np
import pytest

from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.policies.xhand_raw_tactile import extract_raw_tactile_state
from openpi.training import config
from openpi.training import weight_loaders
from openpi.transforms import PadStatesAndActions

TASKS = ("peg_all", "press_the_button", "yellow_socket", "eraser")


def test_extract_exact_indices_and_padding():
    state = np.arange(1972, dtype=np.float32)
    actual = extract_raw_tactile_state(state)
    expected = np.concatenate((state[:6], state[28:52:2], *(state[76 + i * 384:436 + i * 384] for i in range(5))))
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (1818,)
    result = PadStatesAndActions(32, model_state_dim=1818)({"state": actual, "actions": np.zeros((50, 18))})
    assert result["state"].shape == (1818,)
    assert result["actions"].shape == (50, 32)
    with pytest.raises(ValueError, match="1972"):
        extract_raw_tactile_state(np.zeros(1971))


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("pi05", [False, True])
def test_registered_tactile_cli(task, pi05, monkeypatch):
    name = f"{'pi05' if pi05 else 'pi0'}_xhand_rgb_tactile_state/{task}"
    cfg = config.get_config(name)
    monkeypatch.setattr(config, "_CONFIGS_DICT", {name: cfg})
    monkeypatch.setattr(sys, "argv", ["train.py", name, "--exp-name", "seed42"])
    parsed = config.cli()
    assert parsed.name == name
    assert parsed.model.action_dim == 32
    assert parsed.model.state_input_dim == 1818
    assert parsed.data.raw_tactile_state
    assert parsed.freeze_filter == parsed.model.get_freeze_filter()
    assert parsed.model.strict_state_prompt == pi05
    assert parsed.data.spatial is None
    assert parsed.data.object_geometry is None
    obs, actions = parsed.model.inputs_spec()
    assert obs.state.shape == (1, 1818)
    assert actions.shape[-1] == 32


def test_tokenizer_round_trip_and_no_truncation():
    tokenizer = PaligemmaTokenizer(8192, strict_state_prompt=True)
    state = np.linspace(-3, 3, 1818)
    tokens, mask = tokenizer.tokenize("insert the object", state)
    decoded = tokenizer._tokenizer.decode(tokens[mask].tolist())
    assert decoded.endswith(";\nAction: ")
    assert len(decoded.split(", State: ")[1].split(";")[0].split()) == 1818
    with pytest.raises(ValueError, match="truncated"):
        PaligemmaTokenizer(128, strict_state_prompt=True).tokenize("insert", state)


def test_state_projection_extension_inherits_only_proprio(monkeypatch):
    base = {"state_proj": {"kernel": np.arange(32 * 4, dtype=np.float32).reshape(32, 4), "bias": np.ones(4)},
            "other": {"kernel": np.ones((2, 2), np.float32)}}
    target = {"state_proj": {"kernel": jax.ShapeDtypeStruct((1818, 4), np.float32), "bias": jax.ShapeDtypeStruct((4,), np.float64)},
              "other": {"kernel": jax.ShapeDtypeStruct((2, 2), np.float32)}}
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda _: "test")
    monkeypatch.setattr(weight_loaders._model, "restore_params", lambda *a, **k: base)
    loaded = weight_loaders.RawTactileStateWeightLoader("test").load(target)
    np.testing.assert_array_equal(loaded["state_proj"]["kernel"][:18], base["state_proj"]["kernel"][:18])
    assert not loaded["state_proj"]["kernel"][18:].any()
    assert loaded["other"]["kernel"].shape == (2, 2)
    broken = dataclasses.replace(config.get_config("pi0_xhand_rgb_tactile_state/peg_all").model, paligemma_variant="dummy", action_expert_variant="dummy")
    model = nnx.eval_shape(lambda: broken.create(jax.random.key(0)))
    assert nnx.state(model, nnx.Param).flat_state()[("state_proj", "kernel")].value.shape[0] == 1818


def test_input_transform_end_to_end_without_tactile_tokens():
    from openpi.models.model import ModelType
    from openpi.policies.xhand_policy import XHandInputs
    state = np.arange(1972, dtype=np.float32)
    image = np.zeros((8, 8, 3), np.uint8)
    result = XHandInputs(model_type=ModelType.PI0, raw_tactile_state=True)({"state": state,
        "images": {"cam_front": image, "cam_left": image, "cam_right": image},
        "actions": np.zeros((2, 18), np.float32), "prompt": "insert"})
    np.testing.assert_array_equal(result["state"], extract_raw_tactile_state(state))
    assert result["actions"].shape == (2, 18)
    assert "spatial" not in result


def test_two_eraser_baselines(monkeypatch):
    configs = [config.get_config(f"{prefix}_xhand_rgb_baseline/eraser") for prefix in ("pi0", "pi05")]
    for cfg in configs:
        name = cfg.name
        monkeypatch.setattr(config, "_CONFIGS_DICT", {name: cfg})
        monkeypatch.setattr(sys, "argv", ["train.py", name, "--exp-name", "seed42"])
        assert config.cli().name == name
        assert not cfg.data.raw_tactile_state
        assert cfg.model.action_dim == 32
        assert cfg.data.repo_id.endswith("/data/1102_eraser")
        assert cfg.data.assets.asset_id == "1102_eraser_unified_v1"


@pytest.mark.parametrize("pi05", [False, True])
def test_complete_transform_pipeline(pi05):
    from openpi import transforms
    from openpi.shared import normalize
    name = f"{'pi05' if pi05 else 'pi0'}_xhand_rgb_tactile_state/peg_all"
    cfg = config.get_config(name)
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    state = np.arange(1972, dtype=np.float32) % 3
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    item = {"images": {"cam_front": image, "cam_left": image, "cam_right": image}, "state": state,
            "actions": np.zeros((50, 18), np.float32), "prompt": "move to socket"}
    stats = {"state": normalize.NormStats(mean=np.zeros(1818), std=np.ones(1818),
                q01=-np.ones(1818), q99=np.ones(1818)),
             "actions": normalize.NormStats(mean=np.zeros(18), std=np.ones(18), q01=-np.ones(18), q99=np.ones(18))}
    converted = transforms.compose([*dc.data_transforms.inputs,
        transforms.Normalize(stats, use_quantiles=dc.use_quantile_norm), *dc.model_transforms.inputs])(item)
    assert converted["state"].shape == (1818,)
    assert converted["actions"].shape == (50, 32)
    assert np.isfinite(converted["state"]).all()
    if pi05:
        tokenization = next(x.tokenizer for x in dc.model_transforms.inputs if isinstance(x, transforms.TokenizePrompt))
        decoded = tokenization._tokenizer.decode(converted["tokenized_prompt"][converted["tokenized_prompt_mask"]].tolist())
        assert len(decoded.split(", State: ")[1].split(";")[0].split()) == 1818
        assert decoded.endswith(";\nAction: ")


@pytest.mark.parametrize("pi05", [False, True])
def test_synthetic_model_loss_and_gradients(pi05, monkeypatch):
    import jax.numpy as jnp

    from openpi.models import pi0
    name = f"{'pi05' if pi05 else 'pi0'}_xhand_rgb_tactile_state/peg_all"
    production = config.get_config(name).model
    cfg = dataclasses.replace(production, paligemma_variant="dummy", action_expert_variant="dummy", dtype="float32",
                              max_token_len=16, action_horizon=2)
    siglip = pi0._siglip.Module
    monkeypatch.setattr(pi0._siglip, "Module", lambda num_classes=None, *, variant=None, **kwargs:
                        siglip(num_classes=num_classes, variant="mu/14", **kwargs))
    model = cfg.create(jax.random.key(0))
    obs = cfg.fake_obs(1).replace(state=jnp.ones((1, 1818)))
    actions = jnp.zeros((1, 2, 32))
    def loss(m):
        return jnp.mean(m.compute_loss(jax.random.key(1), obs, actions, train=False))
    value, grads = nnx.value_and_grad(loss, argnums=nnx.DiffState(0, nnx.Param))(model)
    assert bool(jnp.isfinite(value))
    assert all(bool(jnp.isfinite(v.value).all()) for v in grads.flat_state().values())
    if not pi05:
        assert bool(jnp.any(grads.flat_state()[("state_proj", "kernel")].value[18:] != 0))


def test_raw_norm_stats_preserve_baseline_proprio_and_actions(tmp_path, monkeypatch):
    from pathlib import Path
    import runpy

    import pandas as pd

    from openpi.shared import normalize

    dataset = tmp_path / "1018_peg_all"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "data").mkdir()
    (dataset / "meta/info.json").write_text('{"total_frames": 8, "total_episodes": 1}')
    raw = np.zeros((8, 1972), dtype=np.float32)
    raw[:, 76] = np.arange(8)
    pd.DataFrame({"episode_index": [0] * 8, "frame_index": np.arange(8),
                  "observation.state": list(raw), "action": list(np.zeros((8, 18), np.float32))}).to_parquet(dataset / "data/part.parquet")
    assets = tmp_path / "assets"
    baseline = {key: normalize.NormStats(mean=np.arange(18, dtype=float), std=np.ones(18),
                 q01=-np.ones(18), q99=np.ones(18)) for key in ("state", "actions")}
    normalize.save(assets / "1018_peg_all_unified_v1", baseline)
    old = config.get_config("pi0_xhand_rgb_baseline")
    replacement = dataclasses.replace(old, data=dataclasses.replace(old.data,
        assets=dataclasses.replace(old.data.assets, assets_dir=str(assets))))
    monkeypatch.setattr(config, "get_config", lambda _: replacement)
    monkeypatch.setattr(sys, "argv", ["compute", "--dataset", str(dataset), "--asset-id", "1018_peg_all_raw_tactile_v1",
                                     "--raw-tactile-state"])
    runpy.run_path(str(Path(__file__).resolve().parents[3] / "scripts/compute_xhand_norm_stats_fast.py"), run_name="__main__")
    result = normalize.load(assets / "1018_peg_all_raw_tactile_v1")
    assert result["state"].mean.shape == (1818,)
    for field in ("mean", "std", "q01", "q99"):
        np.testing.assert_array_equal(getattr(result["state"], field)[:18], getattr(baseline["state"], field))
        np.testing.assert_array_equal(getattr(result["actions"], field), getattr(baseline["actions"], field))
