import dataclasses
import pathlib

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest

from openpi.models import model as _model


class _ListModel(_model.BaseModel):
    def __init__(self, rng):
        super().__init__(action_dim=2, action_horizon=1, max_token_len=1)
        rngs = nnx.Rngs(rng)
        # Indices above 9 catch lexicographic ordering mistakes when restoring
        # string checkpoint keys into the model's integer list indices.
        self.layers = [nnx.Linear(2, 2, rngs=rngs) for _ in range(12)]

    def compute_loss(self, rng, observation, actions, *, train=False):
        raise NotImplementedError

    def sample_actions(self, rng, observation, **kwargs):
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class _ListModelConfig(_model.BaseModelConfig):
    action_dim: int = 2
    action_horizon: int = 1
    max_token_len: int = 1

    @property
    def model_type(self):
        return _model.ModelType.PI0

    def create(self, rng):
        return _ListModel(rng)

    def inputs_spec(self, *, batch_size=1):
        raise NotImplementedError


@pytest.fixture
def model_config_and_state():
    config = _ListModelConfig()
    return config, nnx.state(config.create(jax.random.key(7)))


def _assert_parameters_equal(expected, model):
    actual = nnx.state(model).to_pure_dict()
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    jax.tree.map(np.testing.assert_array_equal, actual, expected)


@pytest.mark.parametrize("remove_extra_params", [True, False])
def test_load_preserves_integer_list_keys(model_config_and_state, remove_extra_params):
    config, state = model_config_and_state
    params = state.to_pure_dict()

    loaded = config.load(params, remove_extra_params=remove_extra_params)

    _assert_parameters_equal(params, loaded)


@pytest.mark.parametrize("save_nnx_state", [True, False])
@pytest.mark.parametrize("remove_extra_params", [True, False])
def test_load_checkpoint_with_list_layers(
    model_config_and_state, tmp_path: pathlib.Path, save_nnx_state, remove_extra_params
):
    config, state = model_config_and_state
    params = state.to_pure_dict()
    checkpoint = tmp_path / "params"
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(checkpoint, {"params": state if save_nnx_state else params})

    restored = _model.restore_params(checkpoint, restore_type=np.ndarray)
    assert set(restored["layers"]) == {str(i) for i in range(12)}
    loaded = config.load(restored, remove_extra_params=remove_extra_params)

    _assert_parameters_equal(params, loaded)


@pytest.mark.parametrize("remove_extra_params", [True, False])
def test_load_handles_extra_parameters(model_config_and_state, remove_extra_params):
    config, state = model_config_and_state
    params = state.to_pure_dict()
    params_with_extra = {**params, "unused": {"kernel": jnp.ones((2, 2))}}

    if remove_extra_params:
        loaded = config.load(params_with_extra, remove_extra_params=True)
        _assert_parameters_equal(params, loaded)
    else:
        with pytest.raises(ValueError, match="PyTrees have different structure"):
            config.load(params_with_extra, remove_extra_params=False)


@pytest.mark.parametrize("remove_extra_params", [True, False])
def test_load_rejects_missing_parameters(model_config_and_state, remove_extra_params):
    config, state = model_config_and_state
    params = state.to_pure_dict()
    del params["layers"][10]["kernel"]

    with pytest.raises(ValueError, match="PyTrees have different structure"):
        config.load(params, remove_extra_params=remove_extra_params)


@pytest.mark.parametrize("remove_extra_params", [True, False])
def test_load_rejects_shape_mismatch(model_config_and_state, remove_extra_params):
    config, state = model_config_and_state
    params = state.to_pure_dict()
    params["layers"][10]["kernel"] = jnp.ones((3, 2))

    with pytest.raises(ValueError, match="Shape mismatch"):
        config.load(params, remove_extra_params=remove_extra_params)
