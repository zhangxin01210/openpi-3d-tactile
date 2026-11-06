import json

import pytest

from scripts import serve_policy

FAMILY = "pi0_xhand_ho_v4"


def _checkpoint(tmp_path, task: str | None = None):
    path = tmp_path / FAMILY
    if task is not None:
        path /= task
    path /= "seed42/19999"
    path.mkdir(parents=True)
    return path


def _add_norm_stats(checkpoint, asset_id: str):
    directory = checkpoint / asset_id
    directory.mkdir()
    (directory / "norm_stats.json").write_text(json.dumps({}))


def test_exact_config_name_is_unchanged(tmp_path):
    config = serve_policy.resolve_checkpoint_config(
        f"{FAMILY}/yellow_socket", tmp_path / "arbitrary"
    )
    assert config.name == f"{FAMILY}/yellow_socket"


def test_family_resolves_task_from_checkpoint_path(tmp_path):
    checkpoint = _checkpoint(tmp_path, "yellow_socket")
    config = serve_policy.resolve_checkpoint_config(FAMILY, checkpoint)
    assert config.name == f"{FAMILY}/yellow_socket"


def test_family_resolves_task_from_norm_asset_at_arbitrary_path(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    _add_norm_stats(checkpoint, "1030_yellow_socket_unified_v1")
    config = serve_policy.resolve_checkpoint_config(FAMILY, checkpoint)
    assert config.name == f"{FAMILY}/yellow_socket"


def test_family_rejects_checkpoint_without_task_evidence(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    with pytest.raises(ValueError, match="could not be resolved uniquely"):
        serve_policy.resolve_checkpoint_config(FAMILY, checkpoint)


def test_family_rejects_path_and_norm_asset_disagreement(tmp_path):
    checkpoint = _checkpoint(tmp_path, "press_the_button")
    _add_norm_stats(checkpoint, "1030_yellow_socket_unified_v1")
    with pytest.raises(ValueError, match="Refusing to load a mismatched task config"):
        serve_policy.resolve_checkpoint_config(FAMILY, checkpoint)
