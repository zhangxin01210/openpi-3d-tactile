import dataclasses
from types import SimpleNamespace

import preflight_final_v2 as preflight

from openpi.shared import normalize
from openpi.spatial_dataset.source import RawSpatialDataset
from openpi.training import config
from openpi.training import data_loader


def _config_with_assets(name, tmp_path):
    original = config.get_config(name)
    return dataclasses.replace(original, data=dataclasses.replace(
        original.data, assets=dataclasses.replace(original.data.assets, assets_dir=str(tmp_path / "assets"))))


def test_missing_norm_skips_expensive_dataset_load(tmp_path, monkeypatch):
    name = "pi0_xhand_1028_press_button_rgb_v2"
    cfg = _config_with_assets(name, tmp_path)
    monkeypatch.setattr(config, "get_config", lambda _: cfg)
    root = tmp_path / "1028_press_button"
    root.mkdir()
    (root / "meta").mkdir()
    (root / "meta/info.json").touch()
    monkeypatch.setattr(data_loader, "resolve_local_lerobot_root", lambda _: root)
    def unexpected(*args, **kwargs):
        raise AssertionError("dataset load must be skipped")
    monkeypatch.setattr(data_loader, "create_torch_dataset", unexpected)
    result = preflight.inspect(name)
    assert result["checks"]["NORM"] == "FAIL"
    assert result["checks"]["JOIN_AND_MASK"] == "SKIPPED"
    assert all("dataset load must be skipped" not in error for error in result["errors"])


def test_rgb_configs_share_dataset_scan_within_one_preflight(tmp_path, monkeypatch):
    names = ["pi0_xhand_1028_press_button_rgb_v2", "pi05_xhand_1028_press_button_rgb_v2"]
    configs = {name: _config_with_assets(name, tmp_path) for name in names}
    stats_file = tmp_path / "assets/1028_press_button_unified_v1/norm_stats.json"
    stats_file.parent.mkdir(parents=True)
    stats_file.touch()
    monkeypatch.setattr(config, "get_config", lambda name: configs[name])
    root = tmp_path / "1028_press_button"
    root.mkdir()
    (root / "meta").mkdir()
    (root / "meta/info.json").touch()
    monkeypatch.setattr(data_loader, "resolve_local_lerobot_root", lambda _: root)
    stats = SimpleNamespace(mean=[0] * 18, q01=[0] * 18, q99=[1] * 18)
    monkeypatch.setattr(normalize, "load", lambda _: {"state": stats, "actions": stats})
    monkeypatch.setattr(config.LeRobotXHandDataConfig, "create", lambda self, assets, model: SimpleNamespace(
        spatial=None, object_geometry=None))
    monkeypatch.setattr(RawSpatialDataset, "discover_episode_ids", lambda _: range(100))
    calls = []
    class Dataset:
        def __len__(self): return 26485
        def __getitem__(self, index): return {}
    def create(*args):
        calls.append(args)
        return Dataset()
    monkeypatch.setattr(data_loader, "create_torch_dataset", create)
    cache = {}
    first = preflight.inspect(names[0], cache=cache)
    second = preflight.inspect(names[1], cache=cache)
    assert first["checks"]["JOIN_AND_MASK"] == "OK"
    assert second["checks"]["JOIN_AND_MASK"] == "OK (cached)"
    assert len(calls) == 1


def test_main_prints_start_and_finish_for_each_config(monkeypatch, capsys):
    monkeypatch.setattr(preflight, "names", lambda: ["first", "second"])
    monkeypatch.setattr(preflight, "inspect", lambda name, cache: {
        "config": name, "checks": {}, "errors": [], "ready": True})
    monkeypatch.setattr("sys.argv", ["preflight_final_v2.py", "--all-configs"])
    assert preflight.main() == 0
    output = capsys.readouterr().out
    assert output.index("[START] first") < output.index("[READY FOR SMOKE] first")
    assert output.index("[READY FOR SMOKE] first") < output.index("[START] second")
    assert output.index("[START] second") < output.index("[READY FOR SMOKE] second")
