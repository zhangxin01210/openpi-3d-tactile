import json
from pathlib import Path

import pytest

from openpi.training import config
from openpi.training import data_loader


def _mock_lerobot(monkeypatch, calls):
    class Metadata:
        def __init__(self, repo_id, root=None):
            calls.append(("metadata", repo_id, root))
            self.fps = 30
            self.total_episodes = 1
            self.video_keys = ("camera",)
        def get_data_file_path(self, episode):
            return Path(f"data/e{episode}.parquet")
        def get_video_file_path(self, episode, key):
            return Path(f"videos/{key}/e{episode}.mp4")

    class Dataset:
        def __init__(self, repo_id, **kwargs):
            calls.append(("dataset", repo_id, kwargs))

    monkeypatch.setattr(data_loader.lerobot_dataset, "LeRobotDatasetMetadata", Metadata)
    monkeypatch.setattr(data_loader.lerobot_dataset, "LeRobotDataset", Dataset)
    monkeypatch.setattr(data_loader, "_LocalLeRobotMetadata", Metadata)
    monkeypatch.setattr(data_loader, "_LocalLeRobotDataset", Dataset)


def test_v2_local_dataset_passes_absolute_root_to_both_constructors(tmp_path, monkeypatch):
    calls = []
    _mock_lerobot(monkeypatch, calls)
    root = tmp_path / "data/1030_yellow_socket"
    for relative in ("meta/info.json", "data/e0.parquet", "videos/camera/e0.mp4"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setattr(data_loader, "resolve_local_lerobot_root", lambda _: root)
    monkeypatch.chdir(tmp_path)
    cfg = config.DataConfig(repo_id="data/1030_yellow_socket")
    data_loader.create_torch_dataset(cfg, 2, object())
    assert calls[0] == ("metadata", cfg.repo_id, root)
    assert calls[1][1] == cfg.repo_id
    assert calls[1][2]["root"] == root
    assert calls[1][2]["delta_timestamps"] == {"actions": [0.0, 1 / 30]}


def test_v2_missing_local_info_fails_without_calling_lerobot(tmp_path, monkeypatch):
    calls = []
    _mock_lerobot(monkeypatch, calls)
    root = tmp_path / "missing"
    monkeypatch.setattr(data_loader, "resolve_local_lerobot_root", lambda _: root)
    with pytest.raises(FileNotFoundError, match="no Hub lookup"):
        data_loader.create_torch_dataset(config.DataConfig(repo_id="data/1028_press_button"), 2, object())
    assert calls == []


def test_v2_missing_local_episode_fails_before_dataset_constructor(tmp_path, monkeypatch):
    calls = []
    _mock_lerobot(monkeypatch, calls)
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").touch()
    monkeypatch.setattr(data_loader, "resolve_local_lerobot_root", lambda _: root)
    with pytest.raises(FileNotFoundError, match="no Hub lookup"):
        data_loader.create_torch_dataset(config.DataConfig(repo_id="data/1018_peg_all"), 2, object())
    assert len(calls) == 1


def test_v2_missing_other_metadata_does_not_fall_back_to_hub(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps({"codebase_version": "v2.1", "features": {}}))
    monkeypatch.setattr(data_loader, "resolve_local_lerobot_root", lambda _: root)
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected Hugging Face request")
    monkeypatch.setattr(data_loader.lerobot_dataset, "get_safe_version", forbidden)
    monkeypatch.setattr(data_loader.lerobot_dataset, "snapshot_download", forbidden)
    with pytest.raises(RuntimeError, match="Incomplete local LeRobot metadata"):
        data_loader.create_torch_dataset(config.DataConfig(repo_id="data/1018_peg_all"), 2, object())


def test_remote_repo_keeps_original_lerobot_calls(monkeypatch):
    calls = []
    _mock_lerobot(monkeypatch, calls)
    cfg = config.DataConfig(repo_id="remote-owner/remote-dataset")
    data_loader.create_torch_dataset(cfg, 2, object())
    assert calls[0] == ("metadata", cfg.repo_id, None)
    assert calls[1][1] == cfg.repo_id
    assert "root" not in calls[1][2]


def test_explicit_absolute_local_path_fails_without_hub(tmp_path, monkeypatch):
    calls = []
    _mock_lerobot(monkeypatch, calls)
    with pytest.raises(FileNotFoundError, match="no Hub lookup"):
        data_loader.create_torch_dataset(config.DataConfig(repo_id=str(tmp_path / "missing")), 2, object())
    assert calls == []


def test_local_dataset_missing_episode_cannot_enter_hub_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(data_loader.lerobot_dataset.LeRobotDataset, "get_episodes_file_paths",
                        lambda _: [Path("data/missing.parquet")])
    dataset = object.__new__(data_loader._LocalLeRobotDataset)  # noqa: SLF001
    dataset.root = tmp_path
    with pytest.raises(RuntimeError, match="missing.parquet"):
        dataset.get_episodes_file_paths()


def test_local_parquet_failure_cannot_enter_hub_retry(tmp_path, monkeypatch):
    def missing(_):
        raise FileNotFoundError("missing parquet")
    monkeypatch.setattr(data_loader.lerobot_dataset.LeRobotDataset, "load_hf_dataset", missing)
    dataset = object.__new__(data_loader._LocalLeRobotDataset)  # noqa: SLF001
    dataset.root = tmp_path
    with pytest.raises(RuntimeError, match="Cannot load local LeRobot data"):
        dataset.load_hf_dataset()


@pytest.mark.parametrize("name", ["1018_peg_all", "1028_press_button", "1030_yellow_socket"])
def test_v2_root_is_checkout_relative_independent_of_cwd(name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert data_loader.resolve_local_lerobot_root(f"data/{name}") == Path(data_loader.__file__).resolve().parents[3] / "data" / name
