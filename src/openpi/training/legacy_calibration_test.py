import copy
import dataclasses
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from openpi.object_geometry.calibration import front_calibration_provenance
from openpi.training import config
from openpi.training import data_loader


def _config(dataset, *, version="v1"):
    root = f"data/{dataset}"
    return config.DataConfig(
        repo_id=root,
        spatial=config.SpatialDataConfig(
            dataset_root=root, camera_roles=("front",),
            front_calibration_profile="configs/ur7e_xhand/front_calibration_candidate.json",
        ),
        object_geometry=config.ObjectGeometryDataConfig(dataset_root=root, version=version),
    )


@pytest.mark.parametrize("dataset", ["1018_peg_all", "1028_press_button", "1030_yellow_socket"])
def test_accepted_v1_missing_calibration_warns(dataset, caplog):
    with caplog.at_level(logging.WARNING):
        assert data_loader.validate_object_geometry_calibration(_config(dataset), object(), {})
    assert "LEGACY_UNRECORDED" in caplog.text
    assert "cannot be considered verified" in caplog.text


def test_recorded_calibration_conflict_is_rejected():
    data = _config("1028_press_button")
    root = Path(data_loader.__file__).resolve().parents[3]
    calibration = front_calibration_provenance(root, data.spatial.front_calibration_profile)
    assert not data_loader.validate_object_geometry_calibration(data, object(), {"calibration": calibration})
    wrong = copy.deepcopy(calibration)
    wrong["profile"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sha256"):
        data_loader.validate_object_geometry_calibration(data, object(), {"calibration": wrong})
    wrong = copy.deepcopy(calibration)
    wrong["applied_front_extrinsic_translation_base_m"] = [1.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="translation"):
        data_loader.validate_object_geometry_calibration(data, object(), {"calibration": wrong})
    with pytest.raises(ValueError, match="missing calibration"):
        data_loader.validate_object_geometry_calibration(data, object(), {"calibration": None})


def test_other_dataset_or_version_still_rejects_missing_calibration():
    for data in (_config("1028_press_button", version="v2"), _config("other_dataset")):
        with pytest.raises(ValueError, match="missing calibration"):
            data_loader.validate_object_geometry_calibration(data, object(), {})


def test_other_v2_dataset_without_profile_still_rejects_missing_calibration():
    data = _config("other_dataset")
    data = dataclasses.replace(data, spatial=dataclasses.replace(data.spatial, front_calibration_profile=None))
    model = SimpleNamespace(encoder=SimpleNamespace(object_relation_mode="final_v2"))
    with pytest.raises(ValueError, match="missing calibration"):
        data_loader.validate_object_geometry_calibration(data, model, {})
