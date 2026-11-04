"""Synthetic on-disk anchor generation, provenance and online/offline identity."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from openpi.object_geometry.ho_v4_anchors import _exporter
from openpi.object_geometry.ho_v4_anchors import load_assets
from openpi.object_geometry.ho_v4_anchors import validate_online_geometry
from openpi.object_geometry_dataset.adapter import ObjectGeometryAugmentedDataset
from openpi.object_geometry_dataset.derived import ObjectGeometryDerivedDataset
from openpi.policies.spatial_online import XHandSpatialOnlinePreprocess

trimesh = pytest.importorskip("trimesh", reason="anchor generation uses the optional GLB reader")

_export = _exporter()
SURFACE_SAMPLING_SEED = _export.SURFACE_SAMPLING_SEED
load_surface_points = _export.load_surface_points


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(repo, dataset):
    prep = repo / "output/preprocess" / dataset
    approved = prep / "geometry/approved"
    approved.mkdir(parents=True)
    (prep / "rgbtrack").mkdir()
    objects = ["moving", "fixed"]
    for name, extents in zip(objects, ([.04, .07, .1], [.06, .09, .14]), strict=True):
        trimesh.creation.box(extents=extents).export(approved / f"{name}.glb")
    release = {"dataset": dataset, "objects": objects,
               "policy": {"dynamic_object": objects[0], "static_object": objects[1]}}
    release_path = prep / "rgbtrack/release_v1.json"
    release_path.write_text(json.dumps(release))
    approved_path = approved / "manifest.json"
    approved_path.write_text(json.dumps({"status": "human_accepted", "objects": objects,
        "assets": {n: {"sha256": sha(approved / f"{n}.glb"), "scale_m_per_unit": 1.0} for n in objects}}))
    sidecar = repo / "data" / dataset / "object_geometry/v1"
    shard = sidecar / "shard_0"
    shard.mkdir(parents=True)
    arrays = {"episode_index": np.array([3, 3]), "frame_index": np.array([0, 1])}
    for i, role in enumerate(("manipulated", "target")):
        points = load_surface_points(approved / f"{objects[i]}.glb", SURFACE_SAMPLING_SEED + i)
        # A different rigid translation per frame verifies fixed point identity.
        arrays[f"{role}_xyz_m"] = np.stack((np.add(points, [0.4, 0.2, 0.1]), np.add(points, [0.5, 0.2, 0.15]))).astype(np.float32)
        arrays[f"{role}_mask"] = np.ones((2, 512), bool)
        arrays[f"{role}_source"] = np.zeros((2, 512), np.int8)
        arrays[f"{role}_age"] = np.zeros((2, 512), np.int16)
        arrays[f"{role}_confidence"] = np.ones((2, 512), np.float32)
    for name, value in arrays.items():
        np.save(shard / f"{name}.npy", value)
    (sidecar / "manifest.json").write_text(json.dumps({"status": "complete", "dataset": dataset,
        "producer": "scripts/preprocess/export_object_geometry_rgbtrack.py",
        "source_release_sha256": sha(release_path), "approved_geometry_sha256": sha(approved_path),
        "approved_meshes_sha256": {n: sha(approved / f"{n}.glb") for n in objects},
        "role_to_object": {"manipulated": objects[0], "target": objects[1]},
        "surface_sampling_seed": SURFACE_SAMPLING_SEED,
        "object_geometry_contract": {"version": "v1", "point_count": 512},
        "records": [{"episode_index": 3, "frame_index": i, "shard": "shard_0", "row": i} for i in range(2)]}))
    return approved


def test_generator_and_online_offline_contract(tmp_path):
    root = Path(__file__).resolve().parents[3]
    datasets = ("1028_press_button", "1030_yellow_socket")
    for dataset in datasets:
        _write_fixture(tmp_path, dataset)
    cmd = [sys.executable, str(root / "scripts/preprocess/build_ho_v4_anchors.py"), "--repo", str(tmp_path)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    files = sorted((tmp_path / "assets/ho_v4_anchors").rglob("*.json"))
    assert len(files) == 4
    original = {path: path.read_bytes() for path in files}
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert original == {path: path.read_bytes() for path in files}
    assert len(list((tmp_path / "output/analysis/ho_v4_anchor_review").glob("*.html"))) == 4
    for dataset in datasets:
        assets = load_assets(tmp_path, dataset)
        for asset in assets.values():
            assert len(set(asset["indices"].tolist())) == 16
            assert np.all((asset["indices"] >= 0) & (asset["indices"] < 512))
        sidecar = ObjectGeometryDerivedDataset(tmp_path / "data" / dataset)
        raw = [{"episode_index": 3, "frame_index": 1}]
        offline = ObjectGeometryAugmentedDataset(raw, sidecar, fixed_anchor_assets=assets)[0]["spatial"]["object_geometry"]
        online_input = sidecar.get(episode_index=3, frame_index=1).as_dict()
        online_input["surface_points_sha256"] = {role: v["surface_points_sha256"] for role, v in assets.items()}
        transform = XHandSpatialOnlinePreprocess(repo_root=root, camera_roles=("front",), use_visual=False,
            use_tactile=False, require_object_geometry=True, require_hand_keypoints=True,
            anchor_asset_root="assets/ho_v4_anchors", anchor_dataset_root=str(tmp_path / "data" / dataset))
        online = transform({"observation.state": np.zeros(1972, np.float32), "object_geometry": online_input})
        for role in assets:
            np.testing.assert_array_equal(offline[f"{role}_anchor_indices"],
                online["spatial"]["object_geometry"][f"{role}_anchor_indices"])
        bad = dict(online_input, target_xyz_m=online_input["target_xyz_m"][::-1].copy())
        with pytest.raises(ValueError, match="ordered surface"):
            validate_online_geometry(bad, assets)
        with pytest.raises(ValueError, match="provenance"):
            validate_online_geometry(dict(online_input, surface_points_sha256={}), assets)
    mesh = tmp_path / "output/preprocess/1028_press_button/geometry/approved/moving.glb"
    mesh.write_bytes(mesh.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="mesh hash"):
        load_assets(tmp_path, datasets[0])
