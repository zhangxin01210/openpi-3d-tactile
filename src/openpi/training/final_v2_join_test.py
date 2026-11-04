import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
import pytest

from openpi.spatial.kinematics import RobotKinematics
from openpi.training import xhand_multimodal_dataset as multimodal


def test_keypoint_join_uses_episode_and_frame_and_validates_manifest(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[3]
    kin = RobotKinematics(repo / "configs/ur7e_xhand/ur7e_xhand_verified.urdf")
    q = dict.fromkeys(kin.required_measured_joints(root="base_link"), 0.0)
    select = runpy.run_path(str(repo / "scripts/spatial/audit_xhand_keypoints.py"))["select_keypoints"]
    names, groups, *_ = select(kin, kin.compute(q, root="base_link"))
    root = tmp_path / "hand_keypoints/v1"
    root.mkdir(parents=True)
    xyz = np.arange(2 * 18 * 3, dtype=np.float32).reshape(2, 18, 3)
    np.savez(root / "e0.npz", frame_index=np.arange(2), xyz_m=xyz)
    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = {"status": "complete", "keypoint_count": 18, "coordinate_frame": "base_link", "unit": "m",
                "spatial_version": "v1", "object_geometry_version": "v1", "names": names, "groups": groups,
                "urdf_sha256": sha(repo / "configs/ur7e_xhand/ur7e_xhand_verified.urdf"),
                "state_mapping_sha256": sha(repo / "configs/ur7e_xhand/state_mapping.csv"),
                "episodes": [{"episode_index": 3, "frames": 2, "file": "e0.npz"}]}
    (root / "manifest.json").write_text(json.dumps(manifest))

    class Objects:
        episode_ids = (3,)
        def __init__(self, *args, **kwargs): pass
        def __len__(self): return 2

    monkeypatch.setattr(multimodal, "ObjectGeometryDerivedDataset", Objects)
    base = [{"episode_index": 3, "frame_index": frame, "spatial": {"object_geometry": {}}} for frame in (1, 0)]
    joined = multimodal.XHandKeypointsSpatialJoin(base, tmp_path, object_version="v1", spatial_version="v1")
    np.testing.assert_array_equal(joined[0]["spatial"]["hand_keypoints"]["xyz_m"], xyz[1])
    np.testing.assert_array_equal(joined[1]["spatial"]["hand_keypoints"]["xyz_m"], xyz[0])
    # Relation-only joins begin with no spatial modality packet at all.
    from openpi.object_geometry_dataset.adapter import ObjectGeometryAugmentedDataset
    class GeometrySample:
        def as_dict(self):
            return {"manipulated_xyz_m": np.full((512, 3), 17, np.float32)}
    Objects.get = lambda self, *, episode_index, frame_index: GeometrySample()
    raw = [{"episode_index": 3, "frame_index": 1}, {"episode_index": 3, "frame_index": 0}]
    relation_only = ObjectGeometryAugmentedDataset(raw, Objects())
    hand_join = multimodal.XHandKeypointsSpatialJoin(relation_only, tmp_path, object_version="v1", spatial_version="v1")
    assert set(hand_join[0]["spatial"]) == {"object_geometry", "hand_keypoints"}
    np.testing.assert_array_equal(hand_join[0]["spatial"]["hand_keypoints"]["xyz_m"], xyz[1])
    np.testing.assert_array_equal(hand_join[0]["spatial"]["object_geometry"]["manipulated_xyz_m"], 17)
    base[0]["frame_index"] = 2
    with pytest.raises(KeyError, match="No hand keypoints"):
        joined[0]
    manifest["groups"][1] = "palm"
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="names/groups"):
        multimodal.XHandKeypointsSpatialJoin(base, tmp_path, object_version="v1", spatial_version="v1")
