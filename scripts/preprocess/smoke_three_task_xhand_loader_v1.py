#!/usr/bin/env python3
"""Data-only smoke for LeRobot + Spatial + Object Geometry + 18-point Hand.

Uses the EXISTING OpenPI LeRobot ActionChunk / XHand transform pipeline and
three TASK-SPECIFIC state/action norm_stats.json files. Object/hand are joined
only in this QA wrapper; this does NOT modify production data_loader.py or Pi0.

This is not a model forward/backward test, nor a substitute for the complete
93,791-frame dataset QA already run.

Run from repository root:
 PYTHONPATH=src .venv/bin/python scripts/preprocess/smoke_three_task_xhand_loader_v1.py
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from openpi.policies.xhand_policy import extract_joint_state
from openpi.object_geometry_dataset.derived import ObjectGeometryDerivedDataset
from openpi.training import config as config_lib
from openpi.training import data_loader
from openpi import transforms

TASKS = {
    "1018_peg_all": (70, 32343),
    "1028_press_button": (100, 26485),
    "1030_yellow_socket": (111, 34963),
}


def scalar_int(value: Any) -> int:
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"Expected scalar episode/frame index, got {arr.shape}")
    return int(arr.reshape(()))


def arr(value: Any) -> np.ndarray:
    # LeRobot yields torch tensors; .numpy is safe for unbatched CPU observations.
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class GeometryAndHandJoin:
    """Minimal data-only join on top of the REAL OpenPI spatial data loader.

    The package's current data_loader auto-joins Spatial only. This wrapper
    demonstrates exactly the additional data-layer join Codex can later adopt,
    without changing any of the underlying dataset or model code.
    """

    def __init__(self, base, root: Path):
        self.base = base
        self.root = root
        self.geo = ObjectGeometryDerivedDataset(root, version="v1")
        self.hand_root = root / "hand_keypoints" / "v1"
        manifest = json.loads((self.hand_root / "manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] != "complete" or manifest["keypoint_count"] != 18:
            raise ValueError(f"{root}: hand keypoints not complete")
        self.hand_files = {int(r["episode_index"]): self.hand_root / r["file"] for r in manifest["episodes"]}
        self.hand_cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def __len__(self):
        return len(self.base)

    def _hand(self, episode: int, frame: int) -> np.ndarray:
        if episode not in self.hand_cache:
            with np.load(self.hand_files[episode], allow_pickle=False) as z:
                frames = np.array(z["frame_index"], dtype=np.int64, copy=True)
                xyz = np.array(z["xyz_m"], dtype=np.float32, copy=True)
            if xyz.shape != (len(frames), 18, 3) or not np.array_equal(frames, np.arange(len(frames))):
                raise ValueError(f"{self.root.name}: invalid hand sidecar episode {episode}")
            self.hand_cache[episode] = frames, xyz
            while len(self.hand_cache) > 2:
                self.hand_cache.popitem(last=False)
        self.hand_cache.move_to_end(episode)
        frames, xyz = self.hand_cache[episode]
        if not 0 <= frame < len(frames) or int(frames[frame]) != frame:
            raise IndexError(f"Hand sidecar missing e{episode} f{frame}")
        return np.array(xyz[frame], copy=True)

    def __getitem__(self, index):
        base = self.base[index]
        episode = scalar_int(base["episode_index"])
        frame = scalar_int(base["frame_index"])
        item = dict(base)
        if "spatial" not in item:
            raise KeyError("Existing OpenPI data loader did not join spatial")
        if "object_geometry" in item or "hand_keypoints" in item:
            raise KeyError("Refusing to overwrite preexisting geometry or hand key")
        obj = self.geo.get(episode_index=episode, frame_index=frame)
        objects = {}
        for role in ("manipulated", "target"):
            objects[role] = {
                "xyz_m": np.array(getattr(obj, f"{role}_xyz_m"), dtype=np.float32, copy=True),
                "mask": np.array(getattr(obj, f"{role}_mask"), dtype=np.bool_, copy=True),
                "source": np.array(getattr(obj, f"{role}_source"), dtype=np.int8, copy=True),
                "age": np.array(getattr(obj, f"{role}_age"), dtype=np.int16, copy=True),
                "confidence": np.array(getattr(obj, f"{role}_confidence"), dtype=np.float32, copy=True),
            }
        item["object_geometry"] = objects
        item["hand_keypoints"] = {
            "xyz_m": self._hand(episode, frame),
            "point_mask": np.ones(18, dtype=np.bool_),
        }
        return item


def make_data_config(repo: Path, name: str, profile: Path, model):
    # Match the native XHand transform path used by the real training code.
    cfg = config_lib.get_config("pi0_xhand_rgb_baseline")
    root = (repo / "data" / name).resolve()
    asset_id = f"{name}_unified_v1"
    factory = dataclasses.replace(
        cfg.data,
        repo_id=str(root),
        assets=config_lib.AssetsConfig(
            assets_dir=str((repo / "assets" / "xhand").resolve()), asset_id=asset_id,
        ),
        spatial=config_lib.SpatialDataConfig(
            dataset_root=str(root), version="v1", camera_roles=("front",),
            front_calibration_profile=str(profile),
        ),
    )
    if factory.use_delta_arm_actions or factory.delta_hand_actions:
        raise ValueError("Expected absolute 18-D XHand action contract")
    data = factory.create(cfg.assets_dirs, model)
    if data.norm_stats is None or set(data.norm_stats) != {"state", "actions"}:
        raise ValueError(f"{name}: task-specific norm stats failed to load")
    if data.asset_id != asset_id:
        raise ValueError(f"{name}: wrong asset_id {data.asset_id}")
    if data.use_quantile_norm:
        raise ValueError(f"{name}: Pi0 expected z-score but quantile normalization was selected")
    for k in ("state", "actions"):
        if np.asarray(data.norm_stats[k].mean).shape != (18,):
            raise ValueError(f"{name}: norm stats {k} not 18-D")
    return data


def assert_sample(name: str, index: int, raw: dict, sample: dict, stats) -> dict:
    ep, frame = scalar_int(raw["episode_index"]), scalar_int(raw["frame_index"])
    sp = sample["spatial"]
    obj = sample["object_geometry"]
    hand = sample["hand_keypoints"]
    assert arr(sp["visual"]["xyz_m"]).shape == (4096, 3)
    assert arr(sp["tactile"]["xyz_m"]).shape == (600, 3)
    assert arr(hand["xyz_m"]).shape == (18, 3)
    for role in ("manipulated", "target"):
        assert arr(obj[role]["xyz_m"]).shape == (512, 3)
        assert arr(obj[role]["mask"]).shape == (512,)
        assert np.isfinite(arr(obj[role]["xyz_m"])).all()
    state = arr(sample["state"])
    actions = arr(sample["actions"])
    assert state.shape == (32,), f"{name} e{ep} f{frame}: state {state.shape} expected (32,)"
    assert actions.shape == (50, 32), f"{name} e{ep} f{frame}: actions {actions.shape} expected (50,32)"
    expected_state = extract_joint_state(arr(raw["observation.state"]))
    expected_actions = arr(raw["action"])
    assert expected_state.shape == (18,)
    assert expected_actions.shape == (50, 18), f"Raw Action Chunk {expected_actions.shape}"
    for field, observed, expected in (
        ("state", state[:18], expected_state),
        ("actions", actions[:, :18], expected_actions),
    ):
        st = stats[field]
        expected_normalized = (expected - arr(st.mean)) / (arr(st.std) + 1e-6)
        np.testing.assert_allclose(observed, expected_normalized, rtol=1e-5, atol=3e-5,
                                   err_msg=f"{name} e{ep} f{frame}: wrong normalization for {field}")
    # Input sidecars stay metric and are preserved *through* XHand transforms.
    for key in ("spatial", "object_geometry", "hand_keypoints"):
        assert key in sample
    return {"global_index": index, "episode_index": ep, "frame_index": frame,
            "manipulated_valid_points": int(arr(obj["manipulated"]["mask"]).sum()),
            "target_valid_points": int(arr(obj["target"]["mask"]).sum())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path("."))
    p.add_argument("--profile", type=Path,
                   default=Path("output/preprocess/1018_peg_all/migration_audit/front_v1_match.json"))
    p.add_argument("--output", type=Path,
                   default=Path("output/preprocess/training_readiness/dataloader_smoke_v1/report.json"))
    args = p.parse_args()
    repo = args.repo.expanduser().resolve()
    profile = args.profile if args.profile.is_absolute() else repo / args.profile
    profile = profile.resolve()
    if not profile.is_file():
        raise FileNotFoundError(f"Missing verified Front v1 profile: {profile}")
    profile_json = json.loads(profile.read_text(encoding="utf-8"))
    if profile_json.get("front_extrinsic_translation_base_m") != [0.0, -0.012, 0.005]:
        raise ValueError("Front v1 profile differs from approved correction")
    if "front_color_intrinsics" in profile_json:
        raise ValueError("Profile would re-enable old K1. Use the approved extrinsic-only v1 profile")

    cfg = config_lib.get_config("pi0_xhand_rgb_baseline")
    if cfg.model.action_horizon != 50:
        raise ValueError(f"Expected action_horizon=50, got {cfg.model.action_horizon}")
    report = {"status": "running", "model_modified": False,
              "purpose": "data-only join + real LeRobot Action Chunk + task-specific normalization",
              "profile": str(profile), "tasks": {}, "caveat": "No Pi0 forward/backward or training integration"}
    for name, (expected_eps, expected_frames) in TASKS.items():
        print(f"\n[TASK] {name}", flush=True)
        root = (repo / "data" / name).resolve()
        data_config = make_data_config(repo, name, profile, cfg.model)
        base = data_loader.create_torch_dataset(data_config, 50, cfg.model)
        if len(base) != expected_frames:
            raise ValueError(f"{name}: dataset length {len(base)} != {expected_frames}")
        joined = GeometryAndHandJoin(base, root)
        # Use the SAME existing OpenPI XHand/normalization/model transforms.
        # preserve_keys is extended from current ('spatial',) to 3 sidecars;
        # no network, model or training code is changed here.
        transformed = data_loader.TransformedDataset(
            joined,
            [*data_config.repack_transforms.inputs,
             *data_config.data_transforms.inputs,
             transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
             *data_config.model_transforms.inputs],
            preserve_keys=("spatial", "object_geometry", "hand_keypoints"),
        )
        indices = sorted(set([0, min(1,len(base)-1), len(base)//2, len(base)-1]))
        entries = []
        for index in indices:
            raw = joined[index]
            sample = transformed[index]
            rec = assert_sample(name, index, raw, sample, data_config.norm_stats)
            entries.append(rec)
            print(f"[PASS] i={index} e={rec['episode_index']} f={rec['frame_index']} "
                  f"state=(32,) actions=(50,32) spatial/object/hand=YES", flush=True)
        # Exercise the real OpenPI Torch batching/collation path (CPU only).
        # A test with only unbatched samples would miss nested sidecar collation bugs.
        torch_loader = data_loader.TorchDataLoader(
            transformed, local_batch_size=2, sampler=indices[:2],
            num_batches=1, num_workers=0, framework="torch",
        )
        batch = next(iter(torch_loader.torch_loader))
        assert arr(batch["state"]).shape == (2, 32)
        assert arr(batch["actions"]).shape == (2, 50, 32)
        assert arr(batch["spatial"]["visual"]["xyz_m"]).shape == (2, 4096, 3)
        assert arr(batch["spatial"]["tactile"]["xyz_m"]).shape == (2, 600, 3)
        assert arr(batch["object_geometry"]["manipulated"]["xyz_m"]).shape == (2, 512, 3)
        assert arr(batch["object_geometry"]["manipulated"]["mask"]).shape == (2, 512)
        assert arr(batch["object_geometry"]["target"]["xyz_m"]).shape == (2, 512, 3)
        assert arr(batch["hand_keypoints"]["xyz_m"]).shape == (2, 18, 3)
        print(f"[BATCH PASS] {name}: batch=2 state=(2,32) actions=(2,50,32) "
              "spatial/object/hand=YES", flush=True)
        report["tasks"][name] = {"status": "PASS", "episodes": expected_eps,
                                 "frames": expected_frames, "asset_id": data_config.asset_id,
                                 "samples": entries, "batch_size": 2,
                                 "sidecar_join_is_prototype": True}
        del base, joined, transformed, torch_loader
    report["status"] = "PASS"
    out = args.output if args.output.is_absolute() else repo / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\n[DATALOADER DATA-ONLY SMOKE PASS] tasks=3, examples=12", flush=True)
    print(f"[REPORT] {out}", flush=True)


if __name__ == "__main__":
    main()
