#!/usr/bin/env python3
"""Replay one ContactWorld cross-task demo with bilateral TacFF capture.

Some official task classes construct both force-field sensors while publishing
only the right pad.  The private runtime wrapper below ensures both sensors
are constructed, then publishes their separate tensors.  It does not edit the
ContactWorld checkout or alter its physics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import types

from contactworld_cross_task_common import TASKS, get_task


def _stage_released_task_assets(source: Path) -> list[str]:
    """Link only missing released assets into the private replay checkout.

    The ContactWorld trajectory archives intentionally contain Zarr data only.
    Flat-Barb and Valve URDFs live in the separately released 60 MB asset
    bundle.  Isaac Gym resolves assets relative to its private checkout, so an
    overlay is required there.  Existing checkout files are never replaced.
    """
    released = Path("/home/sai/zx/openpi-sim-runtime/downloads/contactworld/released_assets")
    target = source / "thirdparty/manifeel-isaacgymenvs/assets"
    if not released.is_dir():
        raise FileNotFoundError("Missing released task assets: " + str(released))
    created: list[str] = []
    for relative_root in (Path("industreal/urdf"), Path("industreal/mesh"), Path("industreal/yaml")):
        origin = released / relative_root
        if not origin.is_dir():
            continue
        for item in origin.rglob("*"):
            if item.is_dir():
                continue
            destination = target / item.relative_to(released)
            if destination.exists() or destination.is_symlink():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(item)
            created.append(str(destination.relative_to(target)))
    return created


def _enable_bilateral(TaskClass) -> None:
    original_force_field_configs = TaskClass._compose_tactile_force_field_configs
    original = TaskClass.compute_observations
    if getattr(TaskClass, "_openpi_bilateral_wrapper", False):
        return

    def compose_bilateral_force_field_configs(self):
        configs = original_force_field_configs(self)
        names = {config["name"] for config in configs}
        if "tactile_force_field_left" in names:
            return configs
        right = next((config for config in configs if config["name"] == "tactile_force_field_right"), None)
        if right is None:
            raise RuntimeError("Task did not create a right TacFF sensor")
        left = right.copy()
        left["name"] = "tactile_force_field_left"
        left["elastomer_link_name"] = "elastomer_left"
        left["elastomer_tip_link_name"] = "elastomer_tip_left"
        return [left, *configs]

    def compute_observations_bilateral(self):
        observations = original(self)
        fields, _ = self.get_tactile_force_field_tensors_dict()
        key = "tactile_force_field_left"
        if key not in self.obs_dict or key not in fields:
            raise RuntimeError("Left TacFF was not created; force fields=%s" % sorted(fields))
        self.obs_dict[key][:] = fields[key]
        if self.cfg_task.env.zero_out_normal_force_field_obs:
            self.obs_dict[key][..., 0] *= 0.0
        return observations

    TaskClass._compose_tactile_force_field_configs = compose_bilateral_force_field_configs
    TaskClass.compute_observations = compute_observations_bilateral
    TaskClass._openpi_bilateral_wrapper = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--pose-feedback", action="store_true", required=True,
                        help="Replay closed-loop targets and save actual executed actions")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    spec = get_task(args.task)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "task": spec.key, "source_group": spec.source_group,
              "action_dim": spec.action_dim, "stages": []}
    began = time.monotonic()

    def stage(name: str, **extra) -> None:
        row = {"name": name, "elapsed_s": time.monotonic() - began, **extra}
        report["stages"].append(row)
        (args.output / "probe.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)

    env = None
    try:
        from isaacgym import gymapi
        import numpy as np
        import torch
        import hydra
        from omegaconf import OmegaConf

        code = args.source / "thirdparty/manifeel-isaacgymenvs"
        sys.path.insert(0, str(code))
        import isaacgymenvs
        tasks = types.ModuleType("isaacgymenvs.tasks")
        tasks.__path__ = [str(code / "isaacgymenvs/tasks")]
        sys.modules["isaacgymenvs.tasks"] = tasks
        isaacgymenvs.tasks = tasks
        module_name = "isaacgymenvs.tasks.tacsl.tacsl_task_" + spec.hydra_config
        module = __import__(module_name, fromlist=[spec.task_class])
        TaskClass = getattr(module, spec.task_class)
        _enable_bilateral(TaskClass)
        stage("task_imported", torch=torch.__version__)

        cfg_root = args.output / "hydra"
        cfg_root.mkdir(exist_ok=True)
        manifeel = args.source / "thirdparty/manifeel"
        staged = _stage_released_task_assets(args.source)
        stage("released_assets_ready", linked_missing_files=len(staged))
        for config in (manifeel / "manifeel/config").glob("*.yaml"):
            link = cfg_root / config.name
            if not link.exists():
                link.symlink_to(config.resolve())
        for name, target in (("task", manifeel / "manifeel/config/task"),
                             ("assets", manifeel / "assets")):
            link = cfg_root / name
            if not link.exists():
                link.symlink_to(target.resolve())
        with hydra.initialize_config_dir(config_dir=str(cfg_root.resolve()), version_base="1.1"):
            cfg = hydra.compose(config_name="isaacgym_config_" + spec.hydra_config,
                                overrides=["num_envs=1", "headless=true"])
            cfg.capture_video = False
            cfg.force_render = False
            task_cfg = OmegaConf.to_container(cfg.task, resolve=True)
            obs_dims = task_cfg["env"].setdefault("obsDims", {})
            obs_dims["tactile_force_field_left"] = [10, 14, 3]
            (args.output / "task_config.json").write_text(json.dumps(task_cfg, indent=2) + "\n")
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            stage("creating_task")
            env = TaskClass(cfg=task_cfg, rl_device="cuda:0", sim_device="cuda:0",
                            graphics_device_id=0, headless=True,
                            virtual_screen_capture=False, force_render=False)
            if env.num_actions != spec.action_dim:
                raise ValueError("Environment action dimension %d != source %d" % (env.num_actions, spec.action_dim))
            stage("task_created", env_actions=env.num_actions)
            env.reset_idx(torch.arange(1, device=env.device, dtype=torch.long))
            stage("reset_idx_passed")
            from replay_contactworld_episode import replay
            replay(env, args.demo, args.output, pose_feedback=True, capture_v2=True,
                   capture_dual=True)
            report["status"] = "completed_execution_not_validated_reproduction"
            stage("complete_episode_executed", caveat="Review RGB-D, cloud, both tactile pads and executed actions before bulk replay")
    except Exception as error:
        report["status"] = "failed"
        stage("exception", type=type(error).__name__, message=str(error))
        raise
    finally:
        if env is not None:
            env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
