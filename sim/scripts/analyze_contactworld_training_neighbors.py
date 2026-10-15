#!/usr/bin/env python3
"""Match evaluation configurations to train-only USB examples and compare behavior."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import rankdata, spearmanr


def geometry(plug, socket, ee):
    plug, socket, ee = map(np.atleast_2d, (plug, socket, ee))
    pr, sr, er = (Rotation.from_quat(x[:, 3:]) for x in (plug, socket, ee))
    return {
        "target": socket[:, :3], "relative": sr.inv().apply(plug[:, :3] - socket[:, :3]),
        "grasp": er.inv().apply(plug[:, :3] - ee[:, :3]),
        "rotation": (sr.inv() * pr).as_quat(), "grasp_rotation": (er.inv() * pr).as_quat(),
        "ee_relative": sr.inv().apply(ee[:, :3] - socket[:, :3]),
    }


def block_distances(query, library):
    position = [np.linalg.norm(library[k] - query[k], axis=-1) * 1000
                for k in ("target", "relative", "grasp")]
    angles = [np.degrees(2 * np.arccos(np.clip(np.abs(
        np.sum(library[k] * query[k], axis=-1)), 0, 1)))
        for k in ("rotation", "grasp_rotation")]
    return np.stack([position[0], position[1], angles[0], position[2], angles[1]], axis=-1)


def score(blocks, *, grasp=True, angle_scale=5):
    scaled = blocks / np.array([10, 10, angle_scale, 5, angle_scale])
    return np.linalg.norm(scaled if grasp else scaled[..., :3], axis=-1)


def correlation(distance, outcomes):
    rho = float(spearmanr(distance, outcomes).statistic)
    rng = np.random.default_rng(20261003)
    x = rankdata(distance); x -= x.mean(); x /= np.linalg.norm(x)
    y = rankdata(outcomes); y -= y.mean(); y /= np.linalg.norm(y)
    permutations = np.stack([rng.permutation(y) for _ in range(10000)])
    null = permutations @ x
    p = (1 + np.count_nonzero(np.abs(null) >= abs(rho) - 1e-12)) / (1 + len(null))
    samples = rng.integers(len(distance), size=(5000, len(distance)))
    bx = rankdata(np.asarray(distance)[samples], axis=1)
    by = rankdata(np.asarray(outcomes)[samples], axis=1)
    bx -= bx.mean(1, keepdims=True); by -= by.mean(1, keepdims=True)
    bootstrap = np.sum(bx * by, axis=1) / np.sqrt(np.sum(bx * bx, axis=1) * np.sum(by * by, axis=1))
    order = np.argsort(distance)
    return {"spearman_rho": rho, "permutation_p_two_sided": float(p),
            "seed_bootstrap95": np.quantile(bootstrap, [.025, .975]).tolist(),
            "near20_mean_success": float(np.mean(outcomes[order[:20]])),
            "far20_mean_success": float(np.mean(outcomes[order[20:]]))}


def interpolated_path(values, end, samples=30):
    end = max(1, end)
    grid = np.linspace(0, end, samples)
    return np.stack([np.interp(grid, np.arange(len(values)), values[:, dim]) for dim in range(3)], axis=-1)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("data/contactworld_usb_v3_bilateral"))
    args = parser.parse_args()
    out = args.root / "training_neighbors";out.mkdir(exist_ok=True)
    manifest = json.loads((args.dataset / "spatial/manifest.json").read_text())
    entries = [e for e in manifest["episodes"] if e["split"] == "train"]
    demos, metadata, frames = [], [], []
    for entry in entries:
        folder = Path(manifest["capture_root"]) / f"insertion_usb_episode_{entry['source_episode']:03d}"
        with np.load(folder / "v2_capture.npz") as capture:
            demo = {k: capture[k] for k in ("plug_pose", "socket_pose", "ee_pose", "state", "action_executed")}
        replay = json.loads((folder / "replay.json").read_text())
        contact = np.linalg.norm([r["plug_socket_force_post"] for r in replay["rows"]], axis=1)
        ix = np.flatnonzero(contact > .1)
        demo["first_contact"] = int(ix[0]) if len(ix) else None
        demo["geometry"] = geometry(demo["plug_pose"], demo["socket_pose"], demo["ee_pose"])
        demos.append(demo)
        # Training observes only the start frame of valid 16-action windows.
        for frame in range(entry["length"] - 16 + 1):
            metadata.append({"source_episode": entry["source_episode"], "episode_index": entry["episode_index"],
                             "frame": frame, "demo_index": len(demos) - 1})
            frames.append({k: value[frame] for k, value in demo["geometry"].items()})
    library = {k: np.stack([row[k] for row in frames]) for k in frames[0]}
    initial = {k: np.stack([demo["geometry"][k][0] for demo in demos]) for k in library}
    summary_paths = sorted(args.root.glob("pi0_cw3_usb_*/summary.json"))
    summaries = [json.loads(p.read_text()) for p in summary_paths]
    assert summaries, "No model summaries found"
    reference_seeds = [r["seed"] for r in summaries[0]["results"]]
    for summary in summaries:
        assert [r["seed"] for r in summary["results"]] == reference_seeds, "Seed order differs across models"
        for actual, reference in zip(summary["results"], summaries[0]["results"]):
            assert actual["initial_physical_state"] == reference["initial_physical_state"], "Unpaired initial states"
    ever = np.asarray([[r["source_ever"] for r in s["results"]] for s in summaries], float)
    final = np.asarray([[r["source_final"] for r in s["results"]] for s in summaries], float)
    neighbors, top5, behaviors = [], [], []
    distances = {k: [] for k in ("initial_full", "initial_task_only", "initial_full_angle2.5",
                                "initial_full_angle10", "initial_full_top5_mean", "seen_frames_full",
                                "seen_frames_task_only")}
    exact_matches = []
    for seed_index, result in enumerate(summaries[0]["results"]):
        seed = result["seed"];st = result["initial_physical_state"]
        query = geometry(np.r_[st["plug_pos"], st["plug_quat"]],
                         np.r_[st["socket_pos_gt"], st["socket_quat"]], np.r_[st["ee_pos"], st["ee_quat"]])
        blocks = block_distances(query, initial);all_blocks = block_distances(query, library)
        d = score(blocks);order = np.argsort(d);nearest = int(order[0]);demo = demos[nearest]
        df = score(all_blocks);frame_index = int(np.argmin(df));fm = metadata[frame_index]
        # Strict geometry equality tolerance: 1 micrometre position / 0.0001 deg rotation.
        exact_ix = np.flatnonzero(np.all(all_blocks <= [0.001, .001, .0001, .001, .0001], axis=1))
        if len(exact_ix):
            exact_matches.append({"seed": seed, "matches": [metadata[i] for i in exact_ix]})
        values = [float(d[nearest]), float(score(blocks, grasp=False).min()),
                  float(score(blocks, angle_scale=2.5).min()), float(score(blocks, angle_scale=10).min()),
                  float(d[order[:5]].mean()), float(df[frame_index]), float(score(all_blocks, grasp=False).min())]
        for key, value in zip(distances, values):distances[key].append(value)
        info = {"seed": seed, "nearest_source_episode": entries[nearest]["source_episode"],
                "nearest_episode_index": entries[nearest]["episode_index"], "distance": values[0],
                "target_mm": float(blocks[nearest, 0]), "relative_plug_mm": float(blocks[nearest, 1]),
                "relative_rotation_deg": float(blocks[nearest, 2]), "grasp_mm": float(blocks[nearest, 3]),
                "grasp_rotation_deg": float(blocks[nearest, 4]),
                "joint_position_rms_rad": float(np.sqrt(np.mean((demo["state"][0, :7] - st["dof_pos"][:7])**2))),
                "seen_frame_source_episode": fm["source_episode"], "seen_frame": fm["frame"],
                "seen_frame_distance": values[5], "models_ever_success": int(ever[:, seed_index].sum()),
                "models_final_success": int(final[:, seed_index].sum())}
        neighbors.append(info)
        for rank, index in enumerate(order[:5], 1):
            top5.append({"seed": seed, "rank": rank, "source_episode": entries[index]["source_episode"],
                         "episode_index": entries[index]["episode_index"], "distance": float(d[index]),
                         **{key: float(value) for key, value in zip(
                             ("target_mm", "relative_plug_mm", "rotation_deg", "grasp_mm", "grasp_rotation_deg"),
                             blocks[index])}})
        for number, path in enumerate(summary_paths, 1):
            trajectory = json.loads((path.parent / f"seed_{seed}/trajectory.json").read_text())
            rows = trajectory["rows"];init = trajectory["result"]["initial_physical_state"]
            # Compare pre-action test states to pre-action training observations.
            states = [init] + rows[:-1]
            g = geometry(np.asarray([r["plug_pos"] + r["plug_quat"] for r in states]),
                         np.asarray([r["socket_pos_gt"] + r["socket_quat"] for r in states]),
                         np.asarray([r["ee_pos"] + r["ee_quat"] for r in states]))
            contact = np.linalg.norm([r["plug_socket_contact_force_xyz"] for r in rows], axis=1)
            ix = np.flatnonzero(contact > .1);tc = int(ix[0]) if len(ix) else None;dc = demo["first_contact"]
            actions = np.asarray([r["action_sim_clipped"] for r in rows])
            count = min(20, len(demo["action_executed"]))
            action_delta = actions[:count] - demo["action_executed"][:count]
            metrics = {
                "seed": seed, "model": number, "source_episode": info["nearest_source_episode"],
                "ever": bool(ever[number-1, seed_index]), "final": bool(final[number-1, seed_index]),
                "first20_translation_command_rms_mm": float(np.sqrt(np.mean(action_delta[:, :3]**2)) * 10),
                "first20_rotation_command_rms_deg": float(np.sqrt(np.mean(action_delta[:, 3:]**2)) * .05 * 180/np.pi),
                "test_contact_step": tc, "train_contact_step": dc,
                "approach_target_relative_ee_path_rmse_mm": None,
                "approach_displacement_shape_rmse_mm": None,
            }
            if tc is not None and dc is not None:
                testpath = interpolated_path(g["ee_relative"], min(tc+1, len(states)-1))
                trainpath = interpolated_path(demo["geometry"]["ee_relative"], min(dc+1, len(demo["state"])-1))
                metrics["approach_target_relative_ee_path_rmse_mm"] = float(np.sqrt(np.mean(
                    np.sum((testpath-trainpath)**2, axis=1))) * 1000)
                metrics["approach_displacement_shape_rmse_mm"] = float(np.sqrt(np.mean(
                    np.sum(((testpath-testpath[0])-(trainpath-trainpath[0]))**2, axis=1))) * 1000)
            behaviors.append(metrics)
    report = {
        "train_episodes": len(demos), "training_observation_start_frames": len(frames),
        "test_initial_states": len(neighbors), "exact_geometry_matches": exact_matches,
        "metric": "sqrt((target_mm/10)^2+(plug_relative_mm/10)^2+(plug_relative_deg/5)^2+(grasp_mm/5)^2+(grasp_deg/5)^2)",
        "statistics_unit": "40 initial states; model success averaged within each seed. Permute/bootstrap seeds, not 520 trajectories.",
        "statistics": {key: {"ever": correlation(np.array(value), ever.mean(0)),
                              "final": correlation(np.array(value), final.mean(0))}
                       for key, value in distances.items()},
        "distances": distances, "neighbors": neighbors,
        "per_model": [{"model": i+1,
                       "rho_ever": float(spearmanr(distances["initial_full"], ever[i]).statistic),
                       "rho_final": float(spearmanr(distances["initial_full"], final[i]).statistic),
                       "near20_ever": int(ever[i, np.argsort(distances["initial_full"])[:20]].sum()),
                       "far20_ever": int(ever[i, np.argsort(distances["initial_full"])[20:]].sum())}
                      for i in range(len(ever))],
    }
    write_csv(out / "nearest_training_per_seed.csv", neighbors)
    write_csv(out / "top5_training_neighbors.csv", top5)
    write_csv(out / "behavior_comparison_520.csv", behaviors)
    (out / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["statistics"], indent=2))


if __name__ == "__main__":main()
