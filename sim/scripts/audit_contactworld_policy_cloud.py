#!/usr/bin/env python3
"""Audit online ContactWorld policy clouds against front RGB and released clouds."""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def project(xyz: np.ndarray, capture: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inverse = np.linalg.inv(capture["view"])
    camera_rotation = inverse[:3, :3].T
    camera_position = inverse[3, :3]
    optical = (xyz - camera_position) @ camera_rotation @ np.diag([1., -1., -1.])
    height, width = capture["depth"].shape
    focal = np.array([capture["projection"][0, 0] * width / 2,
                      capture["projection"][1, 1] * height / 2])
    uv = optical[:, :2] / optical[:, 2:3] * focal + [width / 2, height / 2] - 0.5
    pixels = np.round(uv).astype(int)
    valid = ((optical[:, 2] > 0) & (pixels >= 0).all(1) &
             (pixels[:, 0] < width) & (pixels[:, 1] < height))
    return uv, pixels, valid


def frame_stats(xyz: np.ndarray, front: np.ndarray, pixels: np.ndarray,
                valid: np.ndarray) -> dict:
    colors = front[pixels[valid, 1], pixels[valid, 0]].astype(np.int16)
    red = (colors[:, 0] > 150) & (colors[:, 1] < 120) & (colors[:, 2] < 120)
    blue = (colors[:, 2] > colors[:, 0] + 20) & (colors[:, 2] > colors[:, 1] + 20)
    return {"points": int(len(xyz)), "points_in_front_image": int(valid.sum()),
            "points_z_below_2cm": int((xyz[:, 2] < 0.02).sum()),
            "projected_red_pixels": int(red.sum()), "projected_blue_pixels": int(blue.sum()),
            "projected_white_pixels": int((colors.min(1) > 240).sum()),
            "front_red_pixels": int(((front[:, :, 0] > 150) & (front[:, :, 1] < 120) &
                                      (front[:, :, 2] < 120)).sum())}


def plot_online(path: Path, front: np.ndarray, xyz: np.ndarray,
                pixels: np.ndarray, valid: np.ndarray, seed: int, step: int) -> None:
    hit = np.zeros(front.shape[:2], bool)
    hit[pixels[valid, 1], pixels[valid, 0]] = True
    colors = front[pixels[valid, 1], pixels[valid, 0]] / 255.
    fig = plt.figure(figsize=(14, 5), constrained_layout=True)
    ax = fig.add_subplot(131)
    ax.imshow(front)
    ax.set(title="Front RGB", xlabel="pixels", ylabel="pixels")
    ax = fig.add_subplot(132)
    ax.imshow(front, alpha=0.55)
    ax.scatter(pixels[valid, 0], pixels[valid, 1], s=8, c="lime", alpha=0.8)
    ax.set(title=f"The {valid.sum()} cloud points projected into RGB", xlabel="pixels", ylabel="pixels",
           xlim=(0, 255), ylim=(255, 0))
    ax = fig.add_subplot(133, projection="3d")
    ax.scatter(xyz[valid, 0], xyz[valid, 1], xyz[valid, 2], c=colors, s=3, depthshade=False)
    ax.set(xlabel="base X (m)", ylabel="base Y (m)", zlabel="base Z (m)",
           title="Same 1024 xyz, RGB sampled only for this plot")
    ax.view_init(elev=22, azim=135)
    fig.suptitle(f"Online policy input · seed {seed} · step {step}; model receives xyz without these colors")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def dense_crop_demo(path: Path, probe_path: Path, geometry: dict) -> dict:
    """Illustrate information lost by sample-before-crop on an archived same-frame depth capture."""
    probe = np.load(probe_path)
    depth = probe["camera_front_depth"][0]
    height, width = depth.shape
    vv, uu = np.indices(depth.shape)
    valid = (depth > 1e-4) & (depth < 2.0)
    z = depth[valid]
    projection = probe["projection_front_depth"]
    cam = np.stack((
        ((uu[valid] - width / 2) / width) * z * (2 / projection[0, 0]),
        ((vv[valid] - height / 2) / height) * z * (2 / projection[1, 1]),
        z, np.ones_like(z)), axis=-1)
    raw = (cam @ np.linalg.inv(probe["view_front_depth"]).T)[:, :3]
    a, t = np.asarray(geometry["A"]), np.asarray(geometry["t"])
    world = raw @ a.T + t
    old = probe["pointcloud"][0]
    old_world = old[:, :3] @ a.T + t
    crop = ((world[:, 0] > 0.25) & (world[:, 0] < 0.62) &
            (np.abs(world[:, 1]) < 0.2) & (world[:, 2] > 0.01) & (world[:, 2] < 0.3))
    chosen = np.random.default_rng(0).choice(np.flatnonzero(crop), 1024, replace=False)
    image = probe["camera_front"][0]
    old_uv, old_px, old_valid = project(old_world, probe_capture(probe))
    sample_uv = np.stack((uu[valid][chosen], vv[valid][chosen]), axis=-1)
    sample_color = image[sample_uv[:, 1], sample_uv[:, 0]]
    old_color = np.rint(np.clip(old[:, 3:6], 0, 1) * 255).astype(np.uint8)

    def categories(color):
        color = color.astype(np.int16)
        return {"red": int(((color[:, 0] > 150) & (color[:, 1] < 120) & (color[:, 2] < 120)).sum()),
                "blue": int(((color[:, 2] > color[:, 0] + 20) &
                              (color[:, 2] > color[:, 1] + 20)).sum()),
                "white": int((color.min(1) > 240).sum())}

    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    ax = fig.add_subplot(221)
    ax.imshow(image)
    ax.set_title("RGB from archived same-frame depth capture")
    ax = fig.add_subplot(222)
    ax.imshow(image, alpha=0.35)
    ax.scatter(old_px[old_valid, 0], old_px[old_valid, 1], s=8, c="lime")
    ax.set_title("Original: 1024 sampled before any crop")
    ax = fig.add_subplot(223)
    ax.imshow(image, alpha=0.35)
    ax.scatter(sample_uv[:, 0], sample_uv[:, 1], s=8, c="lime")
    ax.set_title("Illustration: crop dense depth, then sample 1024")
    ax = fig.add_subplot(224, projection="3d")
    ax.scatter(world[chosen, 0], world[chosen, 1], world[chosen, 2],
               c=sample_color / 255., s=3, depthshade=False)
    ax.set(xlabel="base X", ylabel="base Y", zlabel="base Z",
           title="Illustrative cropped dense cloud")
    ax.view_init(elev=22, azim=135)
    fig.suptitle("Separate archived probe; demonstrates what full depth would allow")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"archived_probe": str(probe_path), "valid_dense_pixels": int(valid.sum()),
            "dense_points_in_illustrative_crop": int(crop.sum()),
            "original_1024_color_counts": categories(old_color),
            "cropped_1024_color_counts": categories(sample_color),
            "crop_bounds_base_m": {"x": [0.25, 0.62], "y": [-0.2, 0.2], "z": [0.01, 0.3]},
            "warning": "Illustration only: crop bounds and samples have not been validated across episodes"}


def probe_capture(probe) -> dict:
    return {"view": probe["view_front_depth"], "projection": probe["projection_front_depth"],
            "depth": probe["camera_front_depth"][0]}


def training_distribution(source, manifest: dict, a: np.ndarray, t: np.ndarray) -> dict:
    counts = []
    for episode in manifest["episodes"]:
        if episode["split"] != "train":
            continue
        start, length = int(episode["source_start"]), int(episode["length"])
        for offset in (0, length // 2, length - 1):
            cloud = np.asarray(source["pointcloud"][start + offset])
            world = cloud[:, :3] @ a.T + t
            color = np.rint(np.clip(cloud[:, 3:6], 0, 1) * 255).astype(np.int16)
            counts.append([
                int((world[:, 2] < 0.02).sum()),
                int(((color[:, 0] > 150) & (color[:, 1] < 120) & (color[:, 2] < 120)).sum()),
                int(((color[:, 2] > color[:, 0] + 20) &
                     (color[:, 2] > color[:, 1] + 20)).sum()),
                int((color.min(1) > 240).sum()),
            ])
    array = np.asarray(counts)
    return {"sampled_training_frames": len(array),
            "sampling": "first, middle and last frame of every train episode",
            "heuristic_counts_per_1024_points": {
                name: {"p10": float(np.quantile(array[:, index], 0.1)),
                       "median": float(np.median(array[:, index])),
                       "p90": float(np.quantile(array[:, index], 0.9))}
                for index, name in enumerate(("z_below_2cm", "red_pixels", "blue_pixels", "white_pixels"))}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, default=Path(
        "/home/sai/zx/openpi-sim-runtime/runs/contactworld_pi0_paired_20261002/pi0_cw_usb_02_rgb_pc"))
    parser.add_argument("--capture", type=Path, default=Path(
        "/home/sai/zx/openpi-sim-runtime/runs/contactworld_compat/front_camera_centers/camera.npz"))
    parser.add_argument("--source", type=Path, default=Path(
        "/home/sai/zx/openpi-sim-runtime/data/contactworld/insertion_usb"))
    parser.add_argument("--training", type=Path, default=Path("data/contactworld_usb_positive_all"))
    parser.add_argument("--probe", type=Path, default=Path(
        "/home/sai/zx/openpi-sim-runtime/runs/contactworld_build_5090/usb_probe_calib/sensors.npz"))
    parser.add_argument("--output", type=Path, default=Path(
        "/home/sai/zx/openpi-sim-runtime/visualizations/contactworld_policy_cloud_audit_20261002"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[10000, 10001, 10002, 10003, 10004])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    capture = np.load(args.capture)
    records = []
    cards = []
    for seed in args.seeds:
        with np.load(args.rollouts / f"seed_{seed}/inputs.npz") as data:
            for query in ([0, min(12, len(data["steps"]) - 1)] if seed == args.seeds[0] else [0]):
                step = int(data["steps"][query])
                xyz, front = data["cloud_xyz"][query], data["front"][query]
                uv, pixels, valid = project(xyz, capture)
                item = {"seed": seed, "step": step, **frame_stats(xyz, front, pixels, valid),
                        "max_pixel_rounding_error": float(np.abs(uv[valid] - pixels[valid]).max())}
                records.append(item)
                filename = f"seed_{seed}_step_{step}.png"
                plot_online(args.output / filename, front, xyz, pixels, valid, seed, step)
                cards.append(f'<section><h2>Seed {seed}, step {step}</h2><p>'
                             f'{item["points_z_below_2cm"]}/1024 below 2 cm; '
                             f'{item["projected_red_pixels"]} points project onto red pixels; '
                             f'{item["projected_blue_pixels"]} onto blue pixels. '
                             f'All {item["points_in_front_image"]} points project inside RGB.</p>'
                             f'<img src="{escape(filename)}"></section>')
    # Independent release check: the stored per-point RGB should exactly match
    # the calibrated projection into the stored front frame, and the converted
    # training xyz should equal the same geometry transform.
    import zarr
    source = zarr.open_group(str(args.source), mode="r")["data"]
    geometry = json.loads((Path(__file__).resolve().parents[1] /
                           "reports/2026-10-01/contactworld/runtime_geometry.json").read_text())
    a, t = np.asarray(geometry["A"], np.float32), np.asarray(geometry["t"], np.float32)
    manifest = json.loads((args.training / "spatial/manifest.json").read_text())
    converted = manifest["episodes"][0]
    first_row = int(converted["source_start"])
    released = []
    for row in (first_row, first_row + 31, first_row + int(converted["length"]) - 1):
        raw = np.asarray(source["pointcloud"][row])
        world = raw[:, :3] @ a.T + t
        rgb = np.asarray(source["front"][row])
        uv, pixels, valid = project(world, capture)
        color_error = np.abs(rgb[pixels[valid, 1], pixels[valid, 0]] - raw[valid, 3:6])
        released.append({"source_row": row, "valid": int(valid.sum()),
                         "rgb_mae_float01": float(color_error.mean()),
                         "max_pixel_rounding_error": float(np.abs(uv[valid] - pixels[valid]).max())})
        if row == first_row:
            train_xyz = np.load(args.training / "spatial/episodes/episode_000000/pointcloud_xyz.npy",
                                mmap_mode="r")[0]
            released[-1]["converted_training_max_xyz_difference_m"] = float(
                np.max(np.abs(world - train_xyz)))
    demo = dense_crop_demo(args.output / "dense_crop_illustration.png", args.probe, geometry)
    distribution = training_distribution(source, manifest, a, t)
    report = {"online": records, "release": released, "training_distribution": distribution,
              "dense_crop_illustration": demo,
              "scope": "Pixel correspondence and sparsity; does not prove that sparse clouds retain useful shape"}
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "index.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Cloud correspondence audit</title>"
        "<style>body{font:16px sans-serif;max-width:1600px;margin:2rem auto;background:#171a1e;color:#eee}"
        "img{width:100%}section{border-top:1px solid #555;padding:1rem 0}</style></head><body>"
        "<h1>Actual point cloud ↔ front RGB</h1><p>The green marks are policy input points "
        "reprojected into the matching front RGB frame. The colors on the 3D plot are sampled "
        "from RGB for display; the model receives only xyz. A matching projection proves alignment, "
        "not adequate object detail.</p><h2>Why cropping after sampling cannot restore detail</h2>"
        "<p>This separate archived depth capture illustrates a fixed geometric crop applied "
        "before sampling. Its crop bounds are only an example, not a validated new policy input.</p>"
        "<img src='dense_crop_illustration.png'>" + "".join(cards) + "</body></html>", encoding="utf-8")
    print(args.output / "index.html")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
