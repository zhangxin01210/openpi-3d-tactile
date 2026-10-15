"""Leakage-free, portable features for the ContactWorld action-mixture audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import resnet18

VERSION = "cw-neighbor-resnet18-grid64-bilateral-v1"
MODES = {"S": ("state",), "R": ("rgb", "state"), "RP": ("rgb", "state", "cloud"),
         "RT": ("rgb", "state", "tactile"), "RPT": ("rgb", "state", "cloud", "tactile")}
DEFAULT_WEIGHTS = Path.home() / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def choose_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    try:
        if torch.cuda.is_available():
            # cuda.is_available alone accepts the old Torch build on a 5090.
            x = torch.ones(4, device="cuda")
            float((x*x).sum().item())
            return torch.device("cuda")
    except RuntimeError as error:
        print(f"CUDA kernel probe failed; using CPU: {str(error).splitlines()[0]}", flush=True)
    return torch.device("cpu")


class Features:
    """Frozen ImageNet features + continuous base-frame geometry + both fingers.

    Fixed random projections do not use dataset labels or held-out statistics.
    No simulator object pose, contact label, time, or action enters the features.
    """
    def __init__(self, weights=DEFAULT_WEIGHTS, device="auto", batch=32):
        if not Path(weights).is_file():
            raise FileNotFoundError(f"Local ResNet18 weights required (no auto-download): {weights}")
        self.device = choose_device(device)
        self.batch = batch
        model = resnet18(weights=None)
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.encoder = torch.nn.Sequential(*list(model.children())[:-2]).eval().to(self.device)
        self.projectors = {}
        self.mean = torch.tensor([.485, .456, .406], device=self.device)[None, :, None, None]
        self.std = torch.tensor([.229, .224, .225], device=self.device)[None, :, None, None]

    def project(self, x, seed, width=128):
        key = (x.shape[-1], seed, width)
        if key not in self.projectors:
            rng = np.random.default_rng(seed)
            self.projectors[key] = rng.choice([-1., 1.], size=(x.shape[-1], width)).astype(np.float32) / np.sqrt(width)
        return (np.asarray(x, np.float32) @ self.projectors[key]).astype(np.float32)

    @torch.inference_mode()
    def rgb(self, images):
        results = []
        for start in range(0, len(images), self.batch):
            x = torch.from_numpy(np.asarray(images[start:start+self.batch]).copy()).to(self.device)
            x = x.permute(0, 3, 1, 2).float()/255
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            x = self.encoder((x-self.mean)/self.std)
            x = F.adaptive_avg_pool2d(x, (4, 4)).flatten(1).cpu().numpy()
            # Retain coarse spatial layout, unlike global pooling.
            results.append(self.project(x, 1701))
        return np.concatenate(results)

    def cloud(self, xyz):
        # Three 64x64 continuous occupancy projections. Bilinear splatting
        # retains sub-bin changes; no per-frame centering or object GT crop.
        low = np.array([-.20, -.35, -.005], np.float32)
        high = np.array([.65, .35, .35], np.float32)
        all_features = []
        for cloud in xyz:
            p = np.clip((cloud-low)/(high-low), 0, 1)*63
            planes = []
            for a, b in ((0, 1), (0, 2), (1, 2)):
                u, v = p[:, a], p[:, b]
                ui, vi = np.floor(u).astype(int), np.floor(v).astype(int)
                du, dv = u-ui, v-vi
                grid = np.zeros(4096, np.float32)
                for i, j, w in ((0, 0, (1-du)*(1-dv)), (1, 0, du*(1-dv)),
                                (0, 1, (1-du)*dv), (1, 1, du*dv)):
                    cell = np.minimum(ui+i, 63)*64 + np.minimum(vi+j, 63)
                    grid += np.bincount(cell, weights=w, minlength=4096).astype(np.float32)
                planes.append(np.sqrt(grid/max(len(cloud), 1)))
            all_features.append(np.concatenate(planes))
        return self.project(np.stack(all_features), 1702)

    def tactile(self, xyz, base, local):
        # Preserve both finger identities and taxel order. Force sign and
        # magnitude survive signed-log compression. No per-frame normalization.
        force = lambda f: np.sign(f)*np.log1p(np.abs(f))
        n = len(xyz)
        x = np.concatenate([(xyz/.1).reshape(n, -1), force(base).reshape(n, -1),
                            force(local).reshape(n, -1)], axis=1)
        return self.project(x, 1703)

    def observation(self, front, wrist, state, cloud_xyz, taxel_xyz_base, force_base, force_local):
        out = {"rgb": np.concatenate([self.rgb(front), self.rgb(wrist)], axis=1),
               "state": np.asarray(state, np.float32), "cloud": self.cloud(cloud_xyz),
               "tactile": self.tactile(taxel_xyz_base, force_base, force_local)}
        for key, x in out.items():
            if not np.isfinite(x).all():
                raise ValueError(f"Nonfinite feature: {key}")
        return out


def decode(path, count):
    reader = cv2.VideoCapture(str(path))
    images = []
    try:
        for _ in range(count):
            ok, bgr = reader.read()
            if not ok:
                raise RuntimeError(f"Missing frame {len(images)}: {path}")
            images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        reader.release()
    return np.stack(images)


def prepare(dataset, output, weights, device, batch, limit=None):
    import pyarrow.parquet as pq
    dataset, output = Path(dataset), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset / "spatial/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    identity = {"version": VERSION, "manifest_sha256": digest(manifest_path),
                "vision_weights_sha256": digest(weights), "limit": limit}
    meta = output / "features_manifest.json"
    if meta.exists() and json.loads(meta.read_text()) != identity:
        raise ValueError("Feature cache protocol differs; choose a new output directory")
    meta.write_text(json.dumps(identity, indent=2)+"\n")
    entries = [e for e in manifest["episodes"] if e["split"] == "train"]
    if limit:
        entries = entries[:limit]
    extractor = None
    pieces = []
    for number, e in enumerate(entries, 1):
        ep, source, count = e["episode_index"], e["source_episode"], e["length"]-15
        file = output / "episodes" / f"source_{source:03d}.npz"
        if not file.exists():
            if extractor is None:
                extractor = Features(weights, device, batch)
            table = pq.read_table(dataset / f"data/chunk-000/episode_{ep:06d}.parquet")
            state = np.asarray(table["observation.state"].to_pylist(), np.float32)
            action = np.asarray(table["action"].to_pylist(), np.float32)
            spatial = dataset / f"spatial/episodes/episode_{ep:06d}"
            load = lambda name: np.load(spatial / (name+".npy"), mmap_mode="r")[:count]
            front = decode(dataset / f"videos/chunk-000/observation.images.front/episode_{ep:06d}.mp4", count)
            wrist = decode(dataset / f"videos/chunk-000/observation.images.wrist/episode_{ep:06d}.mp4", count)
            feat = extractor.observation(front, wrist, state[:count], load("pointcloud_fused_xyz"),
                np.stack([load("tactile_xyz_left_base"), load("tactile_xyz_right_base")], axis=1),
                np.stack([load("tactile_force_left_base"), load("tactile_force_right_base")], axis=1),
                np.stack([load("force_grid_left_local"), load("force_grid_right_local")], axis=1))
            phase = np.full(count, -1, np.int8)
            replay = Path(manifest["capture_root"]) / f"insertion_usb_episode_{source:03d}/replay.json"
            if replay.exists():
                rr = json.loads(replay.read_text())["rows"]
                hit = np.flatnonzero(np.linalg.norm([r["plug_socket_force_post"] for r in rr], axis=1) > .1)
                phase[:] = 0
                if len(hit):
                    phase[int(hit[0])+1:] = 1
            target = np.stack([action[i:i+4] for i in range(count)])
            assert target.shape == (count, 4, 6) and state.shape[1] == 18
            save_npz(file, **feat, action=target, source=np.full(count, source),
                     episode=np.full(count, ep), frame=np.arange(count), phase=phase)
        with np.load(file) as z:
            pieces.append({k: z[k] for k in z.files})
        print(f"features {number}/{len(entries)} source={source}", flush=True)
    combined = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    save_npz(output / "train_features.npz", **combined)
    return combined
