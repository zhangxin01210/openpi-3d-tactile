#!/usr/bin/env python3
"""Staged RGBTrack runtime checks for the computer-4 environment.

The checker is intentionally separate from the real e23 command.  A passing
Python import check is never reported as a successful model or tracking run.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any


REQUIRED_IMPORTS = {
    "torch": "torch",
    "pytorch3d": "pytorch3d",
    "open3d": "open3d",
    "trimesh": "trimesh",
    "omegaconf": "omegaconf",
    "kornia": "kornia",
    "einops": "einops",
    "transformations": "transformations",
    "ruamel.yaml": "ruamel.yaml",
    "opencv": "cv2",
    "scipy": "scipy",
    "nvdiffrast": "nvdiffrast.torch",
    "warp": "warp",
    "filterpy": "filterpy.kalman",
    "bop_toolkit_lib": "bop_toolkit_lib",
}
OPTIONAL_IMPORTS = {
    "tensorrt": "tensorrt",
    "bundlesdf.mycuda.common": "bundlesdf.mycuda.common",
}


def _try_import(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # import errors may be CUDA loader errors
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "version": getattr(module, "__version__", None)}


def setup_weights_link(repo_root: Path, weights_root: Path) -> Path:
    source = weights_root.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"weights root does not exist: {source}")
    target = repo_root / "third_party" / "RGBTrack" / "weights"
    if target.is_symlink():
        resolved = target.resolve()
        if resolved != source:
            raise FileExistsError(f"Refusing to replace weights symlink {target} -> {resolved}")
        return target
    if target.exists():
        raise FileExistsError(
            f"Refusing to replace existing non-symlink {target}; move it aside manually if it is a stale checkout"
        )
    target.symlink_to(source, target_is_directory=True)
    return target


def phase_a(repo_root: Path, weights_root: Path) -> dict[str, Any]:
    del repo_root
    report: dict[str, Any] = {"phase": "A", "imports": {}, "weights": {}}
    for label, module_name in REQUIRED_IMPORTS.items():
        report["imports"][label] = _try_import(module_name)
    for label, module_name in OPTIONAL_IMPORTS.items():
        report["imports"][label] = _try_import(module_name)
    try:
        from rgbtrack_adapter import expected_rgbtrack_weights

        paths = expected_rgbtrack_weights(weights_root)
        report["weights"] = {name: {"path": str(path), "exists": path.is_file()} for name, path in paths.items()}
    except Exception as exc:
        report["weights"] = {"error": f"{type(exc).__name__}: {exc}"}
    report["cuda"] = {"checked": False, "note": "Phase A only checks imports; use phase C for a CUDA render."}
    # TensorRT and bundlesdf are intentionally optional; the ordinary
    # PyTorch path does not require either one.
    report["ok"] = all(report["imports"][label]["ok"] for label in REQUIRED_IMPORTS)
    return report


def phase_b(repo_root: Path, weights_root: Path, mesh_path: Path | None, debug_dir: Path) -> dict[str, Any]:
    from rgbtrack_adapter import _import_rgbtrack
    from rgbtrack_adapter import validate_mycpp
    from rgbtrack_adapter import validate_rgbtrack_weights

    report: dict[str, Any] = {"phase": "B", "ok": False}
    report["weights"] = validate_rgbtrack_weights(repo_root, weights_root)
    report["mycpp"] = validate_mycpp(repo_root)
    FoundationPose, trimesh, compatibility = _import_rgbtrack(repo_root)
    if mesh_path is None:
        mesh = trimesh.creation.box(extents=(0.08, 0.08, 0.08))
        report["mesh"] = "synthetic_box"
    else:
        mesh = trimesh.load(str(mesh_path.expanduser().resolve()), force="mesh", process=False)
        report["mesh"] = str(mesh_path.expanduser().resolve())
    estimator = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        debug=0,
        debug_dir=str(debug_dir),
    )
    del estimator
    report["runtime_compatibility"] = compatibility
    report["ok"] = True
    return report


def phase_c() -> dict[str, Any]:
    import torch
    import nvdiffrast.torch as dr

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    context = dr.RasterizeCudaContext()
    positions = torch.tensor(
        [[[-0.7, -0.7, 0.0, 1.0], [0.7, -0.7, 0.0, 1.0], [0.0, 0.7, 0.0, 1.0]]],
        dtype=torch.float32,
        device="cuda",
    )
    triangles = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")
    raster, _ = dr.rasterize(context, positions, triangles, resolution=[16, 16])
    torch.cuda.synchronize()
    if not bool((raster[..., 3] > 0).any().item()):
        raise RuntimeError("NVDiffRast returned an empty triangle raster")
    return {
        "phase": "C",
        "ok": True,
        "torch": getattr(torch, "__version__", None),
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--weights-root", type=Path, default=None)
    parser.add_argument("--phase", choices=("A", "B", "C", "all"), default="A")
    parser.add_argument("--mesh", type=Path, default=None, help="Optional metric OBJ/PLY for phase B")
    parser.add_argument("--debug-dir", type=Path, default=Path(tempfile.gettempdir()) / "rgbtrack_runtime_check")
    parser.add_argument("--setup-weights-link", action="store_true")
    parser.add_argument("--json-output", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    weights_root = (args.weights_root or (repo_root / "checkpoints" / "rgbtrack")).expanduser().resolve()
    report: dict[str, Any] = {"repo_root": str(repo_root), "weights_root": str(weights_root)}
    try:
        if args.setup_weights_link:
            report["weights_link"] = str(setup_weights_link(repo_root, weights_root))
        if args.phase in ("A", "all"):
            report["A"] = phase_a(repo_root, weights_root)
        if args.phase in ("B", "all"):
            report["B"] = phase_b(repo_root, weights_root, args.mesh, args.debug_dir.expanduser().resolve())
        if args.phase in ("C", "all"):
            report["C"] = phase_c()
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        report["ok"] = False
    else:
        report["ok"] = all(
            value.get("ok", True)
            for key, value in report.items()
            if key in {"A", "B", "C"} and isinstance(value, dict)
        )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.json_output is not None:
        args.json_output.expanduser().resolve().write_text(rendered + "\n", encoding="utf-8")
    return 0 if report.get("ok", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
