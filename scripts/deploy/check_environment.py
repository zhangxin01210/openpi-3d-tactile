#!/usr/bin/env python3
"""Audit deployment dependencies without mutating any environment."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def _run(*args: str, cwd: Path = ROOT) -> dict[str, Any]:
    try:
        result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _module(name: str) -> dict[str, Any]:
    try:
        if importlib.util.find_spec(name) is None:
            return {"installed": False}
        module = importlib.import_module(name)
        return {"installed": True, "version": getattr(module, "__version__", None)}
    except Exception as exc:  # a broken CUDA-linked import is useful audit data
        return {"installed": True, "import_error": f"{type(exc).__name__}: {exc}"}


def _perception_environment() -> dict[str, Any]:
    python = ROOT / ".venv-perception" / "bin" / "python"
    if not python.is_file():
        return {"available": False, "python": str(python)}
    audit_code = r'''
import importlib, importlib.util, json
names = {
    "torch": "torch", "pytorch3d": "pytorch3d", "open3d": "open3d",
    "nvdiffrast": "nvdiffrast.torch", "kornia": "kornia", "warp": "warp",
    "filterpy": "filterpy.kalman", "bop_toolkit_lib": "bop_toolkit_lib",
    "sam2": "sam2", "trimesh": "trimesh", "joblib": "joblib",
    "h5py": "h5py", "pybind11": "pybind11",
}
packages = {}
for label, name in names.items():
    try:
        if importlib.util.find_spec(name) is None:
            packages[label] = {"installed": False}
        else:
            module = importlib.import_module(name)
            packages[label] = {"installed": True, "version": getattr(module, "__version__", None)}
    except Exception as exc:
        packages[label] = {"installed": True, "import_error": f"{type(exc).__name__}: {exc}"}
torch = importlib.import_module("torch") if packages["torch"].get("installed") and "import_error" not in packages["torch"] else None
cuda = {"available": bool(torch and torch.cuda.is_available())}
if torch:
    cuda.update({"version": torch.__version__, "runtime": torch.version.cuda})
    if torch.cuda.is_available():
        cuda.update({"device": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)), "architectures": torch.cuda.get_arch_list()})
        try:
            cuda["kernel_smoke"] = float((torch.ones(1, device="cuda") + 1).item()) == 2.0
        except Exception as exc:
            cuda["kernel_smoke"] = False
            cuda["kernel_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps({"available": True, "python": __import__("sys").executable, "packages": packages, "torch_cuda": cuda}))
'''
    result = _run(str(python), "-c", audit_code)
    if not result["ok"]:
        return {"available": True, "python": str(python), "error": result.get("stderr") or result.get("stdout")}
    try:
        report = json.loads(result["stdout"])
    except json.JSONDecodeError as exc:
        return {"available": True, "python": str(python), "error": f"invalid JSON audit output: {exc}"}
    report["dependency_check"] = _run(str(python), "-m", "pip", "check")
    return report


def audit() -> dict[str, Any]:
    package_names = {
        "torch": "torch",
        "torchvision": "torchvision",
        "jax": "jax",
        "jaxlib": "jaxlib",
        "opencv": "cv2",
        "pytorch3d": "pytorch3d",
        "open3d": "open3d",
        "nvdiffrast": "nvdiffrast.torch",
        "warp_lang": "warp",
        "filterpy": "filterpy",
        "pybind11": "pybind11",
        "trimesh": "trimesh",
        "sam2": "sam2",
        "bop_toolkit_lib": "bop_toolkit_lib",
    }
    packages = {key: _module(module) for key, module in package_names.items()}
    torch_status: dict[str, Any] = {"available": False}
    if packages["torch"].get("installed") and "import_error" not in packages["torch"]:
        torch = importlib.import_module("torch")
        torch_status = {
            "available": bool(torch.cuda.is_available()),
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "architectures": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
        }
        if torch.cuda.is_available():
            torch_status["device_name"] = torch.cuda.get_device_name(0)
            torch_status["compute_capability"] = list(torch.cuda.get_device_capability(0))
            try:
                torch_status["kernel_smoke"] = float((torch.ones(1, device="cuda") + 1).item()) == 2.0
            except Exception as exc:
                torch_status["kernel_smoke"] = False
                torch_status["kernel_error"] = f"{type(exc).__name__}: {exc}"
    jax_status: dict[str, Any] = {"available": False}
    if packages["jax"].get("installed") and "import_error" not in packages["jax"]:
        try:
            import jax

            jax_status = {"devices": [str(device) for device in jax.devices()]}
            gpu = next((device for device in jax.devices() if device.platform == "gpu"), None)
            if gpu is not None:
                jax_status["gpu_kernel_smoke"] = int(jax.device_get(jax.device_put(16, gpu) + 120)) == 136
            jax_status["available"] = True
        except Exception as exc:
            jax_status["error"] = f"{type(exc).__name__}: {exc}"

    submodule = (ROOT / "third_party" / "RGBTrack")
    gpu = _run("nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader")
    df = shutil.disk_usage(ROOT)
    return {
        "root": str(ROOT),
        "git": {
            "head": _run("git", "rev-parse", "HEAD")["stdout"],
            "branch_status": _run("git", "status", "-sb")["stdout"],
            "submodule_status": _run("git", "submodule", "status")["stdout"],
            "rgbtrack_head": _run("git", "-C", str(submodule), "rev-parse", "HEAD")["stdout"],
        },
        "host": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "gpu": gpu,
            "nvcc": _run("nvcc", "--version"),
            "cmake": _run("cmake", "--version"),
            "compiler": _run("g++", "--version"),
            "memory": _run("free", "-h"),
            "disk_total_gib": round(df.total / 1024**3, 1),
            "disk_free_gib": round(df.free / 1024**3, 1),
        },
        "packages": packages,
        "torch_cuda": torch_status,
        "jax_cuda": jax_status,
        "perception_environment": _perception_environment(),
        "environments": {
            ".venv": str((ROOT / ".venv").resolve()) if (ROOT / ".venv").exists() else None,
            ".venv-perception": str(ROOT / ".venv-perception") if (ROOT / ".venv-perception").exists() else None,
            "conda": _run("conda", "env", "list"),
            "uv": _run("uv", "--version"),
            "system_python": shutil.which("python"),
            "system_pip": shutil.which("pip"),
        },
        "assets": {
            "funnel_glb": (ROOT / "assets/object_geometry/synthware_19_22/funnel_F181975.glb").is_file(),
            "target_flask_glb": (ROOT / "assets/object_geometry/synthware_19_22/flask_F301950.glb").is_file(),
            "previous_scale_manifest": (ROOT / "data/pose_eval/hand_scene_metric_mesh_scales.json").is_file(),
            "rgbtrack_score_weight": (ROOT / "checkpoints/rgbtrack/2024-01-11-20-02-45/model_best.pth").is_file(),
            "rgbtrack_refine_weight": (ROOT / "checkpoints/rgbtrack/2023-10-28-18-33-37/model_best.pth").is_file(),
            "sam2_checkpoint_cache_candidates": [
                str(path) for base in (Path.home() / ".cache", ROOT / "checkpoints")
                if base.exists() for path in base.rglob("sam2.1_hiera_small.pt")
            ][:10],
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    host = report["host"]
    torch = report["torch_cuda"]
    jax = report["jax_cuda"]
    packages = report["packages"]
    perception = report.get("perception_environment", {})
    rows = [
        "# Deployment environment audit",
        "",
        f"Generated from `{report['root']}` using `{report['host']['python'].splitlines()[0]}`.",
        "",
        "## Host",
        "",
        f"- Git: `{report['git']['head']}`; RGBTrack submodule `{report['git']['rgbtrack_head']}`.",
        f"- GPU: `{host['gpu'].get('stdout', 'unavailable')}`",
        f"- CUDA compiler: `{host['nvcc'].get('stdout', 'unavailable').replace(chr(10), ' ')}`",
        f"- CPU count: {host['cpu_count']}; free disk: {host['disk_free_gib']} GiB.",
        "",
        "## OpenPI and vision runtime",
        "",
        f"- PyTorch: `{packages['torch'].get('version', 'missing')}`; CUDA runtime `{torch.get('cuda_runtime')}`; architectures `{', '.join(torch.get('architectures', []))}`.",
        f"- PyTorch GPU kernel smoke: **{'PASS' if torch.get('kernel_smoke') else 'FAIL'}**{(' — ' + torch['kernel_error']) if torch.get('kernel_error') else ''}.",
        f"- JAX devices: `{', '.join(jax.get('devices', []))}`; JAX GPU kernel smoke: **{'PASS' if jax.get('gpu_kernel_smoke') else 'FAIL/UNAVAILABLE'}**.",
        "",
        "| Component | Status |",
        "| --- | --- |",
    ]
    for name, info in packages.items():
        status = info.get("version", "missing")
        if info.get("import_error"):
            status = f"import error: {info['import_error']}"
        rows.append(f"| {name} | {status} |")
    perception_cuda = perception.get("torch_cuda", {})
    rows.extend([
        "",
        "## Isolated perception environment",
        "",
        f"- Package requirements: **{'PASS' if perception.get('dependency_check', {}).get('ok') else 'FAIL/UNAVAILABLE'}** — {(perception.get('dependency_check', {}).get('stdout') or perception.get('dependency_check', {}).get('stderr') or perception.get('dependency_check', {}).get('error') or 'not checked').rstrip('.')}.",
        "",
        f"- Python: `{perception.get('python', 'missing')}`.",
        f"- PyTorch: `{perception_cuda.get('version', 'missing')}`; CUDA `{perception_cuda.get('runtime')}`; device `{perception_cuda.get('device')}`; kernel smoke: **{'PASS' if perception_cuda.get('kernel_smoke') else 'FAIL/UNAVAILABLE'}**.",
        "",
        "| Component | Status |",
        "| --- | --- |",
    ])
    for name, info in perception.get("packages", {}).items():
        status = info.get("version", "missing")
        if info.get("import_error"):
            status = f"import error: {info['import_error']}"
        rows.append(f"| {name} | {status} |")
    assets = report["assets"]
    rows.extend([
        "",
        "## Assets and checkpoints",
        "",
        f"- Funnel/flask GLBs: `{assets['funnel_glb']}` / `{assets['target_flask_glb']}`.",
        f"- Prior metric scale manifest: `{assets['previous_scale_manifest']}`.",
        f"- RGBTrack scorer/refiner weights: `{assets['rgbtrack_score_weight']}` / `{assets['rgbtrack_refine_weight']}`.",
        f"- Cached SAM2.1 small checkpoints: `{assets['sam2_checkpoint_cache_candidates']}`.",
        "",
        "The project `.venv` is retained unchanged. Vision dependencies belong in `.venv-perception`.",
        "",
    ])
    return "\n".join(rows)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=ROOT / "docs/deploy/environment_audit.json")
    parser.add_argument("--markdown", type=Path, default=ROOT / "docs/deploy/environment_audit.md")
    args = parser.parse_args()
    report = audit()
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(args.markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
