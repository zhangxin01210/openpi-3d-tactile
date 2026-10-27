#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_DIR="${OPENPI_PERCEPTION_ENV:-$ROOT/.venv-perception}"
PY="$ENV_DIR/bin/python"
CACHE_ROOT="${OPENPI_DEPLOY_CACHE:-$HOME/.cache/openpi_deploy}"
PYPI_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
BUILD_JOBS="${RGBTRACK_BUILD_JOBS:-4}"
NVDIFFRAST_ROOT="$CACHE_ROOT/nvdiffrast"
PYTORCH3D_ROOT="$CACHE_ROOT/pytorch3d"
BOP_ROOT="$CACHE_ROOT/bop_toolkit"
NVDIFFRAST_REVISION="20c4135fda78ae29d7658bea28b6c1d5b5e103e5"
PYTORCH3D_REVISION="204fe5831b3a5bbe079f09d72afb4515178f8892"
BOP_REVISION="cea62d651c7e395b2e1962b9749e4e89693c6ac4"

if [[ ! -x "$PY" ]]; then
  echo "Missing perception Python: $PY. Run scripts/deploy/setup_perception.sh first." >&2
  exit 2
fi
mkdir -p "$CACHE_ROOT"

env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy \
  uv pip install --python "$PY" --index-url "$PYPI_MIRROR" \
  'numpy==2.2.6' 'packaging==25.0' 'setuptools==80.10.2' 'omegaconf==2.3.0' \
  cloudpickle nbformat configargparse ipywidgets \
  'kornia-rs>=0.2' 'open3d==0.20.0' ninja plotly dash joblib h5py

if [[ ! -d "$NVDIFFRAST_ROOT/.git" ]]; then
  env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy \
    git clone https://github.com/NVlabs/nvdiffrast.git "$NVDIFFRAST_ROOT"
fi
if ! git -C "$NVDIFFRAST_ROOT" cat-file -e "$NVDIFFRAST_REVISION^{commit}" 2>/dev/null; then
  env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy \
    git -C "$NVDIFFRAST_ROOT" fetch --depth 1 origin "$NVDIFFRAST_REVISION"
fi
git -C "$NVDIFFRAST_ROOT" checkout --detach "$NVDIFFRAST_REVISION"
env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy \
  PATH="$ENV_DIR/bin:$PATH" TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS="$BUILD_JOBS" \
  uv pip install --python "$PY" --no-build-isolation --no-deps "$NVDIFFRAST_ROOT"

if [[ ! -d "$PYTORCH3D_ROOT/.git" ]]; then
  env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy \
    git clone https://github.com/facebookresearch/pytorch3d.git "$PYTORCH3D_ROOT"
fi
if ! git -C "$PYTORCH3D_ROOT" cat-file -e "$PYTORCH3D_REVISION^{commit}" 2>/dev/null; then
  env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy \
    git -C "$PYTORCH3D_ROOT" fetch --depth 1 origin "$PYTORCH3D_REVISION"
fi
git -C "$PYTORCH3D_ROOT" checkout --detach "$PYTORCH3D_REVISION"
env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy \
  PATH="$ENV_DIR/bin:$PATH" TORCH_CUDA_ARCH_LIST=12.0 FORCE_CUDA=1 MAX_JOBS="$BUILD_JOBS" \
  uv pip install --python "$PY" --no-build-isolation --no-deps "$PYTORCH3D_ROOT"

if [[ ! -d "$BOP_ROOT/.git" ]]; then
  env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy \
    git clone https://github.com/thodan/bop_toolkit.git "$BOP_ROOT"
fi
if ! git -C "$BOP_ROOT" cat-file -e "$BOP_REVISION^{commit}" 2>/dev/null; then
  env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy \
    git -C "$BOP_ROOT" fetch --depth 1 origin "$BOP_REVISION"
fi
git -C "$BOP_ROOT" checkout --detach "$BOP_REVISION"
# Expose only the import package. The source checkout includes stale packaging
# metadata whose broad, obsolete requirements conflict with the shared LeRobot
# environment, while RGBTrack only needs the pinned BOP Python modules.
BOP_IMPORT_ROOT="$CACHE_ROOT/bop_import_path"
mkdir -p "$BOP_IMPORT_ROOT"
if [[ -e "$BOP_IMPORT_ROOT/bop_toolkit_lib" || -L "$BOP_IMPORT_ROOT/bop_toolkit_lib" ]]; then
  if [[ "$(realpath "$BOP_IMPORT_ROOT/bop_toolkit_lib")" != "$(realpath "$BOP_ROOT/bop_toolkit_lib")" ]]; then
    echo "Unexpected BOP import link: $BOP_IMPORT_ROOT/bop_toolkit_lib" >&2
    exit 2
  fi
else
  ln -s "$BOP_ROOT/bop_toolkit_lib" "$BOP_IMPORT_ROOT/bop_toolkit_lib"
fi
SITE_PACKAGES="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
printf '%s\n' "$BOP_IMPORT_ROOT" > "$SITE_PACKAGES/openpi_bop_toolkit_source.pth"

"$PY" -m pip check

PYBIND11_CMAKE="$("$PY" -m pybind11 --cmakedir)"
cmake -S "$ROOT/third_party/RGBTrack/mycpp" -B "$ROOT/third_party/RGBTrack/mycpp/build" \
  -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE="$PY" -Dpybind11_DIR="$PYBIND11_CMAKE"
cmake --build "$ROOT/third_party/RGBTrack/mycpp/build" --parallel "$BUILD_JOBS"

mkdir -p "$ROOT/docs/deploy"
mkdir -p "$ROOT/checkpoints/rgbtrack"
PATH="$ENV_DIR/bin:$PATH" PYTHONPATH="$ROOT/src" "$PY" \
  "$ROOT/scripts/object_geometry/pose_tracking/check_rgbtrack_runtime.py" \
  --repo-root "$ROOT" --weights-root "$ROOT/checkpoints/rgbtrack" --phase A \
  --setup-weights-link \
  --json-output "$ROOT/docs/deploy/rgbtrack_phase_a.json"
PATH="$ENV_DIR/bin:$PATH" PYTHONPATH="$ROOT/src" "$PY" \
  "$ROOT/scripts/object_geometry/pose_tracking/check_rgbtrack_runtime.py" \
  --repo-root "$ROOT" --weights-root "$ROOT/checkpoints/rgbtrack" --phase C \
  --json-output "$ROOT/docs/deploy/rgbtrack_phase_c.json"

SCORE_WEIGHT="$ROOT/checkpoints/rgbtrack/2024-01-11-20-02-45/model_best.pth"
REFINE_WEIGHT="$ROOT/checkpoints/rgbtrack/2023-10-28-18-33-37/model_best.pth"
if [[ -f "$SCORE_WEIGHT" && -f "$REFINE_WEIGHT" ]]; then
  PATH="$ENV_DIR/bin:$PATH" PYTHONPATH="$ROOT/src" "$PY" \
    "$ROOT/scripts/object_geometry/pose_tracking/check_rgbtrack_runtime.py" \
    --repo-root "$ROOT" --weights-root "$ROOT/checkpoints/rgbtrack" --phase B \
    --json-output "$ROOT/docs/deploy/rgbtrack_phase_b.json"
else
  echo "[BLOCKED] Phase B requires both RGBTrack model_best.pth files under $ROOT/checkpoints/rgbtrack."
fi
