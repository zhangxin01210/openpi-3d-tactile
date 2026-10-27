#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_DIR="${OPENPI_PERCEPTION_ENV:-$ROOT/.venv-perception}"
SAM2_ROOT="${OPENPI_SAM2_ROOT:-$HOME/.cache/openpi_deploy/sam2}"
SAM2_REVISION="2b90b9f5ceec907a1c18123530e92e794ad901a4"
PYPI_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
BASE_PYTHON="${OPENPI_PERCEPTION_BASE_PYTHON:-$HOME/.conda/envs/lerobot5090/bin/python}"

if [[ -x "$ENV_DIR/bin/python" ]]; then
  "$ENV_DIR/bin/python" - <<'PY' >/dev/null
import torch
assert torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (12, 0)
assert float((torch.ones(1, device='cuda') + 1).item()) == 2.0
PY
  REUSE_BLACKWELL_TORCH=1
elif [[ -x "$BASE_PYTHON" ]] && "$BASE_PYTHON" - <<'PY' >/dev/null 2>&1
import torch
assert torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (12, 0)
assert float((torch.ones(1, device='cuda') + 1).item()) == 2.0
PY
then
  # Reuse the already installed, tested sm_120 Torch read-only through a new
  # project-owned overlay. Pip writes only into ENV_DIR.
  uv venv --system-site-packages --python "$BASE_PYTHON" "$ENV_DIR"
  REUSE_BLACKWELL_TORCH=1
else
  uv venv --python 3.11 "$ENV_DIR"
  REUSE_BLACKWELL_TORCH=0
fi
PY="$ENV_DIR/bin/python"

env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy \
  uv pip install --python "$PY" --index-url "$PYPI_MIRROR" --upgrade pip
VISION_PACKAGES=(
  'trimesh>=4.0' 'hydra-core==1.3.2' 'iopath>=0.1.10' 'tqdm>=4.65' 'pillow>=10'
  'omegaconf==2.3.0' 'kornia>=0.7' 'einops>=0.7' 'transformations>=2022.9'
  'ruamel.yaml>=0.18' 'filterpy>=1.4.5' 'warp-lang>=1.0' 'pybind11>=2.12'
  'pyyaml>=6' 'antlr4-python3-runtime==4.9.3' 'portalocker>=2.8'
  'numpy==2.2.6' 'packaging==25.0' 'setuptools==80.10.2' 'cloudpickle' 'wheel'
)
if [[ "$REUSE_BLACKWELL_TORCH" == "0" ]]; then
  VISION_PACKAGES+=( 'torch==2.10.0' 'torchvision==0.25.0' 'opencv-python-headless>=4.10,<5' 'scipy>=1.10' )
fi
env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy \
uv pip install --python "$PY" --no-deps --index-url "$PYPI_MIRROR" "${VISION_PACKAGES[@]}"

if [[ ! -d "$SAM2_ROOT/.git" ]]; then
  mkdir -p "$(dirname "$SAM2_ROOT")"
  env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy \
    git clone https://github.com/facebookresearch/sam2.git "$SAM2_ROOT"
fi
if ! git -C "$SAM2_ROOT" cat-file -e "$SAM2_REVISION^{commit}" 2>/dev/null; then
  env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy \
    git -C "$SAM2_ROOT" fetch --depth 1 origin "$SAM2_REVISION"
fi
git -C "$SAM2_ROOT" checkout --detach "$SAM2_REVISION"
env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy \
  uv pip install --python "$PY" --no-deps --no-build-isolation -e "$SAM2_ROOT"

"$PY" - <<'PY'
import torch
print('torch', torch.__version__, 'CUDA', torch.version.cuda)
assert torch.cuda.is_available(), 'PyTorch does not see the GPU'
print('device', torch.cuda.get_device_name(0), 'capability', torch.cuda.get_device_capability(0))
assert torch.cuda.get_device_capability(0) == (12, 0), 'Expected RTX 5090 / sm_120'
assert float((torch.ones(1, device='cuda') + 1).item()) == 2.0, 'CUDA kernel did not execute'
print('CUDA kernel smoke: OK')
import sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
print('SAM2 import: OK', sam2.__file__, SAM2ImagePredictor.__name__)
PY

echo "Perception environment ready: $ENV_DIR"
echo "SAM2 source: $SAM2_ROOT @ $SAM2_REVISION"
echo "SAM2 weights remain a separate verified download step; RGBTrack CUDA extensions require the staged checks in docs/deploy/README.md."
