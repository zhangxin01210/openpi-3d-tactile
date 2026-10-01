#!/usr/bin/env bash
set -euo pipefail
CW_RUNTIME=/home/sai/zx/openpi-sim-runtime
CW_BUILD_ENV="$CW_RUNTIME/envs/contactworld-build-sm120"
export PATH="$CW_BUILD_ENV/bin:/usr/local/cuda-12.8/bin:$PATH"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export CUDA_HOME=/usr/local/cuda-12.8
export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 CUDAHOSTCXX=/usr/bin/g++-11
export TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=16 FORCE_CUDA=1
export BUILD_VERSION=0.19.1+cu128sm120
export TORCHVISION_USE_NVJPEG=0 TORCHVISION_USE_FFMPEG=0 TORCHVISION_USE_VIDEO_CODEC=0
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:$CW_BUILD_ENV/lib"
cd "$CW_RUNTIME/third_party/vision-0.19.1"
"$CW_BUILD_ENV/bin/python" -c 'import torch; assert torch.__version__ == "2.4.1+cu128sm120", torch.__version__'
exec "$CW_BUILD_ENV/bin/python" setup.py bdist_wheel
