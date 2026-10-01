#!/usr/bin/env bash
# Run a Python script in the private build environment with private caches.
set -euo pipefail
CW_RUNTIME=/home/sai/zx/openpi-sim-runtime
CW_BUILD_ENV="$CW_RUNTIME/envs/contactworld-build-sm120"
export PATH="$CW_BUILD_ENV/bin:/usr/local/cuda-12.8/bin:$PATH"
export PYTHONNOUSERSITE=1
export CUDA_HOME=/usr/local/cuda-12.8
export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 CUDAHOSTCXX=/usr/bin/g++-11
export TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=12
export TORCH_EXTENSIONS_DIR="$CW_RUNTIME/build_support/contactworld-sm120/torch_extensions"
export PYTHONPATH="$CW_RUNTIME/third_party/IsaacGym_Preview_TacSL_Package/isaacgym/python"
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:$CW_BUILD_ENV/lib"
ulimit -c 0
exec "$CW_BUILD_ENV/bin/python" "$@"
