#!/usr/bin/env bash
# Configure the vendored UniVTAC/UIPC extension for an RTX 5090 without
# changing an existing Isaac Sim or OpenPI environment.
set -euo pipefail

if [[ "${1:-}" != "" && "${1:-}" != "--build" ]]; then
    echo "Usage: $0 [--build]" >&2
    exit 2
fi

runtime_root="${OPENPI_SIM_RUNTIME_ROOT:-/home/sai/zx/openpi-sim-runtime}"
env_prefix="${UNIVTAC_ENV_PREFIX:-/home/sai/miniconda3/envs/univtac-isaac51}"
cuda_root="${UNIVTAC_CUDA_HOME:-/usr/local/cuda-12.8}"
source_root="${runtime_root}/third_party/UniVTAC-full/third_party/TacEx/source/tacex_uipc"
build_dir="${UNIVTAC_UIPC_BUILD_DIR:-${source_root}/build-openpi-sm120}"
vcpkg_root="${runtime_root}/toolchains/vcpkg"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ ! -x "${env_prefix}/bin/python" || ! -x "${cuda_root}/bin/nvcc" ]]; then
    echo "Missing dedicated Python environment or CUDA toolkit" >&2
    exit 1
fi
if ! "${cuda_root}/bin/nvcc" --list-gpu-arch | grep -qx compute_120; then
    echo "Selected CUDA toolkit cannot compile compute_120" >&2
    exit 1
fi

export UNIVTAC_GCC12="${env_prefix}/bin/x86_64-conda-linux-gnu-gcc"
export UNIVTAC_GXX12="${env_prefix}/bin/x86_64-conda-linux-gnu-c++"
export CC="${runtime_root}/third_party/UniVTAC-full/scripts/toolchains/gcc12-system-ld"
export CXX="${runtime_root}/third_party/UniVTAC-full/scripts/toolchains/gxx12-system-ld"
export CUDAHOSTCXX="${CXX}"
export CUDACXX="${cuda_root}/bin/nvcc"
export CUDA_HOME="${cuda_root}"
export CUDA_PATH="${cuda_root}"
export VCPKG_ROOT="${vcpkg_root}"
export CMAKE_PREFIX_PATH="${env_prefix}/lib/python3.11/site-packages/cmeel.prefix${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
export LD_LIBRARY_PATH="${env_prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PATH="${env_prefix}/bin:${cuda_root}/bin:${PATH}"

cmake -S "${source_root}/libuipc" -B "${build_dir}" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="${vcpkg_root}/scripts/buildsystems/vcpkg.cmake" \
    -DVCPKG_OVERLAY_PORTS="${project_root}/sim/overlays;${source_root}/overlay-ports" \
    -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DUIPC_BUILD_PYBIND=ON \
    -DUIPC_DEV_MODE=OFF \
    -DUIPC_BUILD_GUI=OFF \
    -DUIPC_BUILD_EXAMPLES=OFF \
    -DUIPC_BUILD_TESTS=OFF \
    -DUIPC_BUILD_BENCHMARKS=OFF \
    -DMUDA_BUILD_EXAMPLE=OFF \
    -DMUDA_BUILD_TEST=OFF \
    -DUIPC_PYTHON_EXECUTABLE_PATH="${env_prefix}/bin/python"

if [[ "${1:-}" == "--build" ]]; then
    cmake --build "${build_dir}" --parallel "${UNIVTAC_BUILD_JOBS:-4}"
fi
