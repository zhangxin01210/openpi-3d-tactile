#!/usr/bin/env bash
# Build only inside the dedicated clone and source tree. No global installs.
set -euo pipefail
CW_RUNTIME=/home/sai/zx/openpi-sim-runtime
CW_BUILD_ENV="$CW_RUNTIME/envs/contactworld-build-sm120"
CW_SOURCE="$CW_RUNTIME/third_party/pytorch-cw-sm120"
# CUDA's include directory contains cuDNN 9.21 and precedes separate cuDNN
# includes. Use that existing installation consistently; never modify it.
CW_CUDNN=/usr/local/cuda-12.8
export PATH="$CW_BUILD_ENV/bin:/usr/local/cuda-12.8/bin:$PATH"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11
export CUDA_HOME=/usr/local/cuda-12.8 CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8
export CUDNN_INCLUDE_DIR="$CW_CUDNN/include"
export CUDNN_ROOT="$CW_CUDNN"
export CUDNN_LIBRARY="$CW_CUDNN/lib64/libcudnn.so.9"
export LD_LIBRARY_PATH="$CW_CUDNN/lib64:$CW_BUILD_ENV/lib"
export CMAKE_PREFIX_PATH="$CW_BUILD_ENV"
export TORCH_CUDA_ARCH_LIST=12.0
export PYTORCH_BUILD_VERSION=2.4.1+cu128sm120 PYTORCH_BUILD_NUMBER=1
export MAX_JOBS=24 CMAKE_BUILD_PARALLEL_LEVEL=24
export BUILD_TEST=0 USE_CUDA=1 USE_CUDNN=1
export USE_DISTRIBUTED=0 USE_NCCL=0 USE_GLOO=0 USE_MPI=0 USE_TENSORPIPE=0
export USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0 USE_KINETO=0 USE_ITT=0
export USE_FBGEMM=0 USE_XNNPACK=0 USE_NNPACK=0 USE_QNNPACK=0 USE_PYTORCH_QNNPACK=0
export USE_MKLDNN=0
cd "$CW_SOURCE"
test "$(git rev-parse HEAD)" = ee1b6804381c57161c477caa380a840a84167676
exec "$CW_BUILD_ENV/bin/python" setup.py bdist_wheel
