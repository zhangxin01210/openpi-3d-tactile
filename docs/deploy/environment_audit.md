# Deployment environment audit

Generated from `/home/sai/zx/openpi-3d-tactile` using `3.11.15 (main, Mar 10 2026, 18:16:52) [Clang 21.1.4 ]`.

## Host

- Git: `928898d1c3271be27b67e85581ceebae786aada9`; RGBTrack submodule `23e43f832b82f4813d6ab72773779bf8d2ee4d83`.
- GPU: `NVIDIA GeForce RTX 5090, 32607 MiB, 24634 MiB, 580.178.04`
- CUDA compiler: `nvcc: NVIDIA (R) Cuda compiler driver Copyright (c) 2005-2025 NVIDIA Corporation Built on Fri_Feb_21_20:23:50_PST_2025 Cuda compilation tools, release 12.8, V12.8.93 Build cuda_12.8.r12.8/compiler.35583870_0`
- CPU count: 32; free disk: 94.6 GiB.

## OpenPI and vision runtime

- PyTorch: `2.7.1+cu126`; CUDA runtime `12.6`; architectures `sm_50, sm_60, sm_70, sm_75, sm_80, sm_86, sm_90`.
- PyTorch GPU kernel smoke: **FAIL** — RuntimeError: CUDA error: no kernel image is available for execution on the device
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
.
- JAX devices: `cuda:0`; JAX GPU kernel smoke: **PASS**.

| Component | Status |
| --- | --- |
| torch | 2.7.1+cu126 |
| torchvision | 0.22.1+cu126 |
| jax | 0.5.3 |
| jaxlib | 0.5.3 |
| opencv | 4.11.0 |
| pytorch3d | missing |
| open3d | missing |
| nvdiffrast | import error: ModuleNotFoundError: No module named 'nvdiffrast' |
| warp_lang | missing |
| filterpy | missing |
| pybind11 | missing |
| trimesh | 5.1.0 |
| sam2 | missing |
| bop_toolkit_lib | missing |

## Isolated perception environment

- Package requirements: **PASS** — No broken requirements found.

- Python: `/home/sai/zx/openpi-3d-tactile/.venv-perception/bin/python`.
- PyTorch: `2.10.0+cu128`; CUDA `12.8`; device `NVIDIA GeForce RTX 5090`; kernel smoke: **PASS**.

| Component | Status |
| --- | --- |
| torch | 2.10.0+cu128 |
| pytorch3d | 0.7.9 |
| open3d | 0.20.0 |
| nvdiffrast | None |
| kornia | 0.9.0 |
| warp | 1.18.0 |
| filterpy | None |
| bop_toolkit_lib | None |
| sam2 | None |
| trimesh | 5.2.0 |
| joblib | 1.6.0 |
| h5py | 3.16.0 |
| pybind11 | 3.1.0 |

## Assets and checkpoints

- Funnel/flask GLBs: `True` / `True`.
- Prior metric scale manifest: `True`.
- RGBTrack scorer/refiner weights: `False` / `False`.
- Cached SAM2.1 small checkpoints: `['/home/sai/zx/openpi-3d-tactile/checkpoints/sam2/sam2.1_hiera_small.pt']`.

The project `.venv` is retained unchanged. Vision dependencies belong in `.venv-perception`.
