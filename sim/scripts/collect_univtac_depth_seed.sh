#!/usr/bin/env bash
# Collect one UniVTAC insert_hole seed only when RTX 5090 memory is available.
set -euo pipefail

seed="${1:-0}"
if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 [nonnegative_seed]" >&2
    exit 2
fi

runtime_root="${OPENPI_SIM_RUNTIME_ROOT:-/home/sai/zx/openpi-sim-runtime}"
source_root="${runtime_root}/third_party/UniVTAC-full"
python_bin="${UNIVTAC_PYTHON:-/home/sai/miniconda3/envs/univtac-isaac51/bin/python}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
config_path="${project_root}/sim/configs/univtac_insert_hole_depth.yml"
episode_file="${runtime_root}/data_depth/insert_hole/univtac_insert_hole_depth/hdf5/${seed}.hdf5"
minimum_free_mib="${UNIVTAC_MIN_FREE_GPU_MIB:-12000}"

if [[ ! -x "${python_bin}" || ! -f "${config_path}" || ! -d "${source_root}" ]]; then
    echo "Dedicated UniVTAC environment, source, or config is missing" >&2
    exit 1
fi
if [[ -e "${episode_file}" ]]; then
    echo "Episode already exists: ${episode_file}" >&2
    exit 1
fi
if [[ ! "${minimum_free_mib}" =~ ^[0-9]+$ ]]; then
    echo "UNIVTAC_MIN_FREE_GPU_MIB must be a nonnegative integer" >&2
    exit 2
fi

free_mib="$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
if [[ ! "${free_mib}" =~ ^[0-9]+$ ]]; then
    echo "Could not read free RTX 5090 memory" >&2
    exit 1
fi
if (( free_mib < minimum_free_mib )); then
    echo "Need at least ${minimum_free_mib} MiB free on GPU 0; currently ${free_mib} MiB" >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader >&2
    exit 75
fi

cd "${source_root}"
export OMNI_KIT_ACCEPT_EULA=YES
exec "${python_bin}" scripts/collect_data.py insert_hole "${config_path}" \
    --headless --start_seed "${seed}" --max_seed "${seed}"
