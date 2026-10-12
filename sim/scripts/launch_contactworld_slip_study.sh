#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"
output_dir="${CW_SLIP_OUTPUT:-/home/sai/zx/openpi-sim-runtime/runs/contactworld_slip_study_20261010}"
if [[ " $* " == *" --smoke "* && -z "${CW_SLIP_OUTPUT:-}" ]]; then output_dir="${output_dir}_smoke"; fi
mkdir -p "$output_dir"
if [[ "${1:-}" == "--foreground" ]]; then
  shift
  exec 9>"$output_dir/.run.lock"
  flock -n 9 || { echo "Already running: $output_dir" >&2; exit 1; }
  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
  export MPLCONFIGDIR="$output_dir/.matplotlib"
  exec "$repo_dir/.venv/bin/python" -u sim/scripts/run_contactworld_slip_study.py --output "$output_dir" "$@"
else
  nohup bash "$0" --foreground "$@" >>"$output_dir/launch.log" 2>&1 </dev/null &
  printf 'Started PID %s\nLog: %s\nReport: %s\nFollow: tail -f %q\n' \
    "$!" "$output_dir/launch.log" "$output_dir/index.html" "$output_dir/launch.log"
fi
