#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"
output_dir="${CW_DISCOVERY_OUTPUT:-/home/sai/zx/openpi-sim-runtime/runs/contactworld_sensor_discovery_20261004}"
mkdir -p "$output_dir"
if [[ " ${*} " == *" --smoke "* ]]; then
  exec flock -n "$output_dir/.run.lock" env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 \
    .venv/bin/python -u sim/scripts/run_contactworld_sensor_discovery.py --output "$output_dir" "$@"
fi
launch_log="$output_dir/launch.log"
nohup flock -n "$output_dir/.run.lock" env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 \
  .venv/bin/python -u sim/scripts/run_contactworld_sensor_discovery.py --output "$output_dir" "$@" \
  >> "$launch_log" 2>&1 < /dev/null &
run_pid=$!
printf 'Started PID %s\nLog: %s\nReport: %s\nFollow: tail -f %q\n' \
  "$run_pid" "$launch_log" "$output_dir/index.html" "$launch_log"
