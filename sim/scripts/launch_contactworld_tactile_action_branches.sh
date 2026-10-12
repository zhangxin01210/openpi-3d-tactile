#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"
output_dir="${CW_TACTILE_BRANCH_OUTPUT:-/home/sai/zx/openpi-sim-runtime/runs/contactworld_tactile_action_branches_20261008}"
mkdir -p "$output_dir"
run_all() {
  local extra=("$@")
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 .venv/bin/python -u sim/scripts/run_contactworld_tactile_action_branches.py --stage prepare --output "$output_dir" "${extra[@]}"
  bash sim/scripts/run_contactworld_sm120.sh sim/scripts/run_contactworld_tactile_action_branches.py --stage branch --output "$output_dir" "${extra[@]}"
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 .venv/bin/python -u sim/scripts/run_contactworld_tactile_action_branches.py --stage analyze --output "$output_dir" "${extra[@]}"
}
if [[ "${1:-}" == "--foreground" ]]; then
  shift
  exec 9>"$output_dir/.run.lock"
  flock -n 9 || { echo "Experiment already running: $output_dir" >&2; exit 1; }
  run_all "$@"
elif [[ " ${*} " == *" --smoke "* ]]; then
  exec 9>"$output_dir/.run.lock"
  flock -n 9 || { echo "Experiment already running: $output_dir" >&2; exit 1; }
  run_all "$@"
else
  nohup bash "$0" --foreground "$@" >> "$output_dir/launch.log" 2>&1 < /dev/null &
  printf 'Started PID %s\nLog: %s\nReport: %s\nFollow: tail -f %q\n' "$!" "$output_dir/launch.log" "$output_dir/index.html" "$output_dir/launch.log"
fi
