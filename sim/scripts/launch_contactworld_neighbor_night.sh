#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"
night_log="/home/sai/zx/openpi-sim-runtime/runs/contactworld_neighbor_night_20261003.launch.log"
if [[ " ${*} " == *" --check-only "* ]]; then
  exec .venv/bin/python -u sim/scripts/run_contactworld_neighbor_night.py --device cpu "$@"
fi
mkdir -p "$(dirname "$night_log")"
nohup .venv/bin/python -u sim/scripts/run_contactworld_neighbor_night.py --device cpu "$@" >> "$night_log" 2>&1 < /dev/null &
night_pid=$!
printf 'Started PID %s\nLog: %s\nFollow: tail -f %q\n' "$night_pid" "$night_log" "$night_log"
