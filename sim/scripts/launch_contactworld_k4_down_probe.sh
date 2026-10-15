#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"
output_dir="${CW_K4_DOWN_OUTPUT:-/home/sai/zx/openpi-sim-runtime/runs/contactworld_k4_down_probe_20261009}"
mkdir -p "$output_dir"
run_all() {
  local extra=("$@")
  local offline=(env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 .venv/bin/python -u sim/scripts/run_contactworld_k4_down_probe.py)
  if [[ -e "$output_dir/complete.json" ]]; then
    echo "Already complete: $output_dir/index.html"
    return 0
  fi
  "${offline[@]}" --stage prepare --output "$output_dir" "${extra[@]}"
  local planned_stage
  planned_stage="$("$repo_dir/.venv/bin/python" - "$output_dir" <<'PYTHON'
import json,sys
from pathlib import Path
plan=json.loads((Path(sys.argv[1])/'plan.json').read_text())
print('pilot' if len(plan['selected'])<len(plan['all_selected']) or plan.get('smoke') else 'full')
PYTHON
)"
  if [[ "$planned_stage" == "pilot" ]]; then
    bash sim/scripts/run_contactworld_sm120.sh sim/scripts/run_contactworld_k4_down_probe.py --stage branch --output "$output_dir" "${extra[@]}"
    "${offline[@]}" --stage analyze --output "$output_dir" "${extra[@]}"
    if [[ " ${extra[*]} " != *" --smoke "* ]]; then
      "${offline[@]}" --stage expand --output "$output_dir" "${extra[@]}"
      planned_stage=full
    fi
  fi
  if [[ "$planned_stage" == "full" ]]; then
    bash sim/scripts/run_contactworld_sm120.sh sim/scripts/run_contactworld_k4_down_probe.py --stage branch --output "$output_dir" "${extra[@]}"
    "${offline[@]}" --stage analyze --output "$output_dir" "${extra[@]}"
  fi
  "$repo_dir/.venv/bin/python" - "$output_dir" <<'PYTHON'
import json,sys
from pathlib import Path
out=Path(sys.argv[1]);report=json.loads((out/'analysis.json').read_text())
(out/'complete.json').write_text(json.dumps({'stage':report['stage'],'selected':report['selected'],'paired':report['paired']},ensure_ascii=False,indent=2)+'\n')
PYTHON
  echo "DONE: $output_dir/index.html"
}
if [[ "${1:-}" == "--foreground" ]]; then
  shift
  exec 9>"$output_dir/.run.lock"
  flock -n 9 || { echo "Already running: $output_dir" >&2; exit 1; }
  run_all "$@"
elif [[ " ${*} " == *" --smoke "* ]]; then
  exec 9>"$output_dir/.run.lock"
  flock -n 9 || { echo "Already running: $output_dir" >&2; exit 1; }
  run_all "$@"
else
  nohup bash "$0" --foreground "$@" >> "$output_dir/launch.log" 2>&1 < /dev/null &
  printf 'Started PID %s\nLog: %s\nReport: %s\nFollow: tail -f %q\n' "$!" "$output_dir/launch.log" "$output_dir/index.html" "$output_dir/launch.log"
fi
