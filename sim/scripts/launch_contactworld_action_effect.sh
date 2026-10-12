#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"
output_dir="${CW_ACTION_EFFECT_OUTPUT:-/home/sai/zx/openpi-sim-runtime/runs/contactworld_action_effect_20261009}"
mkdir -p "$output_dir"
run_all() {
  local extra=("$@")
  local offline=(env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1
    "$repo_dir/.venv/bin/python" -u sim/scripts/run_contactworld_action_effect.py)
  if [[ -e "$output_dir/complete.json" ]]; then
    echo "Already finished: $output_dir/complete.json"
    return 0
  fi
  echo "PREPARE: old 520-rollout audit and eight-state pilot plan"
  "${offline[@]}" --stage prepare --output "$output_dir" "${extra[@]}"
  local planned_stage
  planned_stage="$("$repo_dir/.venv/bin/python" - "$output_dir" <<'PYTHON'
import json,sys
from pathlib import Path
p=json.loads((Path(sys.argv[1])/'plan.json').read_text())
print('pilot' if len(p['selected'])<len(p['all_selected']) or p['smoke'] else 'full')
PYTHON
)"
  if [[ "$planned_stage" == "pilot" ]]; then
    echo "PILOT: nine actions, repeated from eight source states"
    bash sim/scripts/run_contactworld_sm120.sh sim/scripts/run_contactworld_action_effect.py \
      --stage branch --output "$output_dir" "${extra[@]}"
    "${offline[@]}" --stage analyze --output "$output_dir" "${extra[@]}"
    if [[ " ${extra[*]} " == *" --smoke "* ]]; then
      "$repo_dir/.venv/bin/python" - "$output_dir" <<'PYTHON'
import json,sys
from pathlib import Path
o=Path(sys.argv[1]);a=json.loads((o/'analysis.json').read_text())
(o/'complete.json').write_text(json.dumps({'status':'smoke_completed','selected':a['selected'],
'usable':a['usable']},indent=2)+'\n')
PYTHON
      echo "SMOKE_DONE: $output_dir/index.html"
      return 0
    fi
    echo "PILOT GATE: require three comparable states across mouth and inside"
    "${offline[@]}" --stage expand --output "$output_dir" "${extra[@]}"
    if ! "$repo_dir/.venv/bin/python" - "$output_dir" <<'PYTHON'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])/'pilot_gate.json'
sys.exit(0 if json.loads(p.read_text())['pass_gate'] else 1)
PYTHON
    then
      "$repo_dir/.venv/bin/python" - "$output_dir" <<'PYTHON'
import json,sys
from pathlib import Path
o=Path(sys.argv[1]);g=json.loads((o/'pilot_gate.json').read_text())
(o/'complete.json').write_text(json.dumps({'status':'pilot_gate_failed',
'gate':g,'report':str(o/'index.html')},indent=2)+'\n')
PYTHON
      echo "PILOT_GATE_FAILED: inspect $output_dir/index.html and pilot_gate.json"
      return 0
    fi
  fi
  echo "FULL: expand to thirty distinct source states, resuming pilot branches"
  bash sim/scripts/run_contactworld_sm120.sh sim/scripts/run_contactworld_action_effect.py \
    --stage branch --output "$output_dir" "${extra[@]}"
  "${offline[@]}" --stage analyze --output "$output_dir" "${extra[@]}"
  echo "PROBE: grouped light predictors and paired modality comparison"
  "${offline[@]}" --stage probe --output "$output_dir" "${extra[@]}"
  "$repo_dir/.venv/bin/python" - "$output_dir" <<'PYTHON'
import json,sys
from pathlib import Path
o=Path(sys.argv[1]);a=json.loads((o/'analysis.json').read_text())
p=json.loads((o/'probe.json').read_text())
(o/'complete.json').write_text(json.dumps({'status':'full_completed','selected':a['selected'],
'usable':a['usable'],'probe_status':p['status'],'report':str(o/'index.html')},indent=2)+'\n')
PYTHON
  echo "DONE: $output_dir/index.html"
  echo "PREDICTION: $output_dir/probe.html"
}
if [[ "${1:-}" == "--foreground" ]]; then
  shift
  exec 9>"$output_dir/.run.lock"
  flock -n 9 || { echo "Already running: $output_dir" >&2; exit 1; }
  run_all "$@"
elif [[ " $* " == *" --smoke "* ]]; then
  exec 9>"$output_dir/.run.lock"
  flock -n 9 || { echo "Already running: $output_dir" >&2; exit 1; }
  run_all "$@"
else
  nohup bash "$0" --foreground "$@" >>"$output_dir/launch.log" 2>&1 </dev/null &
  printf 'Started PID %s\nLog: %s\nReport: %s\nFollow: tail -f %q\n' \
    "$!" "$output_dir/launch.log" "$output_dir/index.html" "$output_dir/launch.log"
fi
