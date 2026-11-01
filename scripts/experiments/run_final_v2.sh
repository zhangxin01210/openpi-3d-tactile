#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

configs=(
  pi0_xhand_1028_press_button_rgb_v2 pi05_xhand_1028_press_button_rgb_v2
  pi0_xhand_1028_press_button_st_v2 pi0_xhand_1028_press_button_full_v2
  pi0_xhand_1030_yellow_socket_rgb_v2 pi05_xhand_1030_yellow_socket_rgb_v2
  pi0_xhand_1030_yellow_socket_st_v2 pi0_xhand_1030_yellow_socket_full_v2
  pi0_xhand_1018_peg_all_st_v2 pi0_xhand_1018_peg_all_full_v2
)
selection=${1:-all}
resume=${2:-}
if [[ -n "$resume" && "$resume" != "--resume" ]]; then
  echo "Usage: $0 [all|CONFIG_NAME] [--resume]" >&2; exit 2
fi
if [[ "$selection" == all ]]; then selected=("${configs[@]}"); else
  selected=()
  for item in "${configs[@]}"; do [[ "$item" == "$selection" ]] && selected+=("$item"); done
  if [[ ${#selected[@]} -ne 1 ]]; then echo "Unknown config: $selection" >&2; exit 2; fi
fi
python_bin=${OPENPI_PYTHON:-.venv/bin/python}
exp_name=${OPENPI_FINAL_V2_EXP_NAME:-final_v2}
mkdir -p logs/final_v2
for config in "${selected[@]}"; do
  log="logs/final_v2/${config}_$(date +%Y%m%d_%H%M%S).log"
  {
    printf 'git_commit: '; git rev-parse HEAD
    printf 'git_status:\n'; git status --short
    printf 'config: %s\nexperiment: %s\n' "$config" "$exp_name"
    PYTHONPATH=src "$python_bin" -c 'import sys; from openpi.training.config import get_config; c=get_config(sys.argv[1]); print("dataset:",c.data.repo_id,"spatial:",c.data.spatial,"object:",c.data.object_geometry,"base:",c.weight_loader)' "$config"
    PYTHONPATH=src "$python_bin" scripts/experiments/preflight_final_v2.py --config "$config"
    cmd=("$python_bin" scripts/train.py "$config" --exp-name "$exp_name")
    [[ "$resume" == "--resume" ]] && cmd+=(--resume)
    PYTHONPATH=src "${cmd[@]}"
  } 2>&1 | tee -a "$log"
done
