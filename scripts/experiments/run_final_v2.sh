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
if [[ $# -gt 0 ]]; then shift; fi
resume=false
skip_preflight=false
for option in "$@"; do
  case "$option" in
    --resume) resume=true ;;
    --skip-preflight) skip_preflight=true ;;
    *) echo "Usage: $0 [all|CONFIG_NAME] [--resume] [--skip-preflight]" >&2; exit 2 ;;
  esac
done
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
    if [[ "$skip_preflight" == true ]]; then
      printf 'Preflight skipped; normal training DataLoader checks remain enabled.\n'
    else
      PYTHONPATH=src "$python_bin" scripts/experiments/preflight_final_v2.py --config "$config"
    fi
    cmd=("$python_bin" scripts/train.py "$config" --exp-name "$exp_name")
    [[ "$resume" == true ]] && cmd+=(--resume)
    PYTHONPATH=src "${cmd[@]}"
  } 2>&1 | tee -a "$log"
done
