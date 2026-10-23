#!/usr/bin/env bash
# No implicit approval; stages are deliberately separated.
set -euo pipefail
HOI=/workspace/mnt/sqzhang26/hoi
REPO=/workspace/mnt/sqzhang26/openpi-3d-tactile
SCRIPTS="$HOI/scripts"
usage() { echo 'Usage: bash level1_ep.sh EPISODE prepare|seed|annotate|preview|approve|track|review-track|build|memory'; exit 2; }
[[ $# -eq 2 ]] || usage
ep="$1";step="$2"
[[ "$ep" =~ ^[0-9]+$ ]] || usage
case "$step" in
  prepare)
    cd "$REPO"
    PYTHONPATH=src "$REPO/.venv/bin/python" "$SCRIPTS/level1_sequence_multi.py" prepare --episode "$ep";;
  seed)
    cd "$REPO"
    PYTHONPATH=src "$REPO/.venv/bin/python" "$SCRIPTS/level1_template_init.py" seed --episode "$ep";;
  annotate)
    cd "$REPO"
    PYTHONPATH=src "$REPO/.venv/bin/python" "$SCRIPTS/level1_template_init.py" annotate --episode "$ep";;
  preview|approve|track|review-track)
    cd "$HOI/repos/sam2"
    "$HOI/envs/sam2/bin/python" "$SCRIPTS/level1_sam2_multi.py" "$step" --episode "$ep";;
  build)
    cd "$REPO"
    PYTHONPATH=src "$REPO/.venv/bin/python" "$SCRIPTS/level1_sequence_multi.py" build --episode "$ep" --stride 3;;
  memory)
    cd "$REPO"
    PYTHONPATH=src "$REPO/.venv/bin/python" "$SCRIPTS/level1_contact_memory_multi.py" --episode "$ep" --run-name abc_v1;;
  *) usage;;
esac
