#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
HOI="${HOI:-/workspace/mnt/sqzhang26/hoi}"
PY="${PY:-$REPO/.venv/bin/python}"
export REPO HOI PY
exec bash "$REPO/tools/hoi_level1_snapshot/scripts/level1_ep.sh" "$@"
