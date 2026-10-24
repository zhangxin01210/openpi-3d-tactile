#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
exec bash "$REPO/tools/hoi_level1_snapshot/scripts/level1_batch_preview_all.sh" "$@"
