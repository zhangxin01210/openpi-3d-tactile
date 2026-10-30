#!/usr/bin/env bash
# Non-destructive inventory for comparing deployment / cloud / laptop checkouts.
# Usage: bash scripts/dev/snapshot_repo_for_unification.sh deploy
# Optional: INCLUDE_BUNDLE=1 to preserve all locally referenced commits offline.
set -euo pipefail
ROLE="${1:-}"
[[ "$ROLE" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "Usage: $0 deploy|cloud|local" >&2; exit 2; }
ROOT="$(git rev-parse --show-toplevel)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${SYNC_OUT_DIR:-$ROOT/output/code_sync/${ROLE}_${STAMP}}"
mkdir -p "$OUT"
cd "$ROOT"
{
  printf 'role=%s\n' "$ROLE"
  printf 'timestamp_utc=%s\n' "$STAMP"
  printf 'head=%s\n' "$(git rev-parse HEAD)"
  printf 'branch=%s\n' "$(git branch --show-current)"
  printf 'git_version=%s\n' "$(git --version)"
  printf 'remote_names:\n'
  git remote
} > "$OUT/identity.txt"
git status --porcelain=v1 --untracked-files=normal > "$OUT/status_porcelain.txt"
git branch -avv --no-color > "$OUT/branches.txt"
git log --all -25 --format='%H %ad %s' --date=iso-strict > "$OUT/recent_commits.txt"
git diff --binary > "$OUT/working_tree.patch"
git diff --cached --binary > "$OUT/staged.patch"
git ls-files --others --exclude-standard > "$OUT/untracked_filenames.txt"
git submodule status --recursive > "$OUT/submodules.txt" 2> "$OUT/submodules_stderr.txt" || true
if [[ "${INCLUDE_BUNDLE:-0}" == 1 ]]; then
  git bundle create "$OUT/repository_refs.bundle" --all
fi
printf '[SYNC INVENTORY] %s\n' "$OUT"
printf 'Share identity.txt, status_porcelain.txt, branches.txt, recent_commits.txt and submodules.txt first.\n'
printf 'Keep patches/bundles local until reviewed: patches can contain credentials or large binary diffs.\n'
printf 'Untracked files are LISTED, not copied; they must be preserved separately before cleanup.\n'
printf 'Nothing was staged, committed, checked out, reset, pulled, or pushed.\n'
