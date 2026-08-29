#!/usr/bin/env bash
# M0 tool: walk torchtitan history forward from the pinned SHA and report the
# furthest commit at which `import torchtitan.trainer` still succeeds on this
# torch, and the first commit that breaks it. Cheap (import only, no NPU).
#   TITAN_DIR=../torchtitan ./scripts/probe_compat.sh [upper-ref, default origin/main]
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
TITAN_DIR=${TITAN_DIR:-"$HERE/../torchtitan"}
UPPER=${1:-origin/main}
PIN=$(grep -E '^torchtitan_sha=' "$HERE/constraints/npu.txt" | cut -d= -f2)

git -C "$TITAN_DIR" fetch -q origin
COMMITS=$(git -C "$TITAN_DIR" rev-list --reverse "$PIN..$UPPER")
N=$(echo "$COMMITS" | grep -c . || true)
echo "probing $N commits after $PIN up to $UPPER"

good=$PIN; bad=""
for c in $COMMITS; do
  git -C "$TITAN_DIR" checkout -q "$c"
  if python -c "import torchtitan.trainer" >/dev/null 2>&1; then
    good=$c
  else
    bad=$c
    echo "FIRST BREAKING COMMIT: $(git -C "$TITAN_DIR" log -1 --format='%h %ad %s' --date=short "$c")"
    python -c "import torchtitan.trainer" 2>&1 | tail -3
    break
  fi
done
git -C "$TITAN_DIR" checkout -q "$PIN"
echo "furthest importable: $(git -C "$TITAN_DIR" log -1 --format='%h %ad %s' --date=short "$good")"
[ -z "$bad" ] && echo "all $N commits import cleanly; safe to bump to $UPPER (run the matrix first)."
