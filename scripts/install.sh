#!/usr/bin/env bash
# Install torchtitan at the pinned SHA without CUDA-only extras (design F2/F3).
#   TITAN_DIR=../torchtitan ./scripts/install.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
TITAN_DIR=${TITAN_DIR:-"$HERE/../torchtitan"}
SHA=$(grep -E '^torchtitan_sha=' "$HERE/constraints/npu.txt" | cut -d= -f2)

if [ ! -d "$TITAN_DIR/.git" ]; then
  git clone https://github.com/pytorch/torchtitan.git "$TITAN_DIR"
fi
git -C "$TITAN_DIR" fetch -q origin
git -C "$TITAN_DIR" checkout -q "$SHA"
echo "torchtitan @ $(git -C "$TITAN_DIR" rev-parse --short HEAD)"

# --no-deps: requirements.txt pins attn-gym[linear], whose [linear] extra pulls
# nvidia-cutlass-dsl[cu13]. We install the dependency list ourselves, minus extras.
pip install --no-deps -e "$TITAN_DIR"
pip install -c "$HERE/constraints/npu.txt" -r "$HERE/constraints/titan-deps.txt"
pip install -e "$HERE[dev]"
ascend-titan-doctor
