#!/usr/bin/env bash
# Install torchtitan at the pinned SHA without CUDA-only extras (design F2/F3).
#   TITAN_DIR=../torchtitan ./scripts/install.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
TITAN_DIR=${TITAN_DIR:-"$HERE/../torchtitan"}
SHA=$(tr -d '[:space:]' < "$HERE/constraints/torchtitan.sha")
CONSTRAINTS=${CONSTRAINTS:-"$HERE/constraints/npu.txt"}
WITH_TORCH=${WITH_TORCH:-0}   # 1 = also install torch/torch_npu pinned by $CONSTRAINTS

if [ ! -d "$TITAN_DIR/.git" ]; then
  git clone https://github.com/pytorch/torchtitan.git "$TITAN_DIR"
fi
git -C "$TITAN_DIR" fetch -q origin
git -C "$TITAN_DIR" checkout -q "$SHA"
echo "torchtitan @ $(git -C "$TITAN_DIR" rev-parse --short HEAD)"

# --no-deps: requirements.txt pins attn-gym[linear], whose [linear] extra pulls
# nvidia-cutlass-dsl[cu13]. We install the dependency list ourselves, minus extras.
if [ "$WITH_TORCH" = 1 ]; then
  # torch: CPU wheel (torch_npu supplies the device backend). torch_npu may be a pre-release.
  pip install -c "$CONSTRAINTS" torch --index-url https://download.pytorch.org/whl/cpu
  pip install --pre -c "$CONSTRAINTS" torch_npu
fi
pip install --no-deps -e "$TITAN_DIR"
pip install -c "$CONSTRAINTS" -r "$HERE/constraints/titan-deps.txt"
pip install -e "$HERE[dev]"
ascend-titan-doctor
