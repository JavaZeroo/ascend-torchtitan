#!/usr/bin/env bash
# Install torchtitan at the pinned SHA without CUDA-only extras (design F2/F3).
#   TITAN_DIR=../torchtitan ./scripts/install.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
TITAN_DIR=${TITAN_DIR:-"$HERE/../torchtitan"}
SHA=$(tr -d '[:space:]' < "$HERE/constraints/torchtitan.sha")
CONSTRAINTS=${CONSTRAINTS:-"$HERE/constraints/nightly.txt"}
WITH_TORCH=${WITH_TORCH:-0}   # 1 = also install torch/torch_npu pinned by $CONSTRAINTS

if [ ! -d "$TITAN_DIR/.git" ]; then
  git clone https://github.com/pytorch/torchtitan.git "$TITAN_DIR"
fi
git -C "$TITAN_DIR" fetch -q origin
git -C "$TITAN_DIR" checkout -q "$SHA"
echo "torchtitan @ $(git -C "$TITAN_DIR" rev-parse --short HEAD)"

# --no-deps: requirements.txt pins attn-gym[linear], whose [linear] extra pulls
# nvidia-cutlass-dsl[cu13]. We install the dependency list ourselves, minus extras.
# NIGHTLY track (constraints/nightly.txt): torch is a dated nightly (".dev" in the pin) from the
# nightly index, torch_npu is a locally built wheel (scripts/build_torch_npu.sh, TORCH_NPU_WHEEL=).
NIGHTLY=0; grep -qE '^torch==.*\.dev' "$CONSTRAINTS" && NIGHTLY=1
if [ "$WITH_TORCH" = 1 ]; then
  # torch: CPU wheel (torch_npu supplies the device backend). torch_npu may be a pre-release.
  if [ "$NIGHTLY" = 1 ]; then
    pip install --pre -c "$CONSTRAINTS" torch --index-url https://download.pytorch.org/whl/nightly/cpu
  else
    pip install -c "$CONSTRAINTS" torch --index-url https://download.pytorch.org/whl/cpu
  fi
  if [ -n "${TORCH_NPU_WHEEL:-}" ]; then
    pip install --no-deps --force-reinstall "$TORCH_NPU_WHEEL"
  elif [ "$NIGHTLY" = 1 ]; then
    echo "NIGHTLY track: set TORCH_NPU_WHEEL=<wheel from scripts/build_torch_npu.sh> (PyPI has no torch_npu for torch nightly)"; exit 1
  else
    pip install --pre -c "$CONSTRAINTS" torch_npu
  fi
fi
pip install --no-deps -e "$TITAN_DIR"
if [ "$NIGHTLY" = 1 ]; then
  # stable torchvision would drag in stable torch; take the nightly CPU build without deps.
  grep -v '^torchvision' "$HERE/constraints/titan-deps.txt" > "$HERE/constraints/.titan-deps.notv"
  pip install -c "$CONSTRAINTS" -r "$HERE/constraints/.titan-deps.notv"
  pip install --no-deps --pre torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
  rm -f "$HERE/constraints/.titan-deps.notv"
else
  pip install -c "$CONSTRAINTS" -r "$HERE/constraints/titan-deps.txt"
fi
pip install -e "$HERE[dev]"
ascend-titan-doctor
