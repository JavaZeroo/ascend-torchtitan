#!/usr/bin/env bash
# Ascend counterpart of torchtitan/run_train.sh.
#   NPU=8 MODULE=ascend_titan.models.qwen3 CONFIG=qwen3_debugmodel_npu ./scripts/run_train.sh [extra tyro args]
#   COMM_MODE=fake_backend NPU=8 ...   -> single device, fake process groups (no HCCL)
set -euo pipefail
NPU=${NPU:-8}
export LOG_RANK=${LOG_RANK:-0}
MODULE=${MODULE:-"ascend_titan.models.qwen3"}
CONFIG=${CONFIG:-"qwen3_debugmodel_npu"}
COMM_MODE=${COMM_MODE:-""}
# debugmodel recipes reference upstream test assets by relative path
# (./tests/assets/tokenizer, ./tests/assets/c4_test), so run from the pinned checkout.
TITAN_DIR=${TITAN_DIR:-"$(cd "$(dirname "$0")/.." && pwd)/../torchtitan"}
cd "$TITAN_DIR"

# Which interpreter runs the training. `torchrun` is just
# `python -m torch.distributed.run`, so naming PYTHON once pins the launcher and
# the ranks to the same environment. Left to PATH they can differ: a caller
# outside the venv gets the system python, which has no Triton-Ascend, and every
# run dies with "0 active drivers" (measured 2026-09-02 -- a whole matrix sweep
# came back red on it). The matrix runner sets PYTHON=sys.executable.
PYTHON=${PYTHON:-python}

if [ "$COMM_MODE" = "fake_backend" ]; then
  NGPU="$NPU" LOCAL_RANK=0 "$PYTHON" -m ascend_titan.train \
    --module "$MODULE" --config "$CONFIG" --comm.mode=fake_backend --training.steps 1 "$@"
else
  "$PYTHON" -m torch.distributed.run \
    --nproc_per_node="$NPU" --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter "$LOG_RANK" --role rank --tee 3 \
    -m ascend_titan.train --module "$MODULE" --config "$CONFIG" "$@"
fi
