#!/usr/bin/env bash
# Ascend counterpart of torchtitan/run_train.sh.
#   NPU=8 MODULE=ascend_titan.recipes.qwen3 CONFIG=qwen3_debugmodel_npu ./scripts/run_train.sh [extra tyro args]
#   COMM_MODE=fake_backend NPU=8 ...   -> single device, fake process groups (no HCCL)
set -euo pipefail
NPU=${NPU:-8}
export LOG_RANK=${LOG_RANK:-0}
MODULE=${MODULE:-"ascend_titan.recipes.qwen3"}
CONFIG=${CONFIG:-"qwen3_debugmodel_npu"}
COMM_MODE=${COMM_MODE:-""}

if [ "$COMM_MODE" = "fake_backend" ]; then
  NGPU="$NPU" LOCAL_RANK=0 python -m ascend_titan.train \
    --module "$MODULE" --config "$CONFIG" --comm.mode=fake_backend --training.steps 1 "$@"
else
  torchrun --nproc_per_node="$NPU" --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter "$LOG_RANK" --role rank --tee 3 \
    -m ascend_titan.train --module "$MODULE" --config "$CONFIG" "$@"
fi
