#!/usr/bin/env bash
# Run a recipe deterministically and diff its loss/grad_norm curve against the frozen NPU golden.
#   NPU=1 ASCEND_RT_VISIBLE_DEVICES=0 ./scripts/check_golden.sh qwen3_debugmodel_npu
#   GOLDEN=qwen3_debugmodel_npu_fsdp2 NPU=2 ASCEND_RT_VISIBLE_DEVICES=0,1 \
#       ./scripts/check_golden.sh qwen3_debugmodel_npu --parallelism.data_parallel_shard_degree 2
# Golden files: tests/assets/losses/npu/<GOLDEN|config>__<tuple>.txt ; the tuple is taken from the installed torch/torch_npu.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
CONFIG=${1:?config name}; shift
# 并行度这类组合不再有专属 recipe（命令行可调），所以曲线名可以与 config 名分开：
#   GOLDEN=qwen3_debugmodel_npu_fsdp2 NPU=2 ./scripts/check_golden.sh qwen3_debugmodel_npu \
#       --parallelism.data_parallel_shard_degree 2
GOLDEN_NAME=${GOLDEN:-$CONFIG}
MODULE=${MODULE:-ascend_titan.models.qwen3}
TUPLE=$(python -c "import importlib.metadata as m; print(f\"torch{m.version('torch').split('+')[0]}_npu{m.version('torch_npu').split('+')[0]}\")")
GOLDEN="$HERE/tests/assets/losses/npu/${GOLDEN_NAME}__${TUPLE}.txt"
OUT=$(mktemp)
MODULE=$MODULE CONFIG=$CONFIG "$HERE/scripts/run_train.sh" --debug.seed 42 --debug.deterministic "$@" 2>&1 \
  | grep '^\[rank0\]' | sed -E 's/\x1b\[[0-9;]*m//g' | grep -oE 'step: +[0-9]+ +loss: +[0-9.]+ +grad_norm: +[0-9.]+' > "$OUT" || true
if [ ! -s "$OUT" ]; then echo "no loss lines captured"; exit 2; fi
if [ ! -f "$GOLDEN" ]; then
  echo "no golden for $CONFIG @ $TUPLE; writing $GOLDEN (review and commit it)"; cp "$OUT" "$GOLDEN"; exit 0
fi
if diff <(tr -s ' ' < "$GOLDEN") <(tr -s ' ' < "$OUT"); then echo "GOLDEN MATCH: $GOLDEN_NAME @ $TUPLE"; else echo "GOLDEN MISMATCH: $GOLDEN_NAME @ $TUPLE"; exit 1; fi
