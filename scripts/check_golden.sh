#!/usr/bin/env bash
# Run a recipe deterministically and diff its loss/grad_norm curve against the frozen NPU golden.
#   NPU=1 ASCEND_RT_VISIBLE_DEVICES=0 ./scripts/check_golden.sh qwen3_debugmodel_npu
#   NPU=2 ASCEND_RT_VISIBLE_DEVICES=0,1 ./scripts/check_golden.sh qwen3_debugmodel_npu_fsdp2
# Golden files: tests/assets/losses/npu/<config>__<tuple>.txt ; the tuple is taken from the installed torch/torch_npu.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
CONFIG=${1:?config name}
MODULE=${MODULE:-ascend_titan.models.qwen3}
TUPLE=$(python -c "import importlib.metadata as m; print(f\"torch{m.version('torch').split('+')[0]}_npu{m.version('torch_npu').split('+')[0]}\")")
GOLDEN="$HERE/tests/assets/losses/npu/${CONFIG}__${TUPLE}.txt"
OUT=$(mktemp)
MODULE=$MODULE CONFIG=$CONFIG "$HERE/scripts/run_train.sh" --debug.seed 42 --debug.deterministic 2>&1 \
  | grep '^\[rank0\]' | sed -E 's/\x1b\[[0-9;]*m//g' | grep -oE 'step: +[0-9]+ +loss: +[0-9.]+ +grad_norm: +[0-9.]+' > "$OUT" || true
if [ ! -s "$OUT" ]; then echo "no loss lines captured"; exit 2; fi
if [ ! -f "$GOLDEN" ]; then
  echo "no golden for $CONFIG @ $TUPLE; writing $GOLDEN (review and commit it)"; cp "$OUT" "$GOLDEN"; exit 0
fi
if diff <(tr -s ' ' < "$GOLDEN") <(tr -s ' ' < "$OUT"); then echo "GOLDEN MATCH: $CONFIG @ $TUPLE"; else echo "GOLDEN MISMATCH: $CONFIG @ $TUPLE"; exit 1; fi
