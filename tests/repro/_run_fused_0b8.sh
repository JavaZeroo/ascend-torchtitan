#!/bin/bash
set -e
. /opt/venv-nightly/bin/activate
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=${ACV:-0}
export NPU=${NPU:-1}
export MODULE=${MODULE:-ascend_titan.models.qwen3_5}
export CONFIG=${CONFIG:-qwen35_0_8b_npu_fused}
cd /data/ljb/projects/create-ascend-titian/ascend-torchtitan
LOG=${LOGFILE:-/data/ljb/projects/create-ascend-titian/ascend-torchtitan/tests/repro/fused_run.log}
./scripts/run_train.sh --training.steps "${STEPS:-3}" --debug.seed 42 --debug.deterministic > "$LOG" 2>&1
echo DONE
grep -aoE 'step: +[0-9]+ +loss: +[0-9.]+' "$LOG"
exit 0
