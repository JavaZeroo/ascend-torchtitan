# release 检查 · qwen3 · torch2.15.0.dev20260812_npu2.15.0

生成于 2026-08-31 01:20。判据见 `docs/model-release-criteria.md`。

| 检查 | 判据 | 结果 | 证据 | 秒 |
|---|:--:|:--:|---|--:|
| `real-size 1 NPU` | R1 | 🟢 | step 1 loss 12.11337 -> step 20 loss 7.75390 | 64 |
| `FSDP2 x8` | R2 | 🟢 | step 1 loss 12.12893 -> step 20 loss 7.72891 | 70 |
| `FSDP2 x4 + TP2` | R2 | 🟢 | step 1 loss 12.12286 -> step 20 loss 7.71191 | 77 |
| `PP2 + FSDP2 x4` | R2 | 🟢 | step 1 loss 12.77976 -> step 20 loss 9.45129 | 161 |
| `checkpoint save/resume` | R4 | 🟢 | uninterrupted step 10 loss 9.42568; resumed from step 5 -> step 10 loss 9.42568 | 152 |
| `HF export/import` | R4 | 🟢 | exported at step 5 loss 10.23167; reloaded loss 9.95988 (untrained was 12.14750) | 84 |

复现命令：

```bash
ASCEND_RT_VISIBLE_DEVICES=... NPU=1 MODULE=ascend_titan.models.qwen3 CONFIG=qwen3_0_6b_npu ./scripts/run_train.sh --training.steps 20 --lr_scheduler.total_steps 1000
ASCEND_RT_VISIBLE_DEVICES=... NPU=8 MODULE=ascend_titan.models.qwen3 CONFIG=qwen3_0_6b_npu_fsdp2 ./scripts/run_train.sh --training.steps 20 --lr_scheduler.total_steps 1000
ASCEND_RT_VISIBLE_DEVICES=... NPU=8 MODULE=ascend_titan.models.qwen3 CONFIG=qwen3_0_6b_npu_tp2 ./scripts/run_train.sh --training.steps 20 --lr_scheduler.total_steps 1000
ASCEND_RT_VISIBLE_DEVICES=... NPU=8 MODULE=ascend_titan.models.qwen3 CONFIG=qwen3_8b_npu_pp2 ./scripts/run_train.sh --training.steps 20 --lr_scheduler.total_steps 1000
ASCEND_RT_VISIBLE_DEVICES=... NPU=1 MODULE=ascend_titan.models.qwen3 CONFIG=qwen3_0_6b_npu ./scripts/run_train.sh --checkpoint.enable --checkpoint.interval 5 --training.steps {N} --lr_scheduler.total_steps 10 --debug.seed 42 --debug.deterministic
ASCEND_RT_VISIBLE_DEVICES=... NPU=1 MODULE=ascend_titan.models.qwen3 CONFIG=qwen3_0_6b_npu ./scripts/run_train.sh --checkpoint.enable --checkpoint.last_save_in_hf --training.steps 5 # then --checkpoint.initial_load_in_hf --checkpoint.initial_load_path <dir>
```