# release 检查 · qwen3_5 · torch2.15.0.dev20260812_npu2.15.0

生成于 2026-08-31 11:59。判据见 `docs/model-release-criteria.md`。

| 检查 | 判据 | 结果 | 证据 | 秒 |
|---|:--:|:--:|---|--:|
| `real-size 1 NPU` | R1 | 🟢 | step 1 loss 12.88826 -> step 20 loss 8.14589 | 1345 |
| `FSDP2 x8` | R2 | 🟢 | step 1 loss 12.90316 -> step 20 loss 8.06005 | 1507 |
| `checkpoint save/resume` | R4 | 🔴 | RuntimeError: [rank0]:[rank0]: RuntimeError: Missing key in checkpoint state_dict: optimizer.state.vision_encoder.pos_embed.step. | 1131 |
| `HF export/import` | R4 | 🟢 | exported at step 5 loss 12.93624; reloaded loss 9.91788 (untrained was 12.93624) | 467 |

复现命令：

```bash
ASCEND_RT_VISIBLE_DEVICES=... NPU=1 MODULE=ascend_titan.models.qwen3_5 CONFIG=qwen35_0_8b_npu ./scripts/run_train.sh --training.steps 20 --lr_scheduler.total_steps 1000
ASCEND_RT_VISIBLE_DEVICES=... NPU=8 MODULE=ascend_titan.models.qwen3_5 CONFIG=qwen35_0_8b_npu_fsdp2 ./scripts/run_train.sh --training.steps 20 --lr_scheduler.total_steps 1000
ASCEND_RT_VISIBLE_DEVICES=... NPU=1 MODULE=ascend_titan.models.qwen3_5 CONFIG=qwen35_0_8b_npu ./scripts/run_train.sh --checkpoint.enable --checkpoint.interval 5 --training.steps {N} --lr_scheduler.total_steps 10 --debug.seed 42 --debug.deterministic
ASCEND_RT_VISIBLE_DEVICES=... NPU=1 MODULE=ascend_titan.models.qwen3_5 CONFIG=qwen35_0_8b_npu ./scripts/run_train.sh --checkpoint.enable --checkpoint.last_save_in_hf --training.steps 5 # then --checkpoint.initial_load_in_hf --checkpoint.initial_load_path <dir>
```