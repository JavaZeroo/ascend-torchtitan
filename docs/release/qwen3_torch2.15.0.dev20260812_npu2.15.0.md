# release 检查 · qwen3 · torch2.15.0.dev20260812_npu2.15.0

生成于 2026-08-31 00:09。判据见 `docs/model-release-criteria.md`。

| 检查 | 判据 | 结果 | 证据 | 秒 |
|---|:--:|:--:|---|--:|
| `real-size 1 NPU` | R1 | 🟢 | step 1 loss 12.14616 -> step 20 loss 7.73866 | 62 |
| `FSDP2 x8` | R2 | 🟢 | step 1 loss 12.13871 -> step 20 loss 7.72968 | 69 |
| `FSDP2 x4 + TP2` | R2 | 🟢 | step 1 loss 12.14696 -> step 20 loss 7.70340 | 89 |
| `PP2 + FSDP2 x4` | R2 | 🔴 | torch.OutOfMemoryError: NPU out of memory. Tried to allocate 1.75 GiB (NPU 4; 60.96 GiB total capacity; 53.34 GiB already allocated; 53.34 GiB current active; 811.62 MiB free; 57.66 GiB reserved in to | 110 |
| `checkpoint save/resume` | R4 | 🟢 | uninterrupted step 10 loss 9.42568; resumed from step 5 -> step 10 loss 9.42568 | 152 |

复现命令：

```bash
ASCEND_RT_VISIBLE_DEVICES=... NPU=1 MODULE=ascend_titan.models.qwen3 CONFIG=qwen3_0_6b_npu ./scripts/run_train.sh --training.steps 20 --lr_scheduler.total_steps 1000
ASCEND_RT_VISIBLE_DEVICES=... NPU=8 MODULE=ascend_titan.models.qwen3 CONFIG=qwen3_0_6b_npu_fsdp2 ./scripts/run_train.sh --training.steps 20 --lr_scheduler.total_steps 1000
ASCEND_RT_VISIBLE_DEVICES=... NPU=8 MODULE=ascend_titan.models.qwen3 CONFIG=qwen3_0_6b_npu_tp2 ./scripts/run_train.sh --training.steps 20 --lr_scheduler.total_steps 1000
ASCEND_RT_VISIBLE_DEVICES=... NPU=8 MODULE=ascend_titan.models.qwen3 CONFIG=qwen3_14b_npu_pp2 ./scripts/run_train.sh --training.steps 20 --lr_scheduler.total_steps 1000
ASCEND_RT_VISIBLE_DEVICES=... NPU=1 MODULE=ascend_titan.models.qwen3 CONFIG=qwen3_0_6b_npu ./scripts/run_train.sh --checkpoint.enable --checkpoint.interval 5 --training.steps {N} --lr_scheduler.total_steps 10 --debug.seed 42 --debug.deterministic
```