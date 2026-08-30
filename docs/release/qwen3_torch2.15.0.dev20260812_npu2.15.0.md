# release 检查 · qwen3 · torch2.15.0.dev20260812_npu2.15.0

生成于 2026-08-30 22:54。判据见 `docs/model-release-criteria.md`。

| 检查 | 判据 | 结果 | 证据 | 秒 |
|---|:--:|:--:|---|--:|
| `real-size 1 NPU` | R1 | 🟢 | step 1 loss 12.16065 -> step 20 loss 8.20184 | 74 |
| `FSDP2 x8` | R2 | 🟢 | step 1 loss 12.13415 -> step 20 loss 8.11547 | 74 |
| `FSDP2 x4 + TP2` | R2 | 🟢 | step 1 loss 12.13927 -> step 20 loss 8.12235 | 73 |
| `PP2 + FSDP2 x4` | R2 | 🔴 | torch.distributed.elastic.multiprocessing.errors.ChildFailedError:  | 19 |

复现命令：

```bash
ASCEND_RT_VISIBLE_DEVICES=... NPU=1 MODULE=ascend_titan.models.qwen3 CONFIG=qwen3_0_6b_npu ./scripts/run_train.sh --training.steps 20
ASCEND_RT_VISIBLE_DEVICES=... NPU=8 MODULE=ascend_titan.models.qwen3 CONFIG=qwen3_0_6b_npu_fsdp2 ./scripts/run_train.sh --training.steps 20
ASCEND_RT_VISIBLE_DEVICES=... NPU=8 MODULE=ascend_titan.models.qwen3 CONFIG=qwen3_0_6b_npu_tp2 ./scripts/run_train.sh --training.steps 20
ASCEND_RT_VISIBLE_DEVICES=... NPU=8 MODULE=ascend_titan.models.qwen3 CONFIG=qwen3_0_6b_npu_pp2 ./scripts/run_train.sh --training.steps 20
```