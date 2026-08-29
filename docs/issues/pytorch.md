# pytorch —— 问题（归属：pytorch/pytorch）

## TORCH-1 FlexAttention 硬编码设备白名单 —— `draft`
- `torch/nn/attention/flex_attention.py`（2.12："only supported on CUDA, CPU or HPU"；2.13 与 nightly 2.15.0.dev20260828："... HPU, or MPS"）：任何 privateuse1 设备在任何后端有机会之前就抛 `ValueError`。
- 影响：torchtitan 默认的 LM 注意力（`attn_backend="flex"`）无论 torch_npu 支持与否都永远无法在 NPU 上运行。叠加在 Flex 上的上游特性（`create_block_mask` 文档掩码、attention sinks）一并被挡。
- 诉求：允许其后端注册了 flex 实现的 privateuse1 设备通过（或下沉到 math 路径），而不是按设备类型直接抛错。

## TORCH-2 `fake` 进程组后端无法被树外后端扩展 —— `question`
- `Backend.backend_capability["fake"]` 是固定列表。若 torch_npu 无法干净地追加（见 torch_npu NPU-2），这里需要一个注册 API。

## TORCH-3 `torch.distributed.set_timeout` 公开 API 仅在 nightly —— `info`
- 约 2026 年中作为 `_set_pg_timeout` 的公开名加入。下游（torchtitan）立刻采用；正式版 torch ≤ 2.13 没有。已在 `ascend_titan/compat/shims/dist_set_timeout.py` polyfill（torch 自带后自动 no-op）。
- 注意：torch ≤ 2.13 上 `_set_pg_timeout` 只处理 nccl/gloo，并警告 `Set timeout is now only supported for either nccl or gloo`——HCCL 组的超时保持不变。nightly 的 `ProcessGroup.set_timeout` 会转发给每个后端，所以在 nightly torch + 实现了它的 torch_npu 上行为会与 CUDA 一致。

## TORCH-4 `PipelineSchedule.step(arg_mbs=...)` 预切分输入契约仅在 nightly —— `info`
- 正式版 ≤ 2.13 的 `step` 没有 `arg_mbs/kwarg_mbs/target_mbs`，但 `_step_microbatches` 参数完全相同。已在 `compat/shims/pp_step_presplit.py` 用 wrap 转发；torch 2.12 连 `loss_kwargs` 也没有，shim 在该步期间把它们绑定到 loss 函数上。
