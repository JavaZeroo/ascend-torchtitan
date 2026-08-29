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

## TORCH-5 torch 2.12 的 pipelining `fork_rng` 默认 cuda —— `fixed@2.13`
- `torch/distributed/pipelining/schedules.py:426`（2.12）对 stage 设备调用 `torch.random.fork_rng(devices=...)` 而未传 `device_type`，NPU 上抛 `AssertionError: Torch not compiled with CUDA enabled`。2.13 已改写。影响 STABLE track 上所有 PP 用例；NEXT track 不受影响。

## TORCH-6 FSDP2 与 spmd_types 的集成仅在 nightly —— `info`
- 见 torchtitan TT-5：nightly `_fsdp_param.py` 读取 spmd_types 注解构建 DTensor；2.12/2.13 要求调用方自己给 DTensor。下游 torchtitan 默认 `spmd_types` 后端因此在正式版 torch 上跑不了 CP。

## TORCH-7 `torch.library.opcheck` 的 autograd 注册检查只支持 CPU/CUDA/XPU —— `draft`
- `torch/testing/_internal/optests/autograd_registration.py:89`：`NotImplementedError: autograd_registration_check: NYI devices other than CPU/CUDA/XPU, got {'npu'}`。privateuse1 设备的自定义算子无法用 opcheck 完整校验；本仓的 NPU 测试跳过 `test_autograd_registration`，反向用数值对比覆盖。
- 诉求：把 privateuse1 加入允许的设备集合（检查本身与设备无关）。
