# pytorch —— 问题（归属：pytorch/pytorch）

## TORCH-1 FlexAttention 硬编码设备白名单 —— `draft`
- `torch/nn/attention/flex_attention.py`（2.12："only supported on CUDA, CPU or HPU"；2.13 与 nightly 2.15.0.dev20260828："... HPU, or MPS"）：任何 privateuse1 设备在任何后端有机会之前就抛 `ValueError`。
- 影响：torchtitan 默认的 LM 注意力（`attn_backend="flex"`）无论 torch_npu 支持与否都永远无法在 NPU 上运行。叠加在 Flex 上的上游特性（`create_block_mask` 文档掩码、attention sinks）一并被挡。
- 诉求：允许其后端注册了 flex 实现的 privateuse1 设备通过（或下沉到 math 路径），而不是按设备类型直接抛错。

## TORCH-2 `fake` 进程组后端无法被树外后端扩展 —— `question`
- `Backend.backend_capability["fake"]` 是固定列表。若 torch_npu 无法干净地追加（见 torch_npu NPU-2），这里需要一个注册 API。

## TORCH-6 FSDP2 与 spmd_types 的集成仅在 nightly —— `info`
- nightly `_fsdp_param.py` 读取 spmd_types 注解构建 DTensor；2.12/2.13 要求调用方自己给 DTensor。下游 torchtitan 默认 `spmd_types` 后端因此在正式版 torch 上跑不了 CP。

## TORCH-7 `torch.library.opcheck` 的 autograd 注册检查只支持 CPU/CUDA/XPU —— `draft`
- `torch/testing/_internal/optests/autograd_registration.py:89`：`NotImplementedError: autograd_registration_check: NYI devices other than CPU/CUDA/XPU, got {'npu'}`。privateuse1 设备的自定义算子无法用 opcheck 完整校验；本仓的 NPU 测试跳过 `test_autograd_registration`，反向用数值对比覆盖。
- 诉求：把 privateuse1 加入允许的设备集合（检查本身与设备无关）。

## TORCH-8 `varlen_attn` 用 uint64 创建 rng_state 占位张量 —— `本地补丁`
- `torch/nn/attention/varlen.py`：`rng_state_ = torch.zeros((2,), dtype=torch.uint64, ...)`；该值只是占位（dropout 固定为 0），uint64 在 NPU 上没有 zero_ 内核（NPU-6）。补丁：`patches/pytorch/0002-TORCH-8-varlen-rng_state-int64.patch`。
