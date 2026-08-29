# torchtitan —— 问题 / 诉求（归属：pytorch/torchtitan）

在 torchtitan `13da2d77c`（2026-08-29）测得。"nightly-only" = 需要任何正式版 torch（≤ 2.13.0）都没有的特性。

## TT-1 core 导入链中无条件 `import triton` —— `draft`
- `torchtitan/models/common/token_dispatcher.py:17 → distributed/minimal_async_ep/api.py:30 → minimal_async_ep/kernels.py:10 import triton`。没有 `triton` wheel 的环境 `import torchtitan.trainer` 直接失败，尽管 MinimalAsyncEP 是可选特性。
- 诉求：在内核函数内部懒加载 triton（像 `overrides/fused_swiglu.py` 那样放在 try/except 后面），或用 `find_spec` 保护。本仓的处理：`constraints/titan-deps.txt` 加了 `triton`（纯 Python wheel；它*不是* Triton-Ascend）。

## TT-2 使用 `torch.distributed.set_timeout` 而没有特性检查 —— `draft` <a name="set-timeout"></a>
- `torchtitan/distributed/utils.py::set_pg_timeouts`（自 #3764，2026-06-27）在第一个训练步后调用 nightly-only 的公开 API → torch ≤ 2.13 上 `AttributeError`。
- 诉求：回退到 `torch.distributed.distributed_c10d._set_pg_timeout`（语义相同；nightly 保留为已弃用别名），或用 `check_if_feature_in_pytorch` 门控。本仓已 polyfill（`compat/shims/dist_set_timeout.py`）。

## TT-8 `PipelineSchedule.step(arg_mbs=..., kwarg_mbs=..., target_mbs=...)` 仅在 nightly —— `draft` <a name="pp-step"></a>
- `trainer.py::pp_forward_backward_step` 通过只有 nightly `step` 才接受的关键字传递预切分的 microbatch。正式版 torch（≤ 2.13）把它们当作模型 kwargs 去切分 → `IndexError: Dimension specified as 0 but tensor has no dimensions`（0 维的 `global_valid_tokens`）。上游集成套件中所有 PP 用例受影响。
- 正式版 torch 有参数完全相同的 `_step_microbatches(arg_mbs, kwarg_mbs, target_mbs, losses, return_outputs, loss_kwargs)`；本仓在 `compat/shims/pp_step_presplit.py` 里 wrap 转发。
- 诉求：用 `inspect.signature(schedule.step)` / `check_if_feature_in_pytorch` 门控并回退到 `_step_microbatches`。

## TT-9 `overrides/fused_mla.py` 使用 nightly-only 的 `torch._C.Tag.inplace` —— `draft`
- `AttributeError: type object 'torch._C.Tag' has no attribute 'inplace'`；影响 `deepseek_v3_fused_mla_swiglu_fsdp+ep`。
- 诉求：`getattr(torch.Tag, "inplace", None)` 保护，或按 torch 版本门控该 override。

## TT-3 `create_block_mask(separate_full_blocks=...)` 仅在 nightly —— `info`
- `models/common/attention.py::create_attention_mask` 自 #4106（2026-08-11）。torch 2.13.0 有，2.12.x 没有。TORCH-1 存在期间在 NPU 上无关紧要。

## TT-4 ChunkedLossWrapper 的 backward 在 NPU 上报 "data is not allocated yet" —— `investigating`
- `components/loss.py` 显式驱动 FSDP 的 lm_head `unshard()`（自 #4143，2026-08-13）。在 torch 2.12.0/2.13.0 + torch_npu 上 backward 抛 `RuntimeError: The tensor has a non-zero number of elements, but its data is not allocated yet`，开不开 AC 都一样；普通 `CrossEntropyLoss` 正常。归因仍开放（torch 版本的 FSDP 语义 vs NPU 分配器/事件处理）。矩阵格 `loss/chunked` = 🔴。

## TT-5 `spmd_types` 后端：NPU 上参数以普通张量到达 `fully_shard(dp_mesh_dims=)` —— `investigating`
- 默认 `parallelism.spmd_backend="spmd_types"`（自 #4085，2026-08-18）且 dp_shard>1 时，FSDP 抛 "all parameters must be DTensors on the full SPMD mesh"。**nightly 有同样的检查**，所以不是 torch 版本差异：是 `Module._distribute_states`/`spmd_distribute_tensor` 在 NPU 上没有产出 DTensor。`partial_dtensor` 可用，recipe 采用它。连锁影响：CP 与 muse_glimmer 要求 spmd_types（13+1 个用例）。矩阵格 `parallel/spmd_types` = 🔴。

## TT-6 kimi_k3 的 attention residual 是自由函数（没有 `Configurable` 节点）—— `ask, deferred`
- `models/kimi_k3/model.py:135 _apply_attention_residual`；override 它需要替换整个 `KimiK3TransformerBlock.Config`。抽成带 `sharding_config` 的 `Module` 也解决上游自己的 `TODO: Add TP Support`。等 kimi_k3 稳定（2026-08-24 落地）。

## TT-7 LM 的 `sdpa` inner attention 已移除 —— `info`
- `models/common/config_utils.py:97`。对非 CUDA 后端的后果：上游不存在 eager 注意力路径。记录在案；override 机制是官方答案。
