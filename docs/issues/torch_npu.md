# torch_npu —— 问题（归属：Ascend/pytorch，gitee.com/ascend/pytorch）

2026-08-29 在 torch 2.12.0 / torch_npu 2.12.0 **与** torch 2.13.0 / torch_npu 2.13.0rc1、CANN 9.1.0、Ascend 910B2 上测得。本仓对这些问题一律不做 workaround（P1）。

## NPU-1 `aten::_flash_attention_forward` 没有 PrivateUse1 内核 —— `draft`
- 现象：`torch.nn.attention.varlen.varlen_attn` → `NotImplementedError: Could not run 'aten::_flash_attention_forward' ... PrivateUse1`（该 op 经自定义 op `torch_attn::_varlen_attn` 分发，随后调用 aten op）。
- 检查：`torch._C._dispatch_has_kernel_for_dispatch_key("aten::_flash_attention_forward", "PrivateUse1")` 在 2.12.0 与 2.13.0rc1 上均为 `False`。
- 影响：torchtitan 的 `VarlenAttention`（上游唯一的非 Flex LM 注意力）无法运行。ascend-torchtitan 改为提供基于 `npu_fusion_attention` 的 override（`ascend_titan.kernels.attention`）——这是内核替换，不是对本问题的绕过。
- 诉求：为 NPU 实现 `_flash_attention_forward/_backward`（varlen 签名：`cum_seq_q/k`、`max_q/k`、`window_size`），例如下沉到 `npu_fusion_attention` TND。这样上游 varlen 无需修改即可运行。

## NPU-2 `npu` 未注册到 `fake` 进程组后端 —— `draft`
- 现象：`--comm.mode=fake_backend`（torchtitan 的单设备干跑）→ `torch.distributed.broadcast` / 函数式 `all_reduce` 抛 `RuntimeError: No backend type associated with device type npu`。
- 检查：`torch.distributed.Backend.backend_capability["fake"]` → `['cpu', 'cuda', 'hpu', 'xpu']`（torch nightly 2.15.0.dev20260828 同样）。HPU 把自己加进去了，torch_npu 也应如此。
- 影响：所有基于 fake PG 的工具（配置干跑、上游 `fake_pg` 集成套件、流水线干跑）在 NPU 上不可用。
- 诉求：`import torch_npu` 时把 `"npu"` 注册进 `Backend.backend_capability["fake"]`（以及函数式集合通信的设备映射），让 fake 后端接受 npu 张量。

## NPU-3 复数张量的高级索引失败（aclnnIndex 161002）—— `draft`
- 现象：`RuntimeError: index_high_dims_op_api: .../op_plugin/ops/opapi/IndexKernelNpuOpApi.cpp:40 NPU function error: call aclnnIndex failed, error code is 161002`。
- 最小复现：`c = torch.view_as_complex(torch.randn(64, 32, 2, device="npu")); c[torch.randint(0, 64, (128,), device="npu")]` 失败；同样的索引在实数张量上正常；`c.index_select(0, idx)` 与复数乘法正常。
- 影响：torchtitan 的 `ComplexRoPE`（llama3、deepseek_v3、gpt_oss、kimi、muse_glimmer 等的默认 RoPE）在 `rope.py:_reshape_for_broadcast` 的 `cache[positions]` 处失败——上游集成套件中 19 个用例。本仓用 `ascend_titan.kernels.rope`（实数缓存、数学相同）作为 override。
- 诉求：让 `aclnnIndex`/`index.Tensor` 支持 complex64（或在 op-plugin 中回退到 `index_select`）。

## NPU-4 `ArgSort` int32/int64 回退到 AiCpu（性能警告）—— `note`
- 在 `torch.distributed`/mesh 建立期间出现 `Warning: kernel [ArgSort] can not support dtype int32 or int64 on AiCore, Now this kernel is running on AiCpu`。不是正确性问题；记录下来避免重复排查。

## NPU-5 `torch_npu==2.13.0rc1` 传递性地要求 `attn-gym` ≥0.0.6？—— `check`
- 安装 torch_npu 2.13.0rc1 时拉入了 `attn-gym 0.0.6`，与 torchtitan 的 `==0.0.5` 冲突。需要确认是哪个包要求的；已在 `constraints/titan-deps.txt` 里 pin 回 0.0.5。
