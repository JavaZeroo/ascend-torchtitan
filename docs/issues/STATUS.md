# 问题处理状态（逐条）

规则：torchtitan / pytorch 的问题 → 本地补丁（`patches/`）供 review，不提上游；torch_npu 的问题 → 解决并验证后提 issue + PR（`gitcode.com/Ascend/pytorch`）。
状态：`待确认` · `已确认` · `已修复（本地补丁）` · `已修复（本仓）` · `无需处理` · `阻塞` · `待提交（torch_npu）`

| 编号 | 问题 | 状态 | 方案 / 位置 | 验证 |
|---|---|---|---|---|
| NPU-1 | `_flash_attention_forward` 无 NPU 内核 | 已修复（本地补丁，待提交 torch_npu） | `patches/torch_npu/0003-npu-implement-aten-_flash_attention_forward-_backwar.patch`：`torch_npu/utils/patch_flash_attention.py` 用 `npu_fusion_attention(_grad)` 实现 PrivateUse1 内核（dense/varlen、causal、左窗口、GQA；LSE ↔ 统计量互转） | ✅ stock `varlen_attn` 前向与本仓算子误差 0、梯度 ≤1e-3；**stock VarlenAttention（无任何 override）的 qwen3 10 步 🟢，loss 5.10302 vs golden 5.10304**（配合 TORCH-8） |
| NPU-2 | fake 进程组未注册 `npu` | 已修复（本地补丁，待提交 torch_npu） | `patches/torch_npu/0001-distributed-register-npu-with-the-fake-process-group-backend.patch`（3 行，`_init/registry/distributed.py`） | ✅ 在已安装的 torch_npu 2.13.0rc1 上应用后 `--comm.mode=fake_backend` 干跑 exit=0 |
| NPU-3 | 复数张量高级索引失败（aclnnIndex 161002） | 已修复（本地补丁，待提交 torch_npu） | `patches/torch_npu/0001-npu-support-advanced-indexing-on-complex-tensors-via-the-real-view.patch`（`Tensor.__getitem__` 经 `view_as_real` 路由；长期解法仍是 aclnnIndex 原生支持 complex） | ✅ 所有索引形式与 CPU 一致；**stock ComplexRoPE 的 llama3（不用 RoPE override）10 步 🟢** |
| NPU-4 | ArgSort int 回退 AiCpu（性能警告） | 无需处理 | 记录 | — |
| NPU-6 | uint64 `zero_` 无内核（挡住 stock varlen 的 rng 占位） | 已确认 | op-plugin 侧需支持 uint64；torch 侧本地补丁 TORCH-8 | 见 TORCH-8 |
| TORCH-8 | `varlen.py` rng_state 占位用 uint64 | 已修复（本地补丁） | `patches/pytorch/0002-TORCH-8-*.patch` | 与 NPU-1 一起验证 |
| NPU-5 | torch_npu 2.13.0rc1 拉入 attn-gym 0.0.6？ | 待确认 | 查 `pip show torch_npu` Requires | 待做 |
| TORCH-1 | FlexAttention 设备白名单 | 上游处理中 | torch_npu **main** 已有 `utils/patch_flexattention.py::_patch_flex_attention_device`（2026-08-13），2.13.0rc1 未包含；无需我们再补 torch 侧补丁 | 等 torch_npu 发版后复测 flex |
| TORCH-2 | fake 后端不可扩展 | 已确认 | 与 NPU-2 一并处理（torch_npu 侧注册即可，无需改 torch） | — |
| TORCH-3 | `set_timeout` 仅 nightly | 无需处理 | 本仓 polyfill；nightly 已有 | ✅ |
| TORCH-4 | PP `step(arg_mbs=)` 仅 nightly | 无需处理 | 本仓 shim；nightly 已有 | ✅ |
| TORCH-5 | 2.12 pipelining `fork_rng` 默认 cuda | 无需处理 | 2.13 已修 | ✅ |
| TORCH-6 | FSDP2 × spmd_types 仅 nightly | 阻塞 | 等 torch ≥ 2.14 | — |
| TORCH-7 | `opcheck` autograd 检查不支持 privateuse1 | 已修复（本地补丁） | `patches/pytorch/0001-TORCH-7-*.patch` | 见下方验证记录 |
| TT-1 | core 无条件 `import triton` | 已修复（本地补丁） | `patches/torchtitan/0004-TT-1-*.patch` | ✅ `sys.modules['triton']=None` 下 `import torchtitan.trainer` 成功；有 triton 时内核不变 |
| TT-2 | `set_timeout` 无特性检查 | 已修复（本地补丁） | `patches/torchtitan/0002-TT-2-*.patch` | ✅ 关闭全部 shim，qwen3 golden 在 2.12/2.13 逐位匹配 |
| TT-3 | `separate_full_blocks` 仅 nightly | 无需处理 | 2.13 已有 | ✅ |
| TT-4 | ChunkedLossWrapper backward "not allocated" | 部分归因 | C++ 栈：`libtorch_npu.so: add_param_to_buf(at::Tensor)` ← `aten::mul.Tensor` 重分发——torch_npu 的 aclnn 参数缓冲在 backward 的某个 `mul` 上读到了存储已释放（FSDP2 reshard）的张量。单独的 ChunkedLossWrapper（含 FSDP2 包装 lm_head、ws=1）可正常前后向，只有接上完整 decoder 图才失败；`fsdp_reshard_after_forward=never` 无效。倾向归因 NPU（torch_npu 与 FSDP2 存储生命周期的交互），需要 torch_npu 侧看 `add_param_to_buf` | 隔离脚本 `outputs/tt4_isolate*.py` |
| TT-5 | spmd_types 需 nightly FSDP2 | 阻塞 | 同 TORCH-6 | — |
| TT-6 | kimi_k3 attn_res 无 Configurable 节点 | 已确认 | 本地补丁：抽成 Module（依赖 TT-11） | 待做 |
| TT-7 | LM 移除 sdpa | 无需处理 | override 机制已覆盖 | ✅ |
| TT-8 | PP `step(arg_mbs=)` 无 fallback | 已修复（本地补丁） | `patches/torchtitan/0003-TT-8-*.patch` | ✅ 关闭全部 shim，NEXT 上 `pp_1f1b` 🟢 |
| TT-9 | fused_mla 用 nightly-only `torch.Tag.inplace` | 已修复（本地补丁） | `patches/torchtitan/0001-TT-9-*.patch` | 模块可导入（该 override 的内核仍是 CUDA-only） |
| TT-10 | 树内 Triton override / DistMuon 写死 CUDA | 无需处理 | 上游按设计 CUDA-only；昇腾替代在 L1 | ✅（swiglu 已替代） |
| TT-11 | kimi_k3 导入需要 `cutlass` | 已修复（本地补丁） | `patches/torchtitan/0005-TT-11-*.patch`：`kda.py` 的 short_conv 导入加 try/except，缺 cutlass 时用纯 torch 的按文档 depthwise 因果卷积回退（chunk_kda 走 attn_gym 自带的 naive 回退） | ✅ 无 cutlass 环境 `import torchtitan.models.kimi_k3` 成功；回退卷积与参考逐元素一致 |
| OURS-1 | attention host offsets D2H | 已修复（本仓） | 已移入 custom_op 内部；每步一次 D2H 仍在 | — |
| OURS-2/4/8/9 | LSE / provenance / compile graph break / override 冲突 | 已修复（本仓） | 见 CHANGELOG | ✅ |
| OURS-3 | 滑窗 `sparse_mode=4` 未测 | 待确认 | 补 NPU 测试 | 待做 |
| OURS-5 | 未与 GPU golden 对比 | 阻塞 | 需 GPU 机器 | — |
| OURS-6 | issue 未提交 | 按新规则关闭 | torchtitan/pytorch 不提；torch_npu 待修复后提 | — |
| OURS-7 | 扫描期间同卡 HCCL 冲突 | 无需处理 | 归因 HARNESS | — |
| OURS-10 | gpt_oss × TP | 已确认 | 排查中 | 待做 |
| OURS-11 | fla-npu 需要 Triton-Ascend | 进行中 | wheel 源已找到（osinfra），安装方式排查中 | 待做 |
