# 问题处理状态（逐条）

规则：torchtitan / pytorch 的问题 → 本地补丁（`patches/`）供 review，不提上游；torch_npu 的问题 → 解决并验证后提 issue + PR（`gitcode.com/Ascend/pytorch`）。
状态：`待确认` · `已确认` · `已修复（本地补丁）` · `已修复（本仓）` · `无需处理` · `阻塞` · `待提交（torch_npu）`

| 编号 | 问题 | 状态 | 方案 / 位置 | 验证 |
|---|---|---|---|---|
| NPU-1 | `_flash_attention_forward` 无 NPU 内核 | 已确认 | 计划：在 torch_npu 内以 `npu_fusion_attention` 实现 aten `_flash_attention_forward/_backward` 的 PrivateUse1 内核 | 待做 |
| NPU-2 | fake 进程组未注册 `npu` | 已确认 | 计划：torch_npu 导入时向 `Backend.backend_capability["fake"]` 注册 `npu` | 待做 |
| NPU-3 | 复数张量高级索引失败（aclnnIndex 161002） | 已确认 | 计划：op-plugin 对 complex 输入回退 `view_as_real` 索引 | 待做 |
| NPU-4 | ArgSort int 回退 AiCpu（性能警告） | 无需处理 | 记录 | — |
| NPU-5 | torch_npu 2.13.0rc1 拉入 attn-gym 0.0.6？ | 待确认 | 查 `pip show torch_npu` Requires | 待做 |
| TORCH-1 | FlexAttention 设备白名单 | 已确认 | 本地补丁（放行 privateuse1）；NPU 上 flex 还需 inductor + Triton-Ascend | 待做 |
| TORCH-2 | fake 后端不可扩展 | 已确认 | 与 NPU-2 一并处理（torch_npu 侧注册即可，无需改 torch） | — |
| TORCH-3 | `set_timeout` 仅 nightly | 无需处理 | 本仓 polyfill；nightly 已有 | ✅ |
| TORCH-4 | PP `step(arg_mbs=)` 仅 nightly | 无需处理 | 本仓 shim；nightly 已有 | ✅ |
| TORCH-5 | 2.12 pipelining `fork_rng` 默认 cuda | 无需处理 | 2.13 已修 | ✅ |
| TORCH-6 | FSDP2 × spmd_types 仅 nightly | 阻塞 | 等 torch ≥ 2.14 | — |
| TORCH-7 | `opcheck` autograd 检查不支持 privateuse1 | 已确认 | 本地补丁 | 待做 |
| TT-1 | core 无条件 `import triton` | 已确认 | 本地补丁：懒加载 | 待做 |
| TT-2 | `set_timeout` 无特性检查 | 已修复（本地补丁） | `patches/torchtitan/0002-TT-2-*.patch` | ✅ 关闭全部 shim，qwen3 golden 在 2.12/2.13 逐位匹配 |
| TT-3 | `separate_full_blocks` 仅 nightly | 无需处理 | 2.13 已有 | ✅ |
| TT-4 | ChunkedLossWrapper backward "not allocated" | 已确认 | 根因排查中 | 待做 |
| TT-5 | spmd_types 需 nightly FSDP2 | 阻塞 | 同 TORCH-6 | — |
| TT-6 | kimi_k3 attn_res 无 Configurable 节点 | 已确认 | 本地补丁：抽成 Module（依赖 TT-11） | 待做 |
| TT-7 | LM 移除 sdpa | 无需处理 | override 机制已覆盖 | ✅ |
| TT-8 | PP `step(arg_mbs=)` 无 fallback | 已修复（本地补丁） | `patches/torchtitan/0003-TT-8-*.patch` | ✅ 关闭全部 shim，NEXT 上 `pp_1f1b` 🟢 |
| TT-9 | fused_mla 用 nightly-only `torch.Tag.inplace` | 已修复（本地补丁） | `patches/torchtitan/0001-TT-9-*.patch` | 模块可导入（该 override 的内核仍是 CUDA-only） |
| TT-10 | 树内 Triton override / DistMuon 写死 CUDA | 无需处理 | 上游按设计 CUDA-only；昇腾替代在 L1 | ✅（swiglu 已替代） |
| TT-11 | kimi_k3 导入需要 `cutlass` | 已确认 | 本地补丁：kda.py 懒加载 attn_gym 的 cute 路径 | 待做 |
| OURS-1 | attention host offsets D2H | 已修复（本仓） | 已移入 custom_op 内部；每步一次 D2H 仍在 | — |
| OURS-2/4/8/9 | LSE / provenance / compile graph break / override 冲突 | 已修复（本仓） | 见 CHANGELOG | ✅ |
| OURS-3 | 滑窗 `sparse_mode=4` 未测 | 待确认 | 补 NPU 测试 | 待做 |
| OURS-5 | 未与 GPU golden 对比 | 阻塞 | 需 GPU 机器 | — |
| OURS-6 | issue 未提交 | 按新规则关闭 | torchtitan/pytorch 不提；torch_npu 待修复后提 | — |
| OURS-7 | 扫描期间同卡 HCCL 冲突 | 无需处理 | 归因 HARNESS | — |
| OURS-10 | gpt_oss × TP | 已确认 | 排查中 | 待做 |
| OURS-11 | fla-npu 需要 Triton-Ascend | 进行中 | wheel 源已找到（osinfra），安装方式排查中 | 待做 |
