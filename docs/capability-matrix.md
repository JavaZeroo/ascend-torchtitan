# 能力矩阵

三态（P2）：🟢 可用 · 🔴 失败（必须归因）· ⚪ 未评估。
归因：**TT** torchtitan（→ `docs/issues/torchtitan.md`）· **NPU** torch_npu（→ `docs/issues/torch_npu.md`，绝不绕过，P1）· **TORCH** pytorch 核心（→ `docs/issues/pytorch.md`）· **CANN** 不支持 · **DEP** 第三方 CUDA-only 依赖 · **OURS** 本仓限制（→ `docs/issues/ours.md`）。

数据来源：M2 全量扫描（`python -m ascend_titan.tools.matrix`，上游 `features` + `models` 套件共 61 个用例，对每个配置施加 `npu_baseline`），2026-08-29，torchtitan `13da2d77c`，Ascend 910B2 ×8 / CANN 9.1.0。原始报告：`docs/matrix/2026-08-29_stable.md`、`docs/matrix/2026-08-29_next.md`。

| track | torch | torch_npu | 🟢 | 🔴 | ⚪ |
|---|---|---|---|---|---|
| **NEXT**（默认） | 2.13.0 | 2.13.0rc1 | **24** | 32 | 5（上游禁用/门控） |
| STABLE | 2.12.0 | 2.12.0 | 18 | 38 | 5 |

NEXT 多出的 6 个 🟢 全是 PP 用例：torch 2.12 的 pipelining `fork_rng` 默认 cuda（TORCH-5），2.13 已修。

## 红格按根因归类（两条 track 合计，去重后 6 个根因解释了全部 32 个 NEXT 红格）

| 根因 | 用例数（NEXT） | 归因 | 谁来修 |
|---|---|---|---|
| `spmd_types` 后端需要 nightly 的 FSDP2（CP、muse_glimmer、validation_tp_cp_pp、qwen3/llama3/deepseek/gpt_oss 的所有 CP 组合） | 14 | TT-5 / TORCH-6 | torch 2.14+ 或 torchtitan 给 CP 一条不依赖 spmd_types 的路径 |
| `torch.compile`：我们的注意力模块在 `fullgraph=True` 下 graph break | 6 | OURS-8 | 本仓 M3（custom_op + register_fake） |
| CUDA-only 依赖缺失：`fla`（qwen3_5 GDN）、`helion`（helion_rope、deepseek MTP）、`torchao`（float8） | 6 | DEP | 昇腾替代属 L1（fla-npu 已在路线图） |
| 上游树内 CUDA-only 组件：`fused_swiglu`/`fused_grouped_experts` Triton 内核、DistMuon | 4 | TT-KERNEL / TT-CUDA | 上游按设计为 CUDA；昇腾替代属 L1 |
| 注意力 LSE 尾部（gpt_oss attention sinks） | 2 | OURS-2 | 本仓 M4 |
| 上游 `fused_mla` override 与我们的 RoPE override 节点冲突（该用例本身 CUDA-only） | 1 | OURS-9 / TT-9 | npu_baseline 检测到上游 override 时跳过 RoPE override |

**结论：NPU/CANN 归因的红格为 0。** torch_npu 的三个缺陷（NPU-1 varlen 内核、NPU-2 fake 进程组、NPU-3 复数索引）都已通过上游本就存在的等价实现（varlen 节点 + `npu_fusion_attention`、实数缓存 RoPE）在 L1 层绕开，剩余红格全部归 torch 版本、上游 CUDA-only 组件或本仓自身待办。

## 运行路径（M1）

| 路径 | NEXT | STABLE | 备注 |
|---|---|---|---|
| 单卡 eager，10 步 | 🟢 | 🟢 | golden `tests/assets/losses/npu/qwen3_debugmodel_npu__*.txt` |
| FSDP2 ×2，10 步 | 🟢 | 🟢 | golden `..._fsdp2__*.txt` |
| fake_backend（单卡模拟 8 卡） | 🔴 | 🔴 | NPU-2：fake 进程组没有 `npu` |

## 并行（上游 `features` 套件，llama3 debugmodel）

| 轴 | NEXT | STABLE | 归因 / 备注 |
|---|---|---|---|
| FSDP2（`default`、`fsdp_reshard_always`、`gradient_accumulation`、`optimizer_bf16_states`） | 🟢 | 🟢 | |
| DDP | 🟢 | 🟢 | |
| HSDP（`hsdp`） | 🟢 | 🟢 | |
| TP2 + SP（`2d_eager`）/ TP2 无 SP（`2d_eager_no_sp`） | 🟢 | 🟢 | |
| HSDP + TP（8 卡） | 🟢 | 🟢 | |
| PP 1F1B / PP+DP / PP+TP GPipe / PP+DP+TP / looped 1F1B | 🟢 | 🔴 | STABLE：TORCH-5（2.12 `fork_rng` 默认 cuda）；两条 track 都依赖 shim `pp_step_presplit`（TT-8） |
| CP（`cp`、`fsdp+cp`、`fsdp+tp+cp`、`hsdp+cp_*`） | 🔴 | 🔴 | TT-5：CP 要求 `spmd_types`，后者需要 nightly FSDP2 |
| `validation_tp_cp_pp` | 🔴 | 🔴 | TT-5 |
| async TP（`2d_asynctp_compile`） | ⚪ | ⚪ | 上游禁用 |
| spmd_backend = spmd_types（上游默认） | 🔴 | 🔴 | TT-5 / TORCH-6（torch 版本差异，与 NPU 无关） |

## 模型（上游 `models` 套件）

| 用例 | NEXT | STABLE | 归因 / 备注 |
|---|---|---|---|
| llama3 fsdp+tp+pp（8 卡） | 🟢 | 🔴 | STABLE：TORCH-5 |
| llama3 fsdp+tp+cp | 🔴 | 🔴 | TT-5 |
| deepseek_v3 fsdp+ep / hsdp+ep（MoE + 专家并行） | 🟢 | 🟢 | |
| deepseek_v3 fsdp+cp+pp+ep | 🔴 | 🔴 | TT-5 |
| deepseek_v3 fused_mla_swiglu | 🔴 | 🔴 | NEXT：OURS-9（override 节点冲突）；STABLE：TT-9（nightly-only `torch.Tag.inplace`） |
| deepseek_v3 mtp + compile（helion_rope） | 🔴 | 🔴 | DEP：helion |
| qwen3 fsdp+tp+cp（含 fused/non-fused qkv、MoE param groups） | 🔴 | 🔴 | TT-5 |
| qwen3_5（GDN，需要 `fla`） | 🔴 | 🔴 | DEP：fla（昇腾侧对应 fla-npu，M4） |
| gpt_oss fsdp+tp+ep | 🔴 | 🔴 | OURS-2：attention sinks 需要 LSE 尾部 |
| gpt_oss pp+fsdp+ep+sacop | 🔴 | 🔴 | NEXT：OURS-2；STABLE：TORCH-5 |
| kimi_k2_5 muon（fsdp+ep、pp+fsdp+ep） | 🔴 | 🔴 | TT-CUDA：`DistMuon requires one CUDA device per process` |
| muse_glimmer（text、mm） | 🔴 | 🔴 | TT-5：模型要求 spmd_types |
| kimi_k3 mm | ⚪ | ⚪ | 上游门控：需要 CUDA 算力 10.0/10.3 |

## 注意力后端

| 后端 | 状态 | 归因 / 备注 |
|---|---|---|
| ascend_fusion（varlen 节点上的 override） | 🟢 | `ascend_titan.kernels.attention`；GQA、按文档因果；op 级误差 8e-3（bf16）；`fsdp+varlen_attn+per_op_sac` 🟢 |
| varlen（stock） | 🔴 | NPU-1：没有 `aten::_flash_attention_forward` 的 NPU 内核 |
| flex（stock，上游默认） | 🔴 | TORCH-1：torch 在 `flex_attention` 里拒绝 npu 设备 |
| flex_flash | 🔴 | TT-by-design：`has_cuda_capability(9,0)` 门控 |
| sdpa | 🔴 | TT-7：上游已为 LM 移除 |
| attention sinks（gpt_oss）/ CP 的 LSE 尾部 | 🔴 | OURS-2 |

## RoPE

| 实现 | 状态 | 归因 / 备注 |
|---|---|---|
| real_cache_rope（ComplexRoPE 节点上的 override） | 🟢 | `ascend_titan.kernels.rope`；与上游 ComplexRoPE 在 CPU 上逐位一致（含 llama/yarn scaling） |
| ComplexRoPE（stock） | 🔴 | NPU-3：torch_npu 不能对复数张量做高级索引（aclnnIndex 161002） |
| CosSinRoPE（qwen3 等） | 🟢 | |

## 损失

| loss | 状态 | 归因 / 备注 |
|---|---|---|
| CrossEntropyLoss | 🟢 | M1 默认；`npu_baseline` 把 ChunkedLossWrapper 展开为其内层 loss |
| ChunkedLossWrapper（上游默认） | 🔴 | TT-4：backward "data is not allocated yet"（lm_head 手动 unshard 路径） |

## 激活检查点

| 模式 | 状态 | 归因 / 备注 |
|---|---|---|
| selective（op）——上游默认 | 🟢 | `activation_checkpoint.py:251` 按 `"cuda"` 字符串的 D2H `MUST_SAVE` 策略在 NPU 上不生效：正确但少一个优化 → 包装型 shim 候选（未做） |
| none | 🟢 | |
| full | 🟢 | `full_checkpoint` 用例 |
| per-op SAC + varlen | 🟢 | `fsdp+varlen_attn+per_op_sac` |

## 编译 / 图模式

| 模式 | 状态 | 归因 / 备注 |
|---|---|---|
| eager | 🟢 | |
| inductor（`1d_compile`、`1d_compile_sac_op`、`2d_compile`、`3d_compile`、gpt_oss compile） | 🔴 | OURS-8：我们的注意力模块在 `fullgraph=True` 下 graph break；M3 改 custom_op 后重测，届时才能看到 inductor 在 NPU 上本身的状态 |
| torchair | ⚪ | M5 |
| CUDA graphs | 🔴 | TT-by-design：上游在非 CUDA 上自动退回 eager 并警告。不是 bug。 |

## 精度

| 模式 | 状态 | 归因 / 备注 |
|---|---|---|
| bf16 | 🟢 | |
| float8（`float8_emulate_lora`） | 🔴 | DEP：torchao 未安装；上游 converter 又有 `has_cuda_capability(8,9)` 门控；昇腾 FP8 = post-converter 树上的 override（M5） |
| mx / nvfp4 | 🔴 | CANN：NVIDIA 专有格式。不计划。 |

## 其它

| 特性 | 状态 | 归因 / 备注 |
|---|---|---|
| import torchtitan.trainer | 🟢 | 需要 `triton` wheel（TT-1）——已在 `constraints/titan-deps.txt` |
| 第 1 步后缩短 PG 超时 | 🟡 | 已 polyfill；torch ≤ 2.13 的 `_set_pg_timeout` 忽略 HCCL（TORCH-3） |
| DCP checkpoint 保存/恢复（`full_checkpoint`、`optional_checkpoint`、`last_save_model_only_bf16`、`seed_checkpoint`） | 🟢 | 两条 track |
| HF 格式 checkpoint 导出 + 回载（`model_only_hf_checkpoint`） | 🟢 | 两条 track（首轮扫描的红是 harness 相对路径错误，已修并补跑） |
| SFT（`sft`） | 🟢 | |
| 上游树内 override（`override_fused_swiglu`、`override_fused_grouped_experts`） | 🔴 | TT-KERNEL：Triton 内核只注册 CUDA |
| profiler | ⚪ | |
| 确定性模式（`--debug.deterministic`） | 🟢 | NEXT/STABLE 曲线逐位一致 |
| 多模态（flux、muse_glimmer mm、kimi_k3 mm） | 🔴/⚪ | muse：TT-5；kimi_k3：上游门控；flux：未评估（M5） |
