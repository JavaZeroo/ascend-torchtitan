# 能力矩阵

三态（P2）：🟢 可用 · 🔴 失败（必须归因）· ⚪ 未评估。
归因：**TT** torchtitan（→ `docs/issues/torchtitan.md`）· **NPU** torch_npu / op-plugin（→ `docs/issues/torch_npu.md`，绝不绕过，P1/P9）· **TORCH** pytorch 核心（→ `docs/issues/pytorch.md`）· **CANN** 不支持 · **DEP** 第三方 CUDA-only 依赖 · **OURS** 本仓限制（→ `docs/issues/ours.md`）。状态以 `docs/issues/STATUS.md` 为准（P11）。

## 基线（2026-08-30 起：NIGHTLY，ADR-006）

| track | torch | torch_npu | torchtitan | 说明 |
|---|---|---|---|---|
| **NIGHTLY**（默认、门禁） | 2.15.0.dev20260812 | master `15514cc70` 源码构建（+ `patches/` 六个在途修复） | `13da2d77c` | 全部 shim 关闭；golden 与 2.13 逐位一致 |
| RELEASE（信息性） | 2.13.0 | 2.13.0rc1 | `13da2d77c` | 下表 2026-08-29 的 NEXT 列 |
| ~~STABLE~~ | 2.12.0 | 2.12.0 | | 废弃；下表 STABLE 列仅作历史 |

**NIGHTLY 对红格的影响（2026-08-30 实测，`docs/baseline.md`）**：
- 版本差红格消失：TT-5/TORCH-6（14 个 CP 用例，待矩阵复测）、TORCH-5（6 个 PP 用例）、TT-9；两条 shim 自动 no-op。
- TT-4（ChunkedLossWrapper）🔴 → 🟢：单卡与 FSDP2×2 通过；`npu_baseline` 不再展开 loss。**2026-08-30 起 chunked loss 是 qwen3 参考 recipe 的默认**（删除 DELTA 4，golden 已重录）；非 chunked 路径保留为探针 `qwen3_debugmodel_npu_ce_loss`。
- torch_npu 侧修复（`patches/torch_npu/`、`patches/op-plugin/`）：NPU-1 stock varlen、NPU-2 fake_backend、NPU-3 stock ComplexRoPE、NPU-6 uint64、NPU-7 inductor 签名、NPU-8 spmd_types 循环导入——修复前后的格子见 `docs/issues/STATUS.md` 第二轮。
- `flex` eager 在 torch_npu master 上可用（TORCH-1 被 torch_npu 侧绕开）；模型级 stock flex 走 inductor，仍需 Triton-Ascend（DEP-INDUCTOR）。

## 历史数据（2026-08-29，正式版 torch，M2 扫描）

数据来源：M2 全量扫描（`python -m ascend_titan.tools.matrix`，上游 `features` + `models` 套件共 61 个用例，对每个配置施加 `npu_baseline`），2026-08-29，torchtitan `13da2d77c`，Ascend 910B2 ×8 / CANN 9.1.0。原始报告：`docs/matrix/2026-08-29_stable.md`、`docs/matrix/2026-08-29_next.md`。

| track | torch | torch_npu | 🟢 | 🔴 | ⚪ |
|---|---|---|---|---|---|
| NEXT（= 现 RELEASE） | 2.13.0 | 2.13.0rc1 | **25** | 31 | 5（上游禁用/门控） |
| STABLE（废弃） | 2.12.0 | 2.12.0 | 18 | 38 | 5 |

NEXT 多出的 6 个 🟢 全是 PP 用例：torch 2.12 的 pipelining `fork_rng` 默认 cuda（TORCH-5），2.13 已修。

## 红格按根因归类（两条 track 合计，去重后 6 个根因解释了全部 32 个 NEXT 红格）

| 根因 | 用例数（NEXT） | 归因 | 谁来修 |
|---|---|---|---|
| `spmd_types` 后端需要 nightly 的 FSDP2（CP、muse_glimmer、validation_tp_cp_pp、qwen3/llama3/deepseek/gpt_oss 的所有 CP 组合） | 14 | TT-5 / TORCH-6 | torch 2.14+ 或 torchtitan 给 CP 一条不依赖 spmd_types 的路径 |
| `torch.compile`：inductor 后端需要 Triton-Ascend（M3 已解决我们自己的 graph break） | 6 | DEP-INDUCTOR | M5：Triton-Ascend / torchair |
| CUDA-only 依赖缺失：`fla`（qwen3_5 GDN）、`helion`（helion_rope、deepseek MTP）、`torchao`（float8） | 6 | DEP | 昇腾替代属 L1（fla-npu 已在路线图） |
| 上游树内 CUDA-only 组件：`fused_swiglu`/`fused_grouped_experts` Triton 内核、DistMuon | 4 | TT-KERNEL / TT-CUDA | 上游按设计为 CUDA；昇腾替代属 L1 |
| gpt_oss + TP：路由 softmax backward 形状不匹配（LSE 尾部已实现后新暴露） | 1 | OURS-10 | 本仓，待查 |
| 上游 `fused_mla` override 与我们的 RoPE override 节点冲突（该用例本身 CUDA-only） | 1 | OURS-9 / TT-9 | npu_baseline 检测到上游 override 时跳过 RoPE override |

**结论：NPU/CANN 归因的红格为 0。** torch_npu 的三个缺陷（NPU-1 varlen 内核、NPU-2 fake 进程组、NPU-3 复数索引）都已通过上游本就存在的等价实现（varlen 节点 + `npu_fusion_attention`、实数缓存 RoPE）在 L1 层绕开，剩余红格全部归 torch 版本、上游 CUDA-only 组件或本仓自身待办。

## 运行路径（M1）

| 路径 | NEXT | STABLE | 备注 |
|---|---|---|---|
| 单卡 eager，10 步 | 🟢 | 🟢 | golden `tests/assets/losses/npu/qwen3_debugmodel_npu__*.txt` |
| FSDP2 ×2，10 步 | 🟢 | 🟢 | golden `..._fsdp2__*.txt` |
| fake_backend（单卡模拟 8 卡） | 🔴 | 🔴 | NPU-2：fake 进程组没有 `npu`；**NIGHTLY + NPU-2 补丁 🟢**（step 1 loss 7.66238） |

## 并行（上游 `features` 套件，llama3 debugmodel）

| 轴 | NEXT | STABLE | 归因 / 备注 |
|---|---|---|---|
| FSDP2（`default`、`fsdp_reshard_always`、`gradient_accumulation`、`optimizer_bf16_states`） | 🟢 | 🟢 | |
| DDP | 🟢 | 🟢 | |
| HSDP（`hsdp`） | 🟢 | 🟢 | |
| TP2 + SP（`2d_eager`）/ TP2 无 SP（`2d_eager_no_sp`） | 🟢 | 🟢 | |
| HSDP + TP（8 卡） | 🟢 | 🟢 | |
| PP 1F1B / PP+DP / PP+TP GPipe / PP+DP+TP / looped 1F1B | 🟢 | 🔴 | STABLE：TORCH-5（2.12 `fork_rng` 默认 cuda）；RELEASE 依赖 shim `pp_step_presplit`（TT-8）；**NIGHTLY 无 shim `pp_1f1b` 🟢** |
| CP（`cp`、`fsdp+cp`、`fsdp+tp+cp`、`hsdp+cp_*`） | 🔴 | 🔴 | RELEASE：TT-5（spmd_types 需 nightly FSDP2）；**NIGHTLY：TT-5 消失，`cp` / `fsdp+cp` 停在 DEP-INDUCTOR（Triton-Ascend）** |
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
| gpt_oss fsdp+tp+ep | 🔴 | 🔴 | OURS-10（M3 后新暴露）：TP2+EP4 下某处 softmax backward 形状 [512,8] vs [256,8]，待查；sinks 尾部本身已实现 |
| gpt_oss pp+fsdp+ep+sacop | 🟢 | 🔴 | NEXT：M3 的 LSE 尾部后通过（PP + EP + attention sinks + per-op SAC）；STABLE：TORCH-5 |
| kimi_k2_5 muon（fsdp+ep、pp+fsdp+ep） | 🔴 | 🔴 | TT-CUDA：`DistMuon requires one CUDA device per process` |
| muse_glimmer（text、mm） | 🔴 | 🔴 | TT-5：模型要求 spmd_types |
| kimi_k3 mm | ⚪ | ⚪ | 上游门控：需要 CUDA 算力 10.0/10.3 |

## 注意力后端

| 后端 | 状态 | 归因 / 备注 |
|---|---|---|
| ascend_fusion（varlen 节点上的 override） | 🟢 | `ascend_titan.kernels.attention`；GQA、按文档因果；op 级误差 8e-3（bf16）；`fsdp+varlen_attn+per_op_sac` 🟢。NIGHTLY 上不再是必需项（stock varlen 可用），保留为性能/兼容项 |
| varlen（stock） | 🟢（NIGHTLY + NPU-1/NPU-6 补丁）/ 🔴（原版 torch_npu：NPU-1） | qwen3 零 override 10 步 loss 5.10302 / grad_norm 3.3060 |
| flex（stock，上游默认） | 🔴 | NIGHTLY：eager op 级 🟢（torch_npu master 绕开 TORCH-1）；模型级走 `torch.compile` → NPU-7 修复后 lowering 通过，停在 Triton-Ascend（DEP-INDUCTOR） |
| flex_flash | 🔴 | TT-by-design：`has_cuda_capability(9,0)` 门控 |
| sdpa | 🔴 | TT-7：上游已为 LM 移除 |
| attention sinks（gpt_oss）/ CP 的 LSE 尾部 | 🟢 | M3：LSE = 按文档还原的 `softmax_max + log(softmax_sum)`（统计量为 head-major 布局），与 `logsumexp` 参考对齐；CP 仍被 TT-5 挡 |

## RoPE

| 实现 | 状态 | 归因 / 备注 |
|---|---|---|
| real_cache_rope（ComplexRoPE 节点上的 override） | 🟢 | `ascend_titan.kernels.rope`；与上游 ComplexRoPE 在 CPU 上逐位一致（含 llama/yarn scaling） |
| ComplexRoPE（stock） | 🟢（NIGHTLY + NPU-3 补丁，op-plugin）/ 🔴（原版：NPU-3） | llama3 零 override 10 步 loss 4.01820（单卡）/ 3.97774（FSDP2×2） |
| CosSinRoPE（qwen3 等） | 🟢 | |

## 归一化

| 实现 | 状态 | 归因 / 备注 |
|---|---|---|
| npu_rms_norm（RMSNorm 节点上的 override，`kernels/rms_norm.py`） | 🟢 | `qwen3_debugmodel_npu_fused_norm` 10 步：loss 5.10306（golden 5.10304）、grad_norm 3.3061 一致；**tps 72k vs 55k（+30%），显存 1.96 vs 2.38 GiB**；op 级对上游 bf16/fp32 对齐测试通过 |
| torch.rms_norm（stock） | 🟢 | M1 默认（golden 基于它） |

## 融合内核（零构建，torch_npu 自带）—— `qwen3_debugmodel_npu_fused`

| 内核 | 状态 | 数值 | 收益（单卡 debugmodel，10 步，seed 42） |
|---|---|---|---|
| `npu_rms_norm` | 🟢 | op 级对齐 | 单独：tps 55k → 72k |
| `npu_swiglu`（上游 FusedSwiGLU 布局） | 🟢 | 对上游 FeedForward 对齐，checkpoint 布局不变 | 三者合计：**tps 77k（+40%），显存 2.38 → 1.89 GiB** |
| `npu_rotary_mul`（half / interleave） | 🟢 | 对上游 CosSinRoPE / ComplexRoPE 对齐（bf16 级） | loss 5.0958 vs golden 5.1030 |
| 融合 recipe golden | 🟢 | `qwen3_debugmodel_npu_fused{,_fsdp2}` 两条 track 逐位一致，已冻结进 `tests/assets/losses/npu/`，nightly 校验 | |

## AscendC 算子库（源码构建）

| 内核 | 状态 | 备注 |
|---|---|---|
| ops-nn `situ_glu` / `situ_glu_grad`（Kimi-K3 SiTU-GLU） | 🟢 op 级 | `scripts/build_kernels.sh ops-nn`（本地盘构建，`--force` 安装：proto 库引用了 CANN 9.1.0 没有的 `aclsysGetVersionNum`，运行正常）；`kernels/situ_glu.py` 封装为可微 custom_op；前向对 fp32 参考误差 0，反向对齐 |
| kimi_k3 模型接入 | 🔴 | TT-11：`import torchtitan.models.kimi_k3` 需要 `cutlass`（attn_gym cute 后端），无 CUDA 环境不可导入 |
| ops-transformer `block_attn_res_update`（attn_res） | 🟡 已构建、已注册 | run 包 + `cann_ops_transformer_ascend_titan` 均安装成功，`torch.ops.cann_ops_transformer.block_attn_res_update` 可见。接入推迟：该算子是**前向、原地、流式**的 online-softmax 形式（partial_block/numerator/logit_max/exp_sum 逐块更新），与上游 `_apply_attention_residual` 的一次性拼接形式不同，且无反向算子；训练接入需要重写 block 循环并自写反向。加之 TT-11。M4。 |
| fla-npu（KDA / causal_conv1d） | 🔴 | OURS-11：需要 `triton-ascend`，无可用 wheel |

## 损失

| loss | 状态 | 归因 / 备注 |
|---|---|---|
| CrossEntropyLoss | 🟢 | M1 默认；`npu_baseline` 把 ChunkedLossWrapper 展开为其内层 loss |
| ChunkedLossWrapper（上游默认） | 🟢（NIGHTLY）/ 🔴（RELEASE：TT-4） | NIGHTLY 单卡 + FSDP2×2 通过；RELEASE 上 backward "data is not allocated yet"（版本差，P8 不处理） |

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
| dynamo + AOTAutograd（`aot_eager`） | 🟢 | M3：注意力 custom_op + `register_fake`，`fullgraph=True` 通过（`tests/npu/test_kernel_attention.py`） |
| inductor（`1d_compile`、`1d_compile_sac_op`、`2d_compile`、`3d_compile`、gpt_oss compile） | 🔴 | DEP-INDUCTOR：torch_npu 的 inductor 后端要求 Triton-Ascend（`triton.language.extra.ascend`），当前只装了纯 Python 的 `triton`；M5 与 torchair 一起处理 |
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
