# 能力矩阵

三态（P2）：🟢 可用 · 🔴 失败（必须归因）· ⚪ 未评估。
归因：**TT** torchtitan（→ `docs/issues/torchtitan.md`）· **NPU** torch_npu / op-plugin（→ `docs/issues/torch_npu.md`，绝不绕过，P1/P9）· **TORCH** pytorch 核心（→ `docs/issues/pytorch.md`）· **CANN** 不支持 · **DEP** 第三方 CUDA-only 依赖 · **OURS** 本仓限制（→ `docs/issues/ours.md`）。状态以 `docs/issues/STATUS.md` 为准（P11）。

## 基线（2026-08-30 起：NIGHTLY，ADR-006）

| track | torch | torch_npu | torchtitan | 说明 |
|---|---|---|---|---|
| **NIGHTLY**（默认、门禁） | 2.15.0.dev20260812 | master `15514cc70` 源码构建（+ `patches/` 六个在途修复） | `13da2d77c` | 全部 shim 关闭；golden 与 2.13 逐位一致 |
|  | 2.13.0 | 2.13.0rc1 | `13da2d77c` | 下表 2026-08-29 的 NEXT 列 |
| ~~STABLE~~ | 2.12.0 | 2.12.0 | | 废弃；下表 STABLE 列仅作历史 |

**NIGHTLY 对红格的影响（2026-08-30 实测，`docs/baseline.md`）**：
- TT-4（ChunkedLossWrapper）🔴 → 🟢：单卡与 FSDP2×2 通过；`npu_baseline` 不再展开 loss。**2026-08-30 起 chunked loss 是 qwen3 参考 recipe 的默认**（删除 DELTA 4，golden 已重录）；非 chunked 路径保留为探针 `qwen3_debugmodel_npu_ce_loss`。
- torch_npu 侧修复（`patches/torch_npu/`、`patches/op-plugin/`）：NPU-1 stock varlen、NPU-2 fake_backend、NPU-3 stock ComplexRoPE、NPU-6 uint64、NPU-7 inductor 签名、NPU-8 spmd_types 循环导入——修复前后的格子见 `docs/issues/STATUS.md` 第二轮。
- `flex` eager 在 torch_npu master 上可用（TORCH-1 被 torch_npu 侧绕开；op 级 fwd+bwd 实测通过），因此 `npu_minimal` 的 flex→varlen 改为特性探测，nightly 上不再转换。
- **模型级 flex（2026-08-30 更新）**：torchtitan 在三处无条件 `torch.compile`（`common/attention.py` 的 `_compiled_create_block_mask` 与 `FlexAttention._compiled_flex_attn`、`common/vision_encoder.py` 的 `compiled_create_block_mask`），都不看 `config.compile.enable`。shim `flex_block_mask_eager` 在 triton 没有可用后端时把这三处换回上游自己的未编译函数，模型级 flex 因此**可达**。但**可达不等于可用**：eager flex 会把 O(T²) 的分数矩阵实体化，qwen3 stock flex 因此由 `DEP-INDUCTOR` 变成 **OOM**。所以 `npu_minimal` 的 flex→varlen 转换条件是"设备白名单已解除**且** inductor 有可用后端"，两者缺一就仍然转换；视觉塔的 flex 节点例外（它拿到的是 BlockMask，那条路径没有 varlen 掩码）。
- **CP 的阻塞链（2026-08-30 实测）**：上游明确 `Context Parallel is not supported with ScaledDotProductAttention or VarlenAttention. Use FlexAttention or disable CP.` → CP 必须用 flex → 模型级 flex 需要 inductor → 需要 Triton-Ascend。flex 在昇腾上是通的（见下文 document mask 一节），所以这条链的前提已经不成立——**CP 需要重测**。

## NIGHTLY 全量扫描（2026-08-30，8 卡，61 个上游用例）

数据来源：`python -m ascend_titan.tools.matrix --suites features,models --cards 0-7 --mode minimal --provenance`，
NIGHTLY track（torch 2.15.0.dev20260812 + torch_npu master + `patches/` 的八个修复），910B2 ×8 / CANN 9.1.0。
原始报告：`docs/matrix/2026-08-30_nightly.md`。

**28 🟢 · 28 🔴 · 5 ⚪**

| 归因 | 数量 | 是什么 | 谁来修 |
|---|:--:|---|---|
| `CANN` | 13 | CP ×12（见下）+ `float8_emulate_lora`（float8 没有 cast 内核，`aclnnInplaceCopy 561103`） | **硬件**：都要 Ascend950，910B2 上没有路径 |
| `DEP-INDUCTOR` | 5 | 四个 `*_compile` 用例 + `gpt_oss_fsdp+tp+ep+compile` | 装 Triton-Ascend（已验证可行，见"Triton-Ascend / inductor"一节） |
| `DEP` | 5 | ~~`fla`（qwen3_5 ×3）~~ 已证伪（2026-08-31，见语言侧那行）、`helion` ×2 | 昇腾替代属 L1 |
| `TT-CUDA` | 2 | `DistMuon requires one CUDA device per process` | 上游按设计 CUDA-only |
| `TT-KERNEL` | 3 | `override_fused_swiglu` / `override_fused_grouped_experts` / `deepseek_v3_fused_mla_swiglu`（上游树内 Triton 内核） | 昇腾替代属 L1（我们已有 `npu_swiglu` 版本） |

**`NPU` / `NPU-OP` 归因的红格 = 0，`UNKNOWN` = 0，`HARNESS` = 0，`OURS-*` = 0。**

> 扫描时 `deepseek_v3_fused_mla_swiglu` 还是 `OURS-9`（我们的 override 与上游 `fused_mla` 抢节点）。
> 当天已修复：`fused_mla` 认领的 `layers.N.attention` 同时是 inner attention 与 RoPE 两个节点的祖先，
> 所以两个 override 都得跳过（此前只跳了 RoPE）。修复后复跑，该用例归因变为 `TT-KERNEL`。

⚪ 5 个：4 个上游自己禁用（`2d_asynctp_compile`、`pp_zbv`、`pp_custom_csv`、`pp_looped_zero_bubble`），
1 个上游写死要 CUDA capability 10.0（`kimi_k3_mm_fsdp`）。

绿的 28 个覆盖：FSDP2 / HSDP / DDP / TP+SP / PP（1F1B、GPipe、looped、PP+DP+TP）/ EP（deepseek_v3、gpt_oss）/
checkpoint（full、optional、seed、HF、bf16-only）/ 梯度累积 / bf16 优化器状态 / varlen+SAC / SFT /
多模态（muse_glimmer text 与 mm）。

## 红格按根因归类

| 根因 | 用例数 | 归因 | 谁来修 |
|---|---|---|---|
| `torch.compile`：inductor 后端需要 Triton-Ascend（M3 已解决我们自己的 graph break） | 6 | DEP-INDUCTOR | M5：Triton-Ascend / torchair |
| CUDA-only 依赖缺失：`helion`（helion_rope、deepseek MTP）、`torchao`（float8） | 6 | DEP | 昇腾替代属 L1 |
| 上游树内 CUDA-only 组件：`fused_swiglu`/`fused_grouped_experts` Triton 内核、DistMuon | 4 | TT-KERNEL / TT-CUDA | 上游按设计为 CUDA；昇腾替代属 L1 |
| gpt_oss + TP：路由 softmax backward 形状不匹配（LSE 尾部已实现后新暴露） | 1 | OURS-10 | 本仓，待查 |
| 上游 `fused_mla` override 与我们的 RoPE override 节点冲突（该用例本身 CUDA-only） | 1 | OURS-9 / TT-9 | `npu_minimal` 检测到上游 override 时跳过 RoPE override |

**结论：NPU/CANN 归因的红格为 0。** torch_npu 的三个缺陷（NPU-1 varlen 内核、NPU-2 fake 进程组、NPU-3 复数索引）都已通过上游本就存在的等价实现（varlen 节点 + `npu_fusion_attention`、实数缓存 RoPE）在 L1 层绕开，剩余红格全部归 torch 版本、上游 CUDA-only 组件或本仓自身待办。

## Triton-Ascend / inductor（2026-08-30 实测）

triton-ascend 3.2.2（自带 triton 3.2.0）**可以和 torch 2.15 nightly + torch_npu master 共存**。
装法与两个坑见 `constraints/npu-triton.txt`；环境 `/opt/venv-triton`（从 `/opt/venv-nightly` 克隆而来）。

| 项 | 结果 |
|---|---|
| `triton.backends` 里有 `ascend` 且 `is_active()` | 🟢 |
| `torch.compile(f, backend="inductor")` 简单逐点 + 归约图（前反向） | 🟢 |
| `torch.compile(flex_attention)` + causal block mask（不读张量） | 🔴 → **🟢（NPU-9 修复后）** |
| 直接调 `flex_attention(...)` + 读张量的 mask_mod | **🟢** 前向 / LSE / 反向实测均通过 |
| `torch.compile(flex_attention, options=...)` + 读张量的 mask_mod | 🔴 `SubgraphLoweringException`，见下 |
| torchtitan `1d_compile` 用例（llama3 + `compile.enable`） | 🔴 同上 |

### document mask（读张量的 mask_mod）

**默认路径是通的**，只有把整个 `flex_attention` 函数包进 `torch.compile` 才会踩坑。
实测（`tests/repro/probe_flex_deterministic.py`，910B2）：

| 怎么调用 | 读张量的 mask_mod |
|---|---|
| `flex_attention(q, k, v, block_mask=bm)` —— 它内部只编译 HOP 包装器 | 🟢 前向 / LSE / 反向全过 |
| `torch.compile(flex_attention, options=FlexAttention.inductor_configs)` | 🔴 `SubgraphLoweringException` |

后者的机制链：

1. flex 的 `mask_mod` 是 **pointwise 子图**，torch 的 `PointwiseSubgraphLowering` 禁止在其中创建 buffer。
2. document mask 里 `segment_ids[q_idx]` 是 `aten.index.Tensor`。
3. torch_npu 把 `aten.index` 放在 `INDIRECT_MEM_FALLBACK_LIST` 里，只有
   `inductor_indirect_memory_mode` 打开时才有真正的 lowering，否则 fallback 成 ExternKernel（=建 buffer）。
4. `torch_npu/_inductor/config.py`：`inductor_indirect_memory_mode` 只在 `is_ascend950` 时赋值，
   910B2 上恒为 `None`（实测 `is_ascend950=False`）。

为什么 HOP 包装器那条路上 `aten.index` 没走到 fallback，尚未查清。**在查清前不要把任何一边
推广成通则**——"910B2 上读张量的 mask_mod 一律编不出来"就是这么推出来的，是错的。

实际会踩到这条的只有一处：torchtitan 的 `set_determinism` 在非 ROCm 分支上正是这么做的（TT-12），
`shims/flex_eager_when_deterministic.py` 已处理。

**CP 需要重测**：此前"CP 在 910B2 上不可能"是从那条过宽的结论推出来的，前提不成立了。
但这不等于 CP 能跑——NIGHTLY 扫描里 CP 的 12 个红格归因是 `CANN`，要实测才算。

## 多模态（M5，2026-08-30 实测）

| 模型 / 路径 | NIGHTLY | 说明 |
|---|:--:|---|
| kimi_k3 debugmodel（视觉塔 + KDA + MoE） | 🟢 | 单卡 10 步 `loss 4.56418`，golden 已冻结并逐位复现。08-31 一度记成 🔴，那是误判：它只在 `--debug.deterministic` 下失败，而 `check_golden.sh` 正好加了这个开关。见下面 TT-12 那行 |
| 视觉塔的 BlockMask 构建 | 🟢 | 靠 shim `flex_block_mask_eager`（上游无条件 `torch.compile`，无开关）。注意它只解决**构建掩码**那一步，`flex_attention` 自身的 lowering 不在它管辖内 |
| 视觉塔的 FlexAttention 节点 | 🟢 | `npu_minimal` 不转换它（那条路径没有 varlen 掩码，转了会撞 `attention_masks must be VarlenMetadata, got BlockMask`，实测过）。torch 的 `flex_attention` 在不处于 dynamo 时会自己 `torch.compile(..., fullgraph=True)`，昇腾上这条路是通的——前向、LSE、反向实测均通过 |
| 视觉塔的 block-diagonal document mask | 🟢 | `common/vision_encoder.py:57` 的 `mask_mod` 读张量（`segment_ids[q_idx] == segment_ids[kv_idx]`）。**在昇腾上能跑**：前向、LSE、反向都通过（`tests/repro/probe_flex_deterministic.py`）。此前记成硬件限制是误判——真正的触发条件是`--debug.deterministic`，见 TT-12 |
| **TT-12** flex + 确定性模式 | 🟢（已加 shim） | `set_determinism` 在非 ROCm 分支上把 `_compiled_flex_attn` 重新编译（并覆盖我们的 shim），这条路在昇腾上不通：读张量的掩码 → `SubgraphLoweringException`，causal 掩码 → `InductorError`。上游对 ROCm 的处理就是改用 eager，昇腾缺这条分支。`shims/flex_eager_when_deterministic.py` 补上后两个模型的 golden 都录了 |
| KDA（Kimi Delta Attention） | 🟢 | `kernels/kda.py`：上游 kernel 要 CUDA + Blackwell，改走 attn_gym 的 `impl="reference"` + 自写 depthwise causal conv1d |
| `nvidia-cutlass-dsl` | 🟢 | 有 aarch64 wheel；只 import 不执行（会执行 cute 内核的节点都被 override 掉） |
| qwen3_5 多模态 collator | 🔴 | 视觉塔的 document mask（上一行）。**不是 DEP-FLA**：`fla-core` 有 aarch64 wheel，装上就能 import，只是它的 Triton 内核编不出来——那条路已由 `kernels/gdn.py` 的 override 接管。qwen3_5 的语言侧真实尺寸可跑，见 `ascend_titan/models/qwen3_5/README.md` |

## 低精度 FP8（M5，2026-08-30 实测）

| 项 | 910B2 | 说明 |
|---|:--:|---|
| `torch.zeros(dtype=torch.float8_e4m3fn, device="npu")`、`zero_` | 🟢 | 需要 op-plugin 的 NPU-6 修复（本轮扩展到 float8：全零字节在两种 float8 格式里都是 +0.0，用 int8 视图零化）。修复前报 `aclnnInplaceZero ... 161002` |
| float8 ↔ 其它 dtype 的转换（`copy_` / `.to()` / `.float()`） | 🔴 | `aclnnInplaceCopy failed, error code is 561103`，双向都不行（bf16↔fp8、fp32↔fp8）。归因 **CANN**：910B2 上没有 float8 的 cast 内核 |
| `torch._scaled_mm`（FP8 GEMM） | 🔴 | `_scaled_mm is supported only on the Ascend950 platform and after` —— **硬件限制**，910B2 没有 FP8 计算单元。归因 CANN/HW，不是缺陷 |
| 上游 `float8_emulate_lora` 用例（torchao 已装） | 🔴 | 停在同一个 561103：converter 树要把权重 cast 成 float8 |
| `torch_npu.npu_quant_matmul` | 🟢（存在） | 昇腾自己的量化矩阵乘，INT8 路径；FP8 训练 recipe 需要 A3/950 才有意义 |

结论：910B2 上 CANN 对 float8 的支持仅限于**按字节存储**——分配可以（靠我们扩展的 NPU-6 修复），
**转换和计算都被 CANN 拒绝**。所以 post-converter 树上的 FP8 override 在这台硬件上无从验证，
按 P13 不写：写了也只能在没有的硬件上跑。这条要到 Ascend950 / A3 上才有意义。
归因链：`aclnnInplaceCopy 561103`（cast）→ `_scaled_mm` 明确要求 Ascend950 → **CANN / 硬件**，
按失败处理表"记录错误码，停"。

## 图模式（M5，2026-08-30）

torchair = 昇腾的 GE 图后端，随 torch_npu 发布，但要 `TORCHAIR=1 ./scripts/build_torch_npu.sh` 才带上；
GE 运行时还需要 venv 里有 `decorator`、`scipy`。用法是上游自带的开关（`compile.backend=npu`），不是 shim。

| 分量 | NIGHTLY | 备注 |
|---|:--:|---|
| `compile.components=["loss"]` | 🟢 | qwen3 10 步，`loss 5.11634`（eager 5.10291，编译后归约重排的 bf16 级差异）；recipe 探针 `qwen3_debugmodel_npu_graph` |
| `compile.components=["model"]` | 🔴 | OURS-13：我们的 varlen 注意力 custom_op 没有 GE converter，torchair 找不到 `FusionAttentionVarlen` |
| inductor 后端（`compile.backend=inductor`） | 🔴 | DEP-INDUCTOR：需要 Triton-Ascend |

## 运行路径（M1）

| 路径 | 结果 | 备注 |
|---|---|---|
| 单卡 eager，10 步 | 🟢 | golden `tests/assets/losses/npu/qwen3_debugmodel_npu__*.txt` |
| FSDP2 ×2，10 步 | 🟢 | golden `..._fsdp2__*.txt` |
| fake_backend（单卡模拟 8 卡） | 🔴 | NPU-2：fake 进程组没有 `npu`；**NIGHTLY + NPU-2 补丁 🟢**（step 1 loss 7.66238） |

## 注意力后端

| 后端 | 状态 | 归因 / 备注 |
|---|---|---|
| ascend_fusion（varlen 节点上的 override） | 🟢 | `ascend_titan.kernels.attention`；GQA、按文档因果；op 级误差 8e-3（bf16）；`fsdp+varlen_attn+per_op_sac` 🟢。NIGHTLY 上不再是必需项（stock varlen 可用），保留为性能/兼容项 |
| varlen（stock） | 🟢（NIGHTLY + NPU-1/NPU-6 补丁）/ 🔴（原版 torch_npu：NPU-1） | qwen3 零 override 10 步 loss 5.10302 / grad_norm 3.3060 |
| flex（stock，上游默认） | 🟢 | 前向 / LSE / 反向实测通过（含读张量的 document mask）。mask 构建需要 `flex_block_mask_eager` shim（上游无条件编译它），确定性模式需要 `flex_eager_when_deterministic`（TT-12） |
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
| fla-npu（Gated DeltaNet 融合递推） | 🟢 模型级 | `kernels/gdn_fla.py`：`chunk_gated_delta_rule` 自定义算子（AscendC，R5），`tests/npu/test_kernel_gdn_fla.py` 4 passed；0.8B 单卡模型级真实执行（provenance `AscendFusedGatedDeltaKernel.Config`×18），step-4 tps 1,420 vs 纯 torch 244（≈5.8×），loss 与纯 torch 对拍 bf16 舍入级（step-1 rel 2.2e-5）；golden 逐位待录 |

## 损失

| loss | 状态 | 归因 / 备注 |
|---|---|---|
| CrossEntropyLoss | 🟢 | 2026-08-30 起不再是默认（上游 `ChunkedLossWrapper` 才是）；保留为探针 `qwen3_debugmodel_npu_ce_loss` |
| ChunkedLossWrapper（上游默认） | 🟢（NIGHTLY）/ 🔴（TT-4） | NIGHTLY 单卡 + FSDP2×2 通过； |

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
