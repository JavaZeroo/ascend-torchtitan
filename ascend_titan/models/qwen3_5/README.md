# Qwen3.5 on Ascend

**状态 🟡 — 语言侧真实尺寸能跑，视觉侧被 910B2 的 flex 限制挡住，性能是主要缺口。**
判据见 `docs/model-release-criteria.md`。

| | |
|---|---|
| 上游模型包 | `torchtitan.models.qwen3_5`（`../torchtitan/torchtitan/models/qwen3_5/`） |
| 我们的 recipe | `ascend_titan/models/qwen3_5/recipes.py` |
| 关键 override | `ascend_titan/kernels/gdn.py`（gated delta net + causal conv1d） |
| 对齐测试 | `tests/unit/test_kernel_gdn.py`（CPU）+ `tests/npu/test_kernel_gdn.py`（910B2）——都对 attn_gym reference 钉前向、反向、chunk 尺寸无关性 |
| 最近验证 | torch 2.15.0.dev20260812 + torch_npu 2.15.0（master 源码构建 + `patches/`），CANN 9.1.0，2026-08-31 |

## 1. `fla` 那条阻塞已经没有了

早先这份文档写着"`import torchtitan.models.qwen3_5` 直接 `ModuleNotFoundError: No module named 'fla'`"。
实测不成立：`fla-core` 有 aarch64 wheel，装上就能 import，模型包正常加载。

真正不能用的是 fla 的**内核**——它们是给 CUDA 写的 Triton，`bishengir-compile` 不收。
所以 `kernels/gdn.py` 用 `@override` 把两个节点换掉：

| 节点 | 上游实现 | 我们的实现 |
|---|---|---|
| `GatedDeltaKernel` | `fla.ops.gated_delta_rule`（Triton/CUDA） | `ascend_chunk_gdn`：纯 torch 的 chunk 并行 delta rule |
| `InnerGatedDeltaNet` 的短卷积 | `fla` 的 `causal_conv1d_varlen`（packed 分支进 `torch.cuda`） | `ascend_causal_conv1d`（与 kimi_k3 的 KDA 共用） |

两个节点在配置树上是父子关系，torchtitan 不允许两条 override 各占一个，所以是**一条**
override（`npu_gated_delta_net`）在父节点上 derive 出子节点的配置。

### `ascend_chunk_gdn` 与 attn_gym 的关系

`attn_gym.linear.gdn.chunk_gdn(impl="reference")` 是**判据**，不是运行时实现。它是写给人读的：
chunk 内那步要对单位下三角的转移矩阵求逆，它用 `for row in range(1, chunk_size)` 做前代回代，
每次迭代都 `clone()` 整块 `[B, H, chunks, C, C]`。debugmodel 尺寸下看不出来；0.8B（24 层、
4096 上下文、外面还套着 SelectiveAC 的 `__torch_dispatch__`）下**一步十分钟跑不完**。

`ascend_chunk_gdn` 是同一个分解，只把那个循环换成闭式：严格下三角矩阵幂零，
`(I - A)^-1 = I + A + A² + …` 是有限和，倍增法 `log2(C)` 步算完，全是 matmul。
`tests/unit/test_kernel_gdn.py`（CPU）与 `tests/npu/test_kernel_gdn.py`（910B2）
逐项对 attn_gym 的 reference 钉住：前向、梯度、bf16。

**chunk 尺寸不是可调的旋钮。** 步时正比于 `tokens / chunk_size`，实测 step-1 tps
64 → 231、128 → 314、256 → 385；但 128 与 256 都在第 4 步 loss 变 inf，64 正常。
原因不在求逆的写法，而在 chunk 内的转移矩阵本身：`(I - A)^-1` 里 A 的元素是
`beta * (k_i · k_j) * decay`（量级 1），逆的最大元素随 C 增长——实测 5.7e3 (C=64)、
5.7e6 (C=128)、5.7e15 (C=256)，后面所有乘法都被它吃掉。前代回代同样如此，所以
fla 与 attn_gym 都用 64。性能要从别处来，见第 5 节。

## 2. 跑起来

```bash
# 语言侧的冒烟（10 步，玩具 tokenizer + 仓库自带的 c4_test）
ASCEND_RT_VISIBLE_DEVICES=0 NPU=1 \
MODULE=ascend_titan.models.qwen3_5 CONFIG=qwen35_debugmodel_npu_text ./scripts/run_train.sh

# 与冻结的 golden 逐位对比
MODULE=ascend_titan.models.qwen3_5 ASCEND_RT_VISIBLE_DEVICES=0 NPU=1 \
./scripts/check_golden.sh qwen35_debugmodel_npu_text

# 真实尺寸（真实 tokenizer + 真实 C4）
./scripts/fetch_assets.sh tokenizer Qwen/Qwen3.5-0.8B
./scripts/fetch_assets.sh c4 1
export ASCEND_TITAN_ASSETS=/opt/assets
ASCEND_RT_VISIBLE_DEVICES=0 NPU=1 \
MODULE=ascend_titan.models.qwen3_5 CONFIG=qwen35_0_8b_npu ./scripts/run_train.sh \
    --training.steps 20 --lr_scheduler.total_steps 1000
```

`--lr_scheduler.total_steps` 不是可选的：它缺省回落到 `training.steps`，而 `warmup_steps`
会被 clamp 到它。0.8B 的配置是 lr 5e-3 + 20 步 warmup + 1000 步，用 `--training.steps 5`
跑就等于第一步直接满学习率。

### 从零训练会发散（不是内核；学习率能推迟，不能消除）

即使把 LR 曲线钉对，`qwen35_0_8b_npu` 从随机初始化跑仍然会炸，且与卡数无关：

| step | 8 卡 FSDP2 loss / grad_norm | 单卡 loss / grad_norm |
|--:|--:|--:|
| 1 | 12.92388 / 0.8510 | 12.90188 / 1.0562 |
| 2 | 12.21161 / 1.1073 | 12.43087 / 1.2797 |
| 3 | 10.83435 / 2.9801 | 11.33137 / 2.3391 |
| 4 | 11.50074 / 19.9772 | 非有限，中止 |
| 5 | 非有限，中止 | — |

梯度范数逐步翻倍、loss 在发散前一步反弹——优化器炸掉的形状。梯度裁剪是开着的
（`training.max_norm` 默认 1.0），而打出来的是裁剪**前**的范数，所以非有限来自反向本身。

**不是 GDN 的数值问题**，前反向都查过：`ascend_chunk_gdn` 在最坏参数（beta→1、decay→0）下，
seq 16384 的前向与 seq 4096 的梯度都与 attn_gym reference 一致（前向差 1.6e-7、梯度差 4.6e-5），
两边都有限。这两条都固化在 `tests/unit/test_kernel_gdn.py` 里。

**凡是缩小更新幅度的旋钮都只能推迟它，不能消除**：

| 改什么 | 结局 |
|---|---|
| 原样（lr 5e-3、warmup 20/1000） | 第 4–5 步非有限 |
| lr → 1e-3 | 第 10 步非有限（第 7 步 grad_norm 已 14.4） |
| lr → 5e-4 | 10 步跑完，但第 9 步 grad_norm 10.5 |
| warmup 20 → 200（lr 不动） | 第 13 步非有限（第 9 步 grad_norm 9.45） |

四次的形状一模一样：grad_norm 先出现一次尖峰（6–14）、回落、再往上，然后非有限。
"越小越晚"是**有东西在累积**的形状，不是"学习率大了一点"。

**我们替换掉的每一块都已经单独验过，都不是它：**

- GDN 递推：最坏门控（beta→1、decay→0）下，seq 16384 的前向与 seq 4096 的梯度都与
  attn_gym reference 一致且有限（`tests/unit/test_kernel_gdn.py`）。
- causal conv1d：与上游 dense 分支自己的 `F.pad` + `F.conv1d` + `silu` 逐位一致
  （`tests/unit/test_kernel_kda.py`）——不是跟我自己写的 naive 版比。
- 门控数学（`g_TN` / `beta_TN`）：从上游逐字抄的。
- 同一套代码在 debugmodel 尺寸、同样 lr 5e-3 下 10 步稳定下降，golden 已冻结。

所以现在的判断是：**不是我们的实现，也不是学习率/warmup**。recipe 保持上游的值——没定位
到之前把 lr 调低只是把它藏起来——这一格记 🟡，探针留着当记录。

还没排除的两条，都需要现在没有的东西：

1. **attn_gym 的 reference 与 fla 的真实内核是否等价。** 我们对拍的是 reference，如果
   reference 本身与 fla 的分块公式有稳定性差异，两边会一起错而我们看不出来。要证伪需要
   一张能跑 fla 的 CUDA 卡。
2. **上游这个配置在 C4 上本来能不能训。** 上游 0.8B 配的是多模态 cc12m；拿它当对照跑一次
   就能把"数据"这个变量分开，但 cc12m 是图文数据集，不在这台机器的下载预算里。

值得先试的低成本方向：GDN 的跨 chunk 状态递推在 decay→0 时不会遗忘，`state` 会沿 64 个
chunk 单调累积——打一下 `state` 的范数随步数的变化，看是不是它在涨。

## 3. recipe

| 函数 | 卡 | 说明 |
|---|:--:|---|
| `qwen35_debugmodel_npu` | 1 | 上游 `qwen35_debugmodel`（多模态）+ varlen 注意力 + GDN override |
| `qwen35_debugmodel_npu_fsdp2` | 2 | 同上 + 2 路 FSDP2 |
| `qwen35_debugmodel_npu_text` | 1 | 同尺寸但**只跑语言侧**——GDN 的廉价回归探测器。golden 已冻结：10 步 13.03767 → 3.54950，`check_golden.sh` 逐位复现 |
| `qwen35_0_8b_npu` | 1 | Qwen3.5-0.8B，真实 tokenizer + 真实 C4 + 4096 上下文 |
| `qwen35_0_8b_npu_fsdp2` | 8 | 上面 × FSDP2 8 路 |

## 4. 视觉侧：910B2 上跑不了，归因硬件

`qwen35_debugmodel_npu` 用 cc12m-test，会走视觉塔，然后死在：

```
InductorError: LoweringException: SubgraphLoweringException:
Buffers cannot be created while lowering a pointwise subgraph.
  File ".../torchtitan/models/common/vision_encoder.py", line 57, in mask_mod
```

`vision_encoder.py` 的 `create_block_diagonal_mask` 里 `mask_mod` 是
`segment_ids[q_idx] == segment_ids[kv_idx]`——**读张量**。flex 的 `mask_mod` 是 pointwise
子图，读张量要走 inductor 的 indirect-memory 路径，而 `torch_npu/_inductor/config.py` 里
`inductor_indirect_memory_mode` 只在 `is_ascend950` 时赋值；910B2 上恒为 `None`。
与 CP、模型级 flex document mask 是同一个根因，详见 `docs/capability-matrix.md`。

所以真实尺寸的证据取在**语言侧**（`qwen35_0_8b_npu` 换成纯文本 C4）。这不是把问题藏起来：
多模态那格是红的，写在这里，也写在能力矩阵里。

## 5. 已知缺口（离 🟢 还差什么）

| 判据 | 状态 | 缺什么 |
|---|:--:|---|
| R1 真实形态 | 🟡 | 形态齐了（0.8B + 真实 tokenizer + 真实 C4 + 4096 上下文），但从零训练第 5 步发散——见上面第 2 节，先要定位到学习率还是别的 |
| R2 并行覆盖 | 🟡 | 单卡与 FSDP2×8 都能起来并推进（8 卡逐步日志见第 2 节），但都在第 5 步撞上同一个发散；TP / PP / EP 未测 |
| R3 数值可信 | 🟡 | 算子级对拍 🟢：`tests/unit/test_kernel_gdn.py`（CPU）+ `tests/npu/test_kernel_gdn.py`（910B2，fp32/bf16 前向 + 梯度）都对 attn_gym reference 通过；语言侧 golden 已冻结并逐位复现（`qwen35_debugmodel_npu_text`）。缺的是**真实尺寸**下的长步数下降曲线——它卡在下面那条发散上 |
| R4 checkpoint | ⚪ | 未跑 |
| R5 性能 | 🔴 | **主要缺口**，见下 |
| R6 长稳 | ⚪ | **被 R5 卡住**：0.8B 一步约 2 分钟，500 步要十几个小时。等 GDN 快起来再取；用 debugmodel 跑 500 步不算数（判据要求真实尺寸） |
| R7 文档 | 🟢 | 本文 |
| R8 无隐藏降级 | 🟢 | `ascend-titan-provenance --module ascend_titan.models.qwen3_5 --config qwen35_0_8b_npu`：`AscendFusionAttention` ×6、`AscendGatedDeltaKernel` ×18、`AscendInnerGatedDeltaNet` ×18，共 42 个 ascend 节点 |

### R5：GDN 没有融合算子

`ascend_chunk_gdn` 是纯 torch 的 chunk 递推：chunk 之间的循环是串行 Python，循环里每个算子
还要过一遍 activation-checkpoint 的 dispatch mode。步时正比于 `tokens / chunk_size`，
而 chunk 尺寸被数值条件卡死在 64（上面第 1 节）——所以这条路走到头了。

三条可能的出路，按可行性排：

1. **昇腾侧的 gated-delta-rule 融合算子**——真正的答案，属 L1 任务。
2. **torchair 图模式**：chunk 循环里全是标准 aten 算子（没有 custom_op），理论上可以整段进图，
   把 per-op dispatch 开销消掉。注意力那条 override 是 `custom_op`，缺 GE converter（OURS-13），
   所以要先确认能不能只把 GDN 子图交给 torchair。
3. **Triton-Ascend 编 fla 的内核**——试过，`bishengir-compile` 不收。

在那之前，这个模型能训、能对，但不快。

## 6. 上游还有什么

`qwen35_debugmodel_moe`、`qwen35_2b` / `4b` / `9b` / `27b`、
`qwen35_35b_a3b` / `122b_a10b` / `397b_a17b`。全部 ⚪：语言侧的路径已经通了，
剩下的是显存与并行配置的事，按需逐个加 recipe。
