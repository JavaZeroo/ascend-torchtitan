# Qwen3.5 on Ascend

**状态 🟡 — 语言侧真实尺寸能训，视觉侧被 910B2 的 flex 限制挡住，性能是主要缺口。**
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

`ascend_chunk_gdn` 是同一个分解，只把那个循环换成**分块**前代回代：对角块小到可以
安全用级数，其余每个块行一次 matmul。（一开始用的是整块的 Neumann 级数倍增，那是错的——
它会溢出，见第 2 节。）
`tests/unit/test_kernel_gdn.py`（CPU）与 `tests/npu/test_kernel_gdn.py`（910B2）
逐项对 attn_gym 的 reference 钉住：前向、梯度、bf16。

**chunk 尺寸留在 64**，与 fla / attn_gym 一致。步时正比于 `tokens / chunk_size`
（实测 step-1 tps 64 → 231、128 → 314、256 → 385），但 chunk 内转移矩阵的逆随 C
迅速变大——实测最大元素 5.7e3 (C=64)、5.7e6 (C=128)、5.7e15 (C=256)，后面所有乘法
都要乘上它。这是分解本身的性质，与求逆写法无关。性能要从别处来，见第 5 节。

（早先这里写着"128/256 会 NaN 所以必须用 64"，是错的：chunk 64 当时同样会 NaN，
那是求逆写法的问题，已修。）

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

### 曾经在第 4–13 步发散——是我们的求逆，已修

`qwen35_0_8b_npu` 从随机初始化跑，一度会在第 4–13 步 loss 变非有限。**根因是
`kernels/gdn.py` 里 chunk 内那个求逆**，2026-08-31 修掉。

定位过程（每一步都排除了一个当时看起来最像的嫌疑）：

| 试的东西 | 结果 |
|---|---|
| 学习率 5e-3 → 1e-3 → 5e-4 | 只把发散推到第 10 / 更晚，消不掉 |
| warmup 20 → 200 | 推到第 13 步 |
| 优化器 fused → foreach | 只差一步（第 4 → 第 5） |
| 参数初始化 | `A_log ∈ [0.17, 2.73]`、`dt_bias=1.0`，与上游 flex 路径一致 |
| 卡数 | 单卡与 8 卡一样 |
| 全模块前向钩子 | 第一处非有限出在 `AscendGatedDeltaKernel`，**输入有限、输出非有限** |
| 同输入喂 attn_gym reference | **我们非有限、它有限**（max 3.3e-2）——是我们的实现 |
| 与 reference 逐行 diff | 数学上唯一的差别就是那个求逆 |

原来的写法是 Neumann 级数倍增：`(I-A)^-1 = I + A + … + A^(C-1)` 对幂零的 A 是精确的，
`log2(C)` 次 matmul 就能算完。但**最终的和有界，中间的 `A^32` 不是**——在训练几步之后
学到的门控值上它溢出 fp32。attn_gym 的前代回代从不形成这些幂，所以它没事。

现在用分块前代回代（`X = I + A X`，对角块用级数、其余每个块行一次 matmul，用的都是
已定的 X），拿到替换法的数值和 matmul 的开销。三种写法在 910B2 上实测：

| 方法 | chunks=1 | chunks=64 | |
|---|--:|--:|---|
| Neumann 倍增 | 0.50 ms | 0.52 ms | 溢出 |
| `torch.linalg.solve_triangular` | 1.05 ms | 54.93 ms | 正确但开销随 chunk 线性增长 |
| **分块前代回代** | 1.50 ms | 2.44 ms | 用它 |

修后 20 步：loss 12.85958 → 8.30913，rc=0。第 4 步 grad_norm 仍然冲到 21.8，但随后
7.5 → 4.4 → 2.2 → 1.3 收回来了——梯度尖峰本身是能扛过去的，是求逆溢出把它变成了 NaN。

`qwen35_debugmodel_npu_text` 的 golden 在这次改动后**逐位不变**：新旧写法在正常区间
数值恒等，只在旧写法会溢出的地方才分道扬镳。

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
| R1 真实形态 | 🟢 | 0.8B + 真实 tokenizer + 真实 C4 + 4096 上下文，20 步 loss 12.85958 → 8.30913（rc=0）|
| R2 并行覆盖 | 🟡 | 单卡 🟢（12.88826 → 8.14589）；FSDP2×8 🟢（12.90316 → 8.06005）；TP / PP / EP 未测 |
| R3 数值可信 | 🟡 | 算子级对拍 🟢：`tests/unit/test_kernel_gdn.py`（CPU）+ `tests/npu/test_kernel_gdn.py`（910B2，fp32/bf16 前向 + 梯度）都对 attn_gym reference 通过；语言侧 golden 已冻结并逐位复现（`qwen35_debugmodel_npu_text`）。缺的是**真实尺寸**下的长步数下降曲线——它卡在下面那条发散上 |
| R4 checkpoint | 🔴 | HF 导出/导入 🟢（读回 loss 9.91788，随机初始 12.93624）；**DCP 续训 🔴**，归因见下 |
| R5 性能 | 🔴 | **主要缺口**，见下 |
| R6 长稳 | ⚪ | **被 R5 卡住**：0.8B 一步约 2 分钟，500 步要十几个小时。等 GDN 快起来再取；用 debugmodel 跑 500 步不算数（判据要求真实尺寸） |
| R7 文档 | 🟢 | 本文 |
| R8 无隐藏降级 | 🟢 | `ascend-titan-provenance --module ascend_titan.models.qwen3_5 --config qwen35_0_8b_npu`：`AscendFusionAttention` ×6、`AscendGatedDeltaKernel` ×18、`AscendInnerGatedDeltaNet` ×18，共 42 个 ascend 节点 |

### R4：纯文本 recipe 的续训会失败，这是我们那条增量的代价

```
RuntimeError: Missing key in checkpoint state_dict:
optimizer.state.vision_encoder.pos_embed.step
```

`qwen35_0_8b_npu` 把多模态 collator 换成了 `TextCollator`（DELTA 3），于是视觉塔**拿不到
梯度**，AdamW 从不为它建优化器状态，保存的 checkpoint 里自然没有这些键；而 DCP 的 load
planner 会按当前 optimizer 的完整参数集去要，于是缺键报错。

两条出路都堵着：

- **冻结视觉塔**是语义上正确的做法（`requires_grad=False` 的参数根本不会进优化器，
  见上游 `optimizer.py:195`），但上游没有通用的冻结开关——只有 LoRA 自己做——所以这
  不是一条 recipe 增量能表达的东西。
- **跑多模态**就不用绕，但视觉塔在 910B2 上跑不了（第 4 节）。

HF 导出/导入不受影响（它只存模型，不存优化器状态），已经 🟢。

### R5：GDN 没有融合算子

`ascend_chunk_gdn` 是纯 torch 的 chunk 递推，而且这里有**两层**串行循环：

1. **每个文档一次**。真实 C4 经 `ConcatThenSplitPacking` 打包后，一个 4096 的窗口里有几十个
   文档边界；delta rule 必须在每个边界重启递推，所以 `AscendGatedDeltaKernel.forward` 按段
   循环。实测每步每层约 74 次调用（探针计数：单步 1332 次 ÷ 18 个 GDN 层）。
2. **每个 chunk 一次**。段内再按 64 个 token 一块串行推。

循环里每个算子还要过一遍 activation-checkpoint 的 dispatch mode，所以步时是被 Python
派发次数支配的，不是算力。chunk 尺寸又被数值条件卡死在 64（上面第 1 节）——这条路走到头了。

（早先我从 dataloader 的 batch 里没看到 `cu_seqlens`，据此以为走的是 dense 单次调用。
错了：模型内部会从 `positions` 推出 `cu_seqlens`，packed 路径一直是激活的。）

出路按可行性排：

1. **`flash-linear-attention-npu`**——这台机器上就有一份（`../flash-linear-attention-npu`）：
   天津大学主导的昇腾原生线性注意力算子库，AscendC 实现，`fla/ops/ascendc/` 下有 `gdn` 与
   `kda`，导出 `causal_conv1d`、`chunk_bwd_dqkwg` 等，`examples/flash_gated_delta_rule.py`
   是整网示例。还没装（venv 里只有 `fla-core`）。入口是 `build.sh --soc`（910B2 选 A2）
   与 `gdn-verify.sh`（一键编译+装包+单算子测试）。
   这条同时解两件事：性能，以及给 GDN 一个**独立于 attn_gym reference 的第二实现**——
   我们现在能证明自己和 reference 一致，证明不了 reference 和 fla 的分块公式等价，而这正是
   上面那个发散还没排除的一条。
2. **torchair 图模式**：chunk 循环里全是标准 aten 算子（没有 custom_op），理论上可以整段进图，
   把 per-op dispatch 开销消掉。注意力那条 override 是 `custom_op`，缺 GE converter（OURS-13），
   所以要先确认能不能只把 GDN 子图交给 torchair。
3. **Triton-Ascend 编 fla（CUDA 版）的内核**——试过，`bishengir-compile` 不收。

在那之前，这个模型能对，但不快，而且训不稳。

## 6. 上游还有什么

`qwen35_debugmodel_moe`、`qwen35_2b` / `4b` / `9b` / `27b`、
`qwen35_35b_a3b` / `122b_a10b` / `397b_a17b`。全部 ⚪：语言侧的路径已经通了，
剩下的是显存与并行配置的事，按需逐个加 recipe。
