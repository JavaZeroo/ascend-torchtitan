# Kimi K3 on Ascend

> 910B2 / CANN 9.1.0 / torch 2.15.0.dev20260812 + torch_npu master + `patches/`，2026-09-04 实测。
> ✅ 实测通过 ｜ ❌ 实测失败 ｜ ❓ 没测过（不代表坏）

## 场景总表

| 模型 | 场景 | 状况 | 限制 |
|---|---|:--:|---|
| **`kimi_k3_debugmodel`**<br>原生上游，零 override | 任何场景 | ❌ | KDA 子树是 CUDA-only：`ValueError: x must be a contiguous CUDA FP16, BF16, or FP32 tensor`（`l2norm` 是 Triton），再往里 `KDAKernel.forward` 还有 Blackwell SM100/SM103 断言 |
| **`kimi_k3_debugmodel_npu`**<br>+ 昇腾融合 attn + KDA override | 单卡（多模态：视觉塔 + LM） | ✅ | 10 步 `loss 4.37603`，显存 12.88 GiB，tps 47 |
| | 确定性模式（录 golden） | ✅ | 10 步 golden 逐位冻结，末步 `4.56418`（`--debug.seed 42 --debug.deterministic`） |
| | FSDP2 × 2 | ✅ | 2 步 `12.40092 → 11.23449` |
| | FSDP2 × 8 | ✅ | 2 步 `12.40720 → 10.68999` |
| | HSDP 2 × 4 | ✅ | 2 步 `12.45858 → 10.63290` |
| | TP | ❌ | **上游不支持**：`parallelize_kimi_k3` 开头就 raise。代码无设备判断，CUDA 上一样 |
| | PP | ❌ | **上游不支持**：`Pipeline Parallel is enabled but kimi_k3 does not support pipelining` |
| | EP | ❌ | **上游不支持**（同 TP 那条 raise）。上游 main 已落地 #4314，升 pin 后要重测 |
| | CP | ❌ | **上游不支持**：配置把 `spmd_backend` 钉成 `partial_dtensor`，CP 要 `spmd_types`，连 `parallelize_kimi_k3` 都没走到 |
| | `torch.compile`（model） | ❌ | **上游不支持**：`Kimi K3 does not support model compilation yet` |
| | DCP 续训 | ❓ | 没测。recipe 里 `checkpoint.enable = False`（冒烟配置） |
| | 长稳 500 步 | ❓ | 一步约 5 秒，没跑 |
| | 真实尺寸 | ❌ | **上游只有 `kimi_k3_debugmodel` 一个 flavor**，没有真实尺寸配置可跑 |
| **`kimi_k3_debugmodel_npu_fused`**<br>+ ops-nn SiTU-GLU（需 ops-nn run 包） | 单卡 | ✅ | 10 步 `loss 4.46498`，显存 12.88 GiB，tps 49——**与参考路径同一量级**，说明瓶颈不在 MoE 的激活 |
| | 其它场景 | ❓ | 与参考路径同族，没有逐格测 |
| **`kimi_k3_mm_fsdp`**（矩阵用例） | 2 卡 | ❌ | **上游写死要 CUDA capability 10.0 / 10.3**，不是昇腾能绕的 |

**五个 ❌ 的并行/编译格子全部是上游按设计不支持**：`parallelize_kimi_k3` 开头一次性挡掉
TP / PP / CP / EP，紧接着挡掉 model 编译，代码里**没有任何设备判断**，CUDA 上报同样的错。
所以这五格与昇腾无关，也不该算进"让 kimi_k3 支持更多并行"的收益里——那件事的前提是上游先支持。
逐格命令与报错原文：`docs/matrix/2026-09-04_kimi_k3_parallel.md`。

## 能替换哪些模块

上游每个 flavor 两个入口：裸名 = 原样，`_npu` = 叠加下面三条。
只想换其中一个？用 `--override.imports`，**不改代码**：

```bash
python -m ascend_titan.train --module ascend_titan.models.kimi_k3 \
  --config kimi_k3_debugmodel \
  --override.imports ascend_titan.kernels.kda.npu_kda
```

| 上游节点 | 换成 | override 目标 | 为什么 |
|---|---|---|---|
| 语言塔 6 个 `FlexAttention`（layers 3/7/11/15/19/23） | 昇腾融合注意力 | `ascend_titan.kernels.attention.npu_fusion_attention_from_flex` | 编译版 flex 在 910B2 上编不出 document mask；eager 版实体化 O(T²) 分数矩阵 |
| 上面转出来的 varlen 节点 | 同上 | `ascend_titan.kernels.attention.npu_fusion_attention` | stock varlen 要 `aten::_flash_attention_forward`，torch_npu 没有（NPU-1） |
| `InnerKDA`（整棵 KDA 子树） | attn_gym reference + 自写短卷积 | `ascend_titan.kernels.kda.npu_kda` | 上游内核要 CUDA + Blackwell；`l2norm` 是 Triton、`causal_conv1d` 是 CuTeDSL |
| `KimiFeedForward`（可选，需 ops-nn） | ops-nn `aclnnSituGlu` 融合 | `ascend_titan.kernels.situ_glu.ops_nn_situ_glu` | 性能。**实测不改变吞吐**（tps 47 → 48） |

**没有替换的**（同样是结论）：视觉塔的 `FlexAttention`（它吃 `BlockMask`，转成 varlen 只会在
塔内部炸一个更难懂的类型错误）、RoPE（kimi_k3 没有 `ComplexRoPE` 节点，用不上 NPU-3 的实数
cache override）、MoE 路由、`_apply_attention_residual`、loss、并行策略、优化器——全是上游默认。

---

## 以下是实现细节

## 1. 跑

```bash
pip install -c constraints/nightly.txt nvidia-cutlass-dsl   # kimi_k3 额外依赖

# 冒烟（10 步，玩具 tokenizer + 仓库自带的 cc12m-test）
ASCEND_RT_VISIBLE_DEVICES=0 NPU=1 \
MODULE=ascend_titan.models.kimi_k3 CONFIG=kimi_k3_debugmodel_npu ./scripts/run_train.sh

# 与冻结的 golden 逐位对比
MODULE=ascend_titan.models.kimi_k3 ASCEND_RT_VISIBLE_DEVICES=0 NPU=1 \
./scripts/check_golden.sh kimi_k3_debugmodel_npu

# FSDP2 × 8
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPU=8 \
MODULE=ascend_titan.models.kimi_k3 CONFIG=kimi_k3_debugmodel_npu ./scripts/run_train.sh \
    --parallelism.data_parallel_shard_degree 8

# 融合 SiTU-GLU（需先装 ops-nn 的 run 包与 torch extension：./scripts/build_kernels.sh ops-nn）
ASCEND_RT_VISIBLE_DEVICES=0 NPU=1 \
MODULE=ascend_titan.models.kimi_k3 CONFIG=kimi_k3_debugmodel_npu_fused ./scripts/run_train.sh
```

没装 ops-nn 时 `kimi_k3_debugmodel_npu_fused` **直接失败**，不是退回参考路径：

```
ValueError: override.imports target 'ascend_titan.kernels.situ_glu.ops_nn_situ_glu'
names no registered @override.
```

`kernels/situ_glu.py` 那条 ADR-004 的 WARNING 会照常打（"situ_glu unavailable …;
KimiFeedForward stays on upstream eager"），但 recipe 仍然把 override 目标写进
`config.override.imports`，上游拒绝未注册的目标。要参考路径就用 `kimi_k3_debugmodel_npu`。

## 2. 上游为什么在昇腾上跑不了，以及我们做了什么

| 上游的东西 | 为什么在昇腾上不行 | 我们的做法 |
|---|---|---|
| `KDAKernel.forward` | 显式要求 CUDA 张量 + Blackwell SM100/SM103，否则 `raise` | override 走 attn_gym 自己的 `impl="reference"`（文档写明是"differentiable eager PyTorch in FP32"） |
| `l2norm` | attn_gym 的 Triton 实现，只认 CUDA 张量 | override 里用等价的 torch 实现（FP32 统计） |
| `causal_conv1d` | attn_gym 的 CuTeDSL（cutlass）内核 | `kernels/kda.py::ascend_causal_conv1d`：W 次移位读 + 边界掩码，dense 与 packed（`cu_seqlens`）都精确 |
| `attn_gym` 顶层 import `cutlass` | 曾被记为 TT-11 阻塞 | **不是阻塞**：`nvidia-cutlass-dsl` 有 aarch64 wheel，装上即可 import；所有会真正执行 cute 内核的节点都被 override 掉了 |
| 无条件 `torch.compile` 的 flex attention | Triton-Ascend 已在基线里、inductor 能编，但 document mask 的间接寻址在 910B2 上 lower 不了（`SubgraphLoweringException`） | shim `flex_attention_eager`。它挂在 `set_determinism` 上——那是唯一能改到 `_compiled_flex_attn` 的钩子，且训练前无条件调用，所以**每一次运行都生效**，不只确定性模式。关掉它 kimi_k3 立刻抛那个异常 |
| 无条件编译的 `create_block_mask` | 同一个硬件门；910B2 上要么把 autotune 逼进 SIMT（NPU-11 / TA-1），要么 `LLVM ERROR: Failed to obtain op buffer shape size` | shim `flex_block_mask_eager` 让掩码构建走 eager |
| `_apply_attention_residual` | 纯 torch，昇腾上直接能跑 | 不动。ops-transformer 的 `block_attn_res_update` 只有前向、没有反向，训练接不进来 |

一条 override 覆盖整棵 KDA 子树：`InnerKDA.Config` 持有 `kernel: KDAKernel.Config`，
torchtitan 拒绝"祖先已被别的 override 认领"的嵌套 override，所以嵌套的 kernel 配置在
工厂里用 `derive` 就地换掉。

## 3. 性能

tps 47 / MFU 0.05%。三个原因，按影响排序：

1. **flex attention 走 eager**（掩码构建也是 eager），比编译版慢一个数量级以上——910B2 上
   编不了 document mask，是硬件门。
2. **KDA 走 attn_gym 的 reference 实现**，FP32 逐块 python 循环。
3. MoE 的 grouped experts 未启用融合算子。

按 P13，在这三条解决之前不记性能基线：数字会误导。融合 SiTU-GLU 实测与参考路径同一量级
（47 vs 49），这本身是个结论：瓶颈不在 MoE 的激活。

## 4. 已知缺口（离 🟢 还差什么）

| 判据 | 状态 | 缺什么 |
|---|:--:|---|
| R1 真实形态 | 🔴 | **上游只有 `kimi_k3_debugmodel` 一个 flavor**，没有真实尺寸的配置可跑。这一条不在我们手上 |
| R2 并行覆盖 | 🟡 | 单卡 🟢、FSDP2×2 🟢、FSDP2×8 🟢、HSDP 2×4 🟢；TP / PP / EP / CP 上游按设计不支持（见场景总表）。上游 main 已落地 EP（#4314），升 pin 后重测 |
| R3 数值可信 | 🟡 | golden 逐位冻结并复现 🟢；算子级对拍只有 **CPU** 的 `tests/unit/test_kernel_kda.py`（覆盖我们自己写的 depthwise causal conv1d，含 packed 边界），**缺 910B2 上的 NPU 对拍**，也缺真实尺寸下的长步数下降曲线（被 R1 卡住） |
| R4 checkpoint | ⚪ | DCP 存取 / 续训 / HF 互操作三件事一件都没测 |
| R5 性能 | 🔴 | 见第 3 节。融合 SiTU-GLU 已跑通但不改变吞吐；瓶颈是 eager flex 与 reference KDA |
| R6 长稳 | ⚪ | 没跑。debugmodel 跑 500 步不算数（判据要求真实尺寸），而真实尺寸被 R1 卡住 |
| R7 文档 | 🟢 | 本文 + `docs/matrix/2026-09-04_kimi_k3_parallel.md` |
| R8 无隐藏降级 | 🟢 | `ascend-titan-provenance --module ascend_titan.models.kimi_k3 --config kimi_k3_debugmodel_npu`：`AscendFusionAttention` ×6、`AscendInnerKDA` ×18、`AscendKDAKernel` ×18，共 42 个 ascend 节点；视觉塔那 1 个 `FlexAttention` 按设计留在上游实现 |

**R1 是这个模型的总闸**：没有真实尺寸配置，R3 的长步数曲线、R6 的长稳都取不到，
R5 也没有值得优化的目标。等上游给出真实尺寸的 flavor，或者我们自己按 model_registry
拼一个——后者要先确认权重规模能在 8×910B2 上装下。

## 5. 待办

- **KDA 融合算子**替换 reference 实现。`../flash-linear-attention-npu` 的 `fla/ops/ascendc/`
  下有 `kda`，和 qwen3.5 用的 `gdn` 是同一个库；qwen3.5 那条路已经跑通（≈5.8×），
  kimi_k3 这条还没接。接的时候注意 `fla_npu` 必须在第一个 NPU 算子执行之前 import。
- **补 910B2 上的 KDA 算子对拍**（`tests/npu/test_kernel_kda.py`），补上 R3 的缺口。
- **`attn_res`**：ops-transformer 的 `block_attn_res_update` 缺反向，只能用于推理；
  要么等它补反向（gitcode 提 issue，P9），要么自己写反向。
- **升 `constraints/torchtitan.sha` 后重测 EP**：上游 main 已落地
  `[Kimi K3] Expert Parallelism support`（#4314），`kda.py` 同时改了 117 行，
  override 的四个类名还在，语义有没有漂要真跑才知道。

## 6. 上游还有什么

只有 `kimi_k3_debugmodel`。真实尺寸的 flavor 上游还没给。
