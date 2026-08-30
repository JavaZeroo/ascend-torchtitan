# 一个模型达到 release 需要什么

torchtitan 的 `debugmodel` 是**冒烟件**：几层、随机权重、`tests/assets` 里的玩具 tokenizer 和
几百条 c4 样本、10 步。它能证明"代码路径通了"，不能证明"这个模型在昇腾上能用"。
本文定义两者之间的界线，作为 `ascend_titan/models/registry.py` 里 🟢 的判据。

判据只有一条元规则：**每一条都要有可复现的命令和记录下来的输出**（P13）。没有输出的条目按未达成算。

## R1 真实形态

| | debugmodel | release |
|---|---|---|
| 权重规模 | 玩具（几层） | 上游 registry 里的**真实尺寸**至少一个（qwen3 → `0.6B` 起） |
| tokenizer | `tests/assets/tokenizer` | 真实 HF tokenizer（`assets/hf/<repo>`） |
| 数据 | `c4_test`（几百条，随仓库） | 真实 C4 分片 |
| 上下文 | 2048 | 该尺寸配置的真实值（0.6B → 4096） |

debugmodel 的 golden 仍然保留——它是最便宜的回归探测器——但**不能**作为"支持该模型"的证据。

**"形态齐了"不等于 R1 达成**：真实尺寸的配置必须真的能训，不是能起来。qwen3.5-0.8B 就是
反例——tokenizer、数据、上下文全对，从零训练却在第 4–10 步 loss 非有限。这种情况记 🟡，
把发散的逐步轨迹（loss + grad_norm）写进模型 README，不要因为"跑起来了"就算 R1。

## R2 并行覆盖

单卡、FSDP2×8 必须绿。TP、PP、（MoE 模型）EP 逐项记录：绿的给命令和 loss，红的给归因。
不做"应该可以"的推断。

取 PP 的 loss 要记录**最后一个 rank**：流水并行下只有最后一级算 loss，其余 rank 打
`loss: -4.00000` 占位符，用 rank 0 的日志会把一个健康的 PP 运行看成坏的。`release_check`
与 `bench` 都已经改成 `LOG_RANK=<ngpu-1>`。另外 PP 与权重共享互斥（上游限制），所以
tie 了 embedding 的尺寸取不到这一格——qwen3 的 PP 证据因此取在 8B 上。

## R3 数值可信

1. 确定性 golden（`--debug.seed 42 --debug.deterministic`）逐位冻结，跨版本可比。
2. 损失在**足够步数**下从随机初始的 `ln(vocab)` 附近稳定下降——10 步只是噪声，不算证据。
3. 与参考实现对齐：算子级对齐测试（`tests/npu/`）+ 我们替换掉的每个上游实现都要有一条对拍。

## R4 checkpoint 完整

三件事，缺一不可：

1. **DCP 存/取**：保存后重启加载，loss 轨迹与不中断的一致；
2. **续训**：从第 N 步的 checkpoint 继续，与一口气跑到底的轨迹一致；
3. **HF 互操作**：`state_dict_adapter` 的导出/导入（这是模型能不能交付出去的分界）。

我们现在的 recipe 一律 `checkpoint.enable = False`——那是冒烟配置，release 配置不能这样。

对比时有两个坑，踩过一次要一天：

- **LR 曲线绑在 `--training.steps` 上**。`lr_scheduler.total_steps` 缺省回落到 `training.steps`，
  而 `warmup_steps`（默认 200）会被 clamp 到它。存档运行用 `--training.steps 5`，前 5 步是
  5 步 warmup；参照运行用 `--training.steps 10`，同样这 5 步却是 10 步 warmup——学习率不同，
  续训自然落在参照轨迹之外，看上去像 checkpoint 坏了。三个运行都要 `--lr_scheduler.total_steps`
  钉死。
- **`checkpoint.folder` 是相对 dump folder 解析的**，所以三个运行必须用 `--dump_folder` 分开
  （参照单独一个，存档与续训共用一个）。复用同一个目录会让续训加载它自己上次写的 checkpoint。

`python -m ascend_titan.tools.release_check` 已经按这个协议实现。

## R5 性能有基线

真实尺寸下的 tps / TFLOPs / MFU / 显存，**带 provenance**（P7）。数字要能回答"跑的是不是我们
声称的那个实现"。有对照更好（同尺寸的公开数字、或理论峰值占比），至少要说明差距来自哪里。

## R6 长稳

一次 ≥500 步的运行：不 OOM、不 hang、无 NaN/Inf，显存曲线平稳（无泄漏）。
10 步跑通和 500 步跑通是两回事。

## R7 文档与可复现

模型 README 里，上面每一条结论都有**一条能照抄的命令**；`recipes.py` 顶部有 `validated:` 版本元组；
限制清单写明什么不支持、归因给谁。

## R8 没有隐藏降级

provenance 表证明生效的实现就是我们声称的；没有静默 fallback（P14/ADR-007 已经从机制上堵住了
基础依赖那一路，这里要确认的是 recipe 层）。

## 三态

| 状态 | 含义 |
|---|---|
| 🟢 release | R1–R8 全部有记录 |
| 🟡 可跑 | debugmodel 或真实尺寸能跑，但 R1–R8 有缺口（缺口必须列出来） |
| 🔴 阻塞 | 跑不起来，必须写归因 |

`registry.py` 里的 🟢 从此**只**给 release 级；今天之前的 🟢 含义是"debugmodel 能跑"，
需要按本文重新评定。
