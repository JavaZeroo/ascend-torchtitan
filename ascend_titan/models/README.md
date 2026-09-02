# models —— 每个模型一个包（L3）

> **上游 torchtitan 模型在昇腾上能不能跑？** 910B2 / CANN 9.1.0 /
> torch 2.15.0.dev20260812 + torch_npu master + `patches/`，2026-09-02 实测。
> ✅ 能跑 ｜ ❌ 不能跑 ｜ ❓ 没测过（不代表坏）

能力矩阵 = 上游 `tests/integration_tests` 的 features + models 两套用例，**61 格：43 ✅ / 13 ❌ / 5 ❓**。

| 上游模型 | ✅ | ❌ | 一句话 | 场景总表 |
|---|--:|--:|---|---|
| **Llama 3**（features 全套跑在它上面） | 28 | 6 | **零 override 原样训练** | [llama3/README](llama3/README.md) |
| **Qwen3** | 3 | 1 | release 级参考模型，R1–R8 全绿 | [qwen3/README](qwen3/README.md) |
| **Qwen3.5** | 3 | 0 | 原生与融合两条路径能力边界一样 | [qwen3_5/README](qwen3_5/README.md) |
| **DeepSeek-V3** | 3 | 2 | MoE + EP 通 | 矩阵覆盖，无专属 recipe |
| **GPT-OSS** | 2 | 2 | PP+FSDP+EP+SAC 通 | 矩阵覆盖 |
| **Muse Glimmer** | 2 | 0 | text 与多模态都通，无阻塞 | 矩阵覆盖 |
| **Kimi K2.5** | 0 | 2 | DistMuon 写死 CUDA | 矩阵覆盖 |
| **Kimi K3** | 0 | 1❓ | debugmodel 单卡通 | [kimi_k3/README](kimi_k3/README.md) |

## 13 个不能跑的，按谁的限制

| 谁的限制 | 数量 | 是什么 |
|---|--:|---|
| 上游按设计 CUDA-only | 5 | 树内 Triton 内核 ×3、DistMuon ×2 |
| CUDA-only 依赖缺失 | 2 | `helion` 内核 DSL |
| 910B2 硬件 | 1 | float8 没有 cast 内核，要 Ascend950 |
| 显存（CP + eager flex 的 O(T²)） | 3 | `fsdp+cp`、`hsdp+cp_with_dp_shard`、`hsdp+cp_without_dp_shard` |
| 待查 | 2 | `gpt_oss_fsdp+tp+ep+compile`、`gpt_oss_pp+fsdp+cp+ep+sacop` |

**没有「torch_npu 缺陷」和「我们的缺陷」两类**——曾经有，都已修掉或被证伪。

## 三个通用限制（跨模型，不是某个模型的问题）

| 限制 | 现在怎么处理 | 什么时候消失 |
|---|---|---|
| 编译版 flex 的 document mask 在 910B2 上 lower 不了（间接寻址只有 Ascend950 有） | 掩码构建与 flex 都走 eager（shim，硬件探测） | 换到 Ascend950 自动让路 |
| eager flex 实体化 O(T²) 分数矩阵 | 非 CP → 换昇腾融合注意力；CP → 只能忍，显存不够就 OOM | 让 CP 走融合算子（新增能力） |
| stock `VarlenAttention` 要 `aten::_flash_attention_forward`，torch_npu 没有 | 换昇腾融合注意力 | NPU-1 合入后可换回 |

## 怎么挑模块

上游每个 flavor 两个入口：裸名 = 原样零 override，`_npu` = 叠加该模型族的增量。
只换其中一个用 `--override.imports`，**不改代码**：

```bash
python -m ascend_titan.train --module ascend_titan.models.qwen3_5 \
  --config qwen35_122b_a10b \
  --override.imports ascend_titan.kernels.attention.npu_fusion_attention_from_flex
```

各模型换了什么、**没换什么**，见上表里对应的 README。

## 支持状态

| 模型 | 状态 | 我们的 recipe | 说明 / 阻塞 |
|---|:--:|---|---|
| **Qwen3** (`qwen3`) | 🟢 | `qwen3` | 参考模型，release 级：0.6B 真实尺寸 + 真实 tokenizer/C4，单卡 / FSDP2×8 / TP2 / PP2 全绿，golden 逐位冻结，500 步长稳、checkpoint 续训逐位一致、HF 权重往返、性能基线带 provenance。 |
| **Qwen3.5** (`qwen3_5`) | 🟡 | `qwen3_5` | 语言侧 0.8B 路径打通（gated delta net 与 causal conv1d 走 `kernels/gdn.py` 的 override，逐项对上游/参考实现对拍），0.8B 20 步 12.88826 → 8.14589、FSDP2×8 12.90316 → 8.06005。多模态 debugmodel 也能跑，golden 已冻结。TP2 / PP2 / MoE dp2×tp2×pp2×ep4 已实测，且与原味上游逐格一致。CP 上游按设计不支持（gated delta net 要整条序列，CUDA 上也一样）。**阻塞：**DCP 续训 🔴（纯文本增量让视觉塔没有优化器状态）。 |
| **Llama 3** (`llama3`) | 🟡 | `llama3` | 零 override 的 stock 参考路径：复数 RoPE + ChunkedLoss + spmd_types 全部走上游默认实现。只有 debugmodel：R1–R8 一条都没取，按新判据是 🟡 而不是 🟢。 |
| **Kimi K3** (`kimi_k3`) | 🟡 | `kimi_k3` | 多模态 + KDA + MoE，单卡 10 步 loss 4.56418，golden 已冻结并逐位复现。只有 debugmodel：R1–R8 一条都没取。 |
| **DeepSeek-V3** (`deepseek_v3`) | 🟡 | —（矩阵覆盖） | MoE + EP 在矩阵扫描里通过（fsdp+ep、hsdp+ep）；没有专属 recipe，通过矩阵 runner 跑上游配置。 **阻塞：**`fused_mla_swiglu`：TT-KERNEL（上游树内 Triton 内核，按设计 CUDA-only）；MTP + compile：缺 `helion`。 |
| **GPT-OSS** (`gpt_oss`) | 🟡 | —（矩阵覆盖） | `fsdp+tp+ep` 与 `pp+fsdp+ep+sacop` 均 🟢（attention sinks 的 LSE 尾部已实现；OURS-10 已修）。 **阻塞：**`fsdp+tp+ep+compile`（COMPILE）与 `pp+fsdp+cp+ep+sacop`（CANN error code），两条都待查。 |
| **Kimi K2.7** (`kimi_k2_7`) | 🟡 | —（矩阵覆盖） | muon / MoE 用例在矩阵里覆盖；无专属 recipe。 **阻塞：**DistMuon 写死 CUDA（TT-CUDA），上游按设计如此。 |
| **Muse Glimmer** (`muse_glimmer`) | 🟡 | —（矩阵覆盖） | text 与多模态两个变体在矩阵里**都已通过**（`muse_glimmer_text_fsdp`、`muse_glimmer_mm_fsdp+tp+sp`）。无阻塞。 |
| **Flux** (`flux`) | ⚪ | —（矩阵覆盖） | 扩散模型，尚未评估。 |

🟢 只给 release 级——`docs/model-release-criteria.md` 的 R1–R8 每一条都有记录下来的命令与输出。

### release 判据逐条

| 模型 | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | 证据 |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|---|
| **Qwen3** | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | docs/release/qwen3_torch2.15.0.dev20260812_npu2.15.0.md |
| **Qwen3.5** | 🟢 | 🟢 | 🟡 | 🔴 | 🟢 | ⚪ | 🟢 | 🟢 | docs/release/qwen3_5_torch2.15.0.dev20260812_npu2.15.0.md |

⚪ 表示没测过，不表示坏（P2）。逐条判据的定义见 `docs/model-release-criteria.md`，
一次跑完 R1 / R2 / R4：

```bash
python -m ascend_titan.tools.release_check --model qwen3 --cards 0-7 --out docs/release
```

特性维度的绿红看 `docs/capability-matrix.md`，问题状态看 `docs/issues/STATUS.md`（P11：这里不复述）。

```bash
python -m ascend_titan.models.registry     # 打印上表
```

## 怎么跑

```bash
MODULE=ascend_titan.models.<model> CONFIG=<fn> NPU=<n> ./scripts/run_train.sh
ASCEND_RT_VISIBLE_DEVICES=0 NPU=1 ./scripts/check_golden.sh <fn>      # 与冻结曲线逐位对比
```

没有专属 recipe 的模型（🟡 那几个）直接跑上游配置 + `npu_minimal`（加 `__fused` 后缀则叠加融合算子，`__stock` 则完全不改）：

```bash
MODULE=ascend_titan.recipes.matrix \
CONFIG=torchtitan.models.<model>.config_registry__<upstream_fn> NPU=8 ./scripts/run_train.sh
```

### 族增量声明一次，任何尺寸自动可用

上游 qwen3_5 有 11 个 flavor 且还在增加。为每个 flavor 抄一个 wrapper 是不可持续的，
所以模型包只声明**一次**族增量：

```python
# models/<model>/recipes.py
def npu_deltas(config: Trainer.Config) -> None:
    """这个族在昇腾上需要什么。与 flavor 无关。"""
    add_override(config, ATTENTION_FROM_FLEX_OVERRIDE)
    add_override(config, ATTENTION_OVERRIDE)


# models/<model>/__init__.py
__getattr__, __dir__ = npu_entry_points(_upstream, npu_deltas)
```

于是每个上游 flavor 有**两个**入口（`dir(包)` 列出全部）：

| `--config` | 给你什么 |
|---|---|
| `<flavor>` | 上游自己的配置函数，**原样**（对照组） |
| `<flavor>_npu` | 同一个配置 + 这个族的昇腾增量 |

两者都能再用 `--override.imports` 精确挑选生效哪些模块。
手写 recipe 只在需要额外东西时才写，且必须调用 `npu_deltas`——族增量只存在一处。
同名时手写的优先（Python 先查模块字典，再走 `__getattr__`）。

### recipe 是预设，不是唯一入口

一个 recipe 把四类选择打包成了一个名字：环境必需的增量、内核选择、数据与资产、并行度。
想要别的组合**不必改代码**——`--override.imports` 是整体替换语义，加、减、清空都行：

```bash
# 在预设基础上自己挑内核（这里去掉 GDN、加上融合 RMSNorm）
... ./scripts/run_train.sh --override.imports ascend_titan.kernels.attention.npu_fusion_attention \
                                              ascend_titan.kernels.rms_norm.npu_rms_norm
# 一个 override 都不要
... ./scripts/run_train.sh --override.imports
# 完全的上游原生（对照组）
MODULE=ascend_titan.recipes.matrix CONFIG=<upstream.module>__<fn>__stock ...
```

目标常量在 `ascend_titan/kernels/__init__.py`；`ascend-titan-provenance` 打印实际生效了什么。
唯一还不能从 CLI 换的是注意力后端（flex / varlen）——它是 `model_spec` 的构造参数。

## 命名约定

| 形式 | 含义 |
|---|---|
| `<model>_<flavor>_npu` | 该模型在昇腾上的参考路径。**通常是自动生成的**（上游任一 flavor + `npu_deltas`）；真实尺寸的 tokenizer 由 `HF_REPOS` 表提供 |
| `..._npu_<variant>` | **只用于算子/数据组合**（`fused`、`text`）。并行度**不写 recipe**——用 `--parallelism.*`；golden 用 `GOLDEN=<名字> ./scripts/check_golden.sh <config> --parallelism...` 校验 |
| `<model>_<flavor>_stock_npu` | 零 override 的上游路径（llama3 用这个形式） |
| probes.py 里的任何函数 | 只用于测量，**不是** recipe |

## 新增一个模型（比如 qwen3.6）

1. `cp -r _template qwen3_6`，改包名与 docstring。
2. `recipes.py`：**调用上游 registry 函数再改结果**，绝不从零构造 `Trainer.Config`（P0，
   `tests/unit/test_recipes.py::test_recipe_is_delta_not_copy` 会检查）。每条增量写
   `# DELTA n:` 注释，说明它改了哪个上游默认值、对应矩阵哪一格、**什么条件下能删**。
3. 先在 NPU 上跑通，再写状态：P13"验证先于断言"。跑不通就写 🔴 + 归因，不写"应该可以"。
4. `README.md`：照 `qwen3/README.md` 的骨架——状态表、五分钟跑起来、recipe 列表、
   增量逐条解释、探针、真实尺寸、待办。
5. 在 `registry.py` 里加一条 `ModelEntry`。
6. 绿了就录 golden（`scripts/check_golden.sh`）并在 `recipes.py` 顶部补 `validated:` 头行；
   同步 `docs/capability-matrix.md`。

`tests/unit/test_models_registry.py` 会强制第 1、4、5 步：每个包必须有 README，
每条 registry 条目必须指向真实存在的模块。

## 为什么不把模型配置塞进一个大文件

上游模型会持续增加（qwen3.5 / 3.6 / 3.8、kimi k3、…），每个模型的阻塞原因、算子依赖、
golden 和使用方式都不一样。一个包一个模型意味着：状态、文档、recipe、探针在同一个目录里，
删一个模型是删一个目录，加一个模型是复制一个目录。
