# models —— 每个模型一个包（L3）

`ascend_titan/models/<model>/` 放一个模型族的全部内容；跨模型的机制（`deltas.py` 原语、矩阵解析器与它的 `npu_minimal` / `npu_fused`）
留在 `ascend_titan/recipes/`。**内容在 `models/`，机制在 `recipes/`。**

```
ascend_titan/models/
├── README.md          ← 你在这里：总览 + 新增模型的流程
├── registry.py        ← 模型元数据（状态、阻塞、golden），纯数据，不 import torch
├── _template/         ← 新模型骨架，复制即用
└── <model>/
    ├── __init__.py    ← 重导出 recipe，使 `--module ascend_titan.models.<model>` 可用
    ├── recipes.py     ← 支持的入口：`<model>_<flavor>_npu[_<variant>]`
    ├── probes.py      ← 可选：只用于矩阵测量的配置（不是给人跑的）
    └── README.md      ← 必需：该模型的完整使用指南
```

## 支持状态

| 模型 | 状态 | 我们的 recipe | 说明 / 阻塞 |
|---|:--:|---|---|
| **Qwen3** (`qwen3`) | 🟢 | `qwen3` | 参考模型，release 级：0.6B 真实尺寸 + 真实 tokenizer/C4，单卡 / FSDP2×8 / TP2 / PP2 全绿，golden 逐位冻结，500 步长稳、checkpoint 续训逐位一致、HF 权重往返、性能基线带 provenance。 |
| **Qwen3.5** (`qwen3_5`) | 🟡 | `qwen3_5` | 语言侧 0.8B 路径打通（gated delta net 与 causal conv1d 走 `kernels/gdn.py` 的 override，逐项对上游/参考实现对拍），0.8B 20 步 12.88826 → 8.14589、FSDP2×8 12.90316 → 8.06005。多模态 debugmodel 也能跑，golden 已冻结（确定性模式需要 TT-12 那条 shim）；DCP 续训 🔴（纯文本增量让视觉塔没有优化器状态）；GDN 没有融合算子，性能是主要缺口。 |
| **Llama 3** (`llama3`) | 🟡 | `llama3` | 零 override 的 stock 参考路径：复数 RoPE + ChunkedLoss + spmd_types 全部走上游默认实现。只有 debugmodel：R1–R8 一条都没取，按新判据是 🟡 而不是 🟢。 |
| **Kimi K3** (`kimi_k3`) | 🟡 | `kimi_k3` | 多模态 + KDA + MoE，单卡 10 步 loss 4.56418，golden 已冻结并逐位复现。只有 debugmodel：R1–R8 一条都没取。 |
| **DeepSeek-V3** (`deepseek_v3`) | 🟡 | —（矩阵覆盖） | MoE + EP 在矩阵扫描里通过（fsdp+ep、hsdp+ep）；没有专属 recipe，通过矩阵 runner 跑上游配置。 **阻塞：**fused_mla_swiglu：OURS-9（override 节点冲突）；MTP + helion_rope：DEP-HELION。 |
| **GPT-OSS** (`gpt_oss`) | 🟡 | —（矩阵覆盖） | pp+fsdp+ep+sacop 在矩阵里 🟢（attention sinks 的 LSE 尾部已实现）。 **阻塞：**fsdp+tp+ep：OURS-10（TP2+EP4 下路由 softmax backward 形状不匹配），待查。 |
| **Kimi K2.7** (`kimi_k2_7`) | 🟡 | —（矩阵覆盖） | muon / MoE 用例在矩阵里覆盖；无专属 recipe。 **阻塞：**DistMuon 是 CUDA-only（TT-CUDA）。 |
| **Muse Glimmer** (`muse_glimmer`) | 🟡 | —（矩阵覆盖） | text 变体在矩阵里覆盖；多模态变体依赖 CP。 **阻塞：**mm 变体走 CP，停在 DEP-INDUCTOR（Triton-Ascend 未装）。 |
| **Flux** (`flux`) | ⚪ | —（矩阵覆盖） | 扩散模型，尚未评估。 |

🟢 只给 release 级——`docs/model-release-criteria.md` 的 R1–R8 每一条都有记录下来的命令与输出。

### release 判据逐条

| 模型 | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | 证据 |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|---|
| **Qwen3** | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | docs/release/qwen3_torch2.15.0.dev20260812_npu2.15.0.md |
| **Qwen3.5** | 🟢 | 🟡 | 🟡 | 🔴 | 🔴 | ⚪ | 🟢 | 🟢 | — |

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

于是 `--config <上游 flavor>_npu` 对**每个**上游 flavor 都有效（`dir(包)` 列出全部）。
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
| `<model>_<flavor>_npu` | 该模型在昇腾上的参考路径。`<flavor>` = 上游 config registry 的名字（`debugmodel`、`0_6b`…） |
| `..._npu_<variant>` | 并行或算子增量：`fsdp2`、`fused`、`fused_fsdp2` |
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
