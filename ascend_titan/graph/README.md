# graph（L2）—— 图模式

昇腾的图模式后端是 **torchair**（GE graph），随 torch_npu 发布，但只有在构建时**没有**
`--disable_torchair` 时才带上。

```bash
TORCHAIR=1 ./scripts/build_torch_npu.sh                                  # 约 9 分钟
pip install --no-deps --force-reinstall /opt/wheels/torch_npu-*.whl
pip install -c constraints/nightly.txt decorator scipy                   # GE 运行时的 python 依赖
```

用法（P0：上游已有开关，所以这是 recipe 增量，不是 shim）：

```python
from ascend_titan.graph import npu_graph

npu_graph(config)  # 默认只编译 loss
npu_graph(config, components=["model"])  # 目前会失败，见下
```

```bash
MODULE=ascend_titan.models.qwen3.probes CONFIG=qwen3_debugmodel_npu_graph NPU=1 ./scripts/run_train.sh
```

## 实测（2026-08-30，torch 2.15.0.dev20260812 + torch_npu master(+torchair)，CANN 9.1.0，910B2）

| 分量 | 结果 |
|---|---|
| `components=["loss"]` | 🟢 10 步跑通，`step: 10 loss: 5.11634`（eager 是 5.10291；编译后归约重排，bf16 级差异） |
| `components=["model"]` | 🔴 `No AscendIR FusionAttentionVarlen was found to be registered` |

## 为什么整模型还进不了图（OURS-13）

`kernels/attention.py` 把昇腾融合注意力封成 `torch.library.custom_op`
（`ascend_titan::fusion_attention_varlen`）。torchair 遇到不认识的自定义算子时，会把算子名
转成驼峰去找同名 AscendIR —— `FusionAttentionVarlen` 在 CANN 9.1.0 里不存在，于是报错。

两条出路：

1. **给自定义算子注册 GE converter**（`torchair.register_fx_node_ge_converter`），映射到
   `ge.FlashAttentionScore` 并带 `input_layout="TND"`。障碍：GE 侧 converter 目前只实现了
   BSH / BNSD 分支（`torch_npu/dynamo/torchair/_ge_concrete_graph/ge_converter/custom/npu_fusion_attention.py`），
   而且 varlen 的 `actual_seq_qlen` 是**数据相关**的（每步不同），静态图里要作为张量输入而不是属性。
2. **等 torch_npu / CANN 补上 TND 的 GE 路径**，届时上面的 converter 直接可写。

这是本仓自己的缺口（OURS-13），不是 torch_npu 的缺陷，所以不走 P9 提 issue；等第 1 条做完再复测。
