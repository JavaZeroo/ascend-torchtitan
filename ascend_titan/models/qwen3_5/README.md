# Qwen3.5 on Ascend

**状态 🔴 — 阻塞在 `fla`（DEP-FLA）。** recipe 已写好，装上昇腾版 fla 即可跑。

| | |
|---|---|
| 上游模型包 | `torchtitan.models.qwen3_5` |
| 我们的 recipe | `ascend_titan/models/qwen3_5/recipes.py` |
| 阻塞 | `ModuleNotFoundError: No module named 'fla'` |
| 归因 | DEP（第三方 CUDA-only 依赖），不是 NPU / CANN 问题 |

## 1. 现在会发生什么

```bash
$ python -c "import torchtitan.models.qwen3_5"
ModuleNotFoundError: No module named 'fla'
```

链路：`qwen3_5/__init__.py` → `from .gdn import GatedDeltaNet, ...` →
`gdn.py:15` `from fla.modules.conv.triton.ops import CausalConv1dFunction`、
`gdn.py:16` `from fla.ops.gated_delta_rule import ...`。

`fla`（flash-linear-attention）是 gated delta net 的 Triton 内核实现，只有 CUDA 版本。
它在**模块级**被导入，所以整个 qwen3_5 包在昇腾上 import 就失败——不只是 GDN 那一层。

导入失败是**故意不吞**的（P14 / ADR-007）：基础依赖缺失必须在导入处报错，
不能静默降级成一次"看起来跑了"的运行。

## 2. 怎么解开

按优先级：

1. **fla-npu（L1 任务，M4 路线图）**：用 `torch_npu` / AscendC 实现
   `chunk_gated_delta_rule`、`fused_recurrent_gated_delta_rule`、`causal_conv1d`，
   以 `fla` 的包名/接口提供，或在 `ascend_titan/kernels/` 里做 `@override` 并请上游把
   GDN 的内核调用收敛到一个 `Configurable` 节点（P6，`docs/upstream-tracking.md`）。
2. **Triton-Ascend（M5）**：若 fla 的 Triton kernel 能直接在 Triton-Ascend 上编译，
   路径最短——需要先装 `constraints/npu-triton.txt` 的 TRITON track 验证。

上游把 `fla` 提到模块级 import 是 torchtitan 的设计选择；按 P10，我们**不给
github.com/pytorch 提 issue/PR**，只在 `docs/issues/torchtitan.md` 记录。

## 3. recipe（就绪，未验证）

| 函数 | 卡数 | 说明 |
|---|:--:|---|
| `qwen35_debugmodel_npu` | 1 | 上游 `qwen35_debugmodel` + varlen 注意力 + 关 checkpoint |
| `qwen35_debugmodel_npu_fsdp2` | 2 | 同上 + 2 路 FSDP2 |

解开阻塞后的第一步：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 NPU=1 \
MODULE=ascend_titan.models.qwen3_5 CONFIG=qwen35_debugmodel_npu ./scripts/run_train.sh
```

跑绿后：补 `validated:` 头行 → 录 golden（`scripts/check_golden.sh`）→ 更新
`ascend_titan/models/registry.py` 的状态 → 更新 `docs/capability-matrix.md`（P13：先验证再断言）。

## 4. 上游还有什么

`qwen35_debugmodel_moe`、`qwen35_0_8b` / `2b` / `4b` / `9b` / `27b`、
`qwen35_35b_a3b` / `122b_a10b` / `397b_a17b`，以及多模态 collator 路径。
全部 ⚪，且都在同一个 `fla` 阻塞之后。
