---
name: override-authoring
description: 用 torchtitan 的 @override 机制在 ascend_titan/kernels 编写 L1 融合算子 override，含 custom_op 注册、硬依赖探测（P14）、opcheck 与 eager 对齐测试。用于把 torchtitan 的 Configurable 节点（feed-forward、norm、RoPE、inner attention、KDA kernel、MoE experts）替换为昇腾实现。
---
# override-authoring

先读一次：`../torchtitan/torchtitan/overrides/README.md` 和 `overrides/fused_swiglu.py`；本仓样例：`kernels/attention.py`（外部内核）、`kernels/rope.py`（纯 torch 的替代实现）。

## 1. 找节点
`grep -n "class .*Config(" ../torchtitan/torchtitan/models/common/<file>.py`（或模型目录）。
确认计算在该节点的 `forward` *内部*。若它是块调用的自由函数（如 kimi_k3 的 `_apply_attention_residual`），停：在 `docs/upstream-tracking.md` 加一条上游 ask（P6）。

## 2. 模块骨架（`ascend_titan/kernels/<op>.py`）
```python
"""<Op> via <kernel>. Targets <upstream file:line> <Node>.Config."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torchtitan.config import derive, override
from <upstream module> import <Node>

from ascend_titan.kernels._probe import require_op, torch_npu

# torch_npu is a base dependency (P14 / ADR-007): a missing module or op raises
# at import. NEVER wrap this in try/except and NEVER add an _AVAILABLE switch.
require_op("<npu_op>")


@torch.library.custom_op("ascend_titan::<op>", mutates_args=())
def _op(...): ...
@_op.register_fake
def _(...): ...
# 训练需要 register_autograd


class Ascend<Node>(<Node>):
    @dataclass(kw_only=True, slots=True)
    class Config(<Node>.Config): ...
    def forward(...): ...


@override(target=<Node>.Config, description="…")
def <op>(cfg: <Node>.Config) -> "Ascend<Node>.Config":
    return derive(cfg, Ascend<Node>.Config)
```
真正可选的加速包（ops-nn 的 `cann_ops_nn`、Triton-Ascend——需要单独构建、不在 NIGHTLY 基线里）才用 `_probe.optional_module(*candidates)` + WARNING 降级（ADR-004），样例见 `kernels/situ_glu.py`。
recipe 里激活：`config.override.imports = ["ascend_titan.kernels.<op>.<op>"]`。若**没有它就跑不起来**，加进 `recipes/transforms.py::npu_minimal`；若只是更快（drop-in），加进 `npu_fused`——性能项进 minimal 是 P12 违规。

## 3. 测试
- `tests/unit/test_kernel_<op>.py`（CPU）：用 `npu_stub` fixture 提供假 `torch_npu`；断言 override 注册与 `derive` 结果。若替代实现是纯 torch（如 rope），在 CPU 上与上游模块做逐位/近似对齐。缺依赖的负面用例统一在 `test_kernel_import_safety.py`（必须抛错，P14）。
- `tests/npu/test_kernel_<op>.py`（`@pytest.mark.npu`）：`torch.library.opcheck`；同输入下与上游模块对齐（fwd + grad，容差写在测试里）。
- 若参数布局改变：与 stock 模块的 checkpoint 往返测试。

## 4. 记录
矩阵格子 + 若出现上游 ask 则更新 `docs/upstream-tracking.md`。benchmark 必须带 provenance 表（P7）。
