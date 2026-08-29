---
name: override-authoring
description: 用 torchtitan 的 @override 机制在 ascend_titan/kernels 编写 L1 融合算子 override，含 custom_op 注册、响亮回退、opcheck 与 eager 对齐测试。用于把 torchtitan 的 Configurable 节点（feed-forward、norm、RoPE、inner attention、KDA kernel、MoE experts）替换为昇腾实现。
---
# override-authoring

先读一次：`../torchtitan/torchtitan/overrides/README.md` 和 `overrides/fused_swiglu.py`；本仓样例：`kernels/attention.py`（外部内核）、`kernels/rope.py`（纯 torch 的替代实现）。

## 1. 找节点
`grep -n "class .*Config(" ../torchtitan/torchtitan/models/common/<file>.py`（或模型目录）。
确认计算在该节点的 `forward` *内部*。若它是块调用的自由函数（如 kimi_k3 的 `_apply_attention_residual`），停：在 `docs/upstream-tracking.md` 加一条上游 ask（P6）。

## 2. 模块骨架（`ascend_titan/kernels/<op>.py`）
```python
"""<Op> via <kernel package>. Targets <upstream file:line> <Node>.Config."""
import logging
logger = logging.getLogger(__name__)
try:
    import <kernel_pkg>
    _AVAILABLE = True
except ImportError as e:
    _AVAILABLE = False
    logger.warning("[ascend_titan] %s unavailable (%s); <Node> stays on upstream eager", "<kernel_pkg>", e)

if _AVAILABLE:
    import torch
    from torchtitan.config import derive, override
    from <upstream module> import <Node>

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
recipe 里激活：`config.override.imports = ["ascend_titan.kernels.<op>.<op>"]`；若所有上游配置都需要它，加进 `recipes/transforms.py::npu_baseline`。

## 3. 测试
- `tests/unit/test_kernel_<op>.py`（CPU）：没有内核包时模块可 import；不注册 override；发出 warning。若替代实现是纯 torch（如 rope），在 CPU 上与上游模块做逐位/近似对齐。
- `tests/npu/test_kernel_<op>.py`（`@pytest.mark.npu`）：`torch.library.opcheck`；同输入下与上游模块对齐（fwd + grad，容差写在测试里）。
- 若参数布局改变：与 stock 模块的 checkpoint 往返测试。

## 4. 记录
矩阵格子 + 若出现上游 ask 则更新 `docs/upstream-tracking.md`。benchmark 必须带 provenance 表（P7）。
