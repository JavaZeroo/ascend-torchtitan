---
name: override-authoring
description: Write an L1 fused-kernel override in ascend_titan/kernels using torchtitan's @override mechanism, with custom_op registration, loud fallback, opcheck and eager-alignment tests. Use when replacing a torchtitan Configurable node (feed-forward, norm, RoPE, inner attention, KDA kernel, MoE experts) with an Ascend kernel.
---
# override-authoring

Read once: `../torchtitan/torchtitan/overrides/README.md` and `overrides/fused_swiglu.py`.

## 1. Find the node
`grep -n "class .*Config(" ../torchtitan/torchtitan/models/common/<file>.py` (or the model dir).
Confirm the computation is *inside* that node's `forward`. If it's a free function called by the
block (e.g. kimi_k3 `_apply_attention_residual`), stop: add an upstream ask in
`docs/upstream-tracking.md` (P6).

## 2. Module skeleton (`ascend_titan/kernels/<op>.py`)
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
    # register_autograd for training

    class Ascend<Node>(<Node>):
        @dataclass(kw_only=True, slots=True)
        class Config(<Node>.Config): ...
        def forward(...): ...

    @override(target=<Node>.Config, description="…")
    def <op>(cfg: <Node>.Config) -> "Ascend<Node>.Config":
        return derive(cfg, Ascend<Node>.Config)
```
Activate from a recipe: `config.override.imports = ["ascend_titan.kernels.<op>.<op>"]`.

## 3. Tests
- `tests/unit/test_kernel_<op>.py` (CPU): module imports without the kernel package; no override
  registered; warning emitted.
- `tests/npu/test_kernel_<op>.py` (`@pytest.mark.npu`): `torch.library.opcheck`; alignment vs the
  upstream module on the same inputs (fwd + grads, tolerance stated in the test).
- If parameter layout changes: a checkpoint round-trip test against the stock module.

## 4. Record
Matrix cell + `docs/upstream-tracking.md` if any upstream ask surfaced. Benchmarks must include the
provenance table (P7).
