"""Qwen3 on Ascend. Guide: ascend_titan/models/qwen3/README.md.

Re-exported so ``--module ascend_titan.models.qwen3`` resolves both recipes and
probes; import the submodule directly when you want only one of them.
"""

from torchtitan.models.qwen3 import config_registry as _upstream

from ascend_titan.models._auto import npu_entry_points
from ascend_titan.models.qwen3.probes import (
    qwen3_debugmodel_npu_ce_loss,
    qwen3_debugmodel_npu_fused_norm,
    qwen3_debugmodel_npu_graph,
    qwen3_debugmodel_stock_flex,
    qwen3_debugmodel_stock_varlen,
)
from ascend_titan.models.qwen3.recipes import (
    npu_deltas,
    qwen3_8b_npu_pp2,
    qwen3_debugmodel_npu,
    qwen3_debugmodel_npu_fused,
)

__all__ = [
    "qwen3_8b_npu_pp2",
    "qwen3_debugmodel_npu",
    "qwen3_debugmodel_npu_ce_loss",
    "qwen3_debugmodel_npu_fused",
    "qwen3_debugmodel_npu_fused_norm",
    "qwen3_debugmodel_npu_graph",
    "qwen3_debugmodel_stock_flex",
    "qwen3_debugmodel_stock_varlen",
]

# 任何上游 flavor 都有 `<flavor>_npu` 入口（见 models/_auto.py）；
# 上面显式导入的手写 recipe 优先。
__getattr__, __dir__ = npu_entry_points(_upstream, npu_deltas)
