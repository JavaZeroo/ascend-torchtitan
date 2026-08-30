"""Qwen3 on Ascend. Guide: ascend_titan/models/qwen3/README.md.

Re-exported so ``--module ascend_titan.models.qwen3`` resolves both recipes and
probes; import the submodule directly when you want only one of them.
"""

from ascend_titan.models.qwen3.probes import (
    qwen3_debugmodel_npu_ce_loss,
    qwen3_debugmodel_npu_fused_norm,
    qwen3_debugmodel_stock_flex,
    qwen3_debugmodel_stock_varlen,
)
from ascend_titan.models.qwen3.recipes import (
    qwen3_debugmodel_npu,
    qwen3_debugmodel_npu_fsdp2,
    qwen3_debugmodel_npu_fused,
    qwen3_debugmodel_npu_fused_fsdp2,
)

__all__ = [
    "qwen3_debugmodel_npu",
    "qwen3_debugmodel_npu_ce_loss",
    "qwen3_debugmodel_npu_fsdp2",
    "qwen3_debugmodel_npu_fused",
    "qwen3_debugmodel_npu_fused_fsdp2",
    "qwen3_debugmodel_npu_fused_norm",
    "qwen3_debugmodel_stock_flex",
    "qwen3_debugmodel_stock_varlen",
]
