"""Kimi K3 on Ascend. Guide: models/kimi_k3/README.md."""

from ascend_titan.models.kimi_k3.recipes import (
    kimi_k3_debugmodel_npu,
    kimi_k3_debugmodel_npu_fused,
)

__all__ = ["kimi_k3_debugmodel_npu", "kimi_k3_debugmodel_npu_fused"]
