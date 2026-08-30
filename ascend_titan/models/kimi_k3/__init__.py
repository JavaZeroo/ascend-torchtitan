"""Kimi K3 on Ascend (blocked on cutlass, TT-11). Guide: models/kimi_k3/README.md."""

from ascend_titan.models.kimi_k3.recipes import kimi_k3_debugmodel_npu

__all__ = ["kimi_k3_debugmodel_npu"]
