"""Qwen3.5 on Ascend (blocked on fla). Guide: models/qwen3_5/README.md."""

from ascend_titan.models.qwen3_5.recipes import (
    qwen35_0_8b_npu,
    qwen35_0_8b_npu_fsdp2,
    qwen35_debugmodel_npu,
    qwen35_debugmodel_npu_fsdp2,
)

__all__ = [
    "qwen35_0_8b_npu",
    "qwen35_0_8b_npu_fsdp2",
    "qwen35_debugmodel_npu",
    "qwen35_debugmodel_npu_fsdp2",
]
