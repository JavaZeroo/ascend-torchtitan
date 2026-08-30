"""Llama 3 on Ascend (zero-override reference). Guide: models/llama3/README.md."""

from ascend_titan.models.llama3.recipes import (
    llama3_debugmodel_stock_npu,
    llama3_debugmodel_stock_npu_fsdp2,
)

__all__ = ["llama3_debugmodel_stock_npu", "llama3_debugmodel_stock_npu_fsdp2"]
