"""Llama 3 on Ascend (zero-override reference). Guide: models/llama3/README.md."""

from torchtitan.models.llama3 import config_registry as _upstream

from ascend_titan.models._auto import npu_entry_points
from ascend_titan.models.llama3.recipes import (
    llama3_debugmodel_stock_npu,
    llama3_debugmodel_stock_npu_fsdp2,
    npu_deltas,
)

__all__ = ["llama3_debugmodel_stock_npu", "llama3_debugmodel_stock_npu_fsdp2"]

# 任何上游 flavor 都有 `<flavor>_npu` 入口（见 models/_auto.py）；
# 上面显式导入的手写 recipe 优先。
__getattr__, __dir__ = npu_entry_points(_upstream, npu_deltas)
