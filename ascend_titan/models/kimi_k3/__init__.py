"""Kimi K3 on Ascend. Guide: models/kimi_k3/README.md."""

from torchtitan.models.kimi_k3 import config_registry as _upstream

from ascend_titan.models._auto import npu_entry_points
from ascend_titan.models.kimi_k3.recipes import (
    kimi_k3_debugmodel_npu,
    kimi_k3_debugmodel_npu_fused,
    npu_deltas,
)

__all__ = ["kimi_k3_debugmodel_npu", "kimi_k3_debugmodel_npu_fused"]

# 任何上游 flavor 都有 `<flavor>_npu` 入口（见 models/_auto.py）；
# 上面显式导入的手写 recipe 优先。
__getattr__, __dir__ = npu_entry_points(_upstream, npu_deltas)
