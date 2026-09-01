"""Qwen3.5 on Ascend. Guide: models/qwen3_5/README.md.

Any upstream flavor works without a function being written for it:
``--config <upstream_flavor>_npu`` (e.g. ``qwen35_27b_npu``) is resolved by
``_auto.npu_entry_points`` from ``recipes.npu_deltas``. The names imported below
are the curated ones -- they add something beyond the family deltas (real assets,
a golden-frozen smoke config, a parallel layout) and take precedence.
"""

from torchtitan.models.qwen3_5 import config_registry as _upstream

from ascend_titan.models._auto import npu_entry_points
from ascend_titan.models.qwen3_5.recipes import (
    npu_deltas,
    qwen35_0_8b_npu,
    qwen35_0_8b_npu_fsdp2,
    qwen35_0_8b_npu_fused,
    qwen35_debugmodel_npu,
    qwen35_debugmodel_npu_fsdp2,
    qwen35_debugmodel_npu_text,
)

__getattr__, __dir__ = npu_entry_points(_upstream, npu_deltas)

__all__ = [
    "qwen35_0_8b_npu",
    "qwen35_0_8b_npu_fsdp2",
    "qwen35_0_8b_npu_fused",
    "qwen35_debugmodel_npu",
    "qwen35_debugmodel_npu_fsdp2",
    "qwen35_debugmodel_npu_text",
]
