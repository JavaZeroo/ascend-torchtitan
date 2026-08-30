"""Qwen3.5 recipes.

status: 🔴 DEP-FLA -- ``torchtitan.models.qwen3_5.__init__`` imports ``gdn.py``,
which imports ``fla`` (CUDA-only Triton kernels for the gated delta net) at
module level. The import below therefore fails on Ascend today, loudly and on
purpose (P14): a missing dependency is not something to paper over, and the
Ascend replacement (fla-npu) is an L1 task on the roadmap (M4).

The recipes are written and kept minimal so that the day ``fla`` has an Ascend
implementation, this file is the only thing that has to be re-run -- not
rewritten. Deltas mirror the qwen3 reference path.

Full guide: ascend_titan/models/qwen3_5/README.md
"""

from torchtitan.models.qwen3_5 import model_registry
from torchtitan.models.qwen3_5.config_registry import qwen35_debugmodel
from torchtitan.trainer import Trainer

ATTENTION_OVERRIDE = "ascend_titan.kernels.attention.npu_fusion_attention"


def qwen35_debugmodel_npu() -> Trainer.Config:
    """Upstream ``qwen35_debugmodel`` + the minimal deltas for an NPU run."""
    config = qwen35_debugmodel()

    # DELTA 1: inner attention = varlen node + Ascend fused-attention override
    # (model-level flex needs inductor/Triton-Ascend). Mirrors qwen3 DELTA 1.
    config.model_spec = model_registry("debugmodel", attn_backend="varlen")
    config.override.imports = [ATTENTION_OVERRIDE]

    # DELTA 2: no checkpoint I/O in a smoke run (DCP on NPU is its own cell).
    config.checkpoint.enable = False

    return config


def qwen35_debugmodel_npu_fsdp2() -> Trainer.Config:
    """2-way FSDP2."""
    config = qwen35_debugmodel_npu()
    config.parallelism.data_parallel_shard_degree = 2
    return config
