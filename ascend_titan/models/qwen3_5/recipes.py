"""Qwen3.5 recipes.

``fla`` (flash-linear-attention) is a plain pip dependency -- ``fla-core`` installs
on aarch64 and imports fine, so the model package loads. Its *kernels* are
another matter: they are Triton written for CUDA and ``bishengir-compile``
rejects them even with Triton-Ascend installed. The gated delta net therefore
runs through ``ascend_titan.kernels.gdn``, which selects attn_gym's own
device-agnostic reference recurrence -- same math, different implementation,
exactly like the KDA override for kimi_k3.

Full guide: ascend_titan/models/qwen3_5/README.md
"""

from torchtitan.models.qwen3_5 import model_registry
from torchtitan.models.qwen3_5.config_registry import qwen35_debugmodel
from torchtitan.trainer import Trainer

ATTENTION_OVERRIDE = "ascend_titan.kernels.attention.npu_fusion_attention"
GDN_OVERRIDE = "ascend_titan.kernels.gdn.npu_gated_delta_net"


def qwen35_debugmodel_npu() -> Trainer.Config:
    """Upstream ``qwen35_debugmodel`` + the minimal deltas for an NPU run."""
    config = qwen35_debugmodel()

    # DELTA 1: inner attention = varlen node + Ascend fused-attention override
    # (model-level flex needs inductor/Triton-Ascend). Mirrors qwen3 DELTA 1.
    config.model_spec = model_registry("debugmodel", attn_backend="varlen")
    config.override.imports = [ATTENTION_OVERRIDE]

    # DELTA 2: gated delta net on attn_gym's reference recurrence (fla's Triton
    # kernels do not compile for Ascend). 消失条件：昇腾有了 GDN 融合算子。
    config.override.imports = [*config.override.imports, GDN_OVERRIDE]

    # DELTA 3: no checkpoint I/O in a smoke run (DCP on NPU is its own cell).
    config.checkpoint.enable = False

    return config


def qwen35_debugmodel_npu_fsdp2() -> Trainer.Config:
    """2-way FSDP2."""
    config = qwen35_debugmodel_npu()
    config.parallelism.data_parallel_shard_degree = 2
    return config
