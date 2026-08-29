"""Qwen3 recipes. M1 target: ``qwen3_debugmodel_npu`` runs 10 steps on NPU.

validated: torchtitan=<unvalidated> torch=<unvalidated> torch_npu=<unvalidated> CANN=<unvalidated>
(the line above is rewritten by CI once a run is green; see docs/capability-matrix.md)

Attention on NPU -- known risk, discovered by the CPU recipe test:
upstream removed the ``sdpa`` backend for language models
(``torchtitan/models/common/config_utils.py:97``). Only ``flex`` (FlexAttention,
torch.compile + inductor) and ``varlen`` (``torch.nn.attention.varlen``) remain,
and neither is a plain-eager path. If both are 🔴 on NPU, the first L1 override
(an Ascend inner-attention module on ``FlexAttention.Config`` /
``VarlenAttention.Config``) moves from M3 into M1. Both are matrix cells.
"""

from torchtitan.models.qwen3 import model_registry
from torchtitan.models.qwen3.config_registry import qwen3_debugmodel
from torchtitan.trainer import Trainer


def qwen3_debugmodel_npu() -> Trainer.Config:
    """Upstream ``qwen3_debugmodel`` with the smallest delta for an NPU smoke run.
    Each delta is a matrix cell, tracked in docs/capability-matrix.md."""
    config = qwen3_debugmodel()

    # (no attention delta) upstream default attn_backend="flex" is kept on purpose:
    # matrix cell attention/flex. See module docstring.

    # DELTA 1: no checkpoint I/O in the smoke run (DCP on NPU is its own cell).
    config.checkpoint.enable = False

    return config


def qwen3_debugmodel_npu_varlen() -> Trainer.Config:
    """Matrix cell attention/varlen."""
    config = qwen3_debugmodel_npu()
    # DELTA 2: attention backend flex -> varlen (torch.nn.attention.varlen).
    config.model_spec = model_registry("debugmodel", attn_backend="varlen")
    return config


def qwen3_debugmodel_npu_fsdp2() -> Trainer.Config:
    """M1 acceptance path (3): real multi-device FSDP2."""
    config = qwen3_debugmodel_npu()
    # DELTA 3: 2-way FSDP.
    config.parallelism.data_parallel_shard_degree = 2
    return config
