"""Qwen3 recipes. M1 target: ``qwen3_debugmodel_npu`` runs 10 steps on NPU.

validated: torchtitan=13da2d77c torch=2.13.0 torch_npu=2.13.0rc1 CANN=9.1.0 date=2026-08-29
validated: torchtitan=13da2d77c torch=2.12.0 torch_npu=2.12.0 CANN=9.1.0 date=2026-08-29
(rewritten by CI when a run is green; golden curves in tests/assets/losses/npu/)

Why the attention delta exists (measured 2026-08-29, torch 2.12.0 / torch_npu 2.12.0):
upstream language models offer only ``flex`` and ``varlen`` inner attention
(``sdpa`` was removed, ``config_utils.py:97``). ``flex`` is rejected by torch on
npu devices; ``varlen`` needs ``aten::_flash_attention_forward`` which torch_npu
does not provide. Hence the stock model cannot run on NPU at all, and the
``ascend_titan.kernels.attention`` override is part of the M1 baseline. The
``_stock_*`` recipes below exist only to keep those two matrix cells measurable.
"""

from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3 import model_registry
from torchtitan.models.qwen3.config_registry import qwen3_debugmodel
from torchtitan.trainer import Trainer

ATTENTION_OVERRIDE = "ascend_titan.kernels.attention.npu_fusion_attention"
RMSNORM_OVERRIDE = "ascend_titan.kernels.rms_norm.npu_rms_norm"
SWIGLU_OVERRIDE = "ascend_titan.kernels.swiglu.npu_fused_swiglu"
ROPE_COSSIN_OVERRIDE = "ascend_titan.kernels.rope.npu_rotary_cossin"


def qwen3_debugmodel_npu() -> Trainer.Config:
    """Upstream ``qwen3_debugmodel`` + the minimal deltas for an NPU smoke run.
    Each delta is a matrix cell, tracked in docs/capability-matrix.md."""
    config = qwen3_debugmodel()

    # DELTA 1: inner attention = varlen node + Ascend fused-attention override.
    # (matrix: attention/ascend_fusion)
    config.model_spec = model_registry("debugmodel", attn_backend="varlen")
    config.override.imports = [ATTENTION_OVERRIDE]

    # DELTA 2: spmd_types -> partial_dtensor. Under spmd_types, fully_shard(dp_mesh_dims=)
    # requires DTensor params; on torch 2.12 + NPU the params arrive plain and FSDP
    # raises. partial_dtensor is the upstream-supported alternative (P0, not a shim).
    # (matrix: parallel/spmd_types)
    config.parallelism.spmd_backend = "partial_dtensor"

    # DELTA 3: no checkpoint I/O in the smoke run (DCP on NPU is its own cell).
    config.checkpoint.enable = False

    # DELTA 4: plain CrossEntropyLoss instead of ChunkedLossWrapper. The chunked
    # loss (since upstream #4143, 2026-08-13) drives FSDP's lm_head unshard by hand
    # and its backward hits "data is not allocated yet" on torch 2.12 + NPU.
    # (matrix: loss/chunked)
    assert config.model_spec is not None
    config.loss = CrossEntropyLoss.Config(global_vocab_size=decoder_vocab_size(config.model_spec))

    return config


def qwen3_debugmodel_npu_fsdp2() -> Trainer.Config:
    """M1 acceptance path (3): real multi-device FSDP2."""
    config = qwen3_debugmodel_npu()
    # DELTA 5: 2-way FSDP.
    config.parallelism.data_parallel_shard_degree = 2
    return config


def qwen3_debugmodel_stock_flex() -> Trainer.Config:
    """Matrix cell attention/flex: upstream default, no override. Expected 🔴."""
    config = qwen3_debugmodel_npu()
    config.model_spec = model_registry("debugmodel", attn_backend="flex")
    config.override.imports = []
    return config


def qwen3_debugmodel_stock_varlen() -> Trainer.Config:
    """Matrix cell attention/varlen: upstream varlen kernel, no override. Expected 🔴."""
    config = qwen3_debugmodel_npu()
    config.override.imports = []
    return config


def qwen3_debugmodel_npu_chunked_loss() -> Trainer.Config:
    """Matrix cell loss/chunked: upstream ChunkedLossWrapper (the upstream default)."""
    config = qwen3_debugmodel_npu()
    config.loss = qwen3_debugmodel().loss
    return config


def qwen3_debugmodel_npu_fused_norm() -> Trainer.Config:
    """Matrix cell norm/npu_rms_norm: M1 recipe + the fused RMSNorm override.
    Kept out of ``qwen3_debugmodel_npu`` so the frozen golden curve stays valid."""
    config = qwen3_debugmodel_npu()
    config.override.imports = [*config.override.imports, RMSNORM_OVERRIDE]
    return config


def qwen3_debugmodel_npu_fused() -> Trainer.Config:
    """All zero-build torch_npu fused kernels: RMSNorm + fused SwiGLU + rotary kernel.
    The perf recipe; numerics differ from golden at bf16-rounding level."""
    config = qwen3_debugmodel_npu()
    config.override.imports = [
        *config.override.imports,
        RMSNORM_OVERRIDE,
        SWIGLU_OVERRIDE,
        ROPE_COSSIN_OVERRIDE,
    ]
    return config
