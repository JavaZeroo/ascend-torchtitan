"""Config-tree transforms that turn an upstream recipe into an NPU-runnable one.

``npu_baseline`` is the M1 delta set applied generically (by traversal, not by
rebuilding the model spec), so any upstream recipe — including the 57 upstream
integration-test configs — can be measured on NPU with the *same* baseline.
Everything else in the config (parallelism, AC, compile, checkpointing) is left
untouched: those are the matrix axes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.models.common.attention import FlexAttention, VarlenAttention
from torchtitan.trainer import Trainer

logger = logging.getLogger(__name__)

ATTENTION_OVERRIDE = "ascend_titan.kernels.attention.npu_fusion_attention"


@dataclass
class Applied:
    flex_to_varlen: int = 0
    attention_override: bool = False
    spmd_backend: str | None = None
    chunked_loss_unwrapped: bool = False
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = []
        if self.flex_to_varlen:
            parts.append(f"flex->varlen x{self.flex_to_varlen}")
        if self.attention_override:
            parts.append("override:npu_fusion_attention")
        if self.spmd_backend:
            parts.append(f"spmd_backend:{self.spmd_backend}")
        if self.chunked_loss_unwrapped:
            parts.append("loss:chunked->inner")
        return ", ".join(parts) or "no-op"


def npu_baseline(config: Trainer.Config) -> Applied:
    """Apply the M1 deltas in place. Idempotent."""
    a = Applied()

    # 1. FlexAttention nodes -> VarlenAttention (torch rejects npu in flex: TORCH-1).
    #    Flex-only fields (block_size, kernel_options) are dropped; models that
    #    depend on Flex-only features (sinks, custom mask mods) will fail later and
    #    get attributed, which is the point of measuring them.
    for _fqn, _cfg, parent, attr in list(config.traverse(FlexAttention.Config)):
        new = VarlenAttention.Config()
        if isinstance(parent, list):
            parent[attr] = new  # type: ignore[index]
        else:
            setattr(parent, attr, new)
        a.flex_to_varlen += 1

    # 2. Ascend fused attention on every VarlenAttention node (NPU-1).
    has_varlen = any(True for _ in config.traverse(VarlenAttention.Config))
    if has_varlen and ATTENTION_OVERRIDE not in config.override.imports:
        config.override.imports = [*config.override.imports, ATTENTION_OVERRIDE]
    a.attention_override = has_varlen

    # 3. spmd_types requires DTensor params under dp_mesh_dims on NPU (TT-5).
    if config.parallelism.spmd_backend != "partial_dtensor":
        config.parallelism.spmd_backend = "partial_dtensor"
        a.spmd_backend = "partial_dtensor"
    if getattr(config.debug, "spmd_typechecking", False):
        config.debug.spmd_typechecking = False
        a.notes.append("spmd_typechecking off")

    # 4. ChunkedLossWrapper backward fails on NPU (TT-4): use its inner loss.
    if isinstance(config.loss, ChunkedLossWrapper.Config):
        config.loss = config.loss.loss_fn
        a.chunked_loss_unwrapped = True

    logger.info("[ascend_titan] npu_baseline: %s", a.summary())
    return a
