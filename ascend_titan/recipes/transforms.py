"""Config-tree transforms that turn an upstream recipe into an NPU-runnable one.

Every delta here must name the issue that makes it necessary and disappear by feature
detection once the fix is present (P12: the goal state is the identity transform).

``npu_baseline`` is the M1 delta set applied generically (by traversal, not by
rebuilding the model spec), so any upstream recipe — including the 57 upstream
integration-test configs — can be measured on NPU with the *same* baseline.
Everything else in the config (parallelism, AC, compile, checkpointing) is left
untouched: those are the matrix axes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from torchtitan.models.common.attention import FlexAttention, VarlenAttention
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import ComplexRoPE
from torchtitan.trainer import Trainer

logger = logging.getLogger(__name__)

ATTENTION_OVERRIDE = "ascend_titan.kernels.attention.npu_fusion_attention"
ROPE_OVERRIDE = "ascend_titan.kernels.rope.real_cache_rope"
RMSNORM_OVERRIDE = "ascend_titan.kernels.rms_norm.npu_rms_norm"
# Upstream overrides that replace a whole attention block (which owns the RoPE node);
# adding our RoPE override on top would be a per-node conflict (OURS-9).
_ATTENTION_BLOCK_OVERRIDES = ("torchtitan.overrides.fused_mla.",)


@dataclass
class Applied:
    flex_to_varlen: int = 0
    attention_override: bool = False
    rope_override: bool = False
    rms_norm_override: bool = False
    spmd_backend: str | None = None
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = []
        if self.flex_to_varlen:
            parts.append(f"flex->varlen x{self.flex_to_varlen}")
        if self.attention_override:
            parts.append("override:npu_fusion_attention")
        if self.rope_override:
            parts.append("override:real_cache_rope")
        if self.rms_norm_override:
            parts.append("override:npu_rms_norm")
        if self.spmd_backend:
            parts.append(f"spmd_backend:{self.spmd_backend}")
        return ", ".join(parts) or "no-op"


def _torch_fsdp_reads_spmd_types() -> bool:
    """True when torch's FSDP2 builds DTensor shards from spmd_types annotations
    (``torch.distributed._is_spmd_types_available`` exists; nightly >= 2.14.0.dev)."""
    import torch.distributed as dist

    return hasattr(dist, "_is_spmd_types_available")


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

    # 2b. ComplexRoPE nodes -> real-valued cache (torch_npu cannot index complex, NPU-3).
    has_complex_rope = any(
        type(c) is ComplexRoPE.Config for _f, c, _p, _a in config.traverse(ComplexRoPE.Config)
    )
    block_override = any(
        str(imp if isinstance(imp, str) else imp[0]).startswith(_ATTENTION_BLOCK_OVERRIDES)
        for imp in config.override.imports
    )
    if block_override:
        a.notes.append("rope override skipped: an upstream override claims the attention block")
    elif has_complex_rope and ROPE_OVERRIDE not in config.override.imports:
        config.override.imports = [*config.override.imports, ROPE_OVERRIDE]
    a.rope_override = has_complex_rope and not block_override

    # 2c. RMSNorm -> npu_rms_norm (fused kernel; pure drop-in).
    has_rms = any(type(c) is RMSNorm.Config for _f, c, _p, _a in config.traverse(RMSNorm.Config))
    if has_rms and RMSNORM_OVERRIDE not in config.override.imports:
        config.override.imports = [*config.override.imports, RMSNORM_OVERRIDE]
    a.rms_norm_override = has_rms

    # 3. spmd_types needs FSDP2 to read spmd_types annotations, which only torch
    #    nightly (>= 2.14.0.dev) does (TT-5 / TORCH-6: a torch-version gap, not an
    #    NPU one). Feature-check, like upstream's check_if_feature_in_pytorch: on a
    #    torch that has the integration the upstream default is left untouched.
    if not _torch_fsdp_reads_spmd_types():
        if config.parallelism.spmd_backend != "partial_dtensor":
            config.parallelism.spmd_backend = "partial_dtensor"
            a.spmd_backend = "partial_dtensor"
        if getattr(config.debug, "spmd_typechecking", False):
            config.debug.spmd_typechecking = False
            a.notes.append("spmd_typechecking off")

    # (Removed 2026-08-30: the ChunkedLossWrapper backward failure TT-4 does not exist on the
    #  NIGHTLY track -- torch 2.15.0.dev + torch_npu master, 1 NPU and FSDP2x2 both pass -- and
    #  unwrapping the loss here was a P1/P9 violation: it worked around a suspected torch_npu
    #  defect inside the baseline.)

    logger.info("[ascend_titan] npu_baseline: %s", a.summary())
    return a
