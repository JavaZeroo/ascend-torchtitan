"""Config-tree transforms that turn an upstream recipe into an NPU-runnable one.

Every delta here must name the issue that makes it necessary and disappear by feature
detection once the fix is present (P12: the goal state is the identity transform).

Two transforms, deliberately separate (P12):

``npu_minimal``  only what an upstream config cannot run without on NPU. This is
                 what the capability matrix applies, so a red cell means "upstream
                 feature X does not work here", never "our perf kernel broke it".
                 The goal state is the identity transform.
``npu_fused``    performance overrides (drop-in fused kernels). Opt-in, never part
                 of a measurement baseline, and numerics may differ at bf16
                 rounding level from the un-fused run.

Both are applied generically by traversal, not by rebuilding the model spec, so any
upstream recipe -- including the upstream integration-test configs -- can be measured
on NPU. Everything else in the config (parallelism, AC, compile, checkpointing) is
left untouched: those are the matrix axes.
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
# Subtrees whose FlexAttention nodes must stay flex: their masks are BlockMasks
# and the surrounding code has no varlen path (kimi_k3 / kimi_k2_7 vision towers).
_KEEP_FLEX_FQNS = ("vision_encoder",)


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


def _flex_attention_is_usable() -> bool:
    """True when upstream's FlexAttention is a *usable* choice on this machine.

    Two conditions, and both matter:

    * torch_npu must have lifted torch's device whitelist (``_validate_device``
      rejects anything outside ``{cuda, cpu, xpu, hpu, mps}``, TORCH-1); master
      patches it and marks the module with ``_npu_device_patched``.
    * inductor must have a backend. Eager flex attention materialises the full
      O(T^2) score matrix -- it passes at op scale (fwd + bwd measured on 910B2,
      tests/repro/probe_npu_gaps.py) and then OOMs at model scale. "Runs" is not
      the same as "usable", so the conversion below stays until Triton-Ascend is
      installed (DEP-INDUCTOR).
    """
    from torch.nn.attention import flex_attention

    if not getattr(flex_attention, "_npu_device_patched", False):
        return False
    try:
        from triton.runtime import driver

        return driver.active is not None
    except Exception:  # noqa: BLE001 - no usable triton backend
        return False


def npu_minimal(config: Trainer.Config) -> Applied:
    """Apply, in place, only the deltas an upstream config cannot run without. Idempotent.

    Every step names the issue that makes it necessary and must disappear by
    feature detection once that issue is fixed (P12). Nothing here may exist for
    performance, and nothing here may hide a torch_npu defect (P9).
    """
    a = Applied()

    # 1. FlexAttention nodes -> VarlenAttention while flex is not usable here
    #    (TORCH-1 + DEP-INDUCTOR, see _flex_attention_is_usable). Flex-only fields
    #    (block_size, kernel_options) are dropped by the conversion; models that
    #    depend on Flex-only features (sinks, custom mask mods) fail later and get
    #    attributed, which is the point of measuring them.
    #    Vision encoders are excluded: their attention is fed a BlockMask built by
    #    `create_block_diagonal_mask`, and there is no varlen mask on that path, so
    #    converting the node just swaps one failure for a worse one
    #    ("attention_masks must be VarlenMetadata, got BlockMask").
    if not _flex_attention_is_usable():
        for _fqn, _cfg, parent, attr in list(config.traverse(FlexAttention.Config)):
            if any(marker in _fqn for marker in _KEEP_FLEX_FQNS):
                a.notes.append(f"flex kept at {_fqn} (no varlen mask on this path)")
                continue
            new = VarlenAttention.Config()
            if isinstance(parent, list):
                parent[attr] = new  # type: ignore[index]
            else:
                setattr(parent, attr, new)
            a.flex_to_varlen += 1

    # An upstream override that replaces a whole attention block (fused_mla) owns
    # everything under it -- the inner attention node and the RoPE node. torchtitan
    # rejects an override that claims a descendant of another override's node
    # ("claims 'layers.0.attention', an ancestor of 'layers.0.attention.inner_attention'"),
    # so neither of our two attention-side overrides may be added on such a config
    # (OURS-9). The block override brings its own implementation of both.
    block_override = any(
        str(imp if isinstance(imp, str) else imp[0]).startswith(_ATTENTION_BLOCK_OVERRIDES)
        for imp in config.override.imports
    )
    if block_override:
        a.notes.append(
            "attention and rope overrides skipped: an upstream override claims the attention block"
        )

    # 2. Ascend fused attention on every VarlenAttention node (NPU-1).
    has_varlen = any(True for _ in config.traverse(VarlenAttention.Config))
    if has_varlen and not block_override and ATTENTION_OVERRIDE not in config.override.imports:
        config.override.imports = [*config.override.imports, ATTENTION_OVERRIDE]
    a.attention_override = has_varlen and not block_override

    # 2b. ComplexRoPE nodes -> real-valued cache (torch_npu cannot index complex, NPU-3).
    has_complex_rope = any(
        type(c) is ComplexRoPE.Config for _f, c, _p, _a in config.traverse(ComplexRoPE.Config)
    )
    if has_complex_rope and not block_override and ROPE_OVERRIDE not in config.override.imports:
        config.override.imports = [*config.override.imports, ROPE_OVERRIDE]
    a.rope_override = has_complex_rope and not block_override

    logger.info("[ascend_titan] npu_minimal: %s", a.summary())
    return a


def npu_fused(config: Trainer.Config) -> Applied:
    """Apply, in place, the drop-in fused-kernel overrides. Idempotent.

    Opt-in only. These change nothing about *whether* a config runs, so they must
    never be part of the measurement baseline: a matrix cell has to be able to
    fail on upstream's own implementation (P12). Numerics differ from the un-fused
    run at bf16 rounding level, which is why the fused recipes carry their own
    golden curves.
    """
    a = Applied()

    # RMSNorm -> npu_rms_norm (pure drop-in: same parameter name, shape and
    # checkpoint layout; Meta + autograd registered in torch_npu).
    has_rms = any(type(c) is RMSNorm.Config for _f, c, _p, _a in config.traverse(RMSNorm.Config))
    if has_rms and RMSNORM_OVERRIDE not in config.override.imports:
        config.override.imports = [*config.override.imports, RMSNORM_OVERRIDE]
    a.rms_norm_override = has_rms

    logger.info("[ascend_titan] npu_fused: %s", a.summary())
    return a
