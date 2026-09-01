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
from collections.abc import Sequence
from dataclasses import dataclass, field

from torchtitan.models.common.attention import FlexAttention, VarlenAttention
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import ComplexRoPE
from torchtitan.trainer import Trainer

from ascend_titan.kernels import (
    ATTENTION_OVERRIDE,
    GDN_FUSED_OVERRIDE,
    GDN_OVERRIDE,
    RMSNORM_OVERRIDE,
    ROPE_OVERRIDE,
)

logger = logging.getLogger(__name__)
# Upstream overrides that replace a whole attention block (which owns the RoPE node);
# adding our RoPE override on top would be a per-node conflict (OURS-9).
_ATTENTION_BLOCK_OVERRIDES = ("torchtitan.overrides.fused_mla.",)
# Subtrees whose FlexAttention nodes must stay flex: their masks are BlockMasks
# and the surrounding code has no varlen path (kimi_k3 / kimi_k2_7 vision towers).
_KEEP_FLEX_FQNS = ("vision_encoder",)

# GDN_OVERRIDE (plain-torch chunk recurrence, kernels/gdn.py) and GDN_FUSED_OVERRIDE
# (fla-npu AscendC, kernels/gdn_fla.py, R5) claim the same InnerGatedDeltaNet.Config
# node, so torchtitan rejects both being active at once. They live in two modules on
# purpose: gdn.py is a base dependency, gdn_fla.py registers only when fla_npu
# imports (ADR-004). Opting into fused is therefore a swap (swap_override), never a
# stack. Keep this pair the only place that owns the sibling relationship.


def add_override(config: Trainer.Config, target: str) -> bool:
    """Activate an override target, in place. Idempotent; ``True`` if it was added."""
    if target in config.override.imports:
        return False
    config.override.imports = [*config.override.imports, target]
    return True


def swap_override(config: Trainer.Config, *, remove: str, add: str) -> bool:
    """Replace one override target with a mutually-exclusive sibling, in place.

    Two overrides that claim the same ``Configurable.Config`` node cannot both be
    active (torchtitan raises a per-node conflict), so activating a drop-in
    variant means removing the original and appending the variant -- never
    stacking. Idempotent: an already-swapped list is left untouched. Returns
    ``True`` when ``remove`` was present (and thus removed).
    """
    if remove not in config.override.imports:
        return False
    config.override.imports = [imp for imp in config.override.imports if imp != remove]
    add_override(config, add)
    return True


@dataclass
class TransformReport:
    flex_to_varlen: int = 0
    attention_override: bool = False
    rope_override: bool = False
    rms_norm_override: bool = False
    gdn_fused_override: bool = False
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
        if self.gdn_fused_override:
            parts.append("override:npu_gated_delta_net_fused")
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


def flex_to_varlen(config: Trainer.Config, *, keep: Sequence[str] = ()) -> list[str]:
    """Convert FlexAttention nodes to VarlenAttention, in place. Returns the fqns converted.

    A **no-op once flex is usable on this machine** (feature detection, see
    :func:`_flex_attention_is_usable`) -- that is this delta's disappearance
    condition (P12), not a version check. Flex-only fields (block_size,
    kernel_options) are dropped by the conversion; a model that depends on
    Flex-only features (sinks, custom mask mods) fails later and gets attributed,
    which is the point of measuring it.

    ``keep`` holds fqn substrings whose nodes must stay flex. The case that
    exists is a vision tower: its attention is fed a ``BlockMask`` built by
    ``create_block_diagonal_mask``, while ``VarlenAttention.forward`` asserts
    ``isinstance(attention_masks, VarlenMetadata)`` -- so converting that node
    swaps a clear failure ("flex needs inductor") for a confusing one deep inside
    the tower ("must be VarlenMetadata but got BlockMask").
    """
    if _flex_attention_is_usable():
        return []
    converted: list[str] = []
    for fqn, _cfg, parent, attr in list(config.traverse(FlexAttention.Config)):
        if any(marker in fqn for marker in keep):
            continue
        new = VarlenAttention.Config()
        if isinstance(parent, list):
            parent[attr] = new  # type: ignore[index]
        else:
            setattr(parent, attr, new)
        converted.append(fqn)
    return converted


def npu_minimal(config: Trainer.Config) -> TransformReport:
    """Apply, in place, only the deltas an upstream config cannot run without. Idempotent.

    **This is the capability matrix's transform, not a recipe's.** It exists to
    carry an arbitrary *upstream* config -- one we ship no recipe for -- onto NPU
    generically, so a red cell means "upstream feature X does not work here".
    A model package that has its own ``recipes.py`` must instead spell out its
    deltas one call at a time (``flex_to_varlen`` / ``add_override``), so that
    reading the recipe tells you exactly which modules we swapped and which we
    left alone (``tests/unit/test_models_registry.py`` enforces this).

    Every step names the issue that makes it necessary and must disappear by
    feature detection once that issue is fixed (P12). Nothing here may exist for
    performance, and nothing here may hide a torch_npu defect (P9).
    """
    a = TransformReport()

    # 1. FlexAttention -> VarlenAttention while flex is not usable here
    #    (TORCH-1 + DEP-INDUCTOR), except the vision towers (see flex_to_varlen).
    kept = [
        fqn
        for fqn, *_ in config.traverse(FlexAttention.Config)
        if any(marker in fqn for marker in _KEEP_FLEX_FQNS)
    ]
    a.flex_to_varlen = len(flex_to_varlen(config, keep=_KEEP_FLEX_FQNS))
    a.notes += [f"flex kept at {fqn} (no varlen mask on this path)" for fqn in kept]

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
    a.attention_override = has_varlen and not block_override
    if a.attention_override:
        add_override(config, ATTENTION_OVERRIDE)

    # 2b. ComplexRoPE nodes -> real-valued cache (torch_npu cannot index complex, NPU-3).
    has_complex_rope = any(
        type(c) is ComplexRoPE.Config for _f, c, _p, _a in config.traverse(ComplexRoPE.Config)
    )
    a.rope_override = has_complex_rope and not block_override
    if a.rope_override:
        add_override(config, ROPE_OVERRIDE)

    logger.info("[ascend_titan] npu_minimal: %s", a.summary())
    return a


def npu_fused(config: Trainer.Config) -> TransformReport:
    """Apply, in place, the drop-in fused-kernel overrides. Idempotent.

    Opt-in only. These change nothing about *whether* a config runs, so they must
    never be part of the measurement baseline: a matrix cell has to be able to
    fail on upstream's own implementation (P12). Numerics differ from the un-fused
    run at bf16 rounding level, which is why the fused recipes carry their own
    golden curves.
    """
    a = TransformReport()

    # RMSNorm -> npu_rms_norm (pure drop-in: same parameter name, shape and
    # checkpoint layout; Meta + autograd registered in torch_npu).
    a.rms_norm_override = any(
        type(c) is RMSNorm.Config for _f, c, _p, _a in config.traverse(RMSNorm.Config)
    )
    if a.rms_norm_override:
        add_override(config, RMSNORM_OVERRIDE)

    # Gated DeltaNet -> fla-npu fused chunk recurrence (R5). Optional add-on: the
    # override registers only when fla_npu is importable, so adding it here is a
    # no-op on a baseline without the wheel (ADR-004). Detect the InnerGatedDeltaNet
    # node by the plain-torch override already being present (the same single-subtree
    # override that qwen3_5 recipes install), so we never claim the node independently.
    has_gdn = GDN_OVERRIDE in config.override.imports
    if has_gdn:
        from ascend_titan.kernels._probe import optional_module

        _fla, _err = optional_module("fla_npu")
        if _fla is not None:
            swap_override(config, remove=GDN_OVERRIDE, add=GDN_FUSED_OVERRIDE)
            a.gdn_fused_override = True
        else:
            a.notes.append("gdn fused override skipped: fla_npu not installed (ADR-004)")

    logger.info("[ascend_titan] npu_fused: %s", a.summary())
    return a
