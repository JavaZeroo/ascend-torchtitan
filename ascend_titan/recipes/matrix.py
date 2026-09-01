"""The capability matrix's dynamic recipe module, and the two generic transforms it applies.

``--module ascend_titan.recipes.matrix --config <upstream.module>__<fn>`` resolves
to the upstream recipe function with :func:`npu_minimal` applied, so every upstream
integration-test config can be launched through the ordinary torchtitan CLI without
copying it. Two suffixes change what is applied:

    ``__stock``   nothing at all -- measures stock upstream on NPU
    ``__fused``   ``npu_minimal`` + ``npu_fused`` -- measures the perf kernels

The default deliberately excludes the fused kernels (P12): a red cell has to mean
"this upstream feature does not work on NPU", not "our drop-in kernel broke it".

``npu_minimal`` and ``npu_fused`` live **here**, next to their only caller, and not
in a module recipes import from. They are generic traversals meant for an upstream
config we ship no recipe for; a model package with its own ``recipes.py`` spells out
its deltas call by call instead (``recipes/deltas.py``), so that reading the recipe
tells you which modules we swapped and which we left alone.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
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
from ascend_titan.recipes.deltas import add_override, flex_to_varlen, swap_override

logger = logging.getLogger(__name__)

SEP = "__"
MODES = ("minimal", "stock", "fused")

# Upstream overrides that replace a whole attention block (which owns the RoPE node);
# adding our RoPE override on top would be a per-node conflict (OURS-9).
_ATTENTION_BLOCK_OVERRIDES = ("torchtitan.overrides.fused_mla.",)
# Subtrees whose FlexAttention nodes must stay flex: their masks are BlockMasks
# and the surrounding code has no varlen path (kimi_k3 / kimi_k2_7 vision towers).
_KEEP_FLEX_FQNS = ("vision_encoder",)


@dataclass
class TransformReport:
    """What a transform did to one config. Logged, and asserted on in tests."""

    flex_to_varlen: int = 0
    attention_override: bool = False
    rope_override: bool = False
    rms_norm_override: bool = False
    gdn_fused_override: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def is_noop(self) -> bool:
        """True when the transform changed nothing about the config."""
        return not (
            self.flex_to_varlen
            or self.attention_override
            or self.rope_override
            or self.rms_norm_override
            or self.gdn_fused_override
        )

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
        return ", ".join(parts) or "no-op"


def npu_minimal(config: Trainer.Config) -> TransformReport:
    """Apply, in place, only the deltas an upstream config cannot run without. Idempotent.

    Every step names the issue that makes it necessary and must disappear by
    feature detection once that issue is fixed (P12). Nothing here may exist for
    performance, and nothing here may hide a torch_npu defect (P9).
    """
    a = TransformReport()

    # 1. FlexAttention -> VarlenAttention while flex is not usable here
    #    (TORCH-1 + the Ascend950-only indirect-memory lowering), except the
    #    vision towers (see flex_to_varlen).
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
    if GDN_OVERRIDE in config.override.imports:
        from ascend_titan.kernels._probe import optional_module

        fla, _err = optional_module("fla_npu")
        if fla is not None:
            swap_override(config, remove=GDN_OVERRIDE, add=GDN_FUSED_OVERRIDE)
            a.gdn_fused_override = True
        else:
            a.notes.append("gdn fused override skipped: fla_npu not installed (ADR-004)")

    if a.is_noop:
        # Nothing here targets a node this config has (qwen3_5, for one, carries no
        # torchtitan.models.common RMSNorm node at all). Running it would measure
        # exactly what `minimal` measures, so it must never be *reported* as a fused
        # result -- say so here, and let the matrix skip the case (P7 / P12).
        a.notes.append(
            "no fused kernel targets any node of this config: a 'fused' run of it "
            "measures exactly what 'minimal' does"
        )
        logger.warning("[ascend_titan] npu_fused: %s", a.notes[-1])
    else:
        logger.info("[ascend_titan] npu_fused: %s", a.summary())
    return a


def encode(fn: Callable[[], Trainer.Config], *, mode: str = "minimal") -> str:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    name = f"{fn.__module__}{SEP}{fn.__name__}"
    return name if mode == "minimal" else f"{name}{SEP}{mode}"


def resolve(name: str) -> Callable[[], Trainer.Config]:
    parts = name.split(SEP)
    mode = "minimal"
    if parts[-1] in ("stock", "fused"):
        mode = parts[-1]
        parts = parts[:-1]
    if len(parts) != 2:
        raise AttributeError(
            f"matrix config must look like '<module>{SEP}<fn>[{SEP}stock|{SEP}fused]', got {name!r}"
        )
    module_path, fn_name = parts
    fn = getattr(importlib.import_module(module_path), fn_name)

    def build() -> Trainer.Config:
        config = fn()
        if mode != "stock":
            npu_minimal(config)
            if mode == "fused":
                npu_fused(config)
        return config

    build.__name__ = name
    build.__qualname__ = name
    return build


def __getattr__(name: str):
    if name.startswith("_") or SEP not in name:
        raise AttributeError(name)
    return resolve(name)
