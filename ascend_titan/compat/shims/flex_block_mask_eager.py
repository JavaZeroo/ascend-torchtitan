"""Build FlexAttention BlockMasks eagerly when inductor has no backend here.

torchtitan compiles the flex path at import time, in three places, with no
config switch and no ``Configurable`` node to override (P0, P6)::

    common/attention.py:      _compiled_create_block_mask = compile(create_block_mask)
    common/attention.py:      FlexAttention._compiled_flex_attn = compile(flex_attention)
    common/vision_encoder.py: compiled_create_block_mask  = compile(create_block_mask)

None of them looks at ``config.compile.enable``: the mask builders are reached by
every model whose inner attention is ``FlexAttention`` and by every vision tower,
and ``FlexAttention.forward`` always calls the compiled attention. On Ascend,
inductor needs Triton-Ascend
(DEP-INDUCTOR); without it the compile raises ``RuntimeError: 0 active
drivers`` before a single step runs -- even though ``create_block_mask`` itself
works perfectly in eager, and torch_npu master makes eager ``flex_attention``
itself work on NPU (fwd + bwd measured, tests/repro/probe_npu_gaps.py).

So the shims substitute upstream's own uncompiled functions -- same functions,
same results, just not traced. This buys reachability, not speed: eager flex
attention and eager mask construction are both much slower than the compiled
versions, and torch itself warns as much. Once Triton-Ascend is installed the
shims step aside on their own.

Feature-gated, not version-gated (P12): as soon as triton reports an active
backend on this machine, both shims return the original compiled callables and
upstream's behaviour is back, with no code change here.

Attribution: TT (unconditional ``torch.compile``, no opt-out) + DEP-INDUCTOR.
"""

import logging

from ascend_titan.compat import shim

logger = logging.getLogger(__name__)

_REASON = (
    "torchtitan compiles create_block_mask unconditionally; on NPU inductor has "
    "no backend without Triton-Ascend (DEP-INDUCTOR), and the eager builder "
    "produces the same BlockMask"
)
_UPSTREAM = "draft:docs/issues/torchtitan.md#compiled-block-mask"
_WHY_NOT_WRAP = (
    "the target *is* the compiled callable; wrapping it would still enter "
    "inductor. The replacement is upstream's own uncompiled function."
)


def _inductor_has_a_backend() -> bool:
    """True when triton can name an active backend for this device.

    ``torch._inductor`` asks ``triton.runtime.driver.active`` while generating
    code; with no NPU-aware triton installed that raises instead of returning,
    which is the failure these shims exist for.
    """
    try:
        from triton.runtime import driver

        return driver.active is not None
    except Exception:  # noqa: BLE001 - any failure means "no usable backend"
        return False


def _eager_or_original(original, where: str):
    if _inductor_has_a_backend():
        logger.info(
            "[shim] flex_block_mask_eager: triton has an active backend, keeping "
            "upstream's compiled mask builder (%s)",
            where,
        )
        return original
    from torch.nn.attention.flex_attention import create_block_mask

    return create_block_mask


@shim(
    target="torchtitan.models.common.attention:_compiled_create_block_mask",
    reason=_REASON,
    upstream=_UPSTREAM,
    kind="replace",
    why_not_wrap=_WHY_NOT_WRAP,
)
def flex_block_mask_eager_lm(original):
    return _eager_or_original(original, "language model")


@shim(
    target="torchtitan.models.common.attention:FlexAttention._compiled_flex_attn",
    reason="FlexAttention.forward always calls a torch.compile'd flex_attention, "
    "regardless of config.compile.enable; torch_npu master makes eager "
    "flex_attention work on NPU (fwd + bwd), inductor does not",
    upstream=_UPSTREAM,
    kind="replace",
    why_not_wrap=_WHY_NOT_WRAP,
)
def flex_attn_eager(original):
    if _inductor_has_a_backend():
        return original
    from torch.nn.attention.flex_attention import flex_attention

    return flex_attention


@shim(
    target="torchtitan.models.common.vision_encoder:compiled_create_block_mask",
    reason=_REASON,
    upstream=_UPSTREAM,
    kind="replace",
    why_not_wrap=_WHY_NOT_WRAP,
)
def flex_block_mask_eager_vision(original):
    return _eager_or_original(original, "vision encoder")
