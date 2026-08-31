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

The two mask builders have a genuine uncompiled twin: ``create_block_mask``
called directly builds the same BlockMask without entering inductor. Those two
shims work.

The third one is subtler. ``torch.nn.attention.flex_attention`` has **no**
uncompiled path::

    with setup_compilation_env() as backend:
        flex_fn = torch.compile(_flex_attention_hop_wrapper, backend=backend,
                                fullgraph=True)

Whenever it is not already under dynamo it compiles itself, so substituting it
for upstream's pre-compiled callable only moves the ``torch.compile`` inside
torch. That turns out to matter anyway: compiling just the HOP wrapper works on
910B2 -- document masks included, forward / LSE / backward all measured -- while
compiling the whole ``flex_attention`` function does not. See
``tests/repro/probe_flex_deterministic.py``.

So this shim does buy something, but it is not what the first version claimed
("910B2 cannot lower a tensor-reading mask_mod" was wrong -- see
docs/capability-matrix.md). The one place the whole function still gets compiled
is ``set_determinism``, which re-assigns the attribute after we set it; that is
handled separately by ``flex_eager_when_deterministic``.

Once Triton-Ascend is installed the shims step aside on their own.

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
    "inductor. For the two mask builders the replacement is upstream's own "
    "uncompiled function; for flex_attention itself there is no such function "
    "(see the module docstring)."
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
    "regardless of config.compile.enable. NOTE: this shim moves the compile "
    "boundary but cannot avoid inductor -- torch's flex_attention compiles "
    "itself (fullgraph=True) when not already under dynamo. Kept because the "
    "smaller graph fails later and more legibly; see the module docstring",
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
