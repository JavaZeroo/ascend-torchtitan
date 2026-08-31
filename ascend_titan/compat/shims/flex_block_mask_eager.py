"""torchtitan 无条件编译 mask 构建，昇腾上没有 inductor 后端时会直接崩。

torchtitan 在两处 import 时就把 ``create_block_mask`` 编译掉，都不看
``config.compile.enable``::

    common/attention.py:      _compiled_create_block_mask = compile(create_block_mask)
    common/vision_encoder.py: compiled_create_block_mask  = compile(create_block_mask)

它们被每个 inner attention 是 ``FlexAttention`` 的模型和每个视觉塔用到。昇腾上
inductor 需要 Triton-Ascend（DEP-INDUCTOR），没装时编译在第一步之前就抛
``RuntimeError: 0 active drivers``——而 ``create_block_mask`` 本身 eager 跑得好好的。
实测：关掉这两条 shim，kimi_k3 立刻死在这个错误上。

所以这两条 shim 换上上游自己的未编译函数——同一个函数、同样的结果，只是不进 inductor。
装上 Triton-Ascend 后它们自动让位。

``FlexAttention._compiled_flex_attn`` **不需要**同等处理：torchtitan 默认的
``torch.compile(flex_attention)`` 在昇腾上是通的（实测 kimi_k3 rc=0）。唯一的例外是
确定性模式，那条由 ``flex_eager_when_deterministic`` 处理。

归因：TT（无条件 ``torch.compile``，没有开关）+ DEP-INDUCTOR。
"""

import logging

from ascend_titan.compat import shim

logger = logging.getLogger(__name__)

_REASON = (
    "torchtitan 无条件编译 create_block_mask；昇腾上没有 Triton-Ascend 就没有 "
    "inductor 后端（DEP-INDUCTOR），而 eager 的构建器给出同样的 BlockMask"
)
_UPSTREAM = "draft:docs/issues/torchtitan.md#compiled-block-mask"
_WHY_NOT_WRAP = (
    "包装没有意义：目标本身就是那个编译后的 callable，包一层照样进 inductor。"
    "替换物是上游自己的未编译函数。"
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
    target="torchtitan.models.common.vision_encoder:compiled_create_block_mask",
    reason=_REASON,
    upstream=_UPSTREAM,
    kind="replace",
    why_not_wrap=_WHY_NOT_WRAP,
)
def flex_block_mask_eager_vision(original):
    return _eager_or_original(original, "vision encoder")
