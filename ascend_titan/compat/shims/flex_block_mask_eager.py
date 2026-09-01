"""torchtitan 无条件编译 mask 构建，而确定性模式下 inductor 拒绝编它。

torchtitan 在两处 import 时就把 ``create_block_mask`` 编译掉，都不看
``config.compile.enable``::

    common/attention.py:      _compiled_create_block_mask = compile(create_block_mask)
    common/vision_encoder.py: compiled_create_block_mask  = compile(create_block_mask)

inductor 本身是好的——Triton-Ascend 在基线里，``scripts/install_triton.sh`` 的验收会
编出前反向内核，非确定性运行下这两处编译版也跑得通（实测：关掉这两条 shim，kimi_k3
10 步正常跑完）。**问题只出在确定性模式**：torch_npu 的 inductor 在
``deterministic`` 下禁止未经认证的 autotune benchmark，直接抛

    RuntimeError: In the deterministic mode of Inductor, we will avoid those
    benchmarkings that would cause non-deterministic results.

而 ``scripts/check_golden.sh`` 正是加 ``--debug.deterministic`` 跑的——不修这条，
用 flex 的模型就录不了逐位 golden（实测 2026-09-01：删掉这两条 shim 后
``check_golden.sh kimi_k3_debugmodel_npu`` 直接 "no loss lines captured"）。

所以这两条 shim 换上上游自己的未编译函数——同一个函数、同样的结果，只是不进 inductor。
消失条件：上游给这个编译加开关（``config.compile.enable`` 就够），或 torch_npu 把
mask 构建的 autotune 标成 vetted。

归因：TT（无条件 ``torch.compile``，没有开关）。按 P10 不向 github.com/pytorch 提 issue，
只在 ``docs/issues/torchtitan.md`` 记录。
"""

import logging

from ascend_titan.compat import shim

logger = logging.getLogger(__name__)

_REASON = (
    "torchtitan 无条件编译 create_block_mask；确定性模式下 torch_npu 的 inductor "
    "拒绝未经认证的 autotune benchmark（golden 就是确定性跑的）；eager 的构建器给出同样的 BlockMask"
)
_UPSTREAM = "draft:docs/issues/torchtitan.md#compiled-block-mask"
_WHY_NOT_WRAP = (
    "包装没有意义：目标本身就是那个编译后的 callable，包一层照样进 inductor。"
    "替换物是上游自己的未编译函数。"
)


def _inductor_is_deterministic() -> bool:
    """True when inductor runs in the mode that rejects un-vetted autotuning.

    Not "is there a triton backend": Triton-Ascend is in the baseline and the
    compiled mask builder works fine in an ordinary run (measured). The failure is
    specific to ``--debug.deterministic``, which is exactly how goldens are recorded.
    """
    import torch

    return bool(getattr(torch._inductor.config, "deterministic", False))


def _eager_or_original(original, where: str):
    """Decide **per call**, not at shim-application time.

    Shims are applied in ``setup()``, long before torchtitan's ``set_determinism``
    turns the flag on, so deciding once at apply time would always read False and
    the golden runs would break. The replacement is therefore a dispatcher.
    """

    def build_block_mask(*args, **kwargs):
        if not _inductor_is_deterministic():
            return original(*args, **kwargs)
        from torch.nn.attention.flex_attention import create_block_mask

        logger.debug("[shim] flex_block_mask_eager: deterministic run, eager builder (%s)", where)
        return create_block_mask(*args, **kwargs)

    return build_block_mask


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
