"""torchtitan 无条件编译 mask 构建；两种情况下它编不出来。

torchtitan 在两处 import 时就把 ``create_block_mask`` 编译掉，都不看
``config.compile.enable``::

    common/attention.py:      _compiled_create_block_mask = compile(create_block_mask)
    common/vision_encoder.py: compiled_create_block_mask  = compile(create_block_mask)

inductor 本身是好的——Triton-Ascend 在基线里，``scripts/install_triton.sh`` 的验收会
编出前反向内核。编不出来的是两种具体情况：

**(1) 掩码本身需要间接寻址**（和 ``flex_attention_eager`` 同一个硬件门）。
document mask 会 index 一个 segment-id 张量，torch_npu 只在
``inductor_indirect_memory_mode`` 打开时才 lower 间接寻址，该开关只在 Ascend950 赋值。
在 910B2 上这个编译要么把 autotune 逼进 SIMT 路径（NPU-11 + TA-1，报
``bishengir-compile: Unknown command line argument '--pure-simt'``），要么走 SIMD 时抛
``LLVM ERROR: Failed to obtain op buffer shape size which should be static.``
实测 2026-09-02：stock ``qwen35_debugmodel`` 死在这；换 eager 构建器后 2 步跑完
（``loss 12.84193 -> 12.59930``）。

kimi_k3 的掩码不落在这个模式里，所以它的编译版构建器能过——**这是模型相关的**，
不能按模型下结论，只能按芯片能力下结论。

**(2) 确定性模式**：torch_npu 的 inductor 在 ``deterministic`` 下禁止未经认证的
autotune benchmark，直接抛

    RuntimeError: In the deterministic mode of Inductor, we will avoid those
    benchmarkings that would cause non-deterministic results.

而 ``scripts/check_golden.sh`` 正是加 ``--debug.deterministic`` 跑的——不修这条，
用 flex 的模型就录不了逐位 golden（实测 2026-09-01：删掉这两条 shim 后
``check_golden.sh kimi_k3_debugmodel_npu`` 直接 "no loss lines captured"）。

所以这两条 shim 换上上游自己的未编译函数——同一个函数、同样的结果，只是不进 inductor。
消失条件：(1) 换到能 lower 间接寻址的芯片（自动让路，特性探测）；(2) 上游给这个编译加开关
（``config.compile.enable`` 就够），或 torch_npu 把 mask 构建的 autotune 标成 vetted。

归因：TT（无条件 ``torch.compile``，没有开关）。按 P10 不向 github.com/pytorch 提 issue，
只在 ``docs/issues/torchtitan.md`` 记录。
"""

import logging

from ascend_titan.compat import shim

logger = logging.getLogger(__name__)

_REASON = (
    "torchtitan 无条件编译 create_block_mask；910B2 lower 不了 document mask 的间接寻址，"
    "确定性模式下 torch_npu 的 inductor 还会拒绝未经认证的 autotune benchmark"
    "（golden 就是确定性跑的）；eager 的构建器给出同样的 BlockMask"
)
_UPSTREAM = "draft:docs/issues/torchtitan.md#compiled-block-mask"
_WHY_NOT_WRAP = (
    "包装没有意义：目标本身就是那个编译后的 callable，包一层照样进 inductor。"
    "替换物是上游自己的未编译函数。"
)


def _inductor_is_deterministic() -> bool:
    """True when inductor runs in the mode that rejects un-vetted autotuning."""
    import torch

    return bool(getattr(torch._inductor.config, "deterministic", False))


def _can_lower_a_document_mask() -> bool:
    """True when inductor can lower the indirect memory access a document mask needs.

    Same probe as ``flex_attention_eager`` -- and it must be the same, because the
    compiled mask builder and the compiled attention fail for the same reason.
    Keeping this shim on the deterministic flag alone was too narrow: it let stock
    ``qwen35_debugmodel`` die on the mask builder in an ordinary run.
    """
    from torch_npu._inductor import config as npu_inductor_config

    return getattr(npu_inductor_config, "inductor_indirect_memory_mode", None) is not None


def _must_build_eagerly() -> bool:
    """Either reason is enough; both are per-run, so this is asked per call."""
    return not _can_lower_a_document_mask() or _inductor_is_deterministic()


def _eager_or_original(original, where: str):
    """Decide **per call**, not at shim-application time.

    Shims are applied in ``setup()``, long before torchtitan's ``set_determinism``
    turns the flag on, so deciding once at apply time would always read False and
    the golden runs would break. The replacement is therefore a dispatcher.
    """

    def build_block_mask(*args, **kwargs):
        if not _must_build_eagerly():
            return original(*args, **kwargs)
        from torch.nn.attention.flex_attention import create_block_mask

        logger.debug("[shim] flex_block_mask_eager: eager builder (%s)", where)
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
