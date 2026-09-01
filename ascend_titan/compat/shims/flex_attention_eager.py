"""910B2 上编译版 flex attention 编不出来，所以让它走 eager。

torchtitan 在 ``distributed/utils.py::set_determinism`` 里给 ``FlexAttention``
装编译版实现（非 ROCm 分支）::

    FlexAttention._compiled_flex_attn = torch.compile(
        flex_attention, options=FlexAttention.inductor_configs)

inductor 本身是好的——Triton-Ascend 在基线里，``scripts/install_triton.sh`` 的验收
会编出前反向内核。编不出来的是 **flex 的 document mask**：它 index 一个 segment-id
张量（``aten.index.Tensor``），而 torch_npu 只在 ``inductor_indirect_memory_mode``
打开时才真正 lower 间接寻址，该开关只在 Ascend950 上赋值，910B2 恒为 ``None``。
于是 fallback 成 ExternKernel，在 flex 的 pointwise 子图里建 buffer，抛
``SubgraphLoweringException: Buffers cannot be created while lowering a pointwise
subgraph``。

实测（2026-09-01，装好 Triton-Ascend 之后）：关掉这条 shim，kimi_k3 立刻死在这个异常上；
装回来 10 步跑完。上游对 ROCm 做的正是同一件事（编译版 flex 不确定 → 改用 eager），
昇腾缺这条分支。

``set_determinism`` 是唯一能改到 ``_compiled_flex_attn`` 的钩子——它在训练开始前无条件
被调用，而那个属性没有配置开关（P0：``debug`` 配置里没有这一项）。所以这条 shim 对
**每一次**运行生效，不只是确定性模式。

消失条件是**硬件**：换到能 lower 间接寻址的芯片（Ascend950），下面的特性探测会让它自动让路。
按 P10 不向 github.com/pytorch 提 issue，只在 ``docs/issues/torchtitan.md`` 记录。
"""

import logging

from ascend_titan.compat import shim

logger = logging.getLogger(__name__)


def _can_lower_a_document_mask() -> bool:
    """True when inductor can lower the indirect memory access a document mask needs.

    Not "is there a triton backend": Triton-Ascend is part of the baseline and
    inductor compiles fine. The gate is torch_npu's own
    ``inductor_indirect_memory_mode``, which it sets only on Ascend950 -- probing
    the real cause means this shim stands down on hardware that can run the
    compiled path, and stays put on hardware that cannot.
    """
    from torch_npu._inductor import config as npu_inductor_config

    return getattr(npu_inductor_config, "inductor_indirect_memory_mode", None) is not None


@shim(
    target="torchtitan.distributed.utils:set_determinism",
    reason="上游给 FlexAttention 装编译版实现；910B2 上 document mask 的间接寻址 lower 不了"
    "（inductor_indirect_memory_mode 只在 Ascend950 赋值），编译期抛 "
    "SubgraphLoweringException。上游对 ROCm 就是改用 eager，昇腾缺这条分支",
    upstream="draft:docs/issues/torchtitan.md#compiled-flex-on-npu",
    kind="wrap",
)
def flex_attention_eager(original):
    """跑完上游的 ``set_determinism``，再把 eager flex 装回去。"""

    def wrapper(*args, **kwargs):
        result = original(*args, **kwargs)
        if _can_lower_a_document_mask():
            return result
        from torch.nn.attention.flex_attention import flex_attention
        from torchtitan.models.common.attention import FlexAttention

        if FlexAttention._compiled_flex_attn is not flex_attention:
            FlexAttention._compiled_flex_attn = flex_attention
            logger.info(
                "[shim] flex_attention_eager: 这颗芯片 lower 不了 document mask 的间接寻址，"
                "flex attention 改走 eager（上游对 ROCm 同样处理）"
            )
        return result

    return wrapper
