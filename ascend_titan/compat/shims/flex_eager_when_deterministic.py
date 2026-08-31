"""确定性模式下让 flex attention 走 eager —— 上游已经为 ROCm 这么做了，昇腾缺这条分支。

``torchtitan/distributed/utils.py::set_determinism`` 在 ``debug.deterministic`` 下会
**重新赋值** ``FlexAttention._compiled_flex_attn``::

    if torch.version.hip is not None:
        # Compiled ROCm flex attention is not deterministic.
        FlexAttention._compiled_flex_attn = flex_attention          # eager
    else:
        FlexAttention.inductor_configs["max_autotune"] = False
        FlexAttention._compiled_flex_attn = torch.compile(
            flex_attention, options=FlexAttention.inductor_configs)

昇腾走 else 分支，于是整个 ``flex_attention`` 函数被带着 ``inductor_configs`` 编译。
实测（910B2 / torch 2.15.0.dev20260812 / torch_npu master，见
``tests/repro/probe_flex_deterministic.py``）这条路在昇腾上不通，且两种掩码败因不同：

* ``mask_mod`` 读张量（视觉塔的 block-diagonal document mask）
  → ``SubgraphLoweringException: Buffers cannot be created while lowering a pointwise subgraph``；
* 纯索引算术的 causal mask → ``InductorError``。

**不开确定性时一切正常**：同一份 ``mask_mod``，前向、LSE、反向在昇腾上全部通过，
kimi_k3 与 qwen3.5 的多模态 debugmodel 也都能正常训练。所以这不是硬件限制，
也不是"读张量的掩码 lower 不出来"——是上游这条确定性专用路径在昇腾上没有对应分支。

影响面正好落在 ``scripts/check_golden.sh`` 上（它加 ``--debug.deterministic``），
也就是说：不修这条，用 flex 的模型就录不了逐位 golden。

修法与上游给 ROCm 的完全一致：确定性模式下改用 eager ``flex_attention``。
上游把设备判断硬编码成 ``torch.version.hip``，没有留扩展点，所以只能包 ``set_determinism``
（P0：先找配置开关——``debug`` 配置里没有这一项）。

消失条件：上游把那个 ``torch.version.hip`` 判断换成"编译版 flex 是否确定"的能力探测，
或显式加上 PrivateUse1 分支。按 P10 不向 github.com/pytorch 提 issue，只在
``docs/issues/torchtitan.md`` 记录。
"""

import logging

from ascend_titan.compat import shim

logger = logging.getLogger(__name__)


@shim(
    target="torchtitan.distributed.utils:set_determinism",
    reason="set_determinism 在非 ROCm 分支上把 _compiled_flex_attn 编译掉，"
    "这条路在昇腾上不通（读张量的掩码 SubgraphLoweringException，causal 掩码 "
    "InductorError）；上游对 ROCm 的处理就是改用 eager，昇腾缺这条分支",
    upstream="draft:docs/issues/torchtitan.md#deterministic-flex-on-npu",
    kind="wrap",
)
def flex_eager_when_deterministic(original):
    """跑完上游的 ``set_determinism``，再把 eager flex 装回去。"""

    def wrapper(*args, **kwargs):
        result = original(*args, **kwargs)
        from torch.nn.attention.flex_attention import flex_attention
        from torchtitan.models.common.attention import FlexAttention

        if FlexAttention._compiled_flex_attn is not flex_attention:
            FlexAttention._compiled_flex_attn = flex_attention
            logger.info(
                "[shim] flex_eager_when_deterministic: 确定性模式下 flex attention 改走 "
                "eager（上游对 ROCm 同样处理）"
            )
        return result

    return wrapper
