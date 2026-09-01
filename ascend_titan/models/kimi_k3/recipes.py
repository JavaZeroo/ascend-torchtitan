"""Kimi K3 recipes.

The whole KDA path is CUDA-only upstream: ``KDAKernel`` requires Blackwell
SM100, ``l2norm`` is a Triton kernel and ``causal_conv1d`` is CuTeDSL. The
overrides in ``ascend_titan.kernels.kda`` put it back on attn_gym's own
device-agnostic reference path plus a torch depthwise convolution, so the model
trains on Ascend without touching upstream.

``cutlass`` (the CuTeDSL package that attn_gym imports at module level, TT-11)
is a plain pip dependency -- ``nvidia-cutlass-dsl`` has aarch64 wheels and
installs on the NPU box. It is only imported, never executed, because every node
that would run a cute kernel is overridden here.

The MoE feed-forward has an Ascend fused kernel too (``kernels.situ_glu``, ops-nn
``aclnnSituGlu``), enabled by ``kimi_k3_debugmodel_npu_fused`` when the ops-nn
run package is installed.

Full guide: ascend_titan/models/kimi_k3/README.md
"""

from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel
from torchtitan.trainer import Trainer

from ascend_titan.kernels import ATTENTION_OVERRIDE, KDA_OVERRIDE, SITU_GLU_OVERRIDE
from ascend_titan.recipes.deltas import add_override, flex_to_varlen

# 视觉塔的注意力吃的是 `create_block_diagonal_mask` 造的 BlockMask，而 VarlenAttention
# 断言 `isinstance(attention_masks, VarlenMetadata)`——转过去只会把"flex 需要 inductor"
# 这个清楚的失败换成视觉塔深处一个更难懂的类型错误。所以它保持 flex（靠
# `flex_block_mask_eager` shim 走 eager）。
_VISION_TOWER = "vision_encoder"


def kimi_k3_debugmodel_npu() -> Trainer.Config:
    """Upstream ``kimi_k3_debugmodel`` + the NPU deltas, one visible call each.

    我们**换了什么**（下面四条 DELTA）与**没换什么**，读这个函数就够；每条的理由见
    README §2。没有动的部分（一并列在这里，因为"没动"同样是结论）：RoPE 走上游实现
    （kimi_k3 没有 ComplexRoPE 节点，不需要 NPU-3 的实数 cache override）、MoE 路由、
    `_apply_attention_residual`、loss、并行策略、优化器，全是上游默认。
    """
    config = kimi_k3_debugmodel()

    # DELTA 1: 语言塔的 6 个 flex 注意力节点（每 4 层一个 full-attention 层，其余是 KDA）
    # -> varlen；视觉塔那一个保持 flex。上游的 flex 掩码走 `create_block_mask`，是
    # torch.compile 的，昇腾上要 inductor（DEP-INDUCTOR）。
    # 消失条件：装上 Triton-Ascend 后 flex 可用，flex_to_varlen 自动变成空操作。
    converted = flex_to_varlen(config, keep=(_VISION_TOWER,))

    # DELTA 2: 转出来的 varlen 节点走昇腾融合注意力——stock varlen 要
    # `aten::_flash_attention_forward`，torch_npu 没有（NPU-1）。
    if converted:
        add_override(config, ATTENTION_OVERRIDE)

    # DELTA 3: KDA 走设备无关路径（上游内核要 CUDA/Blackwell，l2norm 是 Triton、
    # 短卷积是 CuTeDSL）。
    # 消失条件：昇腾有了 KDA 融合算子（fla-npu / ops-nn）之后换成融合实现，override 仍在。
    add_override(config, KDA_OVERRIDE)

    # DELTA 4: 冒烟运行不做 checkpoint I/O（DCP on NPU 是独立的矩阵格）。
    config.checkpoint.enable = False

    return config


def kimi_k3_debugmodel_npu_fused() -> Trainer.Config:
    """Reference path + the ops-nn SiTU-GLU fused kernel (needs the ops-nn run package)."""
    config = kimi_k3_debugmodel_npu()
    add_override(config, SITU_GLU_OVERRIDE)
    return config
