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

from ascend_titan.kernels import (
    ATTENTION_FROM_FLEX_OVERRIDE,
    ATTENTION_OVERRIDE,
    KDA_OVERRIDE,
    SITU_GLU_OVERRIDE,
)
from ascend_titan.recipes.deltas import add_override


def npu_deltas(config: Trainer.Config, flavor: str = "") -> None:
    """What Kimi K3 needs on Ascend. Flavor-independent: written once, applied to all.

    ``models/kimi_k3/__init__.py`` hands this to ``_auto.npu_entry_points``, so any
    upstream flavor gets a ``<flavor>_npu`` entry point with no new function.
    """
    # DELTA 1: 语言塔的 flex 注意力节点 -> 昇腾融合注意力（上游 flex 掩码走
    # `create_block_mask`，是 torch.compile 的，昇腾要 inductor，DEP-INDUCTOR）。
    # override 的 fqns 只认领 `layers.*.attention`，视觉塔那个节点因此保持 flex——
    # 它喂的是 BlockMask，这个内核吃不了。
    # 消失条件：装上 Triton-Ascend 后可换回上游 flex。
    add_override(config, ATTENTION_FROM_FLEX_OVERRIDE)
    add_override(config, ATTENTION_OVERRIDE)

    # DELTA 2: KDA 走设备无关路径（上游内核要 CUDA/Blackwell，l2norm 是 Triton、
    # 短卷积是 CuTeDSL）。消失条件：昇腾有了 KDA 融合算子后换实现，override 仍在。
    add_override(config, KDA_OVERRIDE)


def kimi_k3_debugmodel_npu() -> Trainer.Config:
    """Upstream ``kimi_k3_debugmodel`` + the family deltas + a smoke-run convenience.

    换了什么见 ``npu_deltas``（族增量，任何 flavor 通用）；**没换什么**同样是结论：
    视觉塔的 FlexAttention 保持原样、RoPE 走上游实现（kimi_k3 没有 ComplexRoPE 节点）、
    MoE 路由、``_apply_attention_residual``、loss、并行策略、优化器，全是上游默认。
    """
    config = kimi_k3_debugmodel()
    npu_deltas(config)

    # DELTA 3 (this recipe only): 冒烟运行不做 checkpoint I/O（DCP on NPU 是独立
    # 的矩阵格），且是 CLI 可调的（``--checkpoint.no-enable``）。
    config.checkpoint.enable = False

    return config


def kimi_k3_debugmodel_npu_fused() -> Trainer.Config:
    """Reference path + the ops-nn SiTU-GLU fused kernel (needs the ops-nn run package)."""
    config = kimi_k3_debugmodel_npu()
    add_override(config, SITU_GLU_OVERRIDE)
    return config
