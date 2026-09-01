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

from ascend_titan.kernels import KDA_OVERRIDE, SITU_GLU_OVERRIDE
from ascend_titan.recipes.transforms import npu_minimal


def kimi_k3_debugmodel_npu() -> Trainer.Config:
    """Upstream ``kimi_k3_debugmodel`` + the minimal deltas for an NPU run."""
    config = kimi_k3_debugmodel()

    # DELTA 1: the cross-model minimal transform (flex -> varlen + Ascend fused
    # attention + real-cache RoPE). kimi_k3's MLA path builds its FlexAttention mask
    # through `create_block_mask`, which is torch.compile'd and so needs inductor --
    # Triton-Ascend on NPU (DEP-INDUCTOR). Calling the shared transform keeps one
    # implementation of "what every upstream config needs on NPU" (P11).
    npu_minimal(config)

    # DELTA 2: KDA on the device-agnostic path (upstream kernel is CUDA/Blackwell,
    # its l2norm is Triton and its short conv is CuTeDSL).
    # 消失条件：昇腾有了 KDA 融合算子（fla-npu / ops-nn）之后换成融合实现，override 仍在。
    config.override.imports = [*config.override.imports, KDA_OVERRIDE]

    # DELTA 3: no checkpoint I/O in a smoke run (DCP on NPU is its own cell).
    config.checkpoint.enable = False

    return config


def kimi_k3_debugmodel_npu_fused() -> Trainer.Config:
    """Reference path + the ops-nn SiTU-GLU fused kernel (needs the ops-nn run package)."""
    config = kimi_k3_debugmodel_npu()
    config.override.imports = [*config.override.imports, SITU_GLU_OVERRIDE]
    return config
