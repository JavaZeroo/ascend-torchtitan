"""Qwen3.5 recipes.

``fla`` (flash-linear-attention) is a plain pip dependency -- ``fla-core`` installs
on aarch64 and imports fine, so the model package loads. Its *kernels* are
another matter: they are Triton written for CUDA and ``bishengir-compile``
rejects them even with Triton-Ascend installed. The gated delta net therefore
runs through ``ascend_titan.kernels.gdn``, which selects attn_gym's own
device-agnostic reference recurrence -- same math, different implementation,
exactly like the KDA override for kimi_k3.

Full guide: ascend_titan/models/qwen3_5/README.md
"""

from torchtitan.models.qwen3_5 import model_registry
from torchtitan.models.qwen3_5.config_registry import qwen35_0_8b, qwen35_debugmodel
from torchtitan.trainer import Trainer

ATTENTION_OVERRIDE = "ascend_titan.kernels.attention.npu_fusion_attention"
GDN_OVERRIDE = "ascend_titan.kernels.gdn.npu_gated_delta_net"


def qwen35_debugmodel_npu() -> Trainer.Config:
    """Upstream ``qwen35_debugmodel`` + the minimal deltas for an NPU run."""
    config = qwen35_debugmodel()

    # DELTA 1: inner attention = varlen node + Ascend fused-attention override
    # (model-level flex needs inductor/Triton-Ascend). Mirrors qwen3 DELTA 1.
    config.model_spec = model_registry("debugmodel", attn_backend="varlen")
    config.override.imports = [ATTENTION_OVERRIDE]

    # DELTA 2: gated delta net on attn_gym's reference recurrence (fla's Triton
    # kernels do not compile for Ascend). 消失条件：昇腾有了 GDN 融合算子。
    config.override.imports = [*config.override.imports, GDN_OVERRIDE]

    # DELTA 3: no checkpoint I/O in a smoke run (DCP on NPU is its own cell).
    config.checkpoint.enable = False

    return config


def qwen35_debugmodel_npu_fsdp2() -> Trainer.Config:
    """2-way FSDP2."""
    config = qwen35_debugmodel_npu()
    config.parallelism.data_parallel_shard_degree = 2
    return config


# --- release 级 recipe（docs/model-release-criteria.md R1）---------------------


def qwen35_0_8b_npu() -> Trainer.Config:
    """Qwen3.5-0.8B，真实 tokenizer + 真实 C4 文本。

    上游 0.8B 配的是多模态 cc12m；这里换成真实 C4 文本分片，测的是**语言侧**的
    真实尺寸训练（GDN + MoE + 注意力）。视觉塔的路径由 debugmodel 的 cc12m-test 覆盖——
    真实 cc12m 是图文数据集，不在这台机器的下载预算里，这个缺口记在 README 里。
    """
    from torchtitan.components.data.collators import TextCollator
    from torchtitan.components.data.packing import ConcatThenSplitPackingConfig

    from ascend_titan.models.assets import hf_assets_path, local_c4_dataset

    config = qwen35_0_8b()

    # DELTA 1: varlen 注意力 + 昇腾融合注意力（同 debugmodel）。
    config.model_spec = model_registry("0.8B", attn_backend="varlen")
    config.override.imports = [ATTENTION_OVERRIDE]

    # DELTA 2: GDN 走 attn_gym 的 reference 递推（fla 的 Triton 内核编不出来）。
    config.override.imports = [*config.override.imports, GDN_OVERRIDE]

    # DELTA 3: 真实 tokenizer + 真实 C4 文本（纯文本，去掉多模态 collator）。
    config.hf_assets_path = hf_assets_path("Qwen3.5-0.8B")
    config.dataloader.dataset = ConcatThenSplitPackingConfig(dataset=local_c4_dataset())
    config.dataloader.collator = TextCollator.Config()

    return config


def qwen35_0_8b_npu_fsdp2() -> Trainer.Config:
    """0.8B × FSDP2 8 卡。"""
    config = qwen35_0_8b_npu()
    config.parallelism.data_parallel_shard_degree = 8
    return config
