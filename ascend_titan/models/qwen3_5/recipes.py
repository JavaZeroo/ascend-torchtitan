"""Qwen3.5 recipes.

validated: torchtitan=13da2d77c torch=2.15.0.dev20260812 torch_npu=2.15.0 CANN=9.1.0 date=2026-08-31
(rewritten by CI when a run is green)

``fla`` (flash-linear-attention) is a plain pip dependency -- ``fla-core`` installs
on aarch64 and imports fine, so the model package loads. Its *kernels* are
another matter: they are Triton written for CUDA and ``bishengir-compile``
rejects them even with Triton-Ascend installed. The gated delta net therefore
runs through ``ascend_titan.kernels.gdn``, whose chunk-parallel delta rule is
attn_gym's decomposition in plain torch -- same math, different implementation,
exactly like the KDA override for kimi_k3.

The multimodal flavours do not run on 910B2: the vision tower's block-diagonal
FlexAttention mask indexes a tensor inside a pointwise subgraph, which needs
inductor's indirect-memory path (Ascend950 only). ``qwen35_debugmodel_npu_text``
and the 0.8B recipes are the language-side path, and they do run.

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


def qwen35_debugmodel_npu_text() -> Trainer.Config:
    """The same debugmodel on **text only** -- the cheap regression test for GDN.

    ``qwen35_debugmodel`` is multimodal: it trains on ``cc12m-test`` and runs the
    vision tower, whose block-diagonal FlexAttention mask indexes a segment-id
    tensor inside a pointwise subgraph. That lowering needs inductor's
    indirect-memory path, which torch_npu only enables on Ascend950, so the
    multimodal debugmodel cannot run on 910B2 (same root cause as CP and
    model-level flex; see docs/capability-matrix.md).

    The language side has no such problem, and it is the side the GDN override
    lives on. Swapping in the toy tokenizer's C4 text keeps a 10-step
    smoke config that exercises exactly what we replaced.
    """
    from torchtitan.components.data.collators import TextCollator
    from torchtitan.components.data.packing import ConcatThenSplitPackingConfig
    from torchtitan.hf_datasets.text_datasets import DATASETS

    config = qwen35_debugmodel_npu()

    # DELTA 4: text-only data, so the vision tower stays out of the graph.
    config.dataloader.dataset = ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"])
    config.dataloader.collator = TextCollator.Config()
    config.dataloader.shuffle = False
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
