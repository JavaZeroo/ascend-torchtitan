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

from torchtitan.models.qwen3_5.config_registry import qwen35_0_8b, qwen35_debugmodel
from torchtitan.trainer import Trainer

from ascend_titan.kernels import (
    ATTENTION_FROM_FLEX_OVERRIDE,
    ATTENTION_OVERRIDE,
    GDN_FUSED_OVERRIDE,
    GDN_OVERRIDE,
)
from ascend_titan.recipes.deltas import add_override, swap_override

# 有真实 tokenizer 的尺寸：flavor -> HF 仓库名。**加一个真实尺寸只需在这里加一行**，
# 不需要写函数——`<flavor>_npu` 会自动带上真实资产（见 npu_deltas）。
# 没列在这里的 flavor 用上游自己的资产设置（debugmodel 的玩具 tokenizer 等）。
HF_REPOS = {"qwen35_0_8b": "Qwen3.5-0.8B"}


def _use_real_assets(config: Trainer.Config, repo: str) -> None:
    """真实 tokenizer + 真实 C4 文本（纯文本，去掉多模态 collator）。

    上游的真实尺寸配的是多模态 cc12m；真实 cc12m 是图文数据集，不在这台机器的
    下载预算里，所以测的是**语言侧**的真实尺寸训练。缺口记在 README。
    """
    from torchtitan.components.data.collators import TextCollator
    from torchtitan.components.data.packing import ConcatThenSplitPackingConfig

    from ascend_titan.models.assets import hf_assets_path, local_c4_dataset

    config.hf_assets_path = hf_assets_path(repo)
    config.dataloader.dataset = ConcatThenSplitPackingConfig(dataset=local_c4_dataset())
    config.dataloader.collator = TextCollator.Config()


def npu_deltas(config: Trainer.Config, flavor: str = "") -> None:
    """What Qwen3.5 needs on Ascend. Flavor-independent: written once, applied to all.

    ``models/qwen3_5/__init__.py`` hands this to ``_auto.npu_entry_points``, so every
    upstream flavor -- 0.8B, 2B, 4B, 9B, 27B, 35B-A3B and whatever lands next -- gets
    a working ``<flavor>_npu`` entry point without another function being written.
    The hand-written recipes below call it too, so this list is the single place the
    family's Ascend deltas exist.
    """
    # DELTA 1: decoder-layer attention -> the Ascend fused kernel. Two overrides
    # because upstream flavors differ in their default node: most are FlexAttention
    # (model-level flex needs inductor/Triton-Ascend, DEP-INDUCTOR), while
    # ``*_varlen_attn`` is already varlen and stock varlen needs
    # ``aten::_flash_attention_forward``, which torch_npu lacks (NPU-1). They target
    # different Config classes, so both may be active. A vision tower's flex node is
    # outside the ``fqns`` glob of the first one and stays flex (it is fed a BlockMask).
    add_override(config, ATTENTION_FROM_FLEX_OVERRIDE)
    add_override(config, ATTENTION_OVERRIDE)

    # DELTA 2: gated delta net on attn_gym's reference recurrence (fla's Triton
    # kernels do not compile for Ascend). 消失条件：昇腾有了 GDN 融合算子。
    add_override(config, GDN_OVERRIDE)

    # DELTA 3: 该尺寸有真实 tokenizer 就用真实资产（表在 HF_REPOS）。debugmodel 不在
    # 表里，保留上游的玩具 tokenizer/数据。
    if flavor in HF_REPOS:
        _use_real_assets(config, HF_REPOS[flavor])


def qwen35_debugmodel_npu() -> Trainer.Config:
    """Upstream ``qwen35_debugmodel`` + the family deltas. Golden-frozen smoke config."""
    config = qwen35_debugmodel()
    npu_deltas(config)

    # DELTA 3 (this recipe only): no checkpoint I/O in a smoke run -- DCP on NPU is
    # its own matrix cell. Not a family delta: it is a convenience, and it is
    # CLI-addressable (``--checkpoint.no-enable``).
    config.checkpoint.enable = False

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
# 真实尺寸的入口 `qwen35_0_8b_npu` **不是手写的**：HF_REPOS 里有它，所以 `_auto` 生成的
# 入口已经带上真实 tokenizer + 真实 C4。下面只留真正需要额外东西的组合。


def qwen35_0_8b_npu_fused() -> Trainer.Config:
    """Qwen3.5-0.8B 语言侧 + fla-npu 融合 GDN（R5 性能路径，opt-in）。

    0.8B 的 gated delta net 是 K=V=128、Hk=Hv=16，落在融合 shape gate 内
    （V∈{128,256}、K≤128、chunk_size∈{64,128}）；GDN 换到 ``gdn_fla`` 的
    ``chunk_gated_delta_rule`` 自定义算子（6 核前向 / 7 核反向），conv1d 与
    门控数学不变。数值与 ``qwen35_0_8b_npu`` 在 bf16 舍入级不同，所以带自己的
    golden。未安装 fla_npu 时 override 不注册，等价退化为普通 recipe。
    """
    config = qwen35_0_8b()
    npu_deltas(config, "qwen35_0_8b")
    # The fused override is a SIBLING of the plain-torch GDN override (both claim
    # InnerGatedDeltaNet.Config); swap, never stack -- see deltas.swap_override.
    swap_override(config, remove=GDN_OVERRIDE, add=GDN_FUSED_OVERRIDE)
    return config
