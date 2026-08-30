"""Which models we support on Ascend, and how far each one got.

Plain data on purpose: no torch, no torchtitan import (P0/F4), so ``doctor``,
the docs build and CPU tests can all read it. Runtime status of *features*
lives in ``docs/capability-matrix.md``; issue status lives in
``docs/issues/STATUS.md`` (P11). This table only records, per model family:
what upstream calls it, which of our recipe modules drives it, and whether we
have actually run it -- with the attribution when we have not.

Print it with::

    python -m ascend_titan.models.registry
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 🟢 release 级: every criterion in docs/model-release-criteria.md has a recorded run
# 🟡 runs, but with gaps -- ``criteria`` says which
# 🔴 blocked -- ``blocker`` says by what, with the attribution tag
# ⚪ not evaluated yet (P2: "not tested" and "tested and broken" never share a cell)
STATUS = ("🟢", "🟡", "🔴", "⚪")

CRITERIA = ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8")
"""docs/model-release-criteria.md: 真实形态 / 并行 / 数值 / checkpoint /
性能 / 长稳 / 文档 / 无降级.

Before 2026-08-31 a 🟢 here meant "the debugmodel runs". It now means all eight
have a recorded command and output, which is a much higher bar and the reason
several entries moved to 🟡: nothing regressed, the bar did.
"""


@dataclass(frozen=True)
class ModelEntry:
    """One upstream model family."""

    name: str
    """Directory name under ``ascend_titan/models/``

    (or the upstream name when we have no package yet).
    """

    upstream: str
    """Upstream package, e.g. ``torchtitan.models.qwen3``."""

    title: str
    status: str
    summary: str
    recipes: str | None = None
    """Our recipe module, ``None`` when the model is only reachable through the matrix runner."""

    flavors: tuple[str, ...] = ()
    """Recipe functions we ship (debug flavors first)."""

    blocker: str | None = None
    """Why it is not 🟢, with the attribution tag (see docs/capability-matrix.md)."""

    golden: tuple[str, ...] = ()
    """Frozen loss curves under tests/assets/losses/npu/."""

    notes: str = ""
    owner_docs: list[str] = field(default_factory=list)

    criteria: dict[str, str] = field(default_factory=dict)
    """Per-criterion state, keyed by ``CRITERIA``. Empty = never graded (⚪)."""

    evidence: str | None = None
    """Where the runs are recorded, e.g. ``docs/release/qwen3_<tuple>.md``."""

    @property
    def has_package(self) -> bool:
        return self.recipes is not None

    @property
    def graded_status(self) -> str:
        """🟢 only when every criterion is 🟢 -- the rule the criteria doc sets."""
        if not self.criteria:
            return self.status
        marks = [self.criteria.get(c, "⚪") for c in CRITERIA]
        if all(m == "🟢" for m in marks):
            return "🟢"
        return "🔴" if any(m == "🔴" for m in marks) and self.blocker else "🟡"


MODELS: dict[str, ModelEntry] = {
    "qwen3": ModelEntry(
        name="qwen3",
        upstream="torchtitan.models.qwen3",
        title="Qwen3",
        status="🟢",
        summary=(
            "参考模型，release 级：0.6B 真实尺寸 + 真实 tokenizer/C4，"
            "单卡 / FSDP2×8 / TP2 / PP2 全绿，golden 逐位冻结，"
            "500 步长稳、checkpoint 续训逐位一致、HF 权重往返、性能基线带 provenance。"
        ),
        recipes="ascend_titan.models.qwen3.recipes",
        flavors=(
            "qwen3_debugmodel_npu",
            "qwen3_debugmodel_npu_fsdp2",
            "qwen3_debugmodel_npu_fused",
            "qwen3_debugmodel_npu_fused_fsdp2",
            "qwen3_0_6b_npu",
            "qwen3_0_6b_npu_fsdp2",
            "qwen3_0_6b_npu_tp2",
            "qwen3_8b_npu_pp2",
        ),
        golden=(
            "qwen3_debugmodel_npu",
            "qwen3_debugmodel_npu_fsdp2",
            "qwen3_debugmodel_npu_fused",
            "qwen3_debugmodel_npu_fused_fsdp2",
        ),
        criteria={
            "R1": "🟢",  # 0.6B + 真实 tokenizer + 真实 C4 + 4096 上下文
            "R2": "🟢",  # 1 卡 / FSDP2×8 / FSDP2×4+TP2 / PP2×FSDP2-4（8B）全绿
            "R3": "🟢",  # 四条 golden 逐位；500 步 12.12 → 6.28；tests/npu 算子对拍
            "R4": "🟢",  # DCP 续训逐位一致；HF 导出/导入权重完整往返
            "R5": "🟢",  # 10,307 tps / 65.91 TFLOPs / 19.08 GiB，带 provenance
            "R6": "🟢",  # 500 步 rc=0，显存自第 51 步起恒定，无 NaN
            "R7": "🟢",  # models/qwen3/README.md
            "R8": "🟢",  # provenance：AscendFusionAttention
        },
        evidence="docs/release/qwen3_torch2.15.0.dev20260812_npu2.15.0.md",
        notes="1.7B / 32B / 30B-A3B 等其它真实尺寸仍 ⚪。",
    ),
    "qwen3_5": ModelEntry(
        name="qwen3_5",
        upstream="torchtitan.models.qwen3_5",
        title="Qwen3.5",
        status="🟡",
        summary=(
            "语言侧真实尺寸（0.8B + 真实 tokenizer/C4 + 4096 上下文）能跑，"
            "gated delta net 与 causal conv1d 走 `kernels/gdn.py` 的 override。"
            "视觉侧 🔴（视觉塔的 document mask 撞 910B2 的 indirect-memory 限制），"
            "性能是主要缺口：GDN 没有融合算子。"
        ),
        recipes="ascend_titan.models.qwen3_5.recipes",
        flavors=(
            "qwen35_debugmodel_npu",
            "qwen35_debugmodel_npu_fsdp2",
            "qwen35_debugmodel_npu_text",
            "qwen35_0_8b_npu",
            "qwen35_0_8b_npu_fsdp2",
        ),
        criteria={
            "R1": "🟡",  # 0.8B 语言侧形态齐了，但从零训练第 5 步发散（学习率待定位）
            "R2": "🟡",  # 单卡与 FSDP2×8 都能推进，撞同一个发散；TP/PP/EP 未测
            "R3": "🟡",  # 对 attn_gym reference 的前反向对拍 🟢；golden 未冻结
            "R4": "⚪",
            "R5": "🔴",  # 纯 torch chunk 递推，无融合算子
            "R6": "⚪",  # 被 R5 卡住：一步约 2 分钟，500 步要十几个小时
            "R7": "🟢",  # models/qwen3_5/README.md
            "R8": "🟢",  # provenance：42 个 ascend 节点
        },
        notes=(
            "早先记的 DEP-FLA 阻塞不成立：`fla-core` 有 aarch64 wheel，import 正常，"
            "挡住的只是它的 CUDA Triton 内核。"
        ),
    ),
    "llama3": ModelEntry(
        name="llama3",
        upstream="torchtitan.models.llama3",
        title="Llama 3",
        status="🟡",
        summary=(
            "零 override 的 stock 参考路径："
            "复数 RoPE + ChunkedLoss + spmd_types 全部走上游默认实现。"
            "只有 debugmodel：R1–R8 一条都没取，按新判据是 🟡 而不是 🟢。"
        ),
        recipes="ascend_titan.models.llama3.recipes",
        flavors=("llama3_debugmodel_stock_npu", "llama3_debugmodel_stock_npu_fsdp2"),
        notes=(
            "唯一增量是 flex→varlen（flex 的模型级路径要走 inductor）。"
            "上游 features 套件的并行用例也都跑在 llama3 上。"
        ),
    ),
    "kimi_k3": ModelEntry(
        name="kimi_k3",
        upstream="torchtitan.models.kimi_k3",
        title="Kimi K3",
        status="🔴",
        summary=(
            "多模态 + KDA + MoE。2026-08-30 曾跑通 10 步（单卡 loss 4.10312），"
            "2026-08-31 复测不再复现。"
        ),
        recipes="ascend_titan.models.kimi_k3.recipes",
        flavors=("kimi_k3_debugmodel_npu", "kimi_k3_debugmodel_npu_fused"),
        blocker=(
            "视觉塔的 block-diagonal document mask：保留 flex 撞 "
            "`SubgraphLoweringException`（910B2 无 indirect-memory lowering），"
            "转 varlen 撞 `attention_masks must be VarlenMetadata, got BlockMask`。"
            "两条都实测过；需要二分定位从绿变红的那次改动。"
        ),
        notes=(
            "需要 `nvidia-cutlass-dsl`（有 aarch64 wheel，只 import 不执行）；"
            "KDA 走 `kernels/kda.py` 的 override，flex 路径靠 `flex_block_mask_eager` "
            "shim 走 eager。融合变体（ops-nn SiTU-GLU）实测 loss 4.29434 / tps 48。"
            "性能极低，Triton-Ascend 到位前不做性能基线。"
        ),
    ),
    "deepseek_v3": ModelEntry(
        name="deepseek_v3",
        upstream="torchtitan.models.deepseek_v3",
        title="DeepSeek-V3",
        status="🟡",
        summary=(
            "MoE + EP 在矩阵扫描里通过（fsdp+ep、hsdp+ep）；"
            "没有专属 recipe，通过矩阵 runner 跑上游配置。"
        ),
        recipes=None,
        blocker="fused_mla_swiglu：OURS-9（override 节点冲突）；MTP + helion_rope：DEP-HELION。",
        notes="要专属 recipe 时按 `_template/` 建目录。",
    ),
    "gpt_oss": ModelEntry(
        name="gpt_oss",
        upstream="torchtitan.models.gpt_oss",
        title="GPT-OSS",
        status="🟡",
        summary="pp+fsdp+ep+sacop 在矩阵里 🟢（attention sinks 的 LSE 尾部已实现）。",
        recipes=None,
        blocker="fsdp+tp+ep：OURS-10（TP2+EP4 下路由 softmax backward 形状不匹配），待查。",
    ),
    "kimi_k2_7": ModelEntry(
        name="kimi_k2_7",
        upstream="torchtitan.models.kimi_k2_7",
        title="Kimi K2.7",
        status="🟡",
        summary="muon / MoE 用例在矩阵里覆盖；无专属 recipe。",
        recipes=None,
        blocker="DistMuon 是 CUDA-only（TT-CUDA）。",
    ),
    "muse_glimmer": ModelEntry(
        name="muse_glimmer",
        upstream="torchtitan.models.muse_glimmer",
        title="Muse Glimmer",
        status="🟡",
        summary="text 变体在矩阵里覆盖；多模态变体依赖 CP。",
        recipes=None,
        blocker="mm 变体走 CP，停在 DEP-INDUCTOR（Triton-Ascend 未装）。",
    ),
    "flux": ModelEntry(
        name="flux",
        upstream="torchtitan.models.flux",
        title="Flux",
        status="⚪",
        summary="扩散模型，尚未评估。",
        recipes=None,
    ),
}


def table() -> str:
    """The model support table, as Markdown (README.md quotes this shape)."""
    rows = [
        "| 模型 | 状态 | 我们的 recipe | 说明 / 阻塞 |",
        "|---|:--:|---|---|",
    ]
    for e in MODELS.values():
        recipe = f"`{e.recipes.rsplit('.', 2)[-2]}`" if e.recipes else "—（矩阵覆盖）"
        detail = e.summary + (f" **阻塞：**{e.blocker}" if e.blocker else "")
        rows.append(f"| **{e.title}** (`{e.name}`) | {e.status} | {recipe} | {detail} |")
    return "\n".join(rows)


def criteria_table() -> str:
    """Per-model R1-R8, for the models we have actually graded."""
    rows = [
        "| 模型 | " + " | ".join(CRITERIA) + " | 证据 |",
        "|---|" + ":--:|" * len(CRITERIA) + "---|",
    ]
    for e in MODELS.values():
        if not e.criteria:
            continue
        marks = " | ".join(e.criteria.get(c, "⚪") for c in CRITERIA)
        rows.append(f"| **{e.title}** | {marks} | {e.evidence or '—'} |")
    return "\n".join(rows)


if __name__ == "__main__":
    print(table())
    print()
    print(criteria_table())
