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

🟢 要求八条全部有记录下来的命令与输出。
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
            # 手写的只剩三个；其余（含 qwen3_0_6b_npu）由 _auto 从上游 flavor 生成，
            # 并行度组合走命令行（--parallelism.*），不再各写一个函数。
            "qwen3_debugmodel_npu",
            "qwen3_debugmodel_npu_fused",
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
            "R5": "🟢",  # 单卡 10,186 / FSDP2×8 9,440 / PP2 1,409 tps，均带 provenance
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
            "语言侧 0.8B 路径打通（gated delta net 与 causal conv1d 走 "
            "`kernels/gdn.py` 的 override，逐项对上游/参考实现对拍），"
            "0.8B 20 步 12.88826 → 8.14589、FSDP2×8 12.90316 → 8.06005；"
            "TP2 / PP2 / MoE dp2×tp2×pp2×ep4 也已实测通过，且与 stock 逐格一致"
            "（CP 上游按设计不支持：gated delta net 要整条序列）。"
            "多模态 debugmodel 也能跑，golden 已冻结（确定性模式需要 TT-12 那条 shim）；"
            "DCP 续训 🔴（纯文本增量让视觉塔没有优化器状态）；"
            "fla-npu 融合 GDN（R5）已落地并在模型级真实执行："
            "0.8B 单卡 step-4 tps 1,420 vs 纯 torch 244（≈5.8×），"
            "provenance 见 AscendFusedGatedDeltaKernel×18；"
            "对拍 bf16 舍入级，但 AscendC 内核 run-to-run 非确定（OURS-14），"
            "无逐位 golden，改用纯 torch ± bf16 容差断言。"
        ),
        recipes="ascend_titan.models.qwen3_5.recipes",
        flavors=(
            # 手写的只剩三个；qwen35_0_8b_npu 等由 _auto 生成（HF_REPOS 提供真实资产）。
            "qwen35_debugmodel_npu",
            "qwen35_debugmodel_npu_text",
            "qwen35_0_8b_npu_fused",
        ),
        golden=("qwen35_debugmodel_npu", "qwen35_debugmodel_npu_text"),
        criteria={
            "R1": "🟢",  # 0.8B 20 步 12.88826 → 8.14589
            # 单卡 / FSDP2×8 / TP2 / PP2 / MoE dp2×tp2×pp2×ep4 全绿，stock 与融合算子
            # 逐格一致（docs/matrix/2026-09-02_qwen35_parallel.md）。CP 是上游按设计
            # 不支持（gated delta net 要整条序列），stock 同样跑不了，不算缺口。
            "R2": "🟢",
            "R3": "🟡",  # 对拍 🟢 + 语言侧 golden 已冻结；缺真实尺寸的长步数曲线
            "R4": "🔴",  # HF 往返 🟢；DCP 续训缺视觉塔的优化器状态（纯文本增量的代价）
            "R5": "🟢",  # 融合 GDN 模型级真实执行；tps 1,420 vs 244 ≈5.8×
            # 对拍 bf16 对齐；run-to-run 非确定（OURS-14），无逐位 golden
            "R6": "⚪",  # 被 R5 卡住：一步约 2 分钟，500 步要十几个小时
            "R7": "🟢",  # models/qwen3_5/README.md
            "R8": "🟢",  # provenance：42 个 ascend 节点
        },
        evidence="docs/release/qwen3_5_torch2.15.0.dev20260812_npu2.15.0.md",
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
        flavors=("llama3_debugmodel_stock_npu",),  # 并行度走命令行，不再各写一个函数
        notes=(
            "唯一增量是 flex→varlen（flex 的模型级路径要走 inductor）。"
            "上游 features 套件的并行用例也都跑在 llama3 上。"
        ),
    ),
    "kimi_k3": ModelEntry(
        name="kimi_k3",
        upstream="torchtitan.models.kimi_k3",
        title="Kimi K3",
        status="🟡",
        summary=(
            "多模态 + KDA + MoE，单卡 10 步 loss 4.56418，golden 已冻结并逐位复现。"
            "只有 debugmodel：R1–R8 一条都没取。"
        ),
        recipes="ascend_titan.models.kimi_k3.recipes",
        flavors=("kimi_k3_debugmodel_npu", "kimi_k3_debugmodel_npu_fused"),
        golden=("kimi_k3_debugmodel_npu",),
        notes=(
            "需要 `nvidia-cutlass-dsl`（有 aarch64 wheel，只 import 不执行）；"
            "KDA 走 `kernels/kda.py` 的 override；确定性模式下 flex 靠 "
            "`flex_attention_eager` shim 走 eager（TT-12）。"
            "融合变体（ops-nn SiTU-GLU）实测 loss 4.29434 / tps 48。"
            "性能极低（flex 走 eager，910B2 硬件门），不做性能基线。"
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
        blocker="mm 变体走 CP，停在 CANN/硬件（document mask 的间接寻址）。",
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
        rows.append(f"| **{e.title}** (`{e.name}`) | {e.graded_status} | {recipe} | {detail} |")
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
