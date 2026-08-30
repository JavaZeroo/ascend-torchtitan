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

# 🟢 runs on NPU and is covered by a golden or a matrix cell
# 🟡 runs, but only in a reduced flavor / not gated
# 🔴 blocked -- ``blocker`` says by what, with the attribution tag
# ⚪ not evaluated yet (P2: "not tested" and "tested and broken" never share a cell)
STATUS = ("🟢", "🟡", "🔴", "⚪")


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

    @property
    def has_package(self) -> bool:
        return self.recipes is not None


MODELS: dict[str, ModelEntry] = {
    "qwen3": ModelEntry(
        name="qwen3",
        upstream="torchtitan.models.qwen3",
        title="Qwen3",
        status="🟢",
        summary="参考模型：单卡 / FSDP2×2 golden 逐位冻结，NIGHTLY 门禁跑的就是它。",
        recipes="ascend_titan.models.qwen3.recipes",
        flavors=(
            "qwen3_debugmodel_npu",
            "qwen3_debugmodel_npu_fsdp2",
            "qwen3_debugmodel_npu_fused",
            "qwen3_debugmodel_npu_fused_fsdp2",
        ),
        golden=(
            "qwen3_debugmodel_npu",
            "qwen3_debugmodel_npu_fsdp2",
            "qwen3_debugmodel_npu_fused",
            "qwen3_debugmodel_npu_fused_fsdp2",
        ),
        notes="0.6B / 1.7B / 14B / 32B / 30B-A3B 等真实尺寸尚未跑过（⚪），见 README 的路线。",
    ),
    "qwen3_5": ModelEntry(
        name="qwen3_5",
        upstream="torchtitan.models.qwen3_5",
        title="Qwen3.5",
        status="🔴",
        summary=(
            "上游 `qwen3_5/__init__.py` → `gdn.py` 在模块级 import `fla`"
            "（CUDA-only Triton），整个模型包在昇腾上 import 就失败。"
        ),
        recipes="ascend_titan.models.qwen3_5.recipes",
        flavors=("qwen35_debugmodel_npu", "qwen35_debugmodel_npu_fsdp2"),
        blocker=(
            "DEP-FLA：`ModuleNotFoundError: No module named 'fla'`。"
            "昇腾侧对应物 fla-npu 属 L1 任务（M4）。"
        ),
        notes="recipe 已经写好并保持最小增量；装上 fla 的昇腾实现后即可直接跑。",
    ),
    "llama3": ModelEntry(
        name="llama3",
        upstream="torchtitan.models.llama3",
        title="Llama 3",
        status="🟢",
        summary=(
            "零 override 的 stock 参考路径："
            "复数 RoPE + ChunkedLoss + spmd_types 全部走上游默认实现。"
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
        status="🟢",
        summary=("多模态 + KDA + MoE，2026-08-30 在 910B2 上跑通 10 步（loss 9.51 → 4.35）。"),
        recipes="ascend_titan.models.kimi_k3.recipes",
        flavors=("kimi_k3_debugmodel_npu", "kimi_k3_debugmodel_npu_fused"),
        notes=(
            "需要 `nvidia-cutlass-dsl`（有 aarch64 wheel，只 import 不执行）；"
            "KDA 走 `kernels/kda.py` 的 override，flex 路径靠 `flex_block_mask_eager` "
            "shim 走 eager。性能极低（tps 45），Triton-Ascend 到位前不做性能基线。"
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


if __name__ == "__main__":
    print(table())
