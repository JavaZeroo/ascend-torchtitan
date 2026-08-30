"""Performance baseline: run recipes, record throughput *with* provenance.

    python -m ascend_titan.tools.bench --cards 0 --out docs/perf

P7 says a benchmark without a provenance table is not accepted: a number that
came from an accidental eager fallback looks the same as a number from the fused
kernel. So every row here carries the list of Ascend-backed nodes that were
actually in effect for that run, collected from the very config that ran.

Runs are deterministic (``--debug.seed 42 --debug.deterministic``) so the loss
column doubles as a correctness check against the golden curves; throughput on a
shared box still moves a few percent between runs, which is why the report keeps
the raw per-step numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# (module, config, ngpu) -- the qwen3 reference path and its fused variant, which is
# the comparison the fused kernels exist to justify.
DEFAULT_RECIPES = [
    ("ascend_titan.models.qwen3", "qwen3_debugmodel_npu", 1),
    ("ascend_titan.models.qwen3", "qwen3_debugmodel_npu_fused", 1),
]

_STEP = re.compile(
    r"step:\s*(\d+)\s+loss:\s*([\d.]+)\s+grad_norm:\s*([\d.]+)\s+"
    r"memory:\s*([\d.]+)GiB\([\d.]+%\)\s+tps:\s*([\d,]+)\s+tflops:\s*([\d.]+)\s+mfu:\s*([\d.]+)%"
)


@dataclass
class Row:
    module: str
    config: str
    ngpu: int
    ok: bool = False
    loss: float | None = None
    tps: int | None = None
    tflops: float | None = None
    mfu: float | None = None
    memory_gib: float | None = None
    steps: list[dict] = field(default_factory=list)
    ascend_nodes: list[str] = field(default_factory=list)
    note: str = ""


def parse_steps(log: str) -> list[dict]:
    rows = []
    for m in _STEP.finditer(re.sub(r"\x1b\[[0-9;]*m", "", log)):
        rows.append(
            {
                "step": int(m.group(1)),
                "loss": float(m.group(2)),
                "grad_norm": float(m.group(3)),
                "memory_gib": float(m.group(4)),
                "tps": int(m.group(5).replace(",", "")),
                "tflops": float(m.group(6)),
                "mfu": float(m.group(7)),
            }
        )
    return rows


def steady_state(steps: list[dict]) -> dict:
    """Median of the second half: step 1 carries compile/alloc warm-up."""
    tail = steps[len(steps) // 2 :] or steps
    mid = sorted(tail, key=lambda s: s["tps"])[len(tail) // 2]
    return {**mid, "loss": steps[-1]["loss"]}


def ascend_nodes(module: str, config: str) -> list[str]:
    """Which of our overrides were actually in effect (P7)."""
    from ascend_titan.tools.provenance import build_config_isolated, collect

    rows = collect(build_config_isolated(module, config))
    return sorted({r.config_cls for r in rows if r.origin == "ascend"})


def run_one(repo: Path, module: str, config: str, ngpu: int, cards: str, timeout: int) -> Row:
    row = Row(module=module, config=config, ngpu=ngpu)
    env = os.environ.copy()
    env.update(
        {"MODULE": module, "CONFIG": config, "NPU": str(ngpu), "ASCEND_RT_VISIBLE_DEVICES": cards}
    )
    try:
        out = subprocess.run(
            [str(repo / "scripts/run_train.sh"), "--debug.seed", "42", "--debug.deterministic"],
            env=env,
            timeout=timeout,
            text=True,
            capture_output=True,
        )
        log = out.stdout + out.stderr
    except subprocess.TimeoutExpired:
        row.note = "timeout"
        return row
    row.steps = parse_steps(log)
    if out.returncode != 0 or not row.steps:
        row.note = f"rc={out.returncode}, {len(row.steps)} steps parsed"
        return row
    s = steady_state(row.steps)
    row.ok = True
    row.loss, row.tps, row.tflops = s["loss"], s["tps"], s["tflops"]
    row.mfu, row.memory_gib = s["mfu"], s["memory_gib"]
    try:
        row.ascend_nodes = ascend_nodes(module, config)
    except Exception as e:  # noqa: BLE001 - a missing provenance table is a note, not a crash
        row.note = f"provenance unavailable: {type(e).__name__}"
    return row


def render(rows: list[Row], tag: str) -> str:
    lines = [
        f"# 性能基线 · {tag}",
        "",
        f"生成于 {time.strftime('%Y-%m-%d %H:%M')}，"
        "`--debug.seed 42 --debug.deterministic`，取后半程步的中位数。",
        "",
        "| recipe | 卡 | step10 loss | tps | TFLOPs | MFU | 显存 | 生效的昇腾实现 |",
        "|---|:--:|---|---|---|---|---|---|",
    ]
    for r in rows:
        if not r.ok:
            lines.append(f"| `{r.config}` | {r.ngpu} | 🔴 {r.note} | | | | | |")
            continue
        if r.note:
            # A row whose provenance could not be collected is not a usable
            # benchmark row (P7); say so in the table instead of printing "—".
            nodes = f"⚠️ {r.note}"
        else:
            nodes = "<br>".join(n.split(".")[-2] for n in r.ascend_nodes) or "无（纯上游实现）"
        lines.append(
            f"| `{r.config}` | {r.ngpu} | {r.loss:.5f} | {r.tps:,} | {r.tflops:.2f} | "
            f"{r.mfu:.2f}% | {r.memory_gib:.2f} GiB | {nodes} |"
        )
    lines += [
        "",
        "没有 provenance 列的性能数字不收（P7）：eager 回退和融合算子跑出来的数字长得一样。",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from ascend_titan.tools.matrix import repo_root, tuple_tag

    repo = repo_root()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--cards", default="0")
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--out", default=None, help="directory for the markdown + json (default stdout)")
    p.add_argument(
        "--recipe",
        action="append",
        default=None,
        help="module:config[:ngpu], repeatable; defaults to the qwen3 eager/fused pair",
    )
    a = p.parse_args(argv)

    recipes = DEFAULT_RECIPES
    if a.recipe:
        recipes = []
        for spec in a.recipe:
            parts = spec.split(":")
            recipes.append((parts[0], parts[1], int(parts[2]) if len(parts) > 2 else 1))

    rows = []
    for module, config, ngpu in recipes:
        print(f"[bench] {config} on {ngpu} card(s)", flush=True)
        rows.append(run_one(repo, module, config, ngpu, a.cards, a.timeout))
    tag = tuple_tag()
    report = render(rows, tag)
    print(report)
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{time.strftime('%Y-%m-%d')}_{tag}.md").write_text(report)
        (out / f"{time.strftime('%Y-%m-%d')}_{tag}.json").write_text(
            json.dumps([asdict(r) for r in rows], indent=2)
        )
    return 0 if all(r.ok for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
