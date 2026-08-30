"""Run the release checks for one model and record what actually happened.

    python -m ascend_titan.tools.release_check --model qwen3 --cards 0-7 --out docs/release

docs/model-release-criteria.md defines R1-R8. This driver covers the ones that are
a matter of running something (R1 real-size, R2 parallel coverage, R4 checkpoint
round-trip, R6 long run) and records the evidence; R3/R5/R7/R8 have their own tools
(check_golden.sh, bench.py, the READMEs, provenance) and are cross-referenced here.

Every check writes the command it ran and the output it got, because a release
claim without both is not evidence (P13).
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

_STEP = re.compile(r"step:\s*(\d+)\s+loss:\s*([\d.]+)\s+grad_norm:\s*([\d.na]+)")


@dataclass
class Check:
    name: str
    criterion: str
    command: str
    ok: bool = False
    detail: str = ""
    steps: list[tuple[int, float]] = field(default_factory=list)
    seconds: float = 0.0


# (name, criterion, config, ngpu, extra args)
PLANS: dict[str, list[tuple[str, str, str, int, list[str]]]] = {
    "qwen3": [
        ("real-size 1 NPU", "R1", "qwen3_0_6b_npu", 1, ["--training.steps", "20"]),
        ("FSDP2 x8", "R2", "qwen3_0_6b_npu_fsdp2", 8, ["--training.steps", "20"]),
        ("FSDP2 x4 + TP2", "R2", "qwen3_0_6b_npu_tp2", 8, ["--training.steps", "20"]),
        ("PP2 + FSDP2 x4", "R2", "qwen3_0_6b_npu_pp2", 8, ["--training.steps", "20"]),
    ],
    "qwen3_5": [
        ("real-size 1 NPU", "R1", "qwen35_0_8b_npu", 1, ["--training.steps", "20"]),
        ("FSDP2 x8", "R2", "qwen35_0_8b_npu_fsdp2", 8, ["--training.steps", "20"]),
    ],
}
MODULES = {"qwen3": "ascend_titan.models.qwen3", "qwen3_5": "ascend_titan.models.qwen3_5"}


def run(
    repo: Path, module: str, config: str, ngpu: int, cards: str, extra: list[str], timeout: int
) -> Check:
    cmd = f"NPU={ngpu} MODULE={module} CONFIG={config} ./scripts/run_train.sh {' '.join(extra)}"
    chk = Check(name=config, criterion="", command=cmd)
    env = os.environ.copy()
    env.update(
        {"MODULE": module, "CONFIG": config, "NPU": str(ngpu), "ASCEND_RT_VISIBLE_DEVICES": cards}
    )
    t0 = time.time()
    try:
        out = subprocess.run(
            [str(repo / "scripts/run_train.sh"), *extra],
            env=env,
            timeout=timeout,
            text=True,
            capture_output=True,
        )
        log = re.sub(r"\x1b\[[0-9;]*m", "", out.stdout + out.stderr)
    except subprocess.TimeoutExpired:
        chk.detail = f"timeout after {timeout}s"
        return chk
    finally:
        chk.seconds = round(time.time() - t0, 1)
    chk.steps = [(int(m.group(1)), float(m.group(2))) for m in _STEP.finditer(log)]
    if out.returncode == 0 and chk.steps:
        chk.ok = True
        first, last = chk.steps[0], chk.steps[-1]
        chk.detail = f"step {first[0]} loss {first[1]:.5f} -> step {last[0]} loss {last[1]:.5f}"
        if last[1] != last[1] or last[1] > first[1]:  # NaN or not decreasing
            chk.ok = False
            chk.detail += "  (loss did not decrease)"
    else:
        tail = [ln for ln in log.splitlines() if "Error" in ln or "error is" in ln]
        chk.detail = tail[-1][:200] if tail else f"rc={out.returncode}, no step lines"
    return chk


def render(model: str, checks: list[Check], tag: str) -> str:
    lines = [
        f"# release 检查 · {model} · {tag}",
        "",
        f"生成于 {time.strftime('%Y-%m-%d %H:%M')}。判据见 `docs/model-release-criteria.md`。",
        "",
        "| 检查 | 判据 | 结果 | 证据 | 秒 |",
        "|---|:--:|:--:|---|--:|",
    ]
    for c in checks:
        mark = "🟢" if c.ok else "🔴"
        lines.append(f"| `{c.name}` | {c.criterion} | {mark} | {c.detail} | {c.seconds:.0f} |")
    lines += ["", "复现命令：", "", "```bash"]
    lines += [f"ASCEND_RT_VISIBLE_DEVICES=... {c.command}" for c in checks]
    lines += ["```"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from ascend_titan.tools.matrix import repo_root, tuple_tag

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True, choices=sorted(PLANS))
    p.add_argument("--cards", default="0-7")
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    from ascend_titan.tools.matrix import parse_cards

    repo, cards = repo_root(), parse_cards(a.cards)
    checks: list[Check] = []
    for name, criterion, config, ngpu, extra in PLANS[a.model]:
        if ngpu > len(cards):
            print(f"[skip ] {name}: needs {ngpu} cards", flush=True)
            continue
        print(f"[run  ] {name} ({config}, {ngpu} card(s))", flush=True)
        chk = run(
            repo,
            MODULES[a.model],
            config,
            ngpu,
            ",".join(map(str, sorted(cards[:ngpu]))),
            extra,
            a.timeout,
        )
        chk.name, chk.criterion = name, criterion
        checks.append(chk)
        print(f"[{'ok   ' if chk.ok else 'FAIL '}] {name}: {chk.detail}", flush=True)

    report = render(a.model, checks, tuple_tag())
    print(report)
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{a.model}_{tuple_tag()}.md").write_text(report)
        (out / f"{a.model}_{tuple_tag()}.json").write_text(
            json.dumps([asdict(c) for c in checks], indent=2)
        )
    return 0 if all(c.ok for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
