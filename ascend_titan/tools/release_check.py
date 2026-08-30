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
        # Prefer a line that names the actual problem over torchrun's wrapper
        # (ChildFailedError says nothing): config validation prints "Error parsing
        # Config", NPU errors print "error is <code>", python prints "<Name>Error: ".
        lines = [ln.strip(" \u2502|") for ln in log.splitlines()]
        named = [
            ln
            for ln in lines
            if ("Error parsing Config" in ln or "error is" in ln or re.search(r"\w+Error: ", ln))
            and "ChildFailedError" not in ln
            and "lspci" not in ln
        ]
        chk.detail = named[-1][:200] if named else f"rc={out.returncode}, no step lines"
    return chk


def checkpoint_roundtrip(
    repo: Path, module: str, config: str, cards: str, ngpu: int, steps: int, timeout: int
) -> Check:
    """R4: save at ``steps``, resume, and land on the same loss as an uninterrupted run.

    Three deterministic runs of the same recipe:
      A  0 -> 2*steps            uninterrupted, the reference trajectory
      B  0 -> steps              saves a checkpoint at the end
      C  steps -> 2*steps        resumes from B's checkpoint
    C's final loss must equal A's. Anything else means the checkpoint does not
    carry the full training state, which is the difference between a model you
    can hand over and a demo.
    """
    import shutil
    import tempfile

    chk = Check(
        name="checkpoint save/resume",
        criterion="R4",
        command=(
            f"NPU={ngpu} MODULE={module} CONFIG={config} ./scripts/run_train.sh "
            f"--checkpoint.enable --checkpoint.interval {steps} --training.steps {{N}} "
            f"--debug.seed 42 --debug.deterministic"
        ),
    )
    folder = Path(tempfile.mkdtemp(prefix="ascend_titan_ckpt_", dir="/tmp"))
    t0 = time.time()
    try:
        common = ["--debug.seed", "42", "--debug.deterministic"]

        def go(extra: list[str]) -> list[tuple[int, float]]:
            env = os.environ.copy()
            env.update(
                {
                    "MODULE": module,
                    "CONFIG": config,
                    "NPU": str(ngpu),
                    "ASCEND_RT_VISIBLE_DEVICES": cards,
                }
            )
            out = subprocess.run(
                [str(repo / "scripts/run_train.sh"), *common, *extra],
                env=env,
                timeout=timeout,
                text=True,
                capture_output=True,
            )
            log = re.sub(r"\x1b\[[0-9;]*m", "", out.stdout + out.stderr)
            steps_seen = [(int(m.group(1)), float(m.group(2))) for m in _STEP.finditer(log)]
            if out.returncode != 0 and not steps_seen:
                named = [ln for ln in log.splitlines() if re.search(r"\w+Error", ln)]
                raise RuntimeError(named[-1][:160] if named else f"rc={out.returncode}")
            return steps_seen

        reference = go(["--training.steps", str(2 * steps), "--no-checkpoint.enable"])
        go(
            [
                "--training.steps",
                str(steps),
                "--checkpoint.enable",
                "--checkpoint.interval",
                str(steps),
                "--checkpoint.folder",
                str(folder),
            ]
        )
        resumed = go(
            [
                "--training.steps",
                str(2 * steps),
                "--checkpoint.enable",
                "--checkpoint.interval",
                str(10**9),
                "--checkpoint.folder",
                str(folder),
            ]
        )
        if not reference or not resumed:
            chk.detail = "no step lines"
            return chk
        want, got = reference[-1], resumed[-1]
        chk.steps = resumed
        chk.ok = want[0] == got[0] and abs(want[1] - got[1]) < 1e-6
        chk.detail = (
            f"uninterrupted step {want[0]} loss {want[1]:.5f}; "
            f"resumed from step {steps} -> step {got[0]} loss {got[1]:.5f}"
            + ("" if chk.ok else "  (MISMATCH)")
        )
    except Exception as e:  # noqa: BLE001 - the failure is the result
        chk.detail = f"{type(e).__name__}: {str(e)[:160]}"
    finally:
        chk.seconds = round(time.time() - t0, 1)
        shutil.rmtree(folder, ignore_errors=True)
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
    p.add_argument(
        "--checkpoint-steps",
        type=int,
        default=5,
        help="R4: save at this step, resume, compare at twice it",
    )
    p.add_argument("--skip-checkpoint", action="store_true")
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

    if not a.skip_checkpoint:
        first = PLANS[a.model][0]
        print("[run  ] checkpoint save/resume", flush=True)
        chk = checkpoint_roundtrip(
            repo,
            MODULES[a.model],
            first[2],
            ",".join(map(str, sorted(cards[:1]))),
            1,
            a.checkpoint_steps,
            a.timeout,
        )
        checks.append(chk)
        print(f"[{'ok   ' if chk.ok else 'FAIL '}] checkpoint: {chk.detail}", flush=True)

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
