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


# A short probe must keep the recipe's real LR curve. ``lr_scheduler.total_steps``
# falls back to ``training.steps`` and clamps ``warmup_steps`` down to it, so
# ``--training.steps 20`` on a recipe configured for 1000 steps with 20 warmup
# steps reaches full LR immediately -- qwen3.5-0.8B (lr 5e-3) then goes non-finite
# by step 4, which says nothing about Ascend. Pin the schedule to the recipe's own
# ``training.steps`` default.
_SHORT = ["--training.steps", "20", "--lr_scheduler.total_steps", "1000"]

# (name, criterion, config, ngpu, extra args)
PLANS: dict[str, list[tuple[str, str, str, int, list[str]]]] = {
    "qwen3": [
        ("real-size 1 NPU", "R1", "qwen3_0_6b_npu", 1, _SHORT),
        ("FSDP2 x8", "R2", "qwen3_0_6b_npu_fsdp2", 8, _SHORT),
        ("FSDP2 x4 + TP2", "R2", "qwen3_0_6b_npu_tp2", 8, _SHORT),
        ("PP2 + FSDP2 x4", "R2", "qwen3_8b_npu_pp2", 8, _SHORT),
    ],
    "qwen3_5": [
        ("real-size 1 NPU", "R1", "qwen35_0_8b_npu", 1, _SHORT),
        ("FSDP2 x8", "R2", "qwen35_0_8b_npu_fsdp2", 8, _SHORT),
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
        {
            "MODULE": module,
            "CONFIG": config,
            "NPU": str(ngpu),
            "ASCEND_RT_VISIBLE_DEVICES": cards,
            # Log the LAST rank, not rank 0. Under pipeline parallel only the final
            # stage computes the loss; every other rank reports a placeholder
            # (`loss: -4.00000`), so a rank-0 log makes a healthy PP run look like
            # a broken one. For non-PP runs every rank logs the same number.
            "LOG_RANK": str(ngpu - 1),
        }
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

    Two things make this comparison meaningless if you get them wrong, and both
    cost a day to find:

    * ``lr_scheduler.total_steps`` falls back to ``training.steps``, and
      ``warmup_steps`` (200 by default) is clamped down to it. Run B with
      ``--training.steps 5`` therefore warms up over 5 steps while run A warms up
      over 10 -- different learning rates for the same five steps, so C lands
      somewhere A never visits and the checkpoint looks broken when it is fine.
      Pin ``--lr_scheduler.total_steps`` to 2*steps in all three runs.
    * ``checkpoint.folder`` is resolved *inside* the dump folder, so runs must be
      separated with ``--dump_folder``: A alone, B and C sharing one. Reusing a
      folder makes C resume from the checkpoint C itself wrote last time.
    """
    import shutil
    import tempfile

    chk = Check(
        name="checkpoint save/resume",
        criterion="R4",
        command=(
            f"NPU={ngpu} MODULE={module} CONFIG={config} ./scripts/run_train.sh "
            f"--checkpoint.enable --checkpoint.interval {steps} --training.steps {{N}} "
            f"--lr_scheduler.total_steps {2 * steps} --debug.seed 42 --debug.deterministic"
        ),
    )
    root = Path(tempfile.mkdtemp(prefix="ascend_titan_ckpt_", dir="/tmp"))
    t0 = time.time()
    try:
        # Same LR curve in all three runs regardless of how many steps each takes.
        common = [
            "--debug.seed",
            "42",
            "--debug.deterministic",
            "--lr_scheduler.total_steps",
            str(2 * steps),
        ]

        def go(extra: list[str]) -> list[tuple[int, float]]:
            env = os.environ.copy()
            env.update(
                {
                    "MODULE": module,
                    "CONFIG": config,
                    "NPU": str(ngpu),
                    "ASCEND_RT_VISIBLE_DEVICES": cards,
                    "LOG_RANK": str(ngpu - 1),
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
                named = [
                    ln.strip(" \u2502|")
                    for ln in log.splitlines()
                    if re.search(r"\w+Error: |Error parsing Config|error is ", ln)
                    and "ChildFailedError" not in ln
                ]
                raise RuntimeError(named[-1][:160] if named else f"rc={out.returncode}")
            return steps_seen

        reference = go(
            [
                "--training.steps",
                str(2 * steps),
                "--checkpoint.no-enable",
                "--dump_folder",
                str(root / "reference"),
            ]
        )
        resumable = str(root / "resumed")
        go(
            [
                "--training.steps",
                str(steps),
                "--checkpoint.enable",
                "--checkpoint.interval",
                str(steps),
                "--dump_folder",
                resumable,
            ]
        )
        resumed = go(
            [
                "--training.steps",
                str(2 * steps),
                "--checkpoint.enable",
                "--checkpoint.interval",
                str(10**9),
                "--dump_folder",
                resumable,
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
        shutil.rmtree(root, ignore_errors=True)
    return chk


def hf_roundtrip(
    repo: Path, module: str, config: str, cards: str, ngpu: int, steps: int, timeout: int
) -> Check:
    """R4's third item: can the trained model leave torchtitan and come back?

    Two runs:
      A  0 -> steps, ``--checkpoint.last_save_in_hf``  -> HF safetensors
      B  1 step, ``--checkpoint.initial_load_in_hf`` from A's directory
    An HF export is model-only, so B starts with a fresh optimizer and a fresh
    dataloader; its trajectory cannot match A's. What it *can* show, and what a
    broken ``state_dict_adapter`` would fail, is that B starts from a trained
    model rather than a random one: B's first loss must be near A's last, not
    back up at ``ln(vocab_size)``.
    """
    import shutil
    import tempfile

    chk = Check(
        name="HF export/import",
        criterion="R4",
        command=(
            f"NPU={ngpu} MODULE={module} CONFIG={config} ./scripts/run_train.sh "
            f"--checkpoint.enable --checkpoint.last_save_in_hf --training.steps {steps} "
            f"# then --checkpoint.initial_load_in_hf --checkpoint.initial_load_path <dir>"
        ),
    )
    root = Path(tempfile.mkdtemp(prefix="ascend_titan_hf_", dir="/tmp"))
    t0 = time.time()
    try:
        common = ["--debug.seed", "42", "--lr_scheduler.total_steps", str(2 * steps)]

        def go(extra: list[str]) -> list[tuple[int, float]]:
            env = os.environ.copy()
            env.update(
                {
                    "MODULE": module,
                    "CONFIG": config,
                    "NPU": str(ngpu),
                    "ASCEND_RT_VISIBLE_DEVICES": cards,
                    "LOG_RANK": str(ngpu - 1),
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
            seen = [(int(m.group(1)), float(m.group(2))) for m in _STEP.finditer(log)]
            if out.returncode != 0 and not seen:
                named = [
                    ln.strip(" \u2502|")
                    for ln in log.splitlines()
                    if re.search(r"\w+Error: |Error parsing Config|error is ", ln)
                    and "ChildFailedError" not in ln
                ]
                raise RuntimeError(named[-1][:160] if named else f"rc={out.returncode}")
            return seen

        exported = str(root / "exported")
        trained = go(
            [
                "--training.steps",
                str(steps),
                "--checkpoint.enable",
                "--checkpoint.interval",
                str(steps),
                "--checkpoint.last_save_in_hf",
                # last_save_in_hf is rejected without it: an HF export is a model
                # snapshot, not a resumable checkpoint.
                "--checkpoint.last_save_model_only",
                "--dump_folder",
                exported,
            ]
        )
        loaded = go(
            [
                "--training.steps",
                "1",
                "--checkpoint.enable",
                "--checkpoint.interval",
                str(10**9),
                "--checkpoint.initial_load_in_hf",
                "--checkpoint.initial_load_path",
                f"{exported}/checkpoint/step-{steps}",
                "--dump_folder",
                str(root / "reloaded"),
            ]
        )
        if not trained or not loaded:
            chk.detail = "no step lines"
            return chk
        chk.steps = loaded
        before, after = trained[-1][1], loaded[0][1]
        # Random init sits at ln(vocab); anything near it means the weights did
        # not survive the trip. Half the gap back to random is the line.
        untrained = trained[0][1]
        chk.ok = after < before + 0.5 * max(untrained - before, 0.0)
        chk.detail = (
            f"exported at step {steps} loss {before:.5f}; reloaded loss {after:.5f} "
            f"(untrained was {untrained:.5f})" + ("" if chk.ok else "  (LOST THE WEIGHTS)")
        )
    except Exception as e:  # noqa: BLE001 - the failure is the result
        chk.detail = f"{type(e).__name__}: {str(e)[:160]}"
    finally:
        chk.seconds = round(time.time() - t0, 1)
        shutil.rmtree(root, ignore_errors=True)
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

        print("[run  ] HF export/import", flush=True)
        chk = hf_roundtrip(
            repo,
            MODULES[a.model],
            first[2],
            ",".join(map(str, sorted(cards[:1]))),
            1,
            a.checkpoint_steps,
            a.timeout,
        )
        checks.append(chk)
        print(f"[{'ok   ' if chk.ok else 'FAIL '}] HF: {chk.detail}", flush=True)

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
