"""Bisect an upstream regression: which torchtitan commit turned a case red?

    python -m ascend_titan.tools.bisect --config <matrix config> \\
        --good $(cat constraints/torchtitan.sha) --bad origin/main --cards 0,1

The nightly CI runs two legs (the pinned SHA and the furthest importable one).
When the second leg goes red and the first stays green, the answer is a commit
in between, and finding it by hand costs an afternoon. This drives
``git bisect run`` over a **scratch clone** -- ``../torchtitan`` is the pinned
checkout and is never touched (CLAUDE.md) -- with the case itself as the test.

Each candidate runs in a subprocess with ``PYTHONPATH`` pointing at the scratch
clone, so no reinstall is needed between steps. Exit codes follow git bisect:
0 = good, 1 = bad, 125 = skip (the commit does not even import, which is a
different failure and must not be blamed for this one).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SKIP = 125


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, **kw)


def ensure_clone(source: Path, scratch: Path) -> None:
    """A local clone of the pinned checkout, on local disk (P13: never on NFS)."""
    if (scratch / ".git").is_dir():
        _run(["git", "-C", str(scratch), "fetch", "--all", "--quiet"])
        return
    scratch.parent.mkdir(parents=True, exist_ok=True)
    out = _run(["git", "clone", "--quiet", str(source), str(scratch)])
    if out.returncode != 0:
        raise SystemExit(f"clone failed: {out.stderr[-500:]}")


def probe_one(scratch: Path, repo: Path, config: str, cards: str, ngpu: int, timeout: int) -> int:
    """Run one case against the checked-out scratch tree. Returns a bisect code."""
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": f"{scratch}:{env.get('PYTHONPATH', '')}".rstrip(":"),
            "MODULE": "ascend_titan.recipes.matrix",
            "CONFIG": config,
            "NPU": str(ngpu),
            "ASCEND_RT_VISIBLE_DEVICES": cards,
        }
    )
    importable = _run(
        [sys.executable, "-c", "import torchtitan, torchtitan.trainer  # noqa"], env=env
    )
    if importable.returncode != 0:
        print("  [skip] torchtitan does not import at this commit", flush=True)
        return SKIP
    try:
        out = subprocess.run(
            [str(repo / "scripts/run_train.sh")],
            env=env,
            timeout=timeout,
            text=True,
            capture_output=True,
        )
    except subprocess.TimeoutExpired:
        print("  [bad ] timed out", flush=True)
        return 1
    code = 0 if out.returncode == 0 else 1
    print(f"  [{'good' if code == 0 else 'bad '}] rc={out.returncode}", flush=True)
    return code


def main(argv: list[str] | None = None) -> int:
    from ascend_titan.tools.matrix import repo_root

    repo = repo_root()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", required=True, help="matrix config name (module__fn[__mode])")
    p.add_argument("--good", required=True, help="known-good torchtitan ref")
    p.add_argument("--bad", default="origin/main", help="known-bad torchtitan ref")
    p.add_argument("--cards", default="0")
    p.add_argument("--ngpu", type=int, default=1)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument(
        "--titan-dir", default=os.environ.get("TITAN_DIR", str(repo.parent / "torchtitan"))
    )
    p.add_argument("--scratch", default="/opt/build/torchtitan-bisect")
    p.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)  # used by git bisect run
    a = p.parse_args(argv)

    scratch = Path(a.scratch)
    if a.probe:
        return probe_one(scratch, repo, a.config, a.cards, a.ngpu, a.timeout)

    ensure_clone(Path(a.titan_dir), scratch)
    for step in (
        ["bisect", "reset"],
        ["bisect", "start"],
        ["bisect", "bad", a.bad],
        ["bisect", "good", a.good],
    ):
        out = _run(["git", "-C", str(scratch), *step])
        if out.returncode != 0 and step[1] != "reset":
            raise SystemExit(f"git {' '.join(step)} failed: {out.stderr[-300:]}")

    run = _run(
        [
            "git",
            "-C",
            str(scratch),
            "bisect",
            "run",
            sys.executable,
            "-m",
            "ascend_titan.tools.bisect",
            "--probe",
            "--config",
            a.config,
            "--good",
            a.good,
            "--bad",
            a.bad,
            "--cards",
            a.cards,
            "--ngpu",
            str(a.ngpu),
            "--timeout",
            str(a.timeout),
            "--scratch",
            str(scratch),
        ]
    )
    print(run.stdout[-4000:])
    print(run.stderr[-2000:], file=sys.stderr)
    culprit = [ln for ln in run.stdout.splitlines() if "is the first bad commit" in ln]
    if culprit:
        print("\n" + culprit[0])
        print("record it in docs/issues/torchtitan.md and, if it is a version gap, close it (P8)")
    _run(["git", "-C", str(scratch), "bisect", "reset"])
    return 0 if culprit else 1


if __name__ == "__main__":
    sys.exit(main())
