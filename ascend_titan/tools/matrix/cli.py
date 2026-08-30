"""CLI for the capability-matrix sweep.

    python -m ascend_titan.tools.matrix --suites features,models --cards 0-7 \
        --mode minimal --out outputs/matrix/$(date +%F)

``--mode`` picks what is applied to every upstream config: ``minimal`` (default,
only what NPU cannot run without -- so a red cell blames upstream, not our
kernels, P12), ``stock`` (nothing), ``fused`` (minimal + drop-in fused kernels).
``--retriage`` re-attributes an existing sweep after editing ``triage.toml``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import asdict
from pathlib import Path

from ascend_titan.tools.matrix.cases import load_cases
from ascend_titan.tools.matrix.report import render, tuple_tag
from ascend_titan.tools.matrix.runner import Result, sweep
from ascend_titan.tools.matrix.triage import triage


def parse_cards(spec: str) -> list[int]:
    cards: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            cards += list(range(int(a), int(b) + 1))
        elif part:
            cards.append(int(part))
    return cards


def repo_root() -> Path:
    """The checkout we run from (it owns scripts/run_train.sh and outputs/).

    Derived from the package, not from this file's depth: computing it as
    ``__file__.parents[N]`` breaks silently the moment a module moves, and the
    only symptom is that ``tests.integration_tests`` resolves to the wrong repo.
    """
    import ascend_titan

    return Path(ascend_titan.__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    repo = repo_root()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--suites", default="features,models")
    p.add_argument("--cards", default="0-7", help="NPU ids, e.g. 0-7 or 0,1,4")
    p.add_argument("--out", default=None)
    p.add_argument(
        "--titan-dir", default=os.environ.get("TITAN_DIR", str(repo.parent / "torchtitan"))
    )
    p.add_argument("--timeout", type=int, default=900, help="seconds per run")
    p.add_argument("--jobs", type=int, default=8, help="max concurrent cases")
    p.add_argument("--filter", default=None, help="regex on case name")
    p.add_argument(
        "--mode",
        default="minimal",
        choices=("minimal", "stock", "fused"),
        help="minimal: only what NPU cannot run without (default, P12); "
        "stock: upstream unmodified; fused: minimal + drop-in fused kernels",
    )
    p.add_argument("--list", action="store_true")
    p.add_argument("--retriage", default=None, help="re-run triage over an existing sweep dir")
    a = p.parse_args(argv)

    if a.retriage:
        out = Path(a.retriage)
        results = [Result(**r) for r in json.loads((out / "results.json").read_text())]
        for r in results:
            if r.state != "red":
                continue
            log = "".join(
                Path(run["log"]).read_text(errors="replace")
                for run in r.runs
                if Path(run["log"]).exists()
            )
            r.code, r.note = triage(log)
        (out / "results.json").write_text(json.dumps([asdict(r) for r in results], indent=2))
        (out / "report.md").write_text(render(results, title="Matrix sweep (re-triaged)"))
        print(render(results, title="Matrix sweep (re-triaged)"))
        return 0

    cases = load_cases(Path(a.titan_dir), a.suites.split(","), mode=a.mode)
    if a.filter:
        cases = [c for c in cases if re.search(a.filter, c.name)]
    if a.list:
        for c in cases:
            print(f"{c.suite:8} {c.name:45} ngpu={c.ngpu} {'SKIP: ' + c.skip if c.skip else ''}")
        return 0
    out = Path(
        a.out or repo / "outputs" / "matrix" / f"{time.strftime('%Y%m%d-%H%M')}_{tuple_tag()}"
    ).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(cases)} cases -> {out}", flush=True)
    results = sweep(cases, parse_cards(a.cards), out, repo, a.timeout, a.jobs)
    (out / "results.json").write_text(json.dumps([asdict(r) for r in results], indent=2))
    title = f"Matrix sweep (npu_{a.mode})" if a.mode != "stock" else "Matrix sweep (stock upstream)"
    (out / "report.md").write_text(render(results, title=title))
    print(render(results, title=title))
    return 0
