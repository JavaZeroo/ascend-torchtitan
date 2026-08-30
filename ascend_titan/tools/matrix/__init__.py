"""Capability-matrix sweep: run upstream integration-test configs on NPU, attribute
every failure, write a report.

    python -m ascend_titan.tools.matrix --suites features,models --cards 0-7

Split by responsibility so each piece can be read and tested on its own:

    triage.py    log -> attribution code (rules in triage.toml, data not code)
    cases.py     upstream's test lists -> Case objects
    runner.py    card pool + subprocess execution -> Result objects
    report.py    Results -> Markdown
    cli.py       argument parsing and the sweep/retriage entry points
"""

from ascend_titan.tools.matrix.cases import Case, load_cases
from ascend_titan.tools.matrix.cli import main, parse_cards, repo_root
from ascend_titan.tools.matrix.report import provenance, render, tuple_tag
from ascend_titan.tools.matrix.runner import (
    CardPool,
    Result,
    hccl_base_port,
    run_case,
    sweep,
)
from ascend_titan.tools.matrix.triage import rules, triage

__all__ = [
    "Case",
    "CardPool",
    "hccl_base_port",
    "Result",
    "load_cases",
    "main",
    "parse_cards",
    "provenance",
    "render",
    "repo_root",
    "rules",
    "run_case",
    "sweep",
    "triage",
    "tuple_tag",
]
