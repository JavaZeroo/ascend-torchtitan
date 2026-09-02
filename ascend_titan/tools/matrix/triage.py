"""Attribution: map a failure log to an attribution code.

The rules live in ``triage.toml`` next to this file -- data, not code, so adding
one never touches Python (and never gets reflowed by the formatter). First match
wins, which is why the specific rules are listed first.

Codes and what they oblige us to do are documented in ``docs/capability-matrix.md``;
their status is in ``docs/issues/STATUS.md`` (P11). ``UNKNOWN`` means a failure
shape we have never seen: read the log, attribute it, add a rule, ``--retriage``.
"""

from __future__ import annotations

import re
import tomllib
from functools import lru_cache
from pathlib import Path

RULES_PATH = Path(__file__).with_name("triage.toml")

_ERR_LINE = re.compile(r"([A-Za-z_.]*(?:Error|Exception)): ([^\n]*)")
_RECOVERED = re.compile(r"WARNING|falling back|fallback to")


@lru_cache(maxsize=1)
def rules() -> tuple[tuple[str, str, str], ...]:
    """``(code, pattern, note)`` in file order."""
    data = tomllib.loads(RULES_PATH.read_text())
    return tuple((r["code"], r["pattern"], r["note"]) for r in data["rule"])


def triage(log: str) -> tuple[str, str]:
    """``(code, note)`` for a failed run's combined log."""
    for code, pat, note in rules():
        m = re.search(pat, log)
        if m:
            if code == "NPU-OP":
                note = f"torch_npu op-plugin: {m.group(1)} failed, error {m.group(2)}"
            return code, note
    errs = [
        f"{name}: {msg.strip()}"
        for line in log.splitlines()
        # A line that reports a recovery is not the failure. torch_npu logs
        # "NPU streaming create_block_mask failed; falling back to the PyTorch
        # implementation: TypeError: ..." as a WARNING on every step, and it was
        # winning the fallback over the actual OutOfMemoryError further up
        # (measured 2026-09-02 on features/fsdp+cp).
        if not _RECOVERED.search(line)
        for name, msg in _ERR_LINE.findall(line)
        if "ChildFailedError" not in name and "error_file" not in msg
    ]
    return "UNKNOWN", (errs[-1][:200] if errs else "no error line found")
