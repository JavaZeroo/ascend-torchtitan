"""Report rendering: results -> the Markdown table that lands in docs/matrix/."""

from __future__ import annotations

import time

from ascend_titan.tools.matrix.runner import Result


def tuple_tag() -> str:
    import importlib.metadata as md

    def v(n):
        try:
            return md.version(n).split("+")[0]
        except md.PackageNotFoundError:
            return "none"

    return f"torch{v('torch')}_npu{v('torch_npu')}"


def render(results: list[Result], *, title: str) -> str:
    icon = {"green": "🟢", "red": "🔴", "skip": "⚪"}
    lines = [
        f"# {title}",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} · tuple `{tuple_tag()}`",
        "",
    ]
    counts = {s: sum(1 for r in results if r.state == s) for s in ("green", "red", "skip")}
    lines.append(f"**{counts['green']} 🟢 · {counts['red']} 🔴 · {counts['skip']} ⚪ (skipped)**")
    lines.append("")
    by_code: dict[str, int] = {}
    for r in results:
        if r.state == "red":
            by_code[r.code] = by_code.get(r.code, 0) + 1
    if by_code:
        lines.append(
            "Red by attribution: " + ", ".join(f"`{k}`×{v}" for k, v in sorted(by_code.items()))
        )
        lines.append("")
    for suite in sorted({r.suite for r in results}):
        lines += [
            f"## {suite}",
            "",
            "| case | ngpu | state | attribution | note | s |",
            "|---|---|---|---|---|---|",
        ]
        for r in sorted((x for x in results if x.suite == suite), key=lambda x: x.name):
            note = r.note.replace("|", "\\|")
            lines.append(
                f"| {r.name} | {r.ngpu} | {icon[r.state]} | {r.code} | {note} | {r.seconds:.0f} |"
            )
        lines.append("")
    return "\n".join(lines)
