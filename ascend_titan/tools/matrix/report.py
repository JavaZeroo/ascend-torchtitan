"""Report rendering: results -> the Markdown table that lands in docs/matrix/."""

from __future__ import annotations

import logging
import time

from ascend_titan.tools.matrix.runner import Result

logger = logging.getLogger(__name__)


def provenance(configs: list[str]) -> dict[str, tuple[str, int]]:
    """``{config class: (origin, how many configs use it)}`` over ``configs``.

    P7 asks every benchmark to carry an audit table saying which implementation
    actually backed each node; a sweep is a benchmark. Building a config applies
    its overrides into torchtitan's global registry, so each one is bracketed by
    ``clear_overrides()``; configs that fail to build are skipped (they are the
    red cells, and the report already says why).
    """
    from ascend_titan.tools.provenance import build_config_isolated, collect

    seen: dict[str, tuple[str, int]] = {}
    for name in sorted(set(configs)):
        try:
            rows = collect(build_config_isolated("ascend_titan.recipes.matrix", name))
        except Exception as e:  # noqa: BLE001 - a config that cannot build is a red cell
            logger.info("[provenance] %s: %s", name, type(e).__name__)
            continue
        for cls in {(r.config_cls, r.origin) for r in rows}:
            origin, count = seen.get(cls[0], (cls[1], 0))
            seen[cls[0]] = (origin, count + 1)
    return seen


def render_provenance(table: dict[str, tuple[str, int]]) -> list[str]:
    """The audit table, Ascend-backed nodes first."""
    if not table:
        return []
    order = {"ascend": 0, "upstream-override": 1, "upstream": 2}
    lines = [
        "## provenance",
        "",
        "Which implementation actually backs each overridable node (P7). "
        "`ascend` = our override was in effect.",
        "",
        "| config class | origin | cases |",
        "|---|---|---|",
    ]
    for cls, (origin, count) in sorted(table.items(), key=lambda kv: (order[kv[1][0]], kv[0])):
        lines.append(f"| `{cls}` | {origin} | {count} |")
    lines.append("")
    return lines


def tuple_tag() -> str:
    import importlib.metadata as md

    def v(n):
        try:
            return md.version(n).split("+")[0]
        except md.PackageNotFoundError:
            return "none"

    return f"torch{v('torch')}_npu{v('torch_npu')}"


def render(
    results: list[Result], *, title: str, provenance_table: dict[str, tuple[str, int]] | None = None
) -> str:
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
    lines += render_provenance(provenance_table or {})
    return "\n".join(lines)
