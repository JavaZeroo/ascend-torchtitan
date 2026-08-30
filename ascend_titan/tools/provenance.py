"""Provenance: which implementation actually backs every overridable node.

    ascend-titan-provenance --module ascend_titan.models.qwen3 --config qwen3_debugmodel_npu

Builds the config exactly as ``torchtitan.train`` would (module/config lookup +
``apply_overrides``), then walks the ``Trainer.Config`` tree and lists each
``Configurable`` node with the class that will be built and whether it comes
from ``ascend_titan``. This is the audit table P7 asks benchmarks to carry: a
node that silently stayed on upstream eager because a kernel dependency was
missing shows up here as ``upstream``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass


@dataclass
class Row:
    fqn: str
    config_cls: str
    origin: str  # "ascend" | "upstream-override" | "upstream"


def collect(config) -> list[Row]:
    from torchtitan.config.configurable import Configurable

    rows: list[Row] = []
    for fqn, cfg, _parent, _attr in config.traverse(Configurable.Config, recurse=True):
        cls = type(cfg)
        mod = cls.__module__
        if mod.startswith("ascend_titan."):
            origin = "ascend"
        elif ".overrides." in mod:
            origin = "upstream-override"
        else:
            origin = "upstream"
        rows.append(Row(fqn=fqn or "<root>", config_cls=f"{mod}.{cls.__qualname__}", origin=origin))
    return rows


def summarize(rows: list[Row]) -> dict[str, dict[str, int]]:
    """Per config class: how many nodes, collapsed over layer indices."""
    by_cls: Counter[tuple[str, str]] = Counter()
    for r in rows:
        by_cls[(r.config_cls, r.origin)] += 1
    return {f"{c}": {"origin": o, "nodes": n} for (c, o), n in sorted(by_cls.items())}


def build_config(module: str, config_name: str):
    from torchtitan.config.override import apply_overrides

    mod = importlib.import_module(module)
    fn = getattr(mod, config_name)
    config = fn()
    if config.override.imports:
        apply_overrides(config.override, config)
    return config


def build_config_isolated(module: str, config_name: str):
    """Build one config with a registry that is clean before and after.

    Tools that build several configs in one process (the matrix report, the
    benchmark) must not leak overrides between them -- and must not silently
    produce a config with *no* overrides applied, which is exactly the "looks
    green, measured the wrong thing" failure P7 exists to prevent.
    """
    from torchtitan.config.override import clear_overrides

    clear_overrides()
    try:
        mod = importlib.import_module(module)
        config = getattr(mod, config_name)()
        _reimport_override_modules(config)
        return build_config(module, config_name)
    finally:
        clear_overrides()


def _reimport_override_modules(config) -> None:
    """Re-run the @override decorators for this config's targets.

    ``clear_overrides()`` empties torchtitan's registry, but a module only runs
    its decorators on first import, so a second config in the same process finds
    an empty registry and no way to refill it. Reload the modules it names.
    """
    import importlib
    import sys

    for entry in config.override.imports:
        target = entry if isinstance(entry, str) else entry[0]
        module = target.rpartition(".")[0]
        if module in sys.modules:
            importlib.reload(sys.modules[module])


def render(summary: dict[str, dict[str, int]]) -> str:
    width = max(len(k) for k in summary) if summary else 10
    lines = ["ascend-titan provenance", "=" * (width + 30)]
    for cls, info in summary.items():
        marks = {"ascend": "ASCEND ", "upstream-override": "UP-OVR ", "upstream": "       "}
        mark = marks[info["origin"]]
        lines.append(f"{mark}{cls:<{width}}  x{info['nodes']}")
    n_ascend = sum(v["nodes"] for v in summary.values() if v["origin"] == "ascend")
    lines.append(f"ascend-backed nodes: {n_ascend}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--module", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--json", action="store_true")
    p.add_argument("--full", action="store_true", help="one row per node instead of per class")
    a = p.parse_args(argv)
    import ascend_titan

    ascend_titan.setup()
    rows = collect(build_config(a.module, a.config))
    if a.json:
        print(json.dumps([asdict(r) for r in rows] if a.full else summarize(rows), indent=2))
    elif a.full:
        for r in rows:
            print(f"{r.origin:<18} {r.fqn:<60} {r.config_cls}")
    else:
        print(render(summarize(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
