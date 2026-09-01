"""Give every upstream flavor an NPU entry point, without a function per flavor.

Upstream ships a config function per model size (qwen3_5 alone has nine, and adds
more), while what a model family needs on Ascend is the *same* short list of deltas
for all of them. Writing one wrapper per flavor makes that list a copy-paste that
goes stale the moment upstream lands ``qwen35_397b_a17b``.

So a model package declares its deltas once -- explicitly, one call per delta, in
its ``recipes.py`` -- and this module exposes ``<upstream_flavor>_npu`` for every
flavor upstream has:

    python -m ascend_titan.train --module ascend_titan.models.qwen3_5 \\
        --config qwen35_27b_npu          # never hand-written; resolved here

A hand-written function of the same name always wins: Python looks in the module
dict before calling ``__getattr__``. Those stay for flavors that need more than the
family deltas -- real tokenizer/dataset assets, a pipeline-parallel layout, a golden
-frozen smoke config -- and they call the same ``deltas`` callable, so the family
knowledge still exists in exactly one place.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from types import ModuleType

SUFFIX = "_npu"


def _is_config_factory(fn: object, module_name: str) -> bool:
    """A zero-argument callable defined in that module (not imported into it)."""
    if isinstance(fn, type) or not callable(fn):
        return False
    if getattr(fn, "__module__", None) != module_name:
        return False
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return all(
        p.default is not p.empty or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params
    )


def upstream_flavors(upstream: ModuleType) -> dict[str, Callable[[], object]]:
    """Every config factory upstream defines, by name."""
    return {
        name: fn
        for name, fn in vars(upstream).items()
        if not name.startswith("_") and _is_config_factory(fn, upstream.__name__)
    }


def npu_entry_points(upstream: ModuleType, deltas: Callable[[object, str], None]):
    """Return ``(__getattr__, __dir__)`` for a model package's ``__init__``.

    ``deltas(config, flavor)`` is the family's declaration: what this model needs on
    Ascend, applied in place. It takes the flavor name so per-size *data* (which HF
    tokenizer a real-size run should use, say) stays a table lookup in the family's
    declaration rather than another hand-written function per size.
    """

    def __getattr__(name: str):
        if name.startswith("_") or not name.endswith(SUFFIX):
            raise AttributeError(name)
        flavor = name[: -len(SUFFIX)]
        flavors = upstream_flavors(upstream)
        fn = flavors.get(flavor)
        if fn is None:
            raise AttributeError(
                f"{name}: {upstream.__name__} has no config '{flavor}'. "
                f"Upstream flavors: {sorted(flavors)}"
            )

        def build():
            config = fn()
            deltas(config, flavor)
            return config

        build.__name__ = name
        build.__qualname__ = name
        build.__doc__ = f"Upstream ``{flavor}`` + this model's Ascend deltas (generated)."
        return build

    def __dir__() -> list[str]:
        return sorted(f"{flavor}{SUFFIX}" for flavor in upstream_flavors(upstream))

    return __getattr__, __dir__
