"""Dynamic recipe module for the capability matrix.

``--module ascend_titan.recipes.matrix --config <upstream.module>__<fn>`` resolves
to the upstream recipe function with :func:`npu_baseline` applied, so every
upstream integration-test config can be launched through the ordinary
torchtitan CLI without copying it. ``--config <upstream.module>__<fn>__stock``
launches it unmodified (to measure stock upstream on NPU).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

from torchtitan.trainer import Trainer

SEP = "__"


def encode(fn: Callable[[], Trainer.Config], *, stock: bool = False) -> str:
    name = f"{fn.__module__}{SEP}{fn.__name__}"
    return f"{name}{SEP}stock" if stock else name


def resolve(name: str) -> Callable[[], Trainer.Config]:
    parts = name.split(SEP)
    stock = parts[-1] == "stock"
    if stock:
        parts = parts[:-1]
    if len(parts) != 2:
        raise AttributeError(
            f"matrix config must look like '<module>{SEP}<fn>[{SEP}stock]', got {name!r}"
        )
    module_path, fn_name = parts
    fn = getattr(importlib.import_module(module_path), fn_name)

    def build() -> Trainer.Config:
        config = fn()
        if not stock:
            from ascend_titan.recipes.transforms import npu_baseline

            npu_baseline(config)
        return config

    build.__name__ = name
    build.__qualname__ = name
    return build


def __getattr__(name: str):
    if name.startswith("_") or SEP not in name:
        raise AttributeError(name)
    return resolve(name)
