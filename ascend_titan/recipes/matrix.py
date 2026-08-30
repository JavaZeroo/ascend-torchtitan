"""Dynamic recipe module for the capability matrix.

``--module ascend_titan.recipes.matrix --config <upstream.module>__<fn>`` resolves
to the upstream recipe function with :func:`npu_minimal` applied, so every upstream
integration-test config can be launched through the ordinary torchtitan CLI without
copying it. Two suffixes change what is applied:

    ``__stock``   nothing at all -- measures stock upstream on NPU
    ``__fused``   ``npu_minimal`` + ``npu_fused`` -- measures the perf kernels

The default deliberately excludes the fused kernels (P12): a red cell has to mean
"this upstream feature does not work on NPU", not "our drop-in kernel broke it".
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

from torchtitan.trainer import Trainer

SEP = "__"


MODES = ("minimal", "stock", "fused")


def encode(fn: Callable[[], Trainer.Config], *, mode: str = "minimal") -> str:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    name = f"{fn.__module__}{SEP}{fn.__name__}"
    return name if mode == "minimal" else f"{name}{SEP}{mode}"


def resolve(name: str) -> Callable[[], Trainer.Config]:
    parts = name.split(SEP)
    mode = "minimal"
    if parts[-1] in ("stock", "fused"):
        mode = parts[-1]
        parts = parts[:-1]
    if len(parts) != 2:
        raise AttributeError(
            f"matrix config must look like '<module>{SEP}<fn>[{SEP}stock|{SEP}fused]', got {name!r}"
        )
    module_path, fn_name = parts
    fn = getattr(importlib.import_module(module_path), fn_name)

    def build() -> Trainer.Config:
        config = fn()
        if mode != "stock":
            from ascend_titan.recipes.transforms import npu_fused, npu_minimal

            npu_minimal(config)
            if mode == "fused":
                npu_fused(config)
        return config

    build.__name__ = name
    build.__qualname__ = name
    return build


def __getattr__(name: str):
    if name.startswith("_") or SEP not in name:
        raise AttributeError(name)
    return resolve(name)
