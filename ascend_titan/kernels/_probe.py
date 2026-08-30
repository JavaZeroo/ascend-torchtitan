"""Dependency probes for L1 kernel modules.

``torch_npu`` is a **base dependency**, not an optional accelerator: every L1
module imports it unconditionally (principle P14). A missing ``torch_npu`` is a
broken environment and must fail at import, loudly, with a stack trace -- never
degrade into a silent eager path, because that turns "the fused kernel was not
installed" into an unnoticed performance regression and, worse, into numbers
that look fine but were never measured on the kernel we claim to ship.

The same holds one level down: an op missing from ``torch_npu`` is a gap on the
Ascend side, so :func:`require_op` raises and points at the P9 flow (fix
torch_npu, verify locally, file the issue/PR on gitcode) instead of falling back.

Genuinely optional add-ons -- packages that need a separate build and are not
part of the baseline, e.g. ops-nn's ``cann_ops_nn`` -- go through
:func:`optional_module`, which is the only sanctioned "warn and degrade" path
(ADR-004).
"""

from __future__ import annotations

import importlib
import logging
from types import ModuleType

import torch_npu

logger = logging.getLogger(__name__)

__all__ = [
    "MissingNpuOpError",
    "installed_modules",
    "optional_module",
    "require_op",
    "torch_npu",
]


class MissingNpuOpError(RuntimeError):
    """``torch_npu`` is installed but does not provide an op we need."""


def require_op(name: str):
    """Return ``torch_npu.<name>`` or raise. Never returns ``None``."""
    op = getattr(torch_npu, name, None)
    if op is None:
        raise MissingNpuOpError(
            f"torch_npu {getattr(torch_npu, '__version__', '?')} does not provide "
            f"torch_npu.{name}. This is an Ascend-side gap: fix it in torch_npu / "
            f"op-plugin, verify locally, then file the issue and PR on "
            f"gitcode.com/Ascend (principle P9). Do not work around it here."
        )
    return op


def installed_modules(prefix: str) -> list[str]:
    """Importable top-level module names starting with ``prefix``.

    ops-nn names its torch extension after the vendor it was built for
    (``cann_ops_nn_<vendor>``), so the module name depends on how the run
    package was installed. Discover it instead of guessing.
    """
    import pkgutil

    return sorted(m.name for m in pkgutil.iter_modules() if m.name.startswith(prefix))


def optional_module(*candidates: str) -> tuple[ModuleType | None, Exception | None]:
    """Import the first importable module of ``candidates`` (optional add-ons only).

    Returns ``(module, None)`` or ``(None, last_error)``. Callers must log a
    WARNING naming what is degraded (ADR-004).
    """
    err: Exception | None = None
    for name in candidates:
        try:
            return importlib.import_module(name), None
        except Exception as e:  # noqa: BLE001 - JIT builders raise many kinds of errors
            err = e
    return None, err
