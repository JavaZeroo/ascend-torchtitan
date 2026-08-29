"""Bootstrap: the single place where ascend-torchtitan has side effects.

Ordering constraint (see docs/design, finding F4): ``torchtitan/tools/utils.py``
freezes ``device_type`` at *module import time* via
``torch._utils._get_available_device_type()``, which only reports ``"npu"`` if
``torch_npu`` has already registered itself as the privateuse1 backend. Hence
``setup()`` must run before the first ``import torchtitan``.

PyTorch >= 2.5 can autoload device backends declared through the
``torch.backends`` entry point (``torch._import_device_backends``); whether the
installed ``torch_npu`` uses it is probed here and reported, never assumed.
"""

from __future__ import annotations

import importlib
import logging
import sys
from dataclasses import dataclass, field

logger = logging.getLogger("ascend_titan")

_STATE: SetupReport | None = None


@dataclass
class SetupReport:
    """What ``setup()`` actually did. Logged once and returned for tests/tools."""

    torch_version: str = ""
    torch_npu_imported: bool = False
    torch_npu_autoloaded: bool = False
    device_type: str | None = None
    torchtitan_was_already_imported: bool = False
    shims_applied: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"torch={self.torch_version} device_type={self.device_type}",
            (
                f"torch_npu: imported={self.torch_npu_imported} "
                f"autoloaded={self.torch_npu_autoloaded}"
            ),
            f"shims applied ({len(self.shims_applied)}): {', '.join(self.shims_applied) or '-'}",
        ]
        lines += [f"WARNING: {w}" for w in self.warnings]
        return "\n".join("[ascend_titan] " + line for line in lines)


def _privateuse1_name() -> str | None:
    import torch

    try:
        return torch._C._get_privateuse1_backend_name()
    except Exception:  # pragma: no cover - very old torch
        return None


def _import_torch_npu(report: SetupReport) -> None:
    """Import torch_npu unless torch already autoloaded it."""
    if _privateuse1_name() == "npu" and "torch_npu" in sys.modules:
        report.torch_npu_imported = True
        report.torch_npu_autoloaded = True
        return
    try:
        importlib.import_module("torch_npu")
        report.torch_npu_imported = True
    except ImportError:
        report.warnings.append(
            "torch_npu is not installed; running without an Ascend backend "
            "(fine for CPU unit tests, wrong for training)."
        )


def setup(*, apply_shims: bool = True, require_npu: bool = False) -> SetupReport:
    """Bootstrap the Ascend environment. Idempotent.

    Args:
        apply_shims: apply every shim registered in ``ascend_titan.compat``.
        require_npu: raise if ``torch_npu`` cannot be imported.
    """
    global _STATE
    if _STATE is not None:
        return _STATE

    report = SetupReport()
    report.torchtitan_was_already_imported = any(
        m == "torchtitan" or m.startswith("torchtitan.") for m in sys.modules
    )
    if report.torchtitan_was_already_imported:
        report.warnings.append(
            "torchtitan was imported before ascend_titan.setup(); "
            "torchtitan.tools.utils.device_type is already frozen and shims that "
            "wrap import-time state may not take effect. Call setup() first, or "
            "use `python -m ascend_titan.train`."
        )

    import torch

    report.torch_version = torch.__version__
    _import_torch_npu(report)
    if require_npu and not report.torch_npu_imported:
        raise RuntimeError("torch_npu is required but could not be imported")

    from torch._utils import _get_available_device_type

    report.device_type = _get_available_device_type()

    if apply_shims:
        from ascend_titan.compat import apply_all

        report.shims_applied = [a.name for a in apply_all()]

    logger.info(report.summary())
    _STATE = report
    return report


def reset_for_tests() -> None:
    """Forget setup state. Tests only."""
    global _STATE
    _STATE = None
