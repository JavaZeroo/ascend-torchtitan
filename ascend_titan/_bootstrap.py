"""Bootstrap: the single place where ascend-torchtitan has side effects.

Ordering constraint: ``torchtitan/tools/utils.py``
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
import os
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
    fla_npu_imported: bool = False
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
            f"fla_npu: imported={self.fla_npu_imported}",
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


def _npu_is_available() -> bool:
    try:
        import torch

        return bool(torch.npu.is_available())
    except Exception:  # noqa: BLE001 - only used to enrich an error message
        return False


def _import_torch_npu(report: SetupReport) -> None:
    """Import torch_npu, or raise. ``torch_npu`` is a base dependency (P14).

    There is no "run without an Ascend backend" mode: this package exists to run
    torchtitan on NPU, so a missing torch_npu is a broken environment and the
    ImportError must surface here rather than turn into a silent CPU run.
    """
    # sys.modules can hold a None entry (a blocked import), which is not an import.
    if _privateuse1_name() == "npu" and sys.modules.get("torch_npu") is not None:
        report.torch_npu_imported = True
        report.torch_npu_autoloaded = True
        return
    importlib.import_module("torch_npu")
    report.torch_npu_imported = True


def _import_fla_npu(report: SetupReport) -> None:
    """Import fla-npu, the optional R5 fused-GDN add-on (ADR-004).

    fla_npu must be imported after torch_npu and after the CANN environment has
    been sourced, and before the first NPU op runs (its __init__ sets the op-api
    library search path; a late import gives aclnnStatus=161001). Unlike torch_npu
    it is NOT a base dependency: a missing wheel only means the fused GDN kernel
    stays unavailable and kernels/gdn.py's plain-torch recurrence keeps running.
    This must therefore WARN and degrade, never raise.
    """
    try:
        importlib.import_module("fla_npu")
        report.fla_npu_imported = True
    except Exception as exc:  # noqa: BLE001 - optional add-on probe
        report.warnings.append(f"fla_npu unavailable ({exc}); fused GDN not available (ADR-004)")


def setup(*, apply_shims: bool = True) -> SetupReport:
    """Bootstrap the Ascend environment. Idempotent.

    Raises ``ImportError`` if ``torch_npu`` is not installed (P14).

    Args:
        apply_shims: apply every shim registered in ``ascend_titan.compat``.
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
    _import_fla_npu(report)

    from torch._utils import _get_available_device_type

    report.device_type = _get_available_device_type()
    # `torch.npu` only exists once a real torch_npu has registered its backend; CPU
    # unit tests install a stub module instead, and there is nothing to check there.
    if report.device_type != "npu" and hasattr(torch, "npu"):
        # torch_npu imported but torch does not see a usable NPU. torchtitan freezes
        # `utils.device_type` from this value at import time and falls back to "cuda",
        # so the next thing the user sees is
        # `AttributeError: module 'torch._C' has no attribute '_cuda_setDevice'`
        # several layers away. Fail here, where the cause is still visible.
        raise RuntimeError(
            f"torch_npu is imported but torch reports device_type={report.device_type!r}, "
            f"not 'npu' (torch.npu.is_available()={_npu_is_available()}). Usually the "
            f"selected card is busy or invalid: "
            f"ASCEND_RT_VISIBLE_DEVICES={os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '<unset>')}. "
            "Check `npu-smi info` for another job on that card."
        )

    # ASCEND_TITAN_SKIP_SHIMS=1 disables every shim (used to validate upstream
    # patches that make a shim unnecessary); ASCEND_TITAN_SKIP_SHIMS=a,b skips some.
    skip = os.environ.get("ASCEND_TITAN_SKIP_SHIMS", "")
    if apply_shims and skip != "1":
        from ascend_titan.compat import apply_all, list_shims

        only = None
        if skip:
            from ascend_titan.compat.registry import _discover

            _discover()
            only = {s.name for s in list_shims()} - set(skip.split(","))
        report.shims_applied = [a.name for a in apply_all(only=only)]
    elif apply_shims:
        report.warnings.append("all shims skipped (ASCEND_TITAN_SKIP_SHIMS=1)")
    if report.device_type != "npu":
        report.warnings.append(
            f"device_type={report.device_type!r}: no real Ascend backend registered "
            "(expected only in CPU unit tests, where torch_npu is a stub)"
        )

    logger.info(report.summary())
    _STATE = report
    return report


def reset_for_tests() -> None:
    """Forget setup state. Tests only."""
    global _STATE
    _STATE = None
