"""Shim registry: register, validate, apply and audit monkeypatches."""

from __future__ import annotations

import importlib
import logging
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger("ascend_titan.compat")

Kind = Literal["wrap", "replace", "polyfill"]


class ShimError(RuntimeError):
    pass


@dataclass(frozen=True)
class Shim:
    """One governed monkeypatch.

    ``target`` is ``"dotted.module:attr"``; ``attr`` may itself be dotted to reach
    a class attribute (``"pkg.mod:Class.attr"``). ``fn(original)`` receives the
    current attribute and returns the patched one; for ``kind="wrap"`` it must
    call ``original`` so upstream changes are inherited (P3).

    ``kind="polyfill"`` adds an attribute that the installed torch/torchtitan
    lacks (typically a nightly-only API): ``fn(None)`` returns the implementation
    and the shim is skipped entirely when the attribute already exists, so a
    newer torch makes it a no-op without any code change.
    """

    name: str
    target: str
    reason: str
    upstream: str
    kind: Kind
    fn: Callable[[object], object]
    why_not_wrap: str | None = None

    @property
    def module(self) -> str:
        return self.target.split(":", 1)[0]

    @property
    def attr(self) -> str:
        """The last component -- what ``setattr`` is called with."""
        return self.target.split(":", 1)[1].rsplit(".", 1)[-1]

    @property
    def owner_path(self) -> list[str]:
        """Attribute names between the module and :attr:`attr` (empty for a plain attr)."""
        path = self.target.split(":", 1)[1].split(".")
        return path[:-1]

    def owner(self, module: object) -> object:
        """The object that actually holds :attr:`attr` (the module, or a class in it)."""
        obj = module
        for name in self.owner_path:
            obj = getattr(obj, name)
        return obj


@dataclass(frozen=True)
class Applied:
    name: str
    target: str
    kind: Kind


_REGISTRY: dict[str, Shim] = {}
_APPLIED: dict[str, Applied] = {}


def shim(
    *,
    target: str,
    reason: str,
    upstream: str,
    kind: Kind = "wrap",
    why_not_wrap: str | None = None,
) -> Callable[[Callable[[object], object]], Callable[[object], object]]:
    """Register a shim. Validation happens at import time so a malformed shim
    fails CI, not a training job."""

    if ":" not in target or target.startswith(":") or target.endswith(":"):
        raise ShimError(f"target must be 'module:attr', got {target!r}")
    if not reason.strip():
        raise ShimError(f"shim for {target}: reason is required")
    if not upstream.strip():
        raise ShimError(
            f"shim for {target}: upstream issue/PR link is required (P4). "
            "File one before adding the shim."
        )
    if kind == "replace" and not (why_not_wrap and why_not_wrap.strip()):
        raise ShimError(f"shim for {target}: kind='replace' requires why_not_wrap (P3).")
    if kind not in ("wrap", "replace", "polyfill"):
        raise ShimError(f"shim for {target}: unknown kind {kind!r}")
    if not (upstream.startswith("http") or upstream.startswith("draft:")):
        raise ShimError(
            f"shim for {target}: upstream must be an issue/PR URL or 'draft:<path#anchor>' "
            "pointing at the drafted issue text in docs/"
        )

    def deco(fn: Callable[[object], object]) -> Callable[[object], object]:
        name = fn.__name__
        if name in _REGISTRY:
            raise ShimError(f"duplicate shim name {name!r}")
        _REGISTRY[name] = Shim(
            name=name,
            target=target,
            reason=reason,
            upstream=upstream,
            kind=kind,
            fn=fn,
            why_not_wrap=why_not_wrap,
        )
        return fn

    return deco


def list_shims() -> list[Shim]:
    return list(_REGISTRY.values())


def _discover() -> None:
    """Import every module under ascend_titan.compat.shims so decorators run."""
    import ascend_titan.compat.shims as pkg

    for info in pkgutil.iter_modules(pkg.__path__, pkg.__name__ + "."):
        importlib.import_module(info.name)


def apply_all(*, only: set[str] | None = None) -> list[Applied]:
    """Apply registered shims (idempotent). Returns what was applied this call."""
    _discover()
    done: list[Applied] = []
    for s in _REGISTRY.values():
        if only is not None and s.name not in only:
            continue
        if s.name in _APPLIED:
            continue
        try:
            mod = importlib.import_module(s.module)
        except ImportError as e:
            raise ShimError(f"shim {s.name}: cannot import target module {s.module}") from e
        try:
            owner = s.owner(mod)
        except AttributeError as e:
            raise ShimError(
                f"shim {s.name}: {s.target} does not exist. Upstream probably moved or "
                f"removed it -- check {s.upstream} and either retarget or delete the shim."
            ) from e
        if s.kind == "polyfill":
            if hasattr(owner, s.attr):
                logger.info("[shim] %s: %s already exists, polyfill skipped", s.name, s.target)
                _APPLIED[s.name] = Applied(name=s.name, target=s.target, kind=s.kind)
                continue
            patched = s.fn(None)
            setattr(owner, s.attr, patched)
            patched.__ascend_shim__ = s.name  # type: ignore[attr-defined]
            a = Applied(name=s.name, target=s.target, kind=s.kind)
            _APPLIED[s.name] = a
            done.append(a)
            logger.info("[shim] %s (polyfill) -> %s  [%s]", s.name, s.target, s.upstream)
            continue
        if not hasattr(owner, s.attr):
            raise ShimError(
                f"shim {s.name}: {s.target} does not exist. Upstream probably moved or "
                f"removed it -- check {s.upstream} and either retarget or delete the shim."
            )
        original = getattr(owner, s.attr)
        patched = s.fn(original)
        setattr(owner, s.attr, patched)
        patched.__ascend_shim__ = s.name  # type: ignore[attr-defined]
        a = Applied(name=s.name, target=s.target, kind=s.kind)
        _APPLIED[s.name] = a
        done.append(a)
        logger.info("[shim] %s (%s) -> %s  [%s]", s.name, s.kind, s.target, s.upstream)
    return done


def applied() -> list[Applied]:
    return list(_APPLIED.values())


def reset_for_tests() -> None:
    """Clear registry and applied state. Tests only; does not undo setattr."""
    _REGISTRY.clear()
    _APPLIED.clear()
