import importlib.util
import sys
import types

import pytest

# Ops the L1 modules require at import time (P14: torch_npu is a base dependency,
# so CPU tests must *provide* it -- there is no fallback path left to exercise).
NPU_OPS = (
    "npu_fusion_attention",
    "npu_fusion_attention_grad",
    "npu_rms_norm",
    "npu_swiglu",
    "npu_rotary_mul",
)


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def pytest_collection_modifyitems(config, items):
    skip_titan = pytest.mark.skip(reason="torchtitan not installed")
    skip_npu = pytest.mark.skip(reason="torch_npu not installed")
    for item in items:
        if "titan" in item.keywords and not _has("torchtitan"):
            item.add_marker(skip_titan)
        if "npu" in item.keywords and not _has("torch_npu"):
            item.add_marker(skip_npu)


@pytest.fixture
def clean_registry():
    """Fresh shim registry for each test; discovery of real shims disabled."""
    from ascend_titan.compat import registry

    registry.reset_for_tests()
    orig = registry._discover
    registry._discover = lambda: None
    yield registry
    registry._discover = orig
    registry.reset_for_tests()


def _make_torch_npu_stub(without: str | None = None) -> types.ModuleType:
    """A fake ``torch_npu`` exposing every op we require (minus ``without``).

    The ops raise when called: CPU tests must never reach a kernel.
    """
    fake = types.ModuleType("torch_npu")
    fake.__version__ = "0.0.0+stub"
    for op in NPU_OPS:
        if op == without:
            continue

        def boom(*args, _op=op, **kwargs):
            raise AssertionError(f"torch_npu.{_op} called on CPU")

        setattr(fake, op, boom)
    return fake


def _forget_kernel_modules(monkeypatch) -> None:
    for name in [m for m in list(sys.modules) if m.startswith("ascend_titan.kernels")]:
        monkeypatch.delitem(sys.modules, name, raising=False)


@pytest.fixture
def forget_kernels(monkeypatch):
    """Drop cached ascend_titan.kernels.* so the next import re-runs the probes."""
    _forget_kernel_modules(monkeypatch)
    return lambda: _forget_kernel_modules(monkeypatch)


@pytest.fixture
def npu_stub(monkeypatch, forget_kernels):
    """Install a fake ``torch_npu`` unless a real one is present.

    torch_npu is a base dependency (P14): L1 modules import it unconditionally,
    so CPU tests must provide it -- there is no fallback path left to exercise.
    """
    if _has("torch_npu"):
        return sys.modules.get("torch_npu")
    fake = _make_torch_npu_stub()
    monkeypatch.setitem(sys.modules, "torch_npu", fake)
    return fake


@pytest.fixture
def npu_stub_missing_op(monkeypatch, forget_kernels):
    """Factory: install a torch_npu stub that lacks one op."""

    def install(op: str):
        monkeypatch.setitem(sys.modules, "torch_npu", _make_torch_npu_stub(without=op))
        forget_kernels()

    return install


@pytest.fixture
def no_torch_npu(monkeypatch, forget_kernels):
    """Make ``import torch_npu`` fail."""
    monkeypatch.setitem(sys.modules, "torch_npu", None)


@pytest.fixture
def no_ops_nn(monkeypatch):
    """Block every ops-nn torch extension, whatever vendor suffix it carries.

    The wheel is named ``cann_ops_nn_<vendor>``, so a test that only blocks the
    bare name passes on a machine without ops-nn and fails on one with it.
    """
    import pkgutil

    for name in [m.name for m in pkgutil.iter_modules() if m.name.startswith("cann_ops_nn")]:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "cann_ops_nn", None)
    for name in [m for m in list(sys.modules) if m.startswith("ascend_titan.kernels")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
