"""Every L1 module must import cleanly without torch_npu and register nothing (ADR-004)."""

import importlib
import sys

import pytest

MODULES = [
    "ascend_titan.kernels.attention",
    "ascend_titan.kernels.rms_norm",
    "ascend_titan.kernels.swiglu",
    "ascend_titan.kernels.situ_glu",
]


@pytest.mark.parametrize("name", MODULES)
def test_import_without_torch_npu(monkeypatch, name):
    monkeypatch.setitem(sys.modules, "torch_npu", None)
    monkeypatch.setitem(sys.modules, "cann_ops_nn", None)
    monkeypatch.delitem(sys.modules, name, raising=False)
    mod = importlib.import_module(name)
    assert mod._AVAILABLE is False


@pytest.mark.titan
def test_rope_module_falls_back_without_kernel(monkeypatch):
    """rope.py is always importable (pure torch); the kernel path is opt-in on npu."""
    monkeypatch.setitem(sys.modules, "torch_npu", None)
    monkeypatch.delitem(sys.modules, "ascend_titan.kernels.rope", raising=False)
    mod = importlib.import_module("ascend_titan.kernels.rope")
    assert mod._HAS_ROTARY_KERNEL is False
    import torch

    assert mod._use_kernel(torch.zeros(1)) is False
