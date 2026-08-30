import importlib
import sys

import pytest


@pytest.mark.titan
def test_import_safe_without_ops_nn(monkeypatch, npu_stub):
    """ops-nn is an optional add-on (ADR-004), unlike torch_npu (P14)."""
    monkeypatch.setitem(sys.modules, "cann_ops_nn", None)
    mod = importlib.import_module("ascend_titan.kernels.situ_glu")
    assert mod._AVAILABLE is False
    assert not hasattr(mod, "ops_nn_situ_glu")
