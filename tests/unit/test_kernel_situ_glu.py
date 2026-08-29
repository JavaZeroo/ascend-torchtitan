import importlib
import sys


def test_import_safe_without_ops_nn(monkeypatch):
    monkeypatch.setitem(sys.modules, "cann_ops_nn", None)
    monkeypatch.delitem(sys.modules, "ascend_titan.kernels.situ_glu", raising=False)
    mod = importlib.import_module("ascend_titan.kernels.situ_glu")
    assert mod._AVAILABLE is False
    assert not hasattr(mod, "ops_nn_situ_glu")
