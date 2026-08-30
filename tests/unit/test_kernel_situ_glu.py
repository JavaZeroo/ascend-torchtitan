import importlib

import pytest


@pytest.mark.titan
def test_import_safe_without_ops_nn(npu_stub, no_ops_nn):
    """ops-nn is an optional add-on (ADR-004), unlike torch_npu (P14)."""
    mod = importlib.import_module("ascend_titan.kernels.situ_glu")
    assert mod._AVAILABLE is False
    assert not hasattr(mod, "ops_nn_situ_glu")


@pytest.mark.titan
def test_probe_discovers_vendor_suffixed_extension(npu_stub):
    """ops-nn installs as cann_ops_nn_<vendor>; the probe must not hard-code the name."""
    from ascend_titan.kernels._probe import installed_modules

    names = installed_modules("cann_ops_nn")
    assert all(n.startswith("cann_ops_nn") for n in names)
