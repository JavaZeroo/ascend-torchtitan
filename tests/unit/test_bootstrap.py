import pytest

import ascend_titan
from ascend_titan import _bootstrap


def test_setup_is_idempotent_and_reports(npu_stub):
    _bootstrap.reset_for_tests()
    r1 = ascend_titan.setup(apply_shims=False)
    r2 = ascend_titan.setup(apply_shims=False)
    assert r1 is r2
    assert r1.torch_version
    assert "torch=" in r1.summary()
    _bootstrap.reset_for_tests()


def test_setup_warns_if_torchtitan_imported_first(npu_stub):
    pytest.importorskip("torchtitan")
    _bootstrap.reset_for_tests()
    import torchtitan.tools.utils  # noqa: F401

    r = ascend_titan.setup(apply_shims=False)
    assert r.torchtitan_was_already_imported
    assert any("frozen" in w for w in r.warnings)
    _bootstrap.reset_for_tests()


def test_setup_without_torch_npu_raises(no_torch_npu):
    """torch_npu is a base dependency (P14): no silent CPU-only mode."""
    _bootstrap.reset_for_tests()
    with pytest.raises(ImportError):
        ascend_titan.setup(apply_shims=False)
    _bootstrap.reset_for_tests()


def test_setup_raises_when_backend_registered_but_device_unusable(monkeypatch, npu_stub):
    """The confusing failure this replaces: torchtitan freezes device_type=cuda and
    dies much later with `torch._C has no attribute '_cuda_setDevice'`."""
    import torch

    _bootstrap.reset_for_tests()
    monkeypatch.setattr(torch, "npu", npu_stub, raising=False)
    monkeypatch.setattr("torch._utils._get_available_device_type", lambda: None)
    with pytest.raises(RuntimeError, match="device_type=None"):
        ascend_titan.setup(apply_shims=False)
    _bootstrap.reset_for_tests()
