import ascend_titan
from ascend_titan import _bootstrap


def test_setup_is_idempotent_and_reports(monkeypatch):
    _bootstrap.reset_for_tests()
    r1 = ascend_titan.setup(apply_shims=False)
    r2 = ascend_titan.setup(apply_shims=False)
    assert r1 is r2
    assert r1.torch_version
    assert "torch=" in r1.summary()
    _bootstrap.reset_for_tests()


def test_setup_warns_if_torchtitan_imported_first():
    import pytest

    pytest.importorskip("torchtitan")
    _bootstrap.reset_for_tests()
    import torchtitan.tools.utils  # noqa: F401

    r = ascend_titan.setup(apply_shims=False)
    assert r.torchtitan_was_already_imported
    assert any("frozen" in w for w in r.warnings)
    _bootstrap.reset_for_tests()
