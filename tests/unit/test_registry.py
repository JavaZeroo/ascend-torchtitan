import types

import pytest

from ascend_titan.compat import ShimError, shim

UP = "https://github.com/pytorch/torchtitan/issues/0"


def _fake_module(name: str, **attrs):
    import sys

    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


def test_wrap_shim_calls_original(clean_registry):
    _fake_module("fake_target_a", f=lambda x: x + 1)

    @shim(target="fake_target_a:f", reason="test", upstream=UP)
    def double_after(original):
        return lambda x: original(x) * 2

    applied = clean_registry.apply_all()
    import fake_target_a

    assert [a.name for a in applied] == ["double_after"]
    assert fake_target_a.f(1) == 4
    assert fake_target_a.f.__ascend_shim__ == "double_after"


def test_apply_is_idempotent(clean_registry):
    _fake_module("fake_target_b", f=lambda: 1)

    @shim(target="fake_target_b:f", reason="test", upstream=UP)
    def once(original):
        return lambda: original() + 1

    assert len(clean_registry.apply_all()) == 1
    assert clean_registry.apply_all() == []
    import fake_target_b

    assert fake_target_b.f() == 2  # not 3


def test_replace_requires_why_not_wrap(clean_registry):
    with pytest.raises(ShimError, match="why_not_wrap"):

        @shim(target="x:y", reason="r", upstream=UP, kind="replace")
        def bad(original):
            return original


def test_upstream_link_required(clean_registry):
    with pytest.raises(ShimError, match="upstream"):

        @shim(target="x:y", reason="r", upstream="  ")
        def bad(original):
            return original


def test_bad_target_format(clean_registry):
    with pytest.raises(ShimError, match="module:attr"):

        @shim(target="no_colon", reason="r", upstream=UP)
        def bad(original):
            return original


def test_missing_attr_points_at_upstream(clean_registry):
    _fake_module("fake_target_c")

    @shim(target="fake_target_c:gone", reason="r", upstream=UP)
    def s(original):
        return original

    with pytest.raises(ShimError, match="Upstream probably moved"):
        clean_registry.apply_all()


def test_duplicate_name_rejected(clean_registry):
    @shim(target="a:b", reason="r", upstream=UP)
    def dup(original):
        return original

    with pytest.raises(ShimError, match="duplicate"):

        @shim(target="c:d", reason="r", upstream=UP)  # noqa: F811
        def dup(original):
            return original
