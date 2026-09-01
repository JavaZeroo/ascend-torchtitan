"""Every upstream flavor must have an NPU entry point without a function per flavor.

Upstream qwen3_5 alone ships nine sizes and keeps adding them; a wrapper per flavor
is a copy-paste that goes stale the moment ``qwen35_397b_a17b`` lands.
"""

from types import ModuleType

import pytest

from ascend_titan.models import _auto

pytestmark = pytest.mark.titan


def _fake_upstream(**fns):
    mod = ModuleType("fake.config_registry")
    for name, fn in fns.items():
        fn.__module__ = mod.__name__
        setattr(mod, name, fn)
    return mod


def test_only_zero_arg_factories_defined_in_the_module_count():
    def flavor_a():
        return {"built": "a"}

    def needs_an_argument(size):
        return {}

    imported = lambda: {}  # noqa: E731 - stands in for an imported symbol
    imported.__module__ = "somewhere.else"

    mod = _fake_upstream(flavor_a=flavor_a, needs_an_argument=needs_an_argument)
    mod.imported = imported
    mod._private = lambda: {}

    assert set(_auto.upstream_flavors(mod)) == {"flavor_a"}


def test_entry_point_applies_the_family_deltas():
    applied = []

    def flavor_a():
        return {"name": "a"}

    mod = _fake_upstream(flavor_a=flavor_a)
    getattr_, dir_ = _auto.npu_entry_points(mod, lambda cfg: applied.append(cfg["name"]))

    assert dir_() == ["flavor_a_npu"]
    build = getattr_("flavor_a_npu")
    assert build() == {"name": "a"} and applied == ["a"]

    with pytest.raises(AttributeError, match="has no config 'nope'"):
        getattr_("nope_npu")
    with pytest.raises(AttributeError):
        getattr_("flavor_a")  # missing the _npu suffix


def test_every_upstream_qwen3_5_flavor_is_reachable_and_gets_the_gdn_override():
    """The regression this closes: 7 of 9 flavors used to have no working NPU path."""
    pytest.importorskip("fla", reason="pip install fla-core (qwen3_5 extra)")
    import ascend_titan.models.qwen3_5 as pkg
    from ascend_titan.kernels import GDN_OVERRIDE

    names = dir(pkg)
    assert "qwen35_27b_npu" in names and "qwen35_397b_a17b_npu" in names
    for name in ("qwen35_27b_npu", "qwen35_9b_npu"):
        config = getattr(pkg, name)()
        assert GDN_OVERRIDE in config.override.imports, f"{name} would run fla's CUDA kernels"


def test_a_hand_written_recipe_wins_over_the_generated_one():
    """Module dict beats __getattr__, so a curated preset is never shadowed."""
    pytest.importorskip("fla", reason="pip install fla-core (qwen3_5 extra)")
    import ascend_titan.models.qwen3_5 as pkg

    assert pkg.qwen35_0_8b_npu.__module__.endswith("recipes")
    # the curated one carries deltas the generated entry point does not
    assert pkg.qwen35_0_8b_npu().dataloader.collator is not None
