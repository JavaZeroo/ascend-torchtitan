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


def test_a_flavor_is_reachable_both_stock_and_with_our_deltas():
    """One module answers all three questions people ask about a model:

    stock upstream (the control), upstream + our deltas, and either of those with a
    hand-picked override set from the command line. Stock must not require switching
    to a different module with a longer name.
    """

    def flavor_a():
        return {"name": "a", "touched": False}

    mod = _fake_upstream(flavor_a=flavor_a)

    def deltas(config, flavor):
        config["touched"] = flavor

    getattr_, dir_ = _auto.npu_entry_points(mod, deltas)

    assert dir_() == ["flavor_a", "flavor_a_npu"]
    # bare name: upstream's own function, handed over untouched
    assert getattr_("flavor_a") is flavor_a
    assert getattr_("flavor_a")()["touched"] is False
    # _npu: the same config plus this family's deltas
    assert getattr_("flavor_a_npu")()["touched"] == "flavor_a"


def test_entry_point_applies_the_family_deltas():
    applied = []

    def flavor_a():
        return {"name": "a"}

    mod = _fake_upstream(flavor_a=flavor_a)
    getattr_, dir_ = _auto.npu_entry_points(mod, lambda cfg, flavor: applied.append(flavor))

    build = getattr_("flavor_a_npu")
    # the family declaration is handed the flavor name, so per-size data
    # (which HF tokenizer a real size uses) stays a table lookup, not a function
    assert build() == {"name": "a"} and applied == ["flavor_a"]

    with pytest.raises(AttributeError, match="has no config 'nope'"):
        getattr_("nope_npu")
    with pytest.raises(AttributeError, match="expected an upstream flavor"):
        getattr_("not_a_flavor")


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
    """Module dict beats __getattr__, so a curated preset is never shadowed.

    Only three qwen3_5 entries are still hand-written; the rest (including the
    real-size ``qwen35_0_8b_npu``) are generated, which is the point.
    """
    pytest.importorskip("fla", reason="pip install fla-core (qwen3_5 extra)")
    import ascend_titan.models.qwen3_5 as pkg

    # curated: swaps in the fused GDN sibling, which no generated entry does
    from ascend_titan.kernels import GDN_FUSED_OVERRIDE

    assert pkg.qwen35_0_8b_npu_fused.__module__.endswith("recipes")
    assert GDN_FUSED_OVERRIDE in pkg.qwen35_0_8b_npu_fused().override.imports

    # generated: same real assets, without a function existing for it
    generated = pkg.qwen35_0_8b_npu
    assert generated.__name__ == "qwen35_0_8b_npu"
    assert "Qwen3.5-0.8B" in generated().hf_assets_path


def test_real_size_assets_are_a_table_lookup_not_a_function():
    """Adding a real size should be one line in HF_REPOS, not another function."""
    pytest.importorskip("fla", reason="pip install fla-core (qwen3_5 extra)")
    import ascend_titan.models.qwen3 as p3
    import ascend_titan.models.qwen3_5 as p35

    for pkg in (p3, p35):
        repos = pkg.recipes.HF_REPOS
        assert repos, "family must declare its real-size tokenizers"
        for flavor in repos:
            entry = getattr(pkg, f"{flavor}_npu")
            assert callable(entry), f"{flavor} is not an upstream flavor: dead HF_REPOS row"


def test_no_parallelism_only_recipes_remain():
    """Parallelism is CLI-addressable, so one function per layout is copy-paste.

    Text-only check: importing the model packages would drag in optional extras.
    """
    import re
    from pathlib import Path

    keep = ("config.parallelism", "config =", "return", "npu_deltas(", "_use_real_assets(")
    offenders = []
    for path in sorted(Path(_auto.__file__).parent.rglob("recipes.py")):
        for match in re.finditer(r"^def (\w+)\(.*?(?=^def |\Z)", path.read_text(), re.M | re.S):
            name, body = match.group(1), match.group(0)
            if not name.endswith(("_fsdp2", "_tp2", "_pp2", "_ep2")):
                continue
            lines = [ln.strip() for ln in body.splitlines()[1:] if ln.strip()]
            meat = [ln for ln in lines if not ln.startswith("#") and not ln.startswith('"')]
            if meat and all(ln.startswith(keep) for ln in meat):
                offenders.append(f"{path.parent.name}.{name}")
    assert not offenders, f"parallelism-only recipes; use the CLI instead: {offenders}"
