"""Every model package must be registered, documented and importable.

This is what keeps ``ascend_titan/models/`` honest as models are added: a new
directory without a README, or a registry entry pointing at a module that does
not exist, fails here rather than six months later when someone tries to use it.
"""

import importlib
import pathlib

import pytest

from ascend_titan.models.registry import MODELS, STATUS, table

MODELS_DIR = pathlib.Path(__file__).resolve().parents[2] / "ascend_titan" / "models"


def _packages() -> list[str]:
    return sorted(
        p.name for p in MODELS_DIR.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))
    )


def test_every_package_is_registered():
    missing = set(_packages()) - set(MODELS)
    assert not missing, f"add a ModelEntry for: {sorted(missing)}"


def test_every_package_has_a_readme_and_recipes():
    for name in _packages():
        assert (MODELS_DIR / name / "README.md").is_file(), f"{name}: README.md is required"
        assert (MODELS_DIR / name / "recipes.py").is_file(), f"{name}: recipes.py is required"
        assert (MODELS_DIR / name / "__init__.py").is_file(), f"{name}: __init__.py is required"


def test_registry_entries_are_consistent():
    for key, e in MODELS.items():
        assert key == e.name
        assert e.status in STATUS, f"{key}: status {e.status!r} is not one of {STATUS}"
        # 🔴 without an attribution is exactly the "we don't know why" cell P2 forbids.
        if e.status == "🔴":
            assert e.blocker, f"{key}: a 🔴 entry must name its blocker and attribution"
        if e.recipes:
            assert e.name in e.recipes
            assert (MODELS_DIR / e.name).is_dir(), f"{key}: declares recipes but has no package"
            # find_spec would import the parent package, and a 🔴 model is 🔴
            # precisely because that import raises (fla / cutlass). Check the file.
            path = MODELS_DIR.parent.parent / (e.recipes.replace(".", "/") + ".py")
            assert path.is_file(), f"{key}: {e.recipes} does not exist"


def test_template_is_not_importable():
    """The template holds placeholders; .txt keeps it out of import/lint/collection."""
    assert not (MODELS_DIR / "_template" / "recipes.py").exists()
    assert (MODELS_DIR / "_template" / "recipes.py.txt").is_file()
    assert (MODELS_DIR / "_template" / "README.md").is_file()


def test_table_renders_every_model():
    md = table()
    for e in MODELS.values():
        assert e.title in md


@pytest.mark.titan
@pytest.mark.parametrize("name", ["qwen3", "llama3"])
def test_runnable_models_import(name):
    """Models we claim run must at least import; blocked ones are excluded on purpose."""
    mod = importlib.import_module(f"ascend_titan.models.{name}")
    assert mod.__all__
