import importlib.util

import pytest


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def pytest_collection_modifyitems(config, items):
    skip_titan = pytest.mark.skip(reason="torchtitan not installed")
    skip_npu = pytest.mark.skip(reason="torch_npu not installed")
    for item in items:
        if "titan" in item.keywords and not _has("torchtitan"):
            item.add_marker(skip_titan)
        if "npu" in item.keywords and not _has("torch_npu"):
            item.add_marker(skip_npu)


@pytest.fixture
def clean_registry():
    """Fresh shim registry for each test; discovery of real shims disabled."""
    from ascend_titan.compat import registry

    registry.reset_for_tests()
    orig = registry._discover
    registry._discover = lambda: None
    yield registry
    registry._discover = orig
    registry.reset_for_tests()
