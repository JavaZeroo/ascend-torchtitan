"""CPU tests for the Ascend attention override: import safety and everything
except the kernel call itself (which lives in tests/npu)."""

import importlib
import logging
import sys
import types

import pytest


def _reload(monkeypatch, fake_npu: bool):
    """Import ascend_titan.kernels.attention fresh, with or without a fake torch_npu."""
    monkeypatch.delitem(sys.modules, "ascend_titan.kernels.attention", raising=False)
    monkeypatch.delitem(sys.modules, "ascend_titan.kernels.rms_norm", raising=False)
    if fake_npu:
        fake = types.ModuleType("torch_npu")

        def _boom(*a, **k):
            raise AssertionError("kernel called")

        fake.npu_fusion_attention = _boom
        fake.npu_rms_norm = _boom
        monkeypatch.setitem(sys.modules, "torch_npu", fake)
    else:
        monkeypatch.setitem(sys.modules, "torch_npu", None)  # makes `import torch_npu` fail
    mod = importlib.import_module("ascend_titan.kernels.attention")
    if fake_npu:
        importlib.import_module("ascend_titan.kernels.rms_norm")
    return mod


def test_import_without_torch_npu_is_safe_and_loud(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING):
        mod = _reload(monkeypatch, fake_npu=False)
    assert mod._AVAILABLE is False
    assert not hasattr(mod, "npu_fusion_attention")
    assert any("torch_npu unavailable" in r.message for r in caplog.records)


@pytest.mark.titan
def test_override_registers_and_derives(monkeypatch):
    from torchtitan.config.override import clear_overrides
    from torchtitan.models.common.attention import VarlenAttention

    clear_overrides()
    mod = _reload(monkeypatch, fake_npu=True)
    assert mod._AVAILABLE
    cfg = VarlenAttention.Config(window_size=(-1, 0))
    new = mod.npu_fusion_attention(cfg)
    assert isinstance(new, mod.AscendFusionAttention.Config)
    assert isinstance(new, VarlenAttention.Config)  # decoder's isinstance dispatch still works
    assert new.window_size == (-1, 0)
    module = new.build()
    assert type(module).__name__ == "AscendFusionAttention"
    clear_overrides()


@pytest.mark.titan
def test_unsupported_window_rejected_at_build(monkeypatch):
    from torchtitan.config.override import clear_overrides

    clear_overrides()
    mod = _reload(monkeypatch, fake_npu=True)
    with pytest.raises(NotImplementedError, match="window_size"):
        mod.AscendFusionAttention.Config(window_size=(-1, -1)).build()
    clear_overrides()


@pytest.mark.titan
def test_recipe_activates_override_on_config_tree(monkeypatch):
    """apply_overrides must find our module and swap every VarlenAttention node."""
    from torchtitan.config.override import apply_overrides, clear_overrides

    from ascend_titan.recipes.qwen3 import qwen3_debugmodel_npu

    clear_overrides()
    _reload(monkeypatch, fake_npu=True)
    cfg = qwen3_debugmodel_npu()
    assert "ascend_titan.kernels.attention.npu_fusion_attention" in cfg.override.imports
    apply_overrides(cfg.override, cfg)
    inner = cfg.model_spec.model.layers[0].attention.inner_attention
    assert type(inner).__qualname__ == "AscendFusionAttention.Config"
    clear_overrides()
